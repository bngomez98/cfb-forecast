"""Production Sunday service: collect inputs, validate prior runs, calculate forecasts."""

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import os
import shutil
import sys

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

import pandas as pd

from ..evaluation.backtest import run_backtest
from ..modeling.context_model import select_context_model, apply_score_context
from ..ingest.data import fetch, load_data, normalize_schedule, sha256, timestamp
from ..odds.decision import american_to_decimal, moneyline_diagnostics
from ..modeling.forecast import forecast
from ..modeling.forecast_model import METHOD
from ..odds.market import append_market_history
from ..ingest.results import _event_row
from ..ingest.sources import collect, collect_game_inputs, ESPN, utc_now
from ..modeling.strength import train_strength
from ..evaluation.validation import validate_runs


def slate_window(now, timezone_name="America/Chicago"):
    local = timestamp(now).to_pydatetime().astimezone(ZoneInfo(timezone_name))
    monday = local.date() + timedelta(
        days=1 if local.weekday() == 6 else -local.weekday()
    )
    start = datetime.combine(
        monday, datetime.min.time(), tzinfo=ZoneInfo(timezone_name)
    )
    return timestamp(start), timestamp(start + timedelta(days=7))


def select_slate(schedules, season, now, timezone_name="America/Chicago"):
    start, end = slate_window(now, timezone_name)
    at = timestamp(now)
    return schedules[
        schedules.fbs_game
        & (schedules.season == season)
        & ~schedules.completed
        & (schedules.start_date > at)
        & (schedules.start_date >= start)
        & (schedules.start_date < end)
    ].copy()


def sync_results(data_root, season, raw_root, now):
    """Update canonical outcomes by ESPN IDs and verify both team identities."""
    path = Path(data_root) / f"schedule_{season}.csv"
    raw = pd.read_csv(path)
    schedules = normalize_schedule(raw)
    at = timestamp(now)
    past = schedules[
        schedules.fbs_game & (schedules.season == season) & (schedules.start_date < at)
    ]
    if past.empty:
        return
    # Refresh only recent games; the downloaded schedule supplies earlier finals.
    start = max(past.start_date.min(), at - pd.Timedelta(days=14))
    dates = sorted(
        set(past.loc[past.start_date >= start, "start_date"].dt.strftime("%Y%m%d"))
        | set(
            (
                past.loc[past.start_date >= start, "start_date"] - pd.Timedelta(days=1)
            ).dt.strftime("%Y%m%d")
        )
    )

    def scoreboard(date):
        url = f"{ESPN}/scoreboard?dates={date}&groups=80&limit=500"
        payload, receipt = collect(url, Path(raw_root) / f"results_{date}.json")
        if payload is None:
            raise RuntimeError(f"Live results source failed: {receipt.get('error')}")
        return payload, receipt

    responses = list(ThreadPoolExecutor(max_workers=4).map(scoreboard, dates))
    events = [
        (event, receipt)
        for payload, receipt in responses
        for event in payload.get("events", [])
    ]
    if not events:
        raise RuntimeError("Live results source returned no scheduled games")
    raw["completed_at"] = raw.get(
        "completed_at", pd.Series(index=raw.index, dtype=object)
    )
    index = {int(gid): i for i, gid in enumerate(raw.game_id)}
    for event, receipt in events:
        if int(event.get("season", {}).get("year", 0)) != int(season):
            continue
        row = _event_row(event, receipt["retrieved_at"])
        if row is None or row["game_id"] not in index:
            continue
        i = index[row["game_id"]]
        if (int(raw.loc[i, "home_id"]), int(raw.loc[i, "away_id"])) != (
            int(row["home_id"]),
            int(row["away_id"]),
        ):
            raise ValueError(f"Live results team mismatch for game {row['game_id']}")
        if row["is_final"]:
            kickoff = timestamp(row["kickoff"])
            if (
                kickoff >= at
                or row["home_points"] is None
                or row["away_points"] is None
            ):
                raise ValueError(f"Invalid live final for {row['game_id']}")
            was_completed = bool(raw.loc[i, "completed"])
            raw.loc[i, ["completed", "home_points", "away_points"]] = [
                True,
                row["home_points"],
                row["away_points"],
            ]
            if not was_completed or pd.isna(raw.loc[i, "completed_at"]):
                raw.loc[i, "completed_at"] = receipt["retrieved_at"]
    normalize_schedule(raw)
    raw.to_csv(path, index=False)
    manifest_path = Path(data_root) / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    meta = manifest["files"][path.name]
    meta.update(
        sha256=sha256(path),
        bytes=path.stat().st_size,
        downloaded_at=utc_now(),
        results_sources=[r["url"] for _, r in responses],
        results_receipts=[r for _, r in responses],
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))


def capture_prices(games, input_root, store):
    snapshots = []
    for game in games.itertuples(index=False):
        path = Path(input_root) / "raw" / f"game_{int(game.game_id)}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        receipt = json.loads(path.with_suffix(".json.receipt.json").read_text())
        for quote in payload.get("pickcenter", []):
            provider = quote.get("provider", {}).get("name")
            if not provider:
                continue
            row = {
                "game_id": int(game.game_id),
                "provider": f"espn:{provider}",
                "observed_at": receipt["retrieved_at"],
                "source": receipt["url"],
            }
            spread = quote.get("spread")
            h, a = quote.get("homeTeamOdds", {}), quote.get("awayTeamOdds", {})
            if spread is not None and (
                h.get("favorite") or a.get("favorite") or float(spread) == 0
            ):
                row["home_spread"] = (
                    -abs(float(spread)) if h.get("favorite") else abs(float(spread))
                )
            if quote.get("overUnder") is not None:
                row["total_line"] = float(quote["overUnder"])
            for side, values in [("home", h), ("away", a)]:
                for field, key in [
                    ("moneyLine", f"{side}_ml_decimal"),
                    ("spreadOdds", f"{side}_decimal"),
                ]:
                    if values.get(field) is not None:
                        try:
                            row[key] = american_to_decimal(values[field])
                        except ValueError:
                            pass
            if any(
                k in row
                for k in [
                    "home_spread",
                    "total_line",
                    "home_ml_decimal",
                    "away_ml_decimal",
                ]
            ):
                snapshots.append(row)
    if snapshots:
        append_market_history(store, pd.DataFrame(snapshots))
    return len(snapshots)


def publish_latest(root, final):
    latest, temporary = root / "latest", root / ".latest-new"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(final, temporary)
    if latest.exists():
        shutil.rmtree(latest)
    temporary.rename(latest)


def sunday_run(
    data_root=None,
    artifacts_root=None,
    season=2026,
    model_root=None,
    timezone_name="America/Chicago",
    scheduled=False,
    refresh=True,
    adjustment_config=None,
):
    """Execute one real run; persist reports and preserve previous predictions."""
    started = timestamp(utc_now())
    local = started.to_pydatetime().astimezone(ZoneInfo(timezone_name))
    if scheduled and (local.weekday() != 6 or local.hour != 10):
        return {"status": "outside_sunday_schedule", "local_time": local.isoformat()}
    from ..paths import artifacts_root as default_artifacts, data_root as default_data, resolve_path
    root = resolve_path(artifacts_root) if artifacts_root is not None else default_artifacts()
    data = resolve_path(data_root) if data_root is not None else default_data()
    root.mkdir(parents=True, exist_ok=True)
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    monday, end = slate_window(started, timezone_name)
    run_id = f"{season}-{monday.tz_convert(timezone_name):%Y-%m-%d}"
    final = runs / run_id
    if final.exists():
        return {"status": "already_completed", "run_id": run_id}
    work = root / "work" / run_id
    work.mkdir(parents=True, exist_ok=True)
    lock = root / ".run.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    try:
        if sys.platform == "win32":
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                os.close(fd)
                raise RuntimeError(f"Another run is active; check {lock}") from None
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(f"Another run is active; check {lock}") from None
    from .pipeline_status import PipelineJournal
    journal = PipelineJournal(work / "pipeline.json")
    try:
        journal.stage("sources")
        if refresh:
            # Cached historical seasons are retained; the current season always refreshes.
            from ..config import HISTORY_START_SEASON
            fetch(data, list(range(HISTORY_START_SEASON, season)))
            fetch(data, [season], refresh=True)
            fetch(data, [season], raw_pbp=True)
            sync_results(data, season, work / "inputs" / "raw", timestamp(utc_now()))
            # Closing lines teach the forecast how lines relate to final scores.
            # Only completed games read them; upcoming games use captured quotes.
            from ..odds.market import backfill_sportsdataverse_closing
            backfill_sportsdataverse_closing(
                range(HISTORY_START_SEASON, season + 1),
                data / "market_closing_history.csv",
                skip_missing=True,
            )
        schedules, _, paths = load_data(data)
        now = timestamp(utc_now())
        if (schedules.completed & (schedules.start_date >= now)).any():
            raise ValueError(
                "A source marks future games completed; refusing those outcomes"
            )
        games = select_slate(schedules, season, now, timezone_name)
        journal.stage("context")
        # External game inputs are collected before feature/model calculations.
        context_features, _ = (
            collect_game_inputs(games, season, work / "inputs")
            if len(games)
            else (pd.DataFrame(), [])
        )
        from .database_update import update_database
        catalog_database = resolve_path(os.environ.get("CFB_DATABASE", root / "cfb_forecast.sqlite"))
        catalog = update_database(data, catalog_database, season, work / "inputs", fetch_sources=refresh)
        (work / "inputs" / "database_update.json").write_text(json.dumps(catalog, indent=2))
        store = data / "market_history.csv"
        prices = capture_prices(games, work / "inputs", store) if len(games) else 0
        from ..ingest.cfbd_market import capture_cfbd_prices
        prices += capture_cfbd_prices(games, season, work / "inputs" / "raw", store)
        at = timestamp(utc_now())
        if len(context_features):
            from ..ingest.player_matchups import import_player_evidence
            from ..modeling.forecast_model import KNOWN_COLUMNS
            from ..paths import database_path
            features_path = work / "inputs" / "features.csv"
            digest = sha256(features_path)
            context_rows = []
            for row in context_features.to_dict("records"):
                values = {key: float(row[key]) for key in KNOWN_COLUMNS
                          if key in row and row[key] is not None and pd.notna(row[key])}
                context_rows.append({"game_id": int(row["game_id"]), "observed_at": at.isoformat(),
                                     "provider": "archived_run_inputs", "source_url": features_path.resolve().as_uri(),
                                     "source_sha256": digest, "features": values})
            import_player_evidence(database_path(), {"contexts": context_rows})
        journal.stage("validation")
        validated, validation_summary = validate_runs(runs, schedules, at)
        (work / "previous_week_validation.json").write_text(
            json.dumps(validation_summary, indent=2)
        )
        from contextlib import closing
        from ..storage.database import connect, record_weekly_validation
        from ..paths import resolve_path, database_path
        validation_database = database_path()
        with closing(connect(validation_database)) as validation_db:
            for report in validation_summary.get("runs", []):
                if report.get("settled_games"):
                    record_weekly_validation(validation_db, {
                        **report,
                        "season": season,
                        "week": report.get("week", 0),
                        "games": report.get("settled_games", 0),
                        "status": "settled",
                    })
        history = []
        if len(validated):
            for old_id, batch in validated.groupby("run_id"):
                features_path = runs / old_id / "inputs" / "features.csv"
                if features_path.exists():
                    features = pd.read_csv(features_path)
                    # Only pregame saved features, base probability, labels and run identifiers enter fitting.
                    joined = features.merge(
                        batch[
                            [
                                "game_id",
                                "raw_home_win_probability",
                                "home_win",
                                "as_of",
                                "run_id",
                            ]
                        ],
                        on="game_id",
                        validate="one_to_one",
                    )
                    joined = joined.rename(
                        columns={"raw_home_win_probability": "base_probability"}
                    )
                    history.append(joined)
        context_spec, context_report = select_context_model(
            pd.concat(history, ignore_index=True) if history else pd.DataFrame()
        )
        (work / "context_model_selection.json").write_text(
            json.dumps(context_report, indent=2)
        )
        journal.stage("training")
        model_path = resolve_path(model_root) if model_root else resolve_path(os.environ.get("CFB_MODEL_DIR", root / "training" / "model"))
        if model_root is not None and not (model_path / "model.json").exists():
            raise FileNotFoundError(f"Requested model does not exist: {model_path}")
        if not (model_path / "model.json").exists():
            from ..config import HISTORY_START_SEASON
            on_disk = [
                year for year in range(HISTORY_START_SEASON, int(season))
                if (data / f"schedule_{year}.csv").exists() and (data / f"adv_team_{year}.parquet").exists()
            ]
            warmup = on_disk[0] if on_disk else 2021
            train_start = max(warmup + 1, 2022)
            if not warmup < train_start < 2024:
                warmup, train_start = 2021, 2022
            run_backtest(
                data,
                model_path.parent,
                warmup=warmup,
                train_start=train_start,
                validation=2024,
                test=2025,
            )
            strength = train_strength(data, model_path.parent / "strength")
            base = json.loads((model_path / "model.json").read_text())
            evaluation = pd.read_csv(model_path.parent / "metrics.csv")
            base_loss = float(
                evaluation.loc[evaluation.model == base["variant"], "log_loss"].iloc[0]
            )
            adopted = strength["evaluation"]["log_loss"] < base_loss
            selection = {
                "selection_for_season": season,
                "historical_comparison_season": 2025,
                "score_head_log_loss": base_loss,
                "strength_log_loss": strength["evaluation"]["log_loss"],
                "strength_adopted": adopted,
                "sets_win_chances": False,
                "note": "2025 results guide 2026 deployment; they are not an untouched evaluation of that deployment choice. The strength model is saved for comparison; win, spread and total chances come from the same Gaussian score distribution.",
            }
            if adopted:
                base["win_strength_model"] = strength
                (model_path / "model.json").write_text(json.dumps(base, indent=2))
            (model_path.parent / "winner_model_selection.json").write_text(
                json.dumps(selection, indent=2)
            )
        model_meta = json.loads((model_path / "model.json").read_text())
        if model_meta["training_through_season"] >= season:
            raise ValueError("Model training must end before the forecast season")
        from ..modeling.learning import learn
        database = database_path()
        learning = learn(data, model_path, root / "learning", database, at)
        model_path = Path(learning["model_path"])
        (work / "learning_report.json").write_text(json.dumps(learning, indent=2))
        if games.empty:
            receipt = {"status": "no_upcoming_fbs_games", "run_id": run_id, "season": season, "as_of": at.isoformat(), "games": 0, "validation": validation_summary, "model_sha256": sha256(model_path / "model.json"), "source_hashes": {p.name: sha256(p) for p in paths}}
            (work / "run_manifest.json").write_text(json.dumps(receipt, indent=2))
            from .publisher import REQUIRED
            pd.DataFrame(columns=REQUIRED).to_csv(work / "forecasts.csv", index=False)
            journal.stage("publication")
            journal.finish(no_games=True)
            shutil.move(str(work), final)
            publish_latest(root, final)
            from .publisher import publish_forecast_update
            receipt["distribution"] = publish_forecast_update(final, root, run_id=run_id, season=season)
            (final / "run_manifest.json").write_text(json.dumps(receipt, indent=2))
            (root / "latest" / "run_manifest.json").write_text(json.dumps(receipt, indent=2))
            return receipt
        at = timestamp(utc_now())
        journal.stage("probability")
        predictions = forecast(
            data,
            model_path,
            work,
            at,
            games.start_date.max(),
            season,
            market_path=store if store.exists() else None,
            target_game_ids=games.game_id.astype(int).tolist(),
            analytic=True,
            known_future=context_features,
            adjustment_config=adjustment_config,
            database=database,
        )
        ordered = (
            context_features.set_index("game_id")
            .reindex(predictions.game_id)
            .reset_index()
        )
        predictions["context_model_status"] = context_report["status"]
        predictions["context_logit_contribution"] = 0.0
        joint_players = "model_variant" in predictions and bool((predictions.model_variant == "joint_player").all())
        if joint_players:
            predictions["context_model_status"] = "integrated_in_joint_posterior"
        if context_spec and not joint_players:
            selected_model = json.loads((model_path / "model.json").read_text())
            predictions = apply_score_context(
                predictions, ordered, context_spec, selected_model["score_covariance"]
            )
            (work / "context_model.json").write_text(json.dumps(context_spec, indent=2))
        # Incomplete lineup/injury information affects eligibility, not an invented probability adjustment.
        journal.stage("decisions")
        from .decisions import apply_decisions
        predictions = apply_decisions(predictions, ordered.inputs_complete.to_numpy())
        journal.stage("publication")
        predictions.to_csv(work / "forecasts.csv", index=False)
        records = (
            predictions.astype(object)
            .where(pd.notna(predictions), None)
            .to_dict("records")
        )
        (work / "forecasts.json").write_text(
            json.dumps(records, indent=2, allow_nan=False)
        )
        receipt = {
            "schema_version": 1,
            "status": "completed",
            "run_id": run_id,
            "season": season,
            "started_at": started.isoformat(),
            "as_of": at.isoformat(),
            "completed_at": utc_now(),
            "timezone": timezone_name,
            "slate_start": monday.isoformat(),
            "slate_end": end.isoformat(),
            "games": len(predictions),
            "fbs_only": True,
            "price_rows": prices,
            "prediction_sha256": sha256(work / "forecasts.csv"),
            "model_sha256": sha256(model_path / "model.json"),
            "source_hashes": {p.name: sha256(p) for p in paths},
            "validation": validation_summary,
            "context_model_selection": context_report,
            "learning": learning,
            "inputs_collected_before_forecast": True,
            "input_hashes": {
                p.relative_to(work / "inputs").as_posix(): sha256(p)
                for p in (work / "inputs").rglob("*")
                if p.is_file()
            },
            "calculation_method": "joint conditional Gaussian team/player/opponent hierarchy" if joint_players else METHOD,
            "joint_model_path": str(final) if joint_players else None,
        }
        if timestamp(utc_now()) >= games.start_date.min():
            raise ValueError(
                "A target game started during calculation; rerun with the remaining slate"
            )
        from ..storage.database import persist_completed_run
        if joint_players:
            persist_completed_run(database, model_path, records, receipt, schedules, at, joint_model_root=work)
        else:
            persist_completed_run(database, model_path, records, receipt, schedules, at)
        journal.finish()
        shutil.copy2(model_path / "model.json", work / "model.json")
        (work / "run_manifest.json").write_text(json.dumps(receipt, indent=2))
        shutil.move(str(work), final)
        publish_latest(root, final)
        try:
            from .publisher import publish_forecast_update
            receipt["distribution"] = publish_forecast_update(final, root, run_id=run_id, season=season)
        except Exception as exc:
            receipt["distribution"] = {"status": "failed", "error": type(exc).__name__, "message": str(exc)}
        (final / "run_manifest.json").write_text(json.dumps(receipt, indent=2))
        latest_manifest = root / "latest" / "run_manifest.json"
        if latest_manifest.parent.exists():
            latest_manifest.write_text(json.dumps(receipt, indent=2))
        return receipt
    except Exception as error:
        if work.exists():
            journal.fail(error)
        raise
    finally:
        if sys.platform == "win32":
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
