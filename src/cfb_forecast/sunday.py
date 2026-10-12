"""Production Sunday service: collect inputs, validate prior runs, calculate forecasts."""

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import json
import os
import shutil
import sys

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

import pandas as pd

from .backtest import run_backtest
from .context_model import select_context_model, predict_context
from .data import fetch, load_data, normalize_schedule, sha256, timestamp
from .decision import american_to_decimal, moneyline_diagnostics
from .forecast import forecast
from .market import append_market_history
from .results import _event_row
from .sources import collect, collect_game_inputs, ESPN, utc_now
from .strength import train_strength
from .validation import validate_runs


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


def sunday_run(
    data_root="data",
    artifacts_root="artifacts",
    season=2026,
    model_root=None,
    timezone_name="America/Chicago",
    scheduled=False,
    refresh=True,
):
    """Execute one real run; persist reports and preserve previous predictions."""
    started = timestamp(utc_now())
    local = started.to_pydatetime().astimezone(ZoneInfo(timezone_name))
    if scheduled and (local.weekday() != 6 or local.hour != 10):
        return {"status": "outside_sunday_schedule", "local_time": local.isoformat()}
    root, data = Path(artifacts_root), Path(data_root)
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
    try:
        if refresh:
            # Cached historical seasons are retained; the current season always refreshes.
            fetch(data, list(range(2021, season)))
            fetch(data, [season], refresh=True)
            sync_results(data, season, work / "inputs" / "raw", timestamp(utc_now()))
        schedules, _, paths = load_data(data)
        now = timestamp(utc_now())
        if (schedules.completed & (schedules.start_date >= now)).any():
            raise ValueError(
                "A source marks future games completed; refusing those outcomes"
            )
        games = select_slate(schedules, season, now, timezone_name)
        # External game inputs are collected before feature/model calculations.
        context_features, _ = (
            collect_game_inputs(games, season, work / "inputs")
            if len(games)
            else (pd.DataFrame(), [])
        )
        store = data / "market_history.csv"
        prices = capture_prices(games, work / "inputs", store) if len(games) else 0
        at = timestamp(utc_now())
        validated, validation_summary = validate_runs(runs, schedules, at)
        (work / "previous_week_validation.json").write_text(
            json.dumps(validation_summary, indent=2)
        )
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
        if games.empty:
            receipt = {
                "status": "no_upcoming_fbs_games",
                "run_id": run_id,
                "season": season,
                "as_of": at.isoformat(),
                "validation": validation_summary,
            }
            (work / "run_manifest.json").write_text(json.dumps(receipt, indent=2))
            shutil.move(str(work), final)
            return receipt
        model_path = Path(model_root) if model_root else root / "training" / "model"
        if not (model_path / "model.json").exists():
            run_backtest(
                data,
                model_path.parent,
                warmup=2021,
                train_start=2022,
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
                "note": "2025 results guide 2026 deployment; they are not an untouched evaluation of that deployment choice.",
            }
            if adopted:
                base["win_strength_model"] = strength
                (model_path / "model.json").write_text(json.dumps(base, indent=2))
            (model_path.parent / "winner_model_selection.json").write_text(
                json.dumps(selection, indent=2)
            )
        at = timestamp(utc_now())
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
        )
        ordered = (
            context_features.set_index("game_id")
            .reindex(predictions.game_id)
            .reset_index()
        )
        predictions["context_model_status"] = context_report["status"]
        predictions["context_logit_contribution"] = 0.0
        if context_spec:
            (
                predictions["home_win_probability"],
                predictions["context_logit_contribution"],
            ) = predict_context(predictions.home_win_probability, ordered, context_spec)
            (work / "context_model.json").write_text(json.dumps(context_spec, indent=2))
        # Incomplete lineup/injury information affects eligibility, not an invented probability adjustment.
        predictions["inputs_complete"] = ordered.inputs_complete.to_numpy()
        decisions = []
        for row in predictions.to_dict("records"):
            h, a = row.get("home_ml_decimal"), row.get("away_ml_decimal")
            h = None if h is None or pd.isna(h) else h
            a = None if a is None or pd.isna(a) else a
            d = moneyline_diagnostics(row["home_win_probability"], h, a)
            if not row["inputs_complete"]:
                d.update(
                    moneyline_candidate="none",
                    hypothetical_bankroll_fraction=0.0,
                    decision_reason="incomplete_weather_injury_lineup_or_stats_inputs",
                )
            else:
                d["decision_reason"] = "price_and_probability_thresholds"
            decisions.append(d)
        for col in pd.DataFrame(decisions):
            predictions[col] = pd.DataFrame(decisions)[col].to_numpy()
        exposure = float(predictions.hypothetical_bankroll_fraction.sum())
        if exposure > 0.05:
            predictions.hypothetical_bankroll_fraction *= 0.05 / exposure
        predictions["slate_exposure_cap"] = 0.05
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
            "inputs_collected_before_forecast": True,
            "input_hashes": {
                p.relative_to(work / "inputs").as_posix(): sha256(p)
                for p in (work / "inputs").rglob("*")
                if p.is_file()
            },
            "calculation_method": "Gaussian score head; direct probability integration; no generated score samples",
        }
        if timestamp(utc_now()) >= games.start_date.min():
            raise ValueError(
                "A target game started during calculation; rerun with the remaining slate"
            )
        shutil.copy2(model_path / "model.json", work / "model.json")
        (work / "run_manifest.json").write_text(json.dumps(receipt, indent=2))
        shutil.move(str(work), final)
        latest = root / "latest"
        latest_tmp = root / ".latest-new"
        if latest_tmp.exists():
            shutil.rmtree(latest_tmp)
        shutil.copytree(final, latest_tmp)
        if latest.exists():
            shutil.rmtree(latest)
        latest_tmp.rename(latest)
        return receipt
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
