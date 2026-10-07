"""Append-only chronological prospective forecasting ledger.

Unlike a batch forecast that scores every game from one shared as-of state, this
workflow only forecasts games whose configured pregame cutoff has actually
arrived. Each update is frozen as an immutable holdout run with the exact model,
data snapshot, market snapshot and score distribution used at that moment.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

from .data import load_data, normalize_schedule, sha256, timestamp
from .forecast import forecast, advance_forecast_snapshot
from .holdout import freeze_holdout, settle_holdout, verify_holdout
from .evaluate import metrics


LEDGER_COLUMNS = [
    "game_id", "season", "away_team", "home_team", "kickoff",
    "scheduled_cutoff", "actual_forecast_at", "target_lead_hours",
    "actual_lead_hours", "cutoff_delay_minutes", "primary_cutoff_eligible",
    "run_id", "run_integrity_sha256", "model_variant",
    "home_win_probability", "projected_home_margin", "projected_total",
    "expected_home_score", "expected_away_score", "market_status",
    "market_observed_at", "market_provider", "market_snapshot_count",
    "home_spread", "total_line", "market_home_win_probability_no_vig",
]


def _read_json(path: Path):
    return json.loads(path.read_text())


def _ledger_hash(path: Path):
    if not path.exists():
        return None
    return sha256(path)


def select_due_games(schedules: pd.DataFrame, season: int, now, lead_hours: float,
                     already_forecasted=()) -> pd.DataFrame:
    """Select only not-yet-forecast games whose fixed lead-time cutoff is due."""
    at = timestamp(now)
    lead = pd.Timedelta(hours=float(lead_hours))
    f = schedules[(schedules.fbs_game) & (schedules.season == int(season))].copy()
    f["scheduled_cutoff"] = f.start_date - lead
    done = {int(x) for x in already_forecasted}
    due = f[(f.start_date > at) & (f.scheduled_cutoff <= at)]
    if done:
        due = due[~due.game_id.astype(int).isin(done)]
    return due.sort_values(["scheduled_cutoff", "start_date", "game_id"]).reset_index(drop=True)


def _new_manifest(root: Path, season: int, lead_hours: float, model_sha: str,
                  primary_lateness_minutes: float):
    return {
        "schema_version": 1,
        "status": "active_chronological_prospective_holdout",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "season": int(season),
        "target_lead_hours": float(lead_hours),
        "primary_lateness_minutes": float(primary_lateness_minutes),
        "model_sha256": model_sha,
        "policy": (
            "Append-only event-time ledger. A game is forecast only after its configured "
            "pregame cutoff arrives and before kickoff. Every update freezes exact model/data/market "
            "inputs and full score distributions. Forecast timestamps are never backdated."
        ),
        "runs": [],
    }


def _load_or_init_manifest(root: Path, season: int, lead_hours: float, model_sha: str,
                           primary_lateness_minutes: float):
    path = root / "chronology_manifest.json"
    if not path.exists():
        m = _new_manifest(root, season, lead_hours, model_sha, primary_lateness_minutes)
        path.write_text(json.dumps(m, indent=2))
        return m
    m = _read_json(path)
    if int(m["season"]) != int(season):
        raise ValueError("Chronological ledger season cannot change")
    if not np.isclose(float(m["target_lead_hours"]), float(lead_hours)):
        raise ValueError("Chronological ledger lead-time policy cannot change")
    if m["model_sha256"] != model_sha:
        raise ValueError("Chronological prospective holdout model is frozen; start a new ledger for a new model")
    if not np.isclose(float(m.get("primary_lateness_minutes", primary_lateness_minutes)), float(primary_lateness_minutes)):
        raise ValueError("Primary lateness tolerance cannot change inside one ledger")
    return m


def _read_ledger(root: Path) -> pd.DataFrame:
    p = root / "ledger.csv"
    if not p.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    f = pd.read_csv(p)
    if f.game_id.duplicated().any():
        raise ValueError("Chronological ledger contains duplicate game forecasts")
    return f


def _market_status(row):
    observed = row.get("market_observed_at")
    if observed is None or (isinstance(observed, float) and np.isnan(observed)) or pd.isna(observed):
        return "missing_at_forecast"
    return "captured_at_or_before_forecast"


def run_chronological_update(data_root, model_root, ledger_root, season: int,
                             now=None, lead_hours: float = 24.0, market_path=None,
                             draws: int = 50000, market_provider=None,
                             primary_lateness_minutes: float = 60.0):
    """Append all currently due games to a chronological prospective ledger.

    `now` is the real forecast timestamp. If the runner is started after a game's
    scheduled cutoff, the actual timestamp is retained and lateness is recorded;
    the forecast is never backdated to make it look on-time.
    """
    root = Path(ledger_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "work").mkdir(exist_ok=True)
    at = timestamp(now or datetime.now(timezone.utc))
    model_file = Path(model_root) / "model.json"
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    model_sha = sha256(model_file)
    manifest = _load_or_init_manifest(root, season, lead_hours, model_sha, primary_lateness_minutes)
    ledger = _read_ledger(root)

    # Selecting due games only needs the schedule. Avoid reopening Parquet inputs
    # unless a full state replay is actually necessary. This keeps prospective
    # collection operational in minimal runtimes while preserving source hashes.
    schedule_paths = sorted(Path(data_root).glob("schedule_*.csv"))
    if not schedule_paths:
        raise FileNotFoundError("No schedule_*.csv files found")
    schedules = normalize_schedule(pd.concat([pd.read_csv(p) for p in schedule_paths], ignore_index=True))
    done = ledger.game_id.astype(int).tolist() if len(ledger) else []
    due = select_due_games(schedules, season, at, lead_hours, done)
    if due.empty:
        return ledger.copy(), pd.DataFrame(), {"status": "no_due_games", "as_of": at.isoformat(), "games_added": 0}

    run_id = f"run_{at.strftime('%Y%m%dT%H%M%SZ')}"
    if any(r.get("run_id") == run_id for r in manifest.get("runs", [])):
        raise FileExistsError(f"Chronological run already exists: {run_id}")
    work = root / "work" / run_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    through = due.start_date.max() + pd.Timedelta(seconds=1)

    # Prefer an exact saved-state transition when possible. It verifies every
    # prior source hash and refuses to cross a newly available final result.
    # If that refusal occurs, fall back to the full replay path.
    pred = None
    runs = manifest.get("runs", [])
    if runs:
        previous = root / runs[-1]["frozen_run_path"] / "forecast"
        if previous.exists():
            try:
                pred = advance_forecast_snapshot(
                    previous_forecast_dir=previous,
                    data_root=data_root,
                    model_root=model_root,
                    out_root=work,
                    as_of=at,
                    through=through,
                    season=season,
                    market_path=market_path,
                    draws=draws,
                    market_provider=market_provider,
                    save_distribution=True,
                    target_game_ids=due.game_id.astype(int).tolist(),
                )
            except ValueError as exc:
                if "result(s) became available" not in str(exc):
                    raise
                shutil.rmtree(work)
                work.mkdir(parents=True)
    if pred is None:
        pred = forecast(
            data_root=data_root,
            model_root=model_root,
            out_root=work,
            as_of=at,
            through=through,
            season=season,
            market_path=market_path,
            draws=draws,
            market_provider=market_provider,
            save_distribution=True,
            target_game_ids=due.game_id.astype(int).tolist(),
        )
    frozen, receipt = freeze_holdout(
        forecast_dir=work,
        data_root=data_root,
        model_root=model_root,
        holdout_root=root / "runs",
        batch_id=run_id,
        freeze_at=at,
    )
    shutil.rmtree(work)

    due_index = due.set_index(due.game_id.astype(int))
    rows = []
    for rec in pred.to_dict("records"):
        gid = int(rec["game_id"])
        sched = due_index.loc[gid]
        kickoff = timestamp(rec["kickoff"])
        scheduled_cutoff = timestamp(sched.scheduled_cutoff)
        delay_minutes = max(0.0, (at - scheduled_cutoff).total_seconds() / 60.0)
        actual_lead_hours = (kickoff - at).total_seconds() / 3600.0
        row = {
            "game_id": gid,
            "season": int(rec["season"]),
            "away_team": rec["away_team"],
            "home_team": rec["home_team"],
            "kickoff": kickoff.isoformat(),
            "scheduled_cutoff": scheduled_cutoff.isoformat(),
            "actual_forecast_at": at.isoformat(),
            "target_lead_hours": float(lead_hours),
            "actual_lead_hours": actual_lead_hours,
            "cutoff_delay_minutes": delay_minutes,
            "primary_cutoff_eligible": bool(delay_minutes <= float(primary_lateness_minutes)),
            "run_id": run_id,
            "run_integrity_sha256": receipt["integrity_sha256"],
            "model_variant": rec.get("model_variant"),
            "home_win_probability": rec.get("home_win_probability"),
            "projected_home_margin": rec.get("projected_home_margin"),
            "projected_total": rec.get("projected_total"),
            "expected_home_score": rec.get("expected_home_score"),
            "expected_away_score": rec.get("expected_away_score"),
            "market_status": _market_status(rec),
            "market_observed_at": rec.get("market_observed_at"),
            "market_provider": rec.get("market_provider"),
            "market_snapshot_count": rec.get("market_snapshot_count", 0),
            "home_spread": rec.get("home_spread"),
            "total_line": rec.get("total_line"),
            "market_home_win_probability_no_vig": rec.get("market_home_win_probability_no_vig"),
        }
        rows.append(row)

    added = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    combined = pd.concat([ledger, added], ignore_index=True)
    combined["_order"] = pd.to_datetime(combined.actual_forecast_at, utc=True)
    combined = combined.sort_values(["_order", "scheduled_cutoff", "kickoff", "game_id"]).drop(columns="_order")
    if combined.game_id.duplicated().any():
        raise ValueError("Chronological update attempted to duplicate a game forecast")
    if (pd.to_datetime(combined.actual_forecast_at, utc=True) >= pd.to_datetime(combined.kickoff, utc=True)).any():
        raise ValueError("Chronological ledger contains a forecast at/after kickoff")
    prior_hash = _ledger_hash(root / "ledger.csv")
    combined.to_csv(root / "ledger.csv", index=False)
    ledger_hash = sha256(root / "ledger.csv")

    run_rec = {
        "run_id": run_id,
        "actual_forecast_at": at.isoformat(),
        "games_added": int(len(added)),
        "game_ids": added.game_id.astype(int).tolist(),
        "earliest_kickoff": pd.to_datetime(added.kickoff, utc=True).min().isoformat(),
        "latest_kickoff": pd.to_datetime(added.kickoff, utc=True).max().isoformat(),
        "primary_cutoff_eligible_games": int(added.primary_cutoff_eligible.sum()),
        "late_games": int((~added.primary_cutoff_eligible).sum()),
        "market_games": int((added.market_status == "captured_at_or_before_forecast").sum()),
        "prior_ledger_sha256": prior_hash,
        "ledger_sha256_after": ledger_hash,
        "run_integrity_sha256": receipt["integrity_sha256"],
        "frozen_run_path": str(frozen.relative_to(root)),
    }
    manifest["runs"].append(run_rec)
    manifest["last_updated_at"] = at.isoformat()
    manifest["ledger_sha256"] = ledger_hash
    manifest["forecasted_games"] = int(len(combined))
    (root / "chronology_manifest.json").write_text(json.dumps(manifest, indent=2))
    return combined, added, run_rec


def verify_chronological_ledger(ledger_root):
    root = Path(ledger_root)
    manifest = _read_json(root / "chronology_manifest.json")
    ledger = _read_ledger(root)
    if sha256(root / "ledger.csv") != manifest.get("ledger_sha256"):
        raise ValueError("Chronological ledger hash mismatch")
    seen = set()
    last_at = None
    for run in manifest.get("runs", []):
        receipt = verify_holdout(root / run["frozen_run_path"])
        if receipt["integrity_sha256"] != run["run_integrity_sha256"]:
            raise ValueError(f"Run integrity mismatch: {run['run_id']}")
        at = timestamp(run["actual_forecast_at"])
        if last_at is not None and at < last_at:
            raise ValueError("Chronological runs are not ordered in event time")
        last_at = at
        ids = {int(x) for x in run["game_ids"]}
        if seen & ids:
            raise ValueError("A game appears in more than one chronological run")
        seen |= ids
    if seen != set(ledger.game_id.astype(int)):
        raise ValueError("Chronology manifest game IDs do not match ledger")
    return manifest


def settle_chronological_ledger(ledger_root, current_data_root, settled_at=None):
    root = Path(ledger_root)
    manifest = verify_chronological_ledger(root)
    ledger = _read_ledger(root)
    frames = []
    summaries = []
    for run in manifest.get("runs", []):
        games, summary = settle_holdout(root / run["frozen_run_path"], current_data_root, settled_at)
        if len(games):
            games["run_id"] = run["run_id"]
            frames.append(games)
        summaries.append(summary)
    settled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(settled):
        settled = settled.merge(
            ledger[["game_id", "scheduled_cutoff", "actual_forecast_at", "target_lead_hours",
                    "actual_lead_hours", "cutoff_delay_minutes", "primary_cutoff_eligible", "market_status",
                    "home_spread", "total_line", "market_home_win_probability_no_vig"]],
            on="game_id", how="left", validate="one_to_one"
        )
    out = root / "settlements"
    out.mkdir(exist_ok=True)
    tag = timestamp(settled_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    settled.to_csv(out / f"chronological_games_{tag}.csv", index=False)
    summary = {
        "settled_at": timestamp(settled_at or datetime.now(timezone.utc)).isoformat(),
        "forecasted_games": int(len(ledger)),
        "settled_games": int(len(settled)),
        "primary_cutoff_eligible_settled": int(settled.primary_cutoff_eligible.sum()) if len(settled) else 0,
        "market_available_settled": int((settled.market_status == "captured_at_or_before_forecast").sum()) if len(settled) else 0,
        "run_summaries": summaries,
    }
    def scored(frame):
        if frame.empty:
            return None
        return metrics(frame.home_win, frame.forecast_home_win_probability,
                       frame.projected_home_margin, frame.actual_margin,
                       frame.projected_total, frame.actual_total)
    if len(settled):
        summary["all_forecasts_metrics"] = scored(settled)
        primary = settled[settled.primary_cutoff_eligible.astype(bool)]
        summary["primary_fixed_cutoff_metrics"] = scored(primary)
        market_subset = settled[settled.market_status == "captured_at_or_before_forecast"]
        summary["market_available_model_metrics"] = scored(market_subset)
        if len(market_subset) and market_subset.market_home_win_probability_no_vig.notna().any():
            money = market_subset.dropna(subset=["market_home_win_probability_no_vig"])
            summary["market_no_vig_win_probability_metrics"] = metrics(
                money.home_win, money.market_home_win_probability_no_vig
            )
    (out / f"chronological_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    return settled, summary


def bootstrap_from_frozen_batch(frozen_batch, ledger_root, lead_hours: float = 24.0,
                                primary_lateness_minutes: float = 60.0):
    """Start a chronological ledger from an already-frozen pregame batch.

    Only games whose own lead-time cutoff had arrived at the batch's actual
    forecast timestamp are retained. Later games from the original slate are
    deliberately discarded from the chronological holdout rather than treated
    as if they had been forecast at their future cutoffs.
    """
    src = Path(frozen_batch)
    source_receipt = verify_holdout(src)
    actual_at = timestamp(source_receipt["forecast_as_of"])
    pred = pd.read_csv(src / "forecast" / "forecasts.csv")
    pred["kickoff_ts"] = pd.to_datetime(pred.kickoff, utc=True)
    pred["scheduled_cutoff_ts"] = pred.kickoff_ts - pd.Timedelta(hours=float(lead_hours))
    due = pred[(pred.scheduled_cutoff_ts <= actual_at) & (pred.kickoff_ts > actual_at)].copy()
    if due.empty:
        raise ValueError("Frozen batch contains no games whose chronological cutoff had arrived")

    root = Path(ledger_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Chronological bootstrap destination must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir()
    (root / "work").mkdir()
    model_root = src / "inputs" / "model"
    data_root = src / "inputs" / "data"
    model_sha = sha256(model_root / "model.json")
    season = int(due.season.iloc[0])
    manifest = _new_manifest(root, season, lead_hours, model_sha, primary_lateness_minutes)
    manifest["bootstrap_source_batch"] = str(src)
    manifest["bootstrap_source_integrity_sha256"] = source_receipt["integrity_sha256"]
    manifest["bootstrap_rule"] = (
        "Only games already due at the original batch's real forecast timestamp were imported; "
        "not-yet-due games were excluded from the chronological ledger."
    )

    run_id = f"run_{actual_at.strftime('%Y%m%dT%H%M%SZ')}"
    work = root / "work" / run_id
    work.mkdir(parents=True)
    keep = set(due.game_id.astype(int))

    # Preserve exact non-tabular state/availability artifacts from the source run.
    for name in ["data_availability.json", "state_snapshot.npz"]:
        p = src / "forecast" / name
        if p.exists():
            shutil.copy2(p, work / name)

    due.drop(columns=["kickoff_ts", "scheduled_cutoff_ts"]).to_csv(work / "forecasts.csv", index=False)
    (work / "forecasts.json").write_text(json.dumps(due.drop(columns=["kickoff_ts", "scheduled_cutoff_ts"]).to_dict("records"), indent=2))

    for stem in ["score_distribution", "margin_distribution", "total_distribution"]:
        csv = src / "forecast" / f"{stem}.csv"
        pq = src / "forecast" / f"{stem}.parquet"
        if csv.exists():
            f = pd.read_csv(csv)
            f[f.game_id.astype(int).isin(keep)].to_csv(work / csv.name, index=False)
        elif pq.exists():
            # Parquet filtering needs the normal runtime dependency. A CSV source
            # is bundled in the delivered bootstrap, so this branch is mainly for
            # future environments with pyarrow available.
            f = pd.read_parquet(pq)
            f[f.game_id.astype(int).isin(keep)].to_parquet(work / pq.name, index=False)

    fm = _read_json(src / "forecast" / "forecast_manifest.json")
    fm["target_game_ids"] = sorted(keep)
    fm["through"] = (due.kickoff_ts.max() + pd.Timedelta(seconds=1)).isoformat()
    fm["chronological_bootstrap"] = {
        "source_batch_id": source_receipt["batch_id"],
        "source_batch_integrity_sha256": source_receipt["integrity_sha256"],
        "retained_games": int(len(due)),
        "discarded_not_yet_due_games": int(len(pred) - len(due)),
        "actual_forecast_at": actual_at.isoformat(),
        "target_lead_hours": float(lead_hours),
    }
    (work / "forecast_manifest.json").write_text(json.dumps(fm, indent=2))

    frozen, receipt = freeze_holdout(
        forecast_dir=work,
        data_root=data_root,
        model_root=model_root,
        holdout_root=root / "runs",
        batch_id=run_id,
        freeze_at=actual_at,
    )
    shutil.rmtree(work)

    rows = []
    for rec in due.drop(columns=["kickoff_ts", "scheduled_cutoff_ts"]).to_dict("records"):
        kickoff = timestamp(rec["kickoff"])
        scheduled = kickoff - pd.Timedelta(hours=float(lead_hours))
        delay = max(0.0, (actual_at - scheduled).total_seconds() / 60.0)
        rows.append({
            "game_id": int(rec["game_id"]), "season": int(rec["season"]),
            "away_team": rec["away_team"], "home_team": rec["home_team"],
            "kickoff": kickoff.isoformat(), "scheduled_cutoff": scheduled.isoformat(),
            "actual_forecast_at": actual_at.isoformat(), "target_lead_hours": float(lead_hours),
            "actual_lead_hours": (kickoff - actual_at).total_seconds()/3600.0,
            "cutoff_delay_minutes": delay,
            "primary_cutoff_eligible": bool(delay <= float(primary_lateness_minutes)),
            "run_id": run_id, "run_integrity_sha256": receipt["integrity_sha256"],
            "model_variant": rec.get("model_variant"),
            "home_win_probability": rec.get("home_win_probability"),
            "projected_home_margin": rec.get("projected_home_margin"),
            "projected_total": rec.get("projected_total"),
            "expected_home_score": rec.get("expected_home_score"),
            "expected_away_score": rec.get("expected_away_score"),
            "market_status": _market_status(rec),
            "market_observed_at": rec.get("market_observed_at"),
            "market_provider": rec.get("market_provider"),
            "market_snapshot_count": rec.get("market_snapshot_count", 0),
            "home_spread": rec.get("home_spread"),
            "total_line": rec.get("total_line"),
            "market_home_win_probability_no_vig": rec.get("market_home_win_probability_no_vig"),
        })
    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS).sort_values(["scheduled_cutoff","kickoff","game_id"])
    ledger.to_csv(root / "ledger.csv", index=False)
    ledger_hash = sha256(root / "ledger.csv")
    run_rec = {
        "run_id": run_id,
        "actual_forecast_at": actual_at.isoformat(),
        "games_added": int(len(ledger)),
        "game_ids": ledger.game_id.astype(int).tolist(),
        "earliest_kickoff": pd.to_datetime(ledger.kickoff, utc=True).min().isoformat(),
        "latest_kickoff": pd.to_datetime(ledger.kickoff, utc=True).max().isoformat(),
        "primary_cutoff_eligible_games": int(ledger.primary_cutoff_eligible.sum()),
        "late_games": int((~ledger.primary_cutoff_eligible).sum()),
        "market_games": int((ledger.market_status == "captured_at_or_before_forecast").sum()),
        "prior_ledger_sha256": None,
        "ledger_sha256_after": ledger_hash,
        "run_integrity_sha256": receipt["integrity_sha256"],
        "frozen_run_path": str(frozen.relative_to(root)),
        "source_batch_id": source_receipt["batch_id"],
    }
    manifest["runs"] = [run_rec]
    manifest["last_updated_at"] = actual_at.isoformat()
    manifest["ledger_sha256"] = ledger_hash
    manifest["forecasted_games"] = int(len(ledger))
    (root / "chronology_manifest.json").write_text(json.dumps(manifest, indent=2))
    return ledger, manifest
