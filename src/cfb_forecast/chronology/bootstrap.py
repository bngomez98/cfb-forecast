"""Start a chronological ledger from an already-frozen pregame batch.

A pregame slate forecast is useful as a starting point for the chronological
ledger, but only the games whose own lead-time cutoff had actually arrived at
the batch's real forecast timestamp belong in a chronological holdout. Later
games on the same slate are discarded here rather than treated as if they had
been forecast at their future cutoffs; those are reforecast later by
``run_chronological_update`` when their cutoffs arrive.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from ..data import sha256, timestamp
from ..holdout import freeze_holdout, verify_holdout
from ._shared import LEDGER_COLUMNS, market_status
from .manifest import new_manifest

log = logging.getLogger(__name__)

_DISTRIBUTION_STEMS = ("score_distribution", "margin_distribution", "total_distribution")


def _copy_state_artifacts(src_forecast: Path, work: Path) -> None:
    """Carry non-tabular state artifacts (availability JSON, state snapshot) across."""
    for name in ("data_availability.json", "state_snapshot.npz"):
        p = src_forecast / name
        if p.exists():
            shutil.copy2(p, work / name)


def _filter_distribution_files(src_forecast: Path, work: Path, keep: set[int]) -> None:
    """Filter each distribution file to the retained game IDs.

    The CSV branch is the common case for bundled bootstrap sources. The
    parquet branch is used when the source batch was written with pyarrow
    available; we only hit it when the current runtime also has pyarrow.
    """
    for stem in _DISTRIBUTION_STEMS:
        csv = src_forecast / f"{stem}.csv"
        pq = src_forecast / f"{stem}.parquet"
        if csv.exists():
            f = pd.read_csv(csv)
            f[f.game_id.astype(int).isin(keep)].to_csv(work / csv.name, index=False)
        elif pq.exists():
            f = pd.read_parquet(pq)
            f[f.game_id.astype(int).isin(keep)].to_parquet(work / pq.name, index=False)


def _bootstrap_ledger_row(
    rec: dict,
    *,
    actual_at,
    lead_hours: float,
    primary_lateness_minutes: float,
    run_id: str,
    run_integrity: str,
) -> dict:
    kickoff = timestamp(rec["kickoff"])
    scheduled = kickoff - pd.Timedelta(hours=float(lead_hours))
    delay = max(0.0, (actual_at - scheduled).total_seconds() / 60.0)
    return {
        "game_id": int(rec["game_id"]),
        "season": int(rec["season"]),
        "away_team": rec["away_team"],
        "home_team": rec["home_team"],
        "kickoff": kickoff.isoformat(),
        "scheduled_cutoff": scheduled.isoformat(),
        "actual_forecast_at": actual_at.isoformat(),
        "target_lead_hours": float(lead_hours),
        "actual_lead_hours": (kickoff - actual_at).total_seconds() / 3600.0,
        "cutoff_delay_minutes": delay,
        "primary_cutoff_eligible": bool(delay <= float(primary_lateness_minutes)),
        "run_id": run_id,
        "run_integrity_sha256": run_integrity,
        "model_variant": rec.get("model_variant"),
        "home_win_probability": rec.get("home_win_probability"),
        "projected_home_margin": rec.get("projected_home_margin"),
        "projected_total": rec.get("projected_total"),
        "expected_home_score": rec.get("expected_home_score"),
        "expected_away_score": rec.get("expected_away_score"),
        "market_status": market_status(rec),
        "market_observed_at": rec.get("market_observed_at"),
        "market_provider": rec.get("market_provider"),
        "market_snapshot_count": rec.get("market_snapshot_count", 0),
        "home_spread": rec.get("home_spread"),
        "total_line": rec.get("total_line"),
        "market_home_win_probability_no_vig": rec.get("market_home_win_probability_no_vig"),
    }


def bootstrap_from_frozen_batch(
    frozen_batch,
    ledger_root,
    lead_hours: float = 24.0,
    primary_lateness_minutes: float = 60.0,
):
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
    manifest = new_manifest(season, lead_hours, model_sha, primary_lateness_minutes)
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

    _copy_state_artifacts(src / "forecast", work)

    due_clean = due.drop(columns=["kickoff_ts", "scheduled_cutoff_ts"])
    due_clean.to_csv(work / "forecasts.csv", index=False)
    (work / "forecasts.json").write_text(json.dumps(due_clean.to_dict("records"), indent=2))

    _filter_distribution_files(src / "forecast", work, keep)

    fm = json.loads((src / "forecast" / "forecast_manifest.json").read_text())
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

    rows = [
        _bootstrap_ledger_row(
            rec,
            actual_at=actual_at,
            lead_hours=lead_hours,
            primary_lateness_minutes=primary_lateness_minutes,
            run_id=run_id,
            run_integrity=receipt["integrity_sha256"],
        )
        for rec in due_clean.to_dict("records")
    ]
    ledger = (
        pd.DataFrame(rows, columns=LEDGER_COLUMNS)
        .sort_values(["scheduled_cutoff", "kickoff", "game_id"])
    )
    ledger.to_csv(root / "ledger.csv", index=False)
    ledger_sha = sha256(root / "ledger.csv")
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
        "ledger_sha256_after": ledger_sha,
        "run_integrity_sha256": receipt["integrity_sha256"],
        "frozen_run_path": str(frozen.relative_to(root)),
        "source_batch_id": source_receipt["batch_id"],
    }
    manifest["runs"] = [run_rec]
    manifest["last_updated_at"] = actual_at.isoformat()
    manifest["ledger_sha256"] = ledger_sha
    manifest["forecasted_games"] = int(len(ledger))
    (root / "chronology_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("chronology.bootstrap: imported %d games from %s", len(ledger), src)
    return ledger, manifest
