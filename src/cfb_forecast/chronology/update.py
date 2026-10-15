"""Append all currently-due games to the chronological ledger."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..data import normalize_schedule, sha256, timestamp
from ..forecast import advance_forecast_snapshot, forecast
from ..holdout import freeze_holdout
from ._shared import LEDGER_COLUMNS, ledger_hash, market_status, read_ledger
from .manifest import load_or_init_manifest
from .select import select_due_games

log = logging.getLogger(__name__)


def _advance_or_full_forecast(
    manifest: dict,
    ledger_root: Path,
    work: Path,
    *,
    data_root,
    model_root,
    season: int,
    at,
    through,
    market_path,
    draws: int,
    market_provider,
    target_game_ids,
):
    """Prefer the exact-state advance; fall back to a full replay on refusal.

    ``advance_forecast_snapshot`` verifies every prior source hash and refuses
    to cross a newly-available final result. On that specific refusal we drop
    the partial output and rebuild from scratch; anything else propagates.
    """
    runs = manifest.get("runs", [])
    if runs:
        previous = ledger_root / runs[-1]["frozen_run_path"] / "forecast"
        if previous.exists():
            try:
                return advance_forecast_snapshot(
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
                    target_game_ids=target_game_ids,
                )
            except ValueError as exc:
                if "result(s) became available" not in str(exc):
                    raise
                log.info("advance refused due to new results; falling back to full replay")
                shutil.rmtree(work)
                work.mkdir(parents=True)
    return forecast(
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
        target_game_ids=target_game_ids,
    )


def _ledger_row(
    rec: dict,
    *,
    due_index: pd.DataFrame,
    at,
    run_id: str,
    run_integrity: str,
    lead_hours: float,
    primary_lateness_minutes: float,
) -> dict:
    """Turn one forecast record into one ledger row."""
    gid = int(rec["game_id"])
    sched = due_index.loc[gid]
    kickoff = timestamp(rec["kickoff"])
    scheduled_cutoff = timestamp(sched.scheduled_cutoff)
    delay_minutes = max(0.0, (at - scheduled_cutoff).total_seconds() / 60.0)
    actual_lead_hours = (kickoff - at).total_seconds() / 3600.0
    return {
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


def run_chronological_update(
    data_root,
    model_root,
    ledger_root,
    season: int,
    now=None,
    lead_hours: float = 24.0,
    market_path=None,
    draws: int = 50000,
    market_provider=None,
    primary_lateness_minutes: float = 60.0,
):
    """Append all currently due games to a chronological prospective ledger.

    ``now`` is the real forecast timestamp. If the runner is started after a
    game's scheduled cutoff, the actual timestamp is retained and lateness is
    recorded; the forecast is never backdated to make it look on-time.

    Returns a tuple ``(combined_ledger, added_rows, run_record)``. When no
    games are due the run record reports ``status="no_due_games"`` and no run
    directory is created.
    """
    root = Path(ledger_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "work").mkdir(exist_ok=True)
    at = timestamp(now or datetime.now(UTC))
    model_file = Path(model_root) / "model.json"
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    model_sha = sha256(model_file)
    manifest = load_or_init_manifest(root, season, lead_hours, model_sha, primary_lateness_minutes)
    ledger = read_ledger(root)

    # Picking due games only needs the schedule. Avoiding the full data
    # ingestion here keeps prospective collection running in minimal runtimes
    # while preserving source hashes once a full replay is actually required.
    schedule_paths = sorted(Path(data_root).glob("schedule_*.csv"))
    if not schedule_paths:
        raise FileNotFoundError("No schedule_*.csv files found")
    schedules = normalize_schedule(pd.concat([pd.read_csv(p) for p in schedule_paths], ignore_index=True))
    done = ledger.game_id.astype(int).tolist() if len(ledger) else []
    due = select_due_games(schedules, season, at, lead_hours, done)
    log.info(
        "chronology.update: season=%s now=%s due=%d already=%d", season, at.isoformat(), len(due), len(done)
    )
    if due.empty:
        return (
            ledger.copy(),
            pd.DataFrame(),
            {"status": "no_due_games", "as_of": at.isoformat(), "games_added": 0},
        )

    run_id = f"run_{at.strftime('%Y%m%dT%H%M%SZ')}"
    if any(r.get("run_id") == run_id for r in manifest.get("runs", [])):
        raise FileExistsError(f"Chronological run already exists: {run_id}")
    work = root / "work" / run_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    through = due.start_date.max() + pd.Timedelta(seconds=1)

    pred = _advance_or_full_forecast(
        manifest,
        root,
        work,
        data_root=data_root,
        model_root=model_root,
        season=season,
        at=at,
        through=through,
        market_path=market_path,
        draws=draws,
        market_provider=market_provider,
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
    rows = [
        _ledger_row(
            rec,
            due_index=due_index,
            at=at,
            run_id=run_id,
            run_integrity=receipt["integrity_sha256"],
            lead_hours=lead_hours,
            primary_lateness_minutes=primary_lateness_minutes,
        )
        for rec in pred.to_dict("records")
    ]

    added = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    combined = pd.concat([ledger, added], ignore_index=True)
    combined["_order"] = pd.to_datetime(combined.actual_forecast_at, utc=True)
    combined = combined.sort_values(["_order", "scheduled_cutoff", "kickoff", "game_id"]).drop(
        columns="_order"
    )
    if combined.game_id.duplicated().any():
        raise ValueError("Chronological update attempted to duplicate a game forecast")
    if (
        pd.to_datetime(combined.actual_forecast_at, utc=True) >= pd.to_datetime(combined.kickoff, utc=True)
    ).any():
        raise ValueError("Chronological ledger contains a forecast at/after kickoff")
    prior_hash = ledger_hash(root / "ledger.csv")
    combined.to_csv(root / "ledger.csv", index=False)
    ledger_sha = sha256(root / "ledger.csv")

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
        "ledger_sha256_after": ledger_sha,
        "run_integrity_sha256": receipt["integrity_sha256"],
        "frozen_run_path": str(frozen.relative_to(root)),
    }
    manifest["runs"].append(run_rec)
    manifest["last_updated_at"] = at.isoformat()
    manifest["ledger_sha256"] = ledger_sha
    manifest["forecasted_games"] = int(len(combined))
    (root / "chronology_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("chronology.update: appended %d games as run %s", len(added), run_id)
    return combined, added, run_rec
