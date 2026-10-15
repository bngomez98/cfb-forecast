"""Prospective data-availability receipts for a forecast cutoff."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .data import ADV_FIELDS, sha256, timestamp


def _manifest(data_root: Path) -> dict:
    path = data_root / "source_manifest.json"
    if not path.exists():
        return {"files": {}}
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {"files": {}}


def availability_snapshot(data_root, as_of, schedules, summaries, paths) -> dict:
    """Describe exactly what source snapshot existed at forecast time.

    Current-season files with known downloaded_at timestamps must be captured no
    later than the forecast cutoff for a clean prospective receipt. Retained
    historical snapshots are allowed but explicitly marked as having unknown
    original capture times.
    """
    root = Path(data_root)
    at = timestamp(as_of)
    manifest = _manifest(root).get("files", {})
    files = []
    clean_current = True
    forecast_season = (
        int(schedules.loc[schedules.start_date > at, "season"].min())
        if (schedules.start_date > at).any()
        else int(schedules.season.max())
    )
    for p in paths:
        p = Path(p)
        meta = manifest.get(p.name, {})
        downloaded = meta.get("downloaded_at")
        if downloaded:
            captured = timestamp(downloaded) <= at
            status = "captured_before_cutoff" if captured else "captured_after_cutoff"
        elif meta.get("provenance"):
            captured = None
            status = "retained_historical_snapshot_capture_time_unknown"
        else:
            captured = None
            status = "capture_time_unknown"
        year = None
        for token in p.stem.split("_"):
            if token.isdigit() and len(token) == 4:
                year = int(token)
        if year == forecast_season and captured is not True:
            clean_current = False
        files.append(
            {
                "name": p.name,
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
                "source_url": meta.get("url"),
                "downloaded_at": downloaded,
                "capture_status": status,
                "prospective_current_season_eligible": bool(year != forecast_season or captured is True),
                "provenance": meta.get("provenance"),
            }
        )

    prior = schedules[
        schedules.fbs_game & schedules.result_usable & (schedules.result_available_at < at)
    ].copy()
    known = set(summaries.game_id.astype(int)) if summaries is not None and "game_id" in summaries else set()
    metric_coverage = {}
    if summaries is not None:
        for col in ADV_FIELDS[2:]:
            if col in summaries.columns:
                # two team rows are required for a complete game metric snapshot
                g = summaries.dropna(subset=[col]).groupby("game_id").size()
                available = set(g[g >= 2].index.astype(int))
                metric_coverage[col] = {
                    "prior_games_with_two_team_rows": int(prior.game_id.isin(available).sum()),
                    "prior_eligible_games": int(len(prior)),
                    "coverage": float(prior.game_id.isin(available).mean()) if len(prior) else None,
                }
    latest = prior.result_available_at.max() if len(prior) else pd.NaT
    return {
        "schema_version": 1,
        "as_of": at.isoformat(),
        "created_at": datetime.now(UTC).isoformat(),
        "forecast_season": forecast_season,
        "prospective_current_season_source_check": "pass" if clean_current else "fail",
        "prior_eligible_fbs_games": int(len(prior)),
        "prior_games_with_advanced_summary": int(prior.game_id.isin(known).sum()) if known else None,
        "latest_eligible_result_available_at": latest.isoformat() if pd.notna(latest) else None,
        "advanced_metric_coverage": metric_coverage,
        "files": files,
        "known_limitations": [
            "Retained historical files do not reconstruct the original publication time of every statistical correction.",
            "Advanced summaries are game-level provider outputs, not immutable row-level as-known snapshots.",
            "Prospective integrity is strongest from this freeze forward because the exact current-season source files are archived with the holdout.",
        ],
    }
