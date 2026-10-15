"""Verification and settlement of the chronological ledger.

Verification is a replay of the manifest's run order against the stored
holdout artifacts: every run must verify independently, no game appears in
more than one run, and the ledger CSV's hash must match the manifest.

Settlement is the end-of-season step that joins the ledger to final game
results and computes grading metrics on the whole ledger plus two
pre-registered subsets.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..data import sha256, timestamp
from ..evaluate import metrics
from ..holdout import settle_holdout, verify_holdout
from ._shared import read_json, read_ledger

log = logging.getLogger(__name__)

# Columns from the ledger that enrich the settlement output so grading can
# subset by cutoff eligibility and market availability without re-reading the
# ledger file.
_LEDGER_ENRICHMENT_COLUMNS = [
    "game_id",
    "scheduled_cutoff",
    "actual_forecast_at",
    "target_lead_hours",
    "actual_lead_hours",
    "cutoff_delay_minutes",
    "primary_cutoff_eligible",
    "market_status",
    "home_spread",
    "total_line",
    "market_home_win_probability_no_vig",
]


def verify_chronological_ledger(ledger_root) -> dict:
    """Verify every ledger invariant and return the manifest.

    Checks performed, in order: the ledger CSV's current SHA matches the
    manifest's recorded value; each run's frozen holdout verifies against its
    own receipt; runs are ordered in event time; no game appears in more than
    one run; and the game IDs seen across runs match the ledger exactly.
    """
    root = Path(ledger_root)
    manifest = read_json(root / "chronology_manifest.json")
    ledger = read_ledger(root)
    if sha256(root / "ledger.csv") != manifest.get("ledger_sha256"):
        raise ValueError("Chronological ledger hash mismatch")
    seen: set[int] = set()
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


def _scored(frame: pd.DataFrame) -> dict | None:
    """Compute grading metrics on a settled subset, or ``None`` if empty."""
    if frame.empty:
        return None
    return metrics(
        frame.home_win,
        frame.forecast_home_win_probability,
        frame.projected_home_margin,
        frame.actual_margin,
        frame.projected_total,
        frame.actual_total,
    )


def settle_chronological_ledger(ledger_root, current_data_root, settled_at=None):
    """Grade the ledger against final game results and write a summary.

    Produces two CSV outputs under ``settlements/``: the per-game joined table
    and a JSON summary with metrics on the whole ledger, on the primary
    cutoff-eligible subset, and on the market-available subset. A no-vig
    bookmaker baseline is reported when the market subset carries moneyline
    quotes.
    """
    root = Path(ledger_root)
    manifest = verify_chronological_ledger(root)
    ledger = read_ledger(root)
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
            ledger[_LEDGER_ENRICHMENT_COLUMNS],
            on="game_id",
            how="left",
            validate="one_to_one",
        )
    out = root / "settlements"
    out.mkdir(exist_ok=True)
    at = timestamp(settled_at or datetime.now(UTC))
    tag = at.strftime("%Y%m%dT%H%M%SZ")
    settled.to_csv(out / f"chronological_games_{tag}.csv", index=False)
    summary: dict = {
        "settled_at": at.isoformat(),
        "forecasted_games": int(len(ledger)),
        "settled_games": int(len(settled)),
        "primary_cutoff_eligible_settled": int(settled.primary_cutoff_eligible.sum()) if len(settled) else 0,
        "market_available_settled": int((settled.market_status == "captured_at_or_before_forecast").sum())
        if len(settled)
        else 0,
        "run_summaries": summaries,
    }
    if len(settled):
        summary["all_forecasts_metrics"] = _scored(settled)
        primary = settled[settled.primary_cutoff_eligible.astype(bool)]
        summary["primary_fixed_cutoff_metrics"] = _scored(primary)
        market_subset = settled[settled.market_status == "captured_at_or_before_forecast"]
        summary["market_available_model_metrics"] = _scored(market_subset)
        if len(market_subset) and market_subset.market_home_win_probability_no_vig.notna().any():
            money = market_subset.dropna(subset=["market_home_win_probability_no_vig"])
            summary["market_no_vig_win_probability_metrics"] = metrics(
                money.home_win, money.market_home_win_probability_no_vig
            )
    (out / f"chronological_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "chronology.settle: wrote %s and summary for %d games",
        f"chronological_games_{tag}.csv",
        len(settled),
    )
    return settled, summary
