"""Ledger schema and small helpers shared across the chronology submodules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import sha256

#: Columns of the ledger CSV, in the order they are written. New columns go at
#: the end so older ledger files remain readable.
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


def read_json(path: Path) -> dict:
    """Read a JSON file into a dict. Raises if the file does not exist."""
    return json.loads(path.read_text())


def ledger_hash(path: Path) -> str | None:
    """SHA-256 of the ledger CSV, or ``None`` when the file is absent."""
    if not path.exists():
        return None
    return sha256(path)


def read_ledger(root: Path) -> pd.DataFrame:
    """Read the ledger CSV, returning an empty frame when it does not exist.

    Fails if the ledger contains duplicate ``game_id`` rows: a chronological
    ledger must have exactly one forecast per game.
    """
    p = root / "ledger.csv"
    if not p.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    f = pd.read_csv(p)
    if f.game_id.duplicated().any():
        raise ValueError("Chronological ledger contains duplicate game forecasts")
    return f


def market_status(row: dict) -> str:
    """Classify how a forecast row sits relative to its market snapshot."""
    observed = row.get("market_observed_at")
    if observed is None or (isinstance(observed, float) and np.isnan(observed)) or pd.isna(observed):
        return "missing_at_forecast"
    return "captured_at_or_before_forecast"
