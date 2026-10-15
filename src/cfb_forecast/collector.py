"""Operational prospective collection runner.

A single collection cycle records the real execution timestamp, attempts a
point-in-time market capture, then appends only newly due games to the immutable
chronological holdout. Market failure is recorded but does not backdate, delay,
or mutate forecasts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .chronology import run_chronological_update
from .data import timestamp
from .market import capture_sportsdataverse_snapshot


def run_prospective_collection(
    data_root,
    model_root,
    ledger_root,
    season: int,
    now=None,
    lead_hours: float = 24.0,
    market_store="data/market_history.csv",
    market_provider=None,
    draws: int = 50000,
    primary_lateness_minutes: float = 60.0,
    capture_market: bool = True,
):
    """Run one prospective collection cycle and persist an auditable receipt.

    The same real `now` timestamp is used for both market observation and the
    forecast cutoff. If market capture fails, the chronology run proceeds with
    whatever older eligible market rows are already in the store; the failure is
    explicit in the receipt and no quote is invented.
    """
    at = timestamp(now or datetime.now(UTC))
    ledger_root = Path(ledger_root)
    receipt_dir = ledger_root / "collection_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "collection_at": at.isoformat(),
        "season": int(season),
        "target_lead_hours": float(lead_hours),
        "market_store": str(market_store),
        "market_capture": {"status": "skipped"},
    }

    if capture_market:
        try:
            history = capture_sportsdataverse_snapshot(season, market_store, observed_at=at)
            receipt["market_capture"] = {
                "status": "ok",
                "rows_in_store": int(len(history)),
                "provider": "sportsdataverse_espn_resolved",
                "observed_at": at.isoformat(),
            }
        except Exception as exc:  # preserve forecast collection even if public market source is down
            receipt["market_capture"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "observed_at": at.isoformat(),
            }

    ledger, added, chronology_receipt = run_chronological_update(
        data_root=data_root,
        model_root=model_root,
        ledger_root=ledger_root,
        season=season,
        now=at,
        lead_hours=lead_hours,
        market_path=market_store,
        draws=draws,
        market_provider=market_provider,
        primary_lateness_minutes=primary_lateness_minutes,
    )
    receipt["chronology"] = chronology_receipt
    receipt["games_added"] = int(len(added))
    receipt["ledger_games_after"] = int(len(ledger))
    receipt["game_ids_added"] = added.game_id.astype(int).tolist() if len(added) else []

    path = receipt_dir / f"collection_{at.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(receipt, indent=2))
    receipt["receipt_path"] = str(path)
    return ledger, added, receipt
