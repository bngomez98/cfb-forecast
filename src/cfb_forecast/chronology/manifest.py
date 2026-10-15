"""Chronological ledger manifest: on-disk record of policy and runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ._shared import read_json


def new_manifest(
    season: int,
    lead_hours: float,
    model_sha: str,
    primary_lateness_minutes: float,
) -> dict:
    """Build a fresh manifest for a brand-new chronological ledger."""
    return {
        "schema_version": 1,
        "status": "active_chronological_prospective_holdout",
        "created_at": datetime.now(UTC).isoformat(),
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


def load_or_init_manifest(
    root: Path,
    season: int,
    lead_hours: float,
    model_sha: str,
    primary_lateness_minutes: float,
) -> dict:
    """Load an existing manifest, or write a new one when none is present.

    When a manifest already exists, verifies that the fixed policy fields
    (season, lead hours, model hash, lateness tolerance) still match. The
    chronological holdout is only meaningful if those stay constant.
    """
    path = root / "chronology_manifest.json"
    if not path.exists():
        m = new_manifest(season, lead_hours, model_sha, primary_lateness_minutes)
        path.write_text(json.dumps(m, indent=2))
        return m
    m = read_json(path)
    if int(m["season"]) != int(season):
        raise ValueError("Chronological ledger season cannot change")
    if not np.isclose(float(m["target_lead_hours"]), float(lead_hours)):
        raise ValueError("Chronological ledger lead-time policy cannot change")
    if m["model_sha256"] != model_sha:
        raise ValueError(
            "Chronological prospective holdout model is frozen; start a new ledger for a new model"
        )
    if not np.isclose(
        float(m.get("primary_lateness_minutes", primary_lateness_minutes)),
        float(primary_lateness_minutes),
    ):
        raise ValueError("Primary lateness tolerance cannot change inside one ledger")
    return m
