"""Read-only HTTP service over authoritative saved Python forecast runs.

This module never fits ratings and never calculates a new forecast. It exposes the
immutable output produced by ``cfb_forecast.sunday`` so web clients cannot drift
onto a second implementation.
"""

from __future__ import annotations

from pathlib import Path
import json
import os

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .data import normalize_schedule


app = FastAPI(title="CFB Forecast API", version="1")


def _artifact_root() -> Path:
    return Path(os.environ.get("CFB_ARTIFACTS_DIR", "artifacts"))


def _data_root() -> Path:
    return Path(os.environ.get("CFB_DATA_DIR", "data"))


def _latest() -> Path:
    root = _artifact_root() / "latest"
    if not root.exists():
        raise HTTPException(status_code=503, detail="No authoritative forecast run is published")
    return root


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def _schedule_index(season: int) -> dict[int, dict]:
    path = _data_root() / f"schedule_{season}.csv"
    if not path.exists():
        return {}
    frame = normalize_schedule(pd.read_csv(path))
    return {
        int(r.game_id): {
            "week": int(r.week),
            "home_id": int(r.home_id),
            "away_id": int(r.away_id),
            "home_team": r.home_team,
            "away_team": r.away_team,
        }
        for r in frame.itertuples(index=False)
    }


def _game_inputs() -> dict[int, dict]:
    rows = _read_json(_latest() / "inputs" / "game_inputs.json", [])
    return {int(row["game_id"]): row for row in rows if "game_id" in row}


def _number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(value) else value


def _forecast_payload(row: dict, schedule: dict, inputs: dict) -> dict:
    game_id = int(row["game_id"])
    meta = schedule.get(game_id, {})
    context = inputs.get(game_id, {})
    home_name = row.get("home_team") or meta.get("home_team") or str(meta.get("home_id", "home"))
    away_name = row.get("away_team") or meta.get("away_team") or str(meta.get("away_id", "away"))
    home_p = _number(row.get("home_win_probability"))
    if home_p is None:
        raise HTTPException(status_code=500, detail=f"Forecast {game_id} has no home win probability")
    return {
        "id": str(game_id),
        "date": row.get("kickoff"),
        "week": int(meta.get("week", 0)),
        "status": "pre",
        "home": {
            "id": str(meta.get("home_id", "")),
            "name": home_name,
            "abbreviation": home_name,
            "score": None,
            "rank": None,
        },
        "away": {
            "id": str(meta.get("away_id", "")),
            "name": away_name,
            "abbreviation": away_name,
            "score": None,
            "rank": None,
        },
        "venue": (context.get("venue") or {}).get("fullName") if isinstance(context.get("venue"), dict) else context.get("venue") or "Venue unavailable",
        "spread": _number(row.get("home_spread")),
        "marketTotal": _number(row.get("total_line")),
        "homeWin": home_p,
        "awayWin": 1.0 - home_p,
        "margin": _number(row.get("projected_home_margin")),
        "total": _number(row.get("projected_total")),
        "homeExpectedPoints": _number(row.get("expected_home_score")),
        "awayExpectedPoints": _number(row.get("expected_away_score")),
        "marginSd": _number(row.get("margin_sd")),
        "totalSd": _number(row.get("total_sd")),
        "cover": _number(row.get("home_cover_probability")),
        "over": _number(row.get("over_probability")),
        "kelly": _number(row.get("hypothetical_bankroll_fraction")),
        "poissonHomeWin": None,
        "model": row.get("prediction_method") or row.get("model_variant") or "authoritative Python forecast",
        "cutoff": row.get("as_of"),
        "inputsComplete": bool(row.get("inputs_complete", False)),
        "contextLogitContribution": _number(row.get("context_logit_contribution")),
        "probabilitySource": row.get("probability_source"),
        "forecastRecord": row,
    }


def _validation_payload() -> dict:
    report = _read_json(_latest() / "previous_week_validation.json", {})
    runs = report.get("runs", []) if isinstance(report, dict) else []
    if not runs:
        return {"games": 0, "accuracy": 0.0, "logLoss": 0.0, "spreadGames": 0, "spreadMae": 0.0, "rows": []}
    latest = runs[-1]
    return {
        "games": int(latest.get("games", latest.get("graded_games", 0)) or 0),
        "accuracy": float(latest.get("winner_accuracy", latest.get("accuracy", 0.0)) or 0.0),
        "logLoss": float(latest.get("log_loss", 0.0) or 0.0),
        "spreadGames": int(latest.get("spread_games", 0) or 0),
        "spreadMae": float(latest.get("margin_mae", latest.get("spread_mae", 0.0)) or 0.0),
        "rows": [],
        "raw": latest,
    }


@app.get("/health")
def health():
    latest = _artifact_root() / "latest"
    return {"status": "ok" if latest.exists() else "no_forecast", "authoritative_engine": "python"}


@app.get("/v1/forecast")
def forecast_feed(season: int = Query(default=2026), week: int | None = Query(default=None)):
    latest = _latest()
    rows = _read_json(latest / "forecasts.json", None)
    if rows is None:
        raise HTTPException(status_code=503, detail="Published run has no forecasts.json")
    schedule = _schedule_index(season)
    inputs = _game_inputs()
    forecasts = [_forecast_payload(row, schedule, inputs) for row in rows]
    if week is not None:
        forecasts = [row for row in forecasts if row["week"] == week]
    manifest = _read_json(latest / "run_manifest.json", {})
    return {
        "forecasts": forecasts,
        "ratings": [],
        "fittedGames": int(forecasts[0]["forecastRecord"].get("prior_fbs_games", 0)) if forecasts else 0,
        "backtest": _validation_payload(),
        "dataCutoff": manifest.get("as_of") or (forecasts[0]["cutoff"] if forecasts else None),
        "source": "authoritative saved Python forecast run",
        "sources": [],
        "run": manifest,
        "architecture": "Python inference -> immutable artifacts -> read-only API -> UI",
    }
