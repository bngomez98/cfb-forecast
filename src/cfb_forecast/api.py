"""FastAPI application for the established college-football forecast model."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .database import connect, insert_forecasts, list_published_board, register_model
from .forecast import forecast

from .paths import data_root, artifacts_root, model_root, database_path

DATA_ROOT = data_root()
ARTIFACT_ROOT = artifacts_root()
MODEL_ROOT = model_root()
DATABASE_PATH = database_path()

app = FastAPI(title="CFB Win Probability Forecasting API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ForecastRequest(BaseModel):
    season: int = Field(..., ge=2000)
    as_of: str
    through: str
    model_path: str | None = None
    market_path: str | None = None
    draws: int = Field(50000, ge=1000, le=500000)


def _model_manifest(model_path: Path) -> dict[str, Any]:
    model_file = model_path / "model.json"
    if not model_file.exists():
        raise HTTPException(status_code=404, detail=f"Established model not found: {model_file}")
    return json.loads(model_file.read_text())


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


@app.get("/health")
def health():
    model = MODEL_ROOT / "model.json"
    return {"status": "ok", "model_available": model.exists(), "database": str(DATABASE_PATH)}


@app.get("/ready")
def ready():
    model = MODEL_ROOT / "model.json"
    if not model.exists():
        raise HTTPException(status_code=503, detail="Established model artifact unavailable")
    try:
        connect(DATABASE_PATH).close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
    return {"status": "ready", "model": str(model), "database": str(DATABASE_PATH)}


@app.get("/awareness")
def awareness():
    from .awareness import awareness_report
    report = awareness_report(DATA_ROOT, MODEL_ROOT, DATABASE_PATH)
    if report["overall"] != "pass":
        raise HTTPException(status_code=503, detail=report)
    return report


@app.get("/api/model")
def model_info():
    model = _model_manifest(MODEL_ROOT)
    return {
        "variant": model.get("variant"),
        "training_through_season": model.get("training_through_season"),
        "warmup_season": model.get("warmup_season"),
        "calibration_season": model.get("calibration_season"),
        "model_path": str(MODEL_ROOT),
        "established_model": True,
    }


@app.post("/api/forecasts")
def create_forecasts(request: ForecastRequest):
    from uuid import uuid4
    from .data import timestamp
    if request.model_path is not None or request.market_path is not None:
        raise HTTPException(status_code=400, detail="Server controls model and market artifact paths")
    try:
        at, through = timestamp(request.as_of), timestamp(request.through)
        if through <= at or through - at > pd.Timedelta(days=14):
            raise ValueError("Forecast window must be positive and no longer than 14 days")
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid forecast time window") from exc
    # Forecasts are fitted live from historical observations on every request.
    # The configured model path supplies priors and state hyperparameters only.
    model_path = MODEL_ROOT
    model = _model_manifest(model_path)
    if model.get("training_through_season", request.season) >= request.season:
        raise HTTPException(status_code=400, detail="Established model must be trained before forecast season")
    output = ARTIFACT_ROOT / "api-forecasts" / uuid4().hex
    try:
        frame = forecast(
            DATA_ROOT,
            model_path,
            output,
            request.as_of,
            request.through,
            request.season,
            request.market_path,
            request.draws,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    manifest = {"request": request.model_dump(), "model": model}
    from contextlib import closing
    from .data import sha256
    with closing(connect(DATABASE_PATH)) as conn:
        model_id = register_model(conn, sha256(model_path / "model.json"), str(model_path), manifest)
        insert_forecasts(conn, model_id, _records(frame))
    return {"model_id": model_id, "count": len(frame), "forecasts": _records(frame)}


@app.get("/api/forecasts")
def get_forecasts(season: int | None = Query(default=None)):
    from contextlib import closing
    with closing(connect(DATABASE_PATH)) as conn:
        return {"forecasts": [dict(row) for row in list_published_board(conn, season)]}


@app.get("/api/learning")
def learning_status():
    from contextlib import closing
    from .learning import SCHEMA
    with closing(connect(DATABASE_PATH)) as conn:
        conn.executescript(SCHEMA)
        rows = conn.execute("SELECT report_json FROM learning_versions ORDER BY created_at DESC LIMIT 20").fetchall()
        return {"versions": [json.loads(row[0]) for row in rows]}


@app.get("/api/pipeline")
def pipeline_status():
    root = ARTIFACT_ROOT / "latest"
    if not (root / "run_manifest.json").exists():
        raise HTTPException(status_code=404, detail="No completed pipeline run")
    return {"manifest": json.loads((root / "run_manifest.json").read_text()), "learning": json.loads((root / "learning_report.json").read_text()) if (root / "learning_report.json").exists() else None}


@app.get("/api/watchdog")
def watchdog_status():
    from contextlib import closing
    from .data_watchdog import WATCHDOG_SCHEMA
    with closing(connect(DATABASE_PATH)) as conn:
        conn.executescript(WATCHDOG_SCHEMA)
        row = conn.execute("SELECT details_json FROM watchdog_events WHERE event_type='check' ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="No watchdog check recorded")
        return json.loads(row[0])


def run() -> None:
    import uvicorn
    uvicorn.run("cfb_forecast.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    run()
