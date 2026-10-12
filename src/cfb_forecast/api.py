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

from .database import connect, insert_forecasts, list_forecasts, register_model
from .forecast import forecast

DATA_ROOT = Path(os.getenv("CFB_DATA_DIR", "/app/data"))
ARTIFACT_ROOT = Path(os.getenv("CFB_ARTIFACTS_DIR", "/app/artifacts"))
MODEL_ROOT = Path(os.getenv("CFB_MODEL_DIR", str(ARTIFACT_ROOT / "backtest" / "model")))
DATABASE_PATH = Path(os.getenv("CFB_DATABASE", str(ARTIFACT_ROOT / "cfb_forecast.sqlite3")))

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
        connect(DATABASE_PATH)
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
    model_path = Path(request.model_path) if request.model_path else MODEL_ROOT
    model = _model_manifest(model_path)
    if model.get("training_through_season", request.season) >= request.season:
        raise HTTPException(status_code=400, detail="Established model must be trained before forecast season")
    output = ARTIFACT_ROOT / "api-forecasts" / f"{request.season}-{request.as_of.replace(':', '').replace('+', '')}"
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
    conn = connect(DATABASE_PATH)
    model_id = register_model(conn, model_path.name, str(model_path), manifest)
    insert_forecasts(conn, model_id, frame.to_dict(orient="records"))
    return {"model_id": model_id, "count": len(frame), "forecasts": _records(frame)}


@app.get("/api/forecasts")
def get_forecasts(season: int | None = Query(default=None)):
    conn = connect(DATABASE_PATH)
    return {"forecasts": [dict(row) for row in list_forecasts(conn, season)]}


def run() -> None:
    import uvicorn
    uvicorn.run("cfb_forecast.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    run()
