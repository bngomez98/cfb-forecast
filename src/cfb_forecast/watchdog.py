"""Persist completed forecast runs and watch the artifacts directory."""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from .database import connect, insert_forecasts, register_model


def _json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _row_payload(row: pd.Series, season: int) -> dict:
    data = {key: _json_value(value) for key, value in row.to_dict().items()}
    data.setdefault("season", season)
    data.setdefault("as_of", data.get("forecast_cutoff") or data.get("as_of") or "unknown")
    data.setdefault("kickoff", data.get("kickoff") or data.get("start_date") or "unknown")
    data.setdefault("home_team", data.get("home_team", ""))
    data.setdefault("away_team", data.get("away_team", ""))
    for key in (
        "expected_home_score", "expected_away_score", "projected_home_margin",
        "projected_total", "home_win_probability",
    ):
        data[key] = float(data.get(key) or 0.0)
    return data


def forecast_files(artifacts: str | Path) -> list[Path]:
    root = Path(artifacts)
    files = list(root.glob("latest/forecasts.csv"))
    files.extend(root.glob("runs/*/forecasts.csv"))
    return sorted({path.resolve() for path in files if path.is_file()})


def sync_file(
    conn: sqlite3.Connection,
    forecast_path: str | Path,
    database_path: str | Path,
    model_path: str | Path | None = None,
) -> int:
    path = Path(forecast_path)
    frame = pd.read_csv(path)
    if frame.empty:
        return 0
    season = int(frame["season"].iloc[0]) if "season" in frame else 0
    manifest_path = path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"source": str(path)}
    model_dir = Path(model_path) if model_path else path.parents[2] / "training" / "model"
    model_json = model_dir / "model.json"
    if not model_json.exists():
        model_dir = path.parents[1] / "model"
    if not (model_dir / "model.json").exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.json").write_text(json.dumps({
            "training_through_season": season,
            "variant": "saved-forecast",
        }))
    model_id = register_model(conn, f"saved-forecast-{season}", str(model_dir), manifest)
    rows = (_row_payload(row, season) for _, row in frame.iterrows())
    return insert_forecasts(conn, model_id, rows)


def sync_artifacts(artifacts: str | Path, database: str | Path, model_path: str | Path | None = None) -> dict:
    conn = connect(database)
    files = forecast_files(artifacts)
    inserted = sum(sync_file(conn, path, database, model_path) for path in files)
    conn.close()
    return {"database": str(database), "files_seen": len(files), "rows_inserted": inserted}


def watch(artifacts: str | Path, database: str | Path, model_path: str | Path | None, interval: float) -> None:
    seen: dict[str, int] = {}
    while True:
        for path in forecast_files(artifacts):
            stamp = path.stat().st_mtime_ns
            if seen.get(str(path)) != stamp:
                result = sync_artifacts(artifacts, database, model_path)
                print(json.dumps(result), flush=True)
                seen[str(path)] = stamp
        time.sleep(interval)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persist forecast artifacts in SQLite and watch for new runs")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--database", default="artifacts/cfb_forecast.sqlite3")
    parser.add_argument("--model", default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.watch:
        watch(args.artifacts, args.database, args.model, args.interval)
    else:
        print(json.dumps(sync_artifacts(args.artifacts, args.database, args.model)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
