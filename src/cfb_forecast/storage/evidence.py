"""Immutable, content-addressed forecast inputs and later evaluations.

Operational tables still serve the UI. These records retain the full precision
forecast and exact input bytes independently of the replaceable latest board.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_evidence (
  run_key TEXT NOT NULL,
  game_id INTEGER NOT NULL,
  recorded_at TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (run_key, game_id)
);
CREATE TABLE IF NOT EXISTS forecast_inputs (
  run_key TEXT NOT NULL,
  name TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  byte_count INTEGER NOT NULL,
  artifact_path TEXT NOT NULL,
  PRIMARY KEY (run_key, name)
);
CREATE TABLE IF NOT EXISTS forecast_evaluations (
  run_key TEXT NOT NULL,
  game_id INTEGER NOT NULL,
  outcome_sha256 TEXT NOT NULL,
  evaluated_at TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (run_key, game_id, outcome_sha256),
  FOREIGN KEY (run_key, game_id) REFERENCES forecast_evidence(run_key, game_id)
);
"""
for _table in ("forecast_evidence", "forecast_inputs", "forecast_evaluations"):
    for _operation in ("UPDATE", "DELETE"):
        SCHEMA += f"""
CREATE TRIGGER IF NOT EXISTS immutable_{_table}_{_operation.lower()}
BEFORE {_operation} ON {_table} BEGIN
  SELECT RAISE(ABORT, 'immutable forecast evidence');
END;
"""


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_json(value):
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def freeze_forecast(conn, run_key, frame, run_dir, evidence_dir):
    """Serialize competing publishers before checking and freezing a whole batch."""
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        return _freeze_forecast(conn, run_key, frame, run_dir, evidence_dir)


def _freeze_forecast(conn, run_key, frame, run_dir, evidence_dir):
    """Freeze a complete batch; identical retries succeed, conflicts fail atomically."""
    if frame.game_id.duplicated().any():
        raise ValueError("Duplicate game IDs in forecast batch")
    now = datetime.now(timezone.utc).isoformat()
    forecasts = []
    for record in frame.to_dict("records"):
        payload = canonical_json(record)
        digest = _digest(payload.encode())
        key = (str(run_key), int(record["game_id"]))
        existing = conn.execute("SELECT payload_sha256 FROM forecast_evidence WHERE run_key=? AND game_id=?", key).fetchone()
        if existing and existing[0] != digest:
            raise ValueError(f"Saved forecast changed for {key}; use a new run ID")
        forecasts.append((*key, now, digest, payload))
    old_ids = {row[0] for row in conn.execute("SELECT game_id FROM forecast_evidence WHERE run_key=?", (str(run_key),))}
    if old_ids and old_ids != set(frame.game_id.astype(int)):
        raise ValueError("Saved forecast population changed; use a new run ID")
    inputs = []
    directory = Path(evidence_dir)
    # Include calculation artifacts and retained sources, not just rounded UI rows.
    for path in sorted(Path(run_dir).rglob("*")):
        if not path.is_file() or path.suffix not in {".csv", ".json", ".npz", ".parquet"}:
            continue
        if path.name.startswith(("validation", "evaluation", "publish", "graded")):
            continue
        name = path.relative_to(run_dir).as_posix()
        content = path.read_bytes()
        digest = _digest(content)
        existing = conn.execute("SELECT content_sha256 FROM forecast_inputs WHERE run_key=? AND name=?", (str(run_key), name)).fetchone()
        if existing and existing[0] != digest:
            raise ValueError(f"Saved forecast input changed: {path.name}; use a new run ID")
        target = directory / digest
        if target.exists() and _digest(target.read_bytes()) != digest:
            raise ValueError(f"Corrupt evidence artifact: {target}")
        inputs.append((str(run_key), name, digest, len(content), str(target.resolve()), content))
    old_names = {row[0] for row in conn.execute("SELECT name FROM forecast_inputs WHERE run_key=?", (str(run_key),))}
    if old_names and old_names != {row[1] for row in inputs}:
        raise ValueError("Saved forecast input set changed; use a new run ID")
    directory.mkdir(parents=True, exist_ok=True)
    for *_, target, content in inputs:
        path = Path(target)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(content)
    with conn:
        conn.executemany("INSERT OR IGNORE INTO forecast_evidence VALUES(?,?,?,?,?)", forecasts)
        conn.executemany("INSERT OR IGNORE INTO forecast_inputs VALUES(?,?,?,?,?)", [row[:-1] for row in inputs])
    return len(forecasts)


def verify_evidence(conn, run_key, evidence_dir=None):
    rows = conn.execute("SELECT * FROM forecast_inputs WHERE run_key=?", (str(run_key),)).fetchall()
    if not rows:
        raise ValueError(f"No frozen inputs for run {run_key}")
    for row in rows:
        path = Path(evidence_dir) / row["content_sha256"] if evidence_dir is not None else Path(row["artifact_path"])
        if not path.is_file() or _digest(path.read_bytes()) != row["content_sha256"]:
            raise ValueError(f"Saved forecast input missing or changed: {row['name']}")
    for row in conn.execute("SELECT * FROM forecast_evidence WHERE run_key=?", (str(run_key),)):
        if _digest(row["payload_json"].encode()) != row["payload_sha256"]:
            raise ValueError("Saved forecast payload changed")
    return {"run_id": str(run_key), "status": "verified", "input_files": len(rows)}
