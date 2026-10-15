#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/c/Users/brian/cfb-forecast}"
VENV_DIR="${VENV_DIR:-/home/brian/.venvs/cfb-forecast}"
DATA_DIR="${CFB_DATA_DIR:-$PROJECT_DIR/data}"

[[ -x "$VENV_DIR/bin/python" ]] || { echo "Python venv not found: $VENV_DIR" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "Dataset directory not found: $DATA_DIR" >&2; exit 1; }

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

DATA_DIR="$DATA_DIR" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from cfb_forecast.ingest.data import ADV_FIELDS, SCHEDULE_FIELDS, load_data, normalize_schedule

root = Path(os.environ["DATA_DIR"])
required_schedules = sorted(root.glob("schedule_*.csv"))
required_advanced = sorted(root.glob("adv_team_*.parquet"))

if not required_schedules:
    raise SystemExit(f"No schedule_*.csv files found in {root}")
if not required_advanced:
    raise SystemExit(f"No adv_team_*.parquet files found in {root}")

for path in required_schedules:
    frame = pd.read_csv(path, low_memory=False)
    normalize_schedule(frame)
    print(f"OK schedule: {path.name} ({path.stat().st_size} bytes)")

for path in required_advanced:
    frame = pd.read_parquet(path)
    missing = sorted(set(ADV_FIELDS) - set(frame.columns))
    if missing:
        raise SystemExit(f"{path.name} missing columns: {missing}")
    print(f"OK advanced team: {path.name} ({path.stat().st_size} bytes)")

schedules, summaries, loaded = load_data(root)
print(f"Loaded schedules: {len(schedules)} rows")
print(f"Loaded advanced-team rows: {len(summaries)} rows")
print(f"Loaded files: {len(loaded)}")

for optional in ["available_data_catalog.json", "source_manifest.json", "market_history.csv"]:
    path = root / optional
    if path.exists():
        print(f"OK optional dataset: {optional}")

manifest = root / "source_manifest.json"
if manifest.exists():
    data = json.loads(manifest.read_text())
    for name, metadata in data.get("files", {}).items():
        path = root / name
        expected = metadata.get("sha256")
        if expected and not path.exists():
            raise SystemExit(f"Manifest file is missing: {name}")
        if path.exists():
            actual_bytes = path.stat().st_size
            expected_bytes = metadata.get("bytes")
            if expected_bytes is not None and actual_bytes != expected_bytes:
                raise SystemExit(f"Byte-size mismatch: {name} (expected {expected_bytes}, got {actual_bytes})")
            if expected:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != expected:
                    raise SystemExit(f"SHA-256 mismatch: {name}")
    print("OK source manifest checksums and sizes")

print("Dataset verification passed")
PY
