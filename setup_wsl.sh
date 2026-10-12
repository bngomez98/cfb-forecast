#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
DATA_DIR="${CFB_DATA_DIR:-${PROJECT_ROOT}/data}"
ARTIFACTS_DIR="${CFB_ARTIFACTS_DIR:-${PROJECT_ROOT}/artifacts}"
MODEL_DIR="${CFB_MODEL_DIR:-${ARTIFACTS_DIR}/backtest/model}"
START_STACK="${START_STACK:-0}"
RUN_TESTS="${RUN_TESTS:-1}"

log() { printf '\n[cfb] %s\n' "$*"; }
die() { printf '\n[cfb] ERROR: %s\n' "$*" >&2; exit 1; }

cd "$PROJECT_ROOT"

command -v python3 >/dev/null || die "python3 is required in WSL"
command -v docker >/dev/null || die "Docker Desktop integration is required"
command -v docker compose >/dev/null || die "Docker Compose is required"

log "Checking Docker Desktop connectivity"
docker info >/dev/null 2>&1 || die "Docker daemon is unavailable; start Docker Desktop and enable WSL integration"

log "Creating Python virtual environment at ${VENV_DIR}"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

log "Installing application and test dependencies"
python -m pip install -e '.[test]'

[[ -d "$DATA_DIR" ]] || die "Data directory not found: ${DATA_DIR}"
[[ -d "$ARTIFACTS_DIR" ]] || mkdir -p "$ARTIFACTS_DIR"

log "Checking source data and established model"
python - <<'PY'
import os
from pathlib import Path
from cfb_forecast.data import load_data
from cfb_forecast.model import load_model

root = Path(os.environ.get("CFB_DATA_DIR", "data"))
model_root = Path(os.environ.get("CFB_MODEL_DIR", "artifacts/backtest/model"))
schedules, summaries, paths = load_data(root)
model = load_model(model_root)
print(f"schedule rows: {len(schedules)}")
print(f"advanced summary rows: {len(summaries)}")
print(f"source files: {len(paths)}")
print(f"model variant: {model['variant']}")
print(f"model trained through: {model['training_through_season']}")
PY

log "Running continuous-awareness check"
CFB_DATA_DIR="$DATA_DIR" CFB_MODEL_DIR="$MODEL_DIR" CFB_DATABASE="${ARTIFACTS_DIR}/cfb_forecast.sqlite3" \
python - <<'PY'
import os
from cfb_forecast.awareness import awareness_report

report = awareness_report(
    os.environ["CFB_DATA_DIR"],
    os.environ["CFB_MODEL_DIR"],
    os.environ["CFB_DATABASE"],
)
print(report)
if report["overall"] != "pass":
    raise SystemExit("awareness check failed")
PY

if [[ "$RUN_TESTS" == "1" ]]; then
  log "Running tests"
  python -m pytest -q
fi

log "Validating Docker Compose configuration"
docker compose config >/dev/null

if [[ "$START_STACK" == "1" ]]; then
  log "Starting API and dashboard"
  docker compose up --build -d
  log "Waiting for API readiness"
  for attempt in {1..30}; do
    if curl -fsS http://localhost:8000/ready >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  curl -fsS http://localhost:8000/ready >/dev/null || die "API did not become ready"
  curl -fsS http://localhost:8000/awareness >/dev/null || die "Awareness endpoint failed"
  log "Stack is ready: http://localhost:8000 and http://localhost:8501"
else
  log "Setup complete. Set START_STACK=1 to start Docker Compose."
fi
