#!/usr/bin/env bash
set -Eeuo pipefail

# One-command WSL build for the CFB forecasting project.
# Run from any WSL directory:
#   bash /mnt/c/Users/brian/cfb-forecast/scripts/run/wsl_forecast.sh

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-/home/brian/.venvs/cfb-forecast}"
DATA_DIR="${CFB_DATA_DIR:-$PROJECT_DIR/data}"
ARTIFACT_DIR="${CFB_ARTIFACTS_DIR:-$PROJECT_DIR/artifacts}"
SEASON="${CFB_SEASON:-2026}"
SEASONS="${CFB_SEASONS:-2021 2022 2023 2024 2025 2026}"
DRAWS="${CFB_DRAWS:-50000}"
LOG_DIR="$ARTIFACT_DIR/logs"
LOG_FILE="$LOG_DIR/full_build_$(date -u +%Y%m%dT%H%M%SZ).log"
STATUS_FILE="$ARTIFACT_DIR/full_build_status.json"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAGE="starting"

fail() {
  local message="$1"
  printf '\nBUILD FAILED\nStage: %s\nReason: %s\nLog: %s\n' "$STAGE" "$message" "$LOG_FILE" >&2
  python3 - "$STATUS_FILE" "$STARTED_AT" "$STAGE" "$message" <<'PY' 2>/dev/null || true
import json, sys
from pathlib import Path
path, started, stage, message = sys.argv[1:]
Path(path).write_text(json.dumps({"status":"failed","started_at":started,"stage":stage,"error":message}, indent=2))
PY
  exit 1
}

trap 'fail "Command failed at line $LINENO: $BASH_COMMAND"' ERR

say() { printf '\n\033[1;36m[%s] %s\033[0m\n' "$(date -u +%H:%M:%S)" "$*"; }

STAGE="repository"
say "Checking Git repository"
if [[ -d "$PROJECT_DIR/.git" ]]; then
  cd "$PROJECT_DIR"
  git fetch --all --prune
  if [[ -z "$(git status --porcelain)" ]]; then
    branch="$(git branch --show-current)"
    if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
      git pull --ff-only origin "$branch"
    fi
  else
    say "Local Git changes detected; fetch completed and pull skipped"
    git status --short
  fi
else
  say "No Git metadata found; continuing with local project files"
fi
cd "$PROJECT_DIR"

STAGE="project validation"
say "Validating project paths"
[[ -d "$PROJECT_DIR" ]] || fail "Project directory not found: $PROJECT_DIR"
[[ -f "$PROJECT_DIR/pyproject.toml" ]] || fail "pyproject.toml not found"
[[ -f "$PROJECT_DIR/dashboard/requirements.txt" ]] || fail "Dashboard requirements not found"
mkdir -p "$DATA_DIR" "$ARTIFACT_DIR"

STAGE="system packages"
say "Checking WSL system packages"
if command -v git >/dev/null && command -v python3 >/dev/null; then
  say "git and python3 are already installed; skipping apt"
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo apt-get update
  sudo apt-get install -y git python3 python3-dev python3-venv build-essential ca-certificates curl
else
  command -v git >/dev/null || fail "git is missing and passwordless sudo is unavailable"
  command -v python3 >/dev/null || fail "python3 is missing and passwordless sudo is unavailable"
fi

STAGE="virtual environment"
say "Creating or updating $VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m ensurepip --upgrade || true
python --version
python -m pip --version

STAGE="Python dependencies"
say "Installing package, dashboard, Jupyter, test, and build dependencies"
python -m pip install --upgrade pip setuptools wheel build
python -m pip install --upgrade jupyterlab ipykernel
python -m pip install -e "$PROJECT_DIR[test]"
python -m pip install -r "$PROJECT_DIR/dashboard/requirements.txt"
python -m ipykernel install --user --name cfb-forecast --display-name "Python (cfb-forecast)"
python -m pip check
python - <<'PY'
import enum, subprocess, sys
import IPython, cfb_forecast, jupyterlab, numpy, pandas, pyarrow, scipy, sklearn, streamlit
print(f"Python {sys.version.split()[0]} and all required imports are available")
PY

STAGE="source connectivity"
say "Checking ESPN and public repository connectivity"
python "$PROJECT_DIR/scripts/run/update_forecast.py" "$SEASON" "$DRAWS" \
  --no-fetch --no-results --no-forecast --no-tests --no-build

STAGE="dataset refresh"
say "Refreshing all integrated seasons: $SEASONS"
FETCH_ARGS=("-m" "cfb_forecast" "fetch" "--data" "$DATA_DIR" "--seasons")
for year in $SEASONS; do FETCH_ARGS+=("$year"); done
FETCH_ARGS+=("--refresh" "--raw-pbp")
python "${FETCH_ARGS[@]}"

STAGE="dataset validation"
say "Validating schedules, advanced-team, PBP, markets, manifests, and checksums"
CFB_DATA_DIR="$DATA_DIR" bash "$PROJECT_DIR/scripts/checks/verify_datasets.sh"

STAGE="ESPN results"
say "Refreshing ESPN results for season $SEASON"
python -m cfb_forecast results-refresh \
  --data "$DATA_DIR" \
  --out "$ARTIFACT_DIR/results" \
  --season "$SEASON"

STAGE="forecast"
say "Generating weekly forecast for season $SEASON"
python -m cfb_forecast weekly-forecast \
  --data "$DATA_DIR" \
  --model "$ARTIFACT_DIR/backtest/model" \
  --out "$ARTIFACT_DIR" \
  --season "$SEASON" \
  --draws "$DRAWS"

STAGE="tests"
say "Running test suite"
python -m pytest -q

STAGE="package build"
say "Building wheel and source distribution"
rm -rf "$PROJECT_DIR/build" "$PROJECT_DIR/dist"
python -m build
WHEEL="$(find "$PROJECT_DIR/dist" -maxdepth 1 -type f -name '*.whl' -print -quit)"
[[ -n "$WHEEL" ]] || fail "No wheel was created"
python -m pip install --force-reinstall "$WHEEL"
python -m pip check
python -m cfb_forecast --help >/dev/null

STAGE="final report"
say "Writing final build report"
python - "$STATUS_FILE" "$PROJECT_DIR" "$VENV_DIR" "$DATA_DIR" "$ARTIFACT_DIR" "$LOG_FILE" "$WHEEL" "$STARTED_AT" "$SEASON" "$DRAWS" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
status, project, venv, data, artifacts, log, wheel, started, season, draws = sys.argv[1:]
files = sorted(str(p) for p in Path(artifacts).rglob("*") if p.is_file())
Path(status).write_text(json.dumps({
    "status": "success",
    "started_at": started,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "project": project,
    "venv": venv,
    "data": data,
    "artifacts": artifacts,
    "season": int(season),
    "draws": int(draws),
    "wheel": wheel,
    "log": log,
    "artifact_count": len(files),
    "latest_weekly": str(Path(artifacts) / "latest-weekly"),
}, indent=2))
PY

printf '\nBUILD COMPLETE\n'
printf 'Python environment: %s\n' "$VENV_DIR"
printf 'Forecast output:    %s/latest-weekly\n' "$ARTIFACT_DIR"
printf 'Package output:     %s/dist\n' "$PROJECT_DIR"
printf 'Build report:       %s\n' "$STATUS_FILE"
printf 'Full log:           %s\n' "$LOG_FILE"
printf '\nStart the GUI with:\n  source "%s/bin/activate"\n  streamlit run "%s/dashboard/app.py"\n' "$VENV_DIR" "$PROJECT_DIR"
