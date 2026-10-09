#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/c/Users/brian/Downloads/cfb_forecast/cfb_forecast}"
VENV_DIR="${VENV_DIR:-$HOME/pythonlib1}"
DATA_DIR="${CFB_DATA_DIR:-$PROJECT_DIR/data}"
ARTIFACT_DIR="${CFB_ARTIFACTS_DIR:-$PROJECT_DIR/artifacts}"
SEASON="${CFB_FORECAST_SEASON:-2026}"
DRAWS="${CFB_FORECAST_DRAWS:-50000}"
REFRESH_DATA="${CFB_REFRESH_DATA:-1}"
REFRESH_RESULTS="${CFB_REFRESH_RESULTS:-1}"
RUN_TESTS="${CFB_RUN_TESTS:-1}"
RUN_FORECAST="${CFB_RUN_FORECAST:-1}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$PROJECT_DIR" ]] || fail "Project directory not found: $PROJECT_DIR"
[[ -x "$VENV_DIR/bin/python" ]] || fail "Virtual environment not found: $VENV_DIR"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
mkdir -p "$DATA_DIR" "$ARTIFACT_DIR/results"

step() { printf '\n==> %s\n' "$*"; }

step 'Verify Python dependencies'
python -m pip check
python -c 'import cfb_forecast, numpy, pandas, scipy, sklearn, pyarrow, streamlit; print("Python dependencies available")'

step 'Verify public ESPN and GitHub sources'
bash "$PROJECT_DIR/scripts_verify_sources.sh"

if [[ "$REFRESH_DATA" == "1" ]]; then
  step "Refresh schedule and advanced-team data for season $SEASON"
  python -m cfb_forecast fetch \
    --data "$DATA_DIR" \
    --seasons "$SEASON" \
    --refresh
fi

step 'Verify local datasets'
CFB_DATA_DIR="$DATA_DIR" bash "$PROJECT_DIR/scripts_verify_datasets.sh"

if [[ "$REFRESH_RESULTS" == "1" ]]; then
  step "Refresh ESPN results for season $SEASON"
  python -m cfb_forecast results-refresh \
    --data "$DATA_DIR" \
    --out "$ARTIFACT_DIR/results" \
    --season "$SEASON"
fi

if [[ "$RUN_FORECAST" == "1" ]]; then
  step "Generate weekly forecast for season $SEASON"
  python -m cfb_forecast weekly-forecast \
    --data "$DATA_DIR" \
    --out "$ARTIFACT_DIR/latest-weekly" \
    --season "$SEASON" \
    --draws "$DRAWS"
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  step 'Run tests'
  python -m pytest -q
fi

if [[ "${CFB_BUILD_PACKAGE:-1}" == "1" ]]; then
  step 'Build and verify package'
  bash "$PROJECT_DIR/scripts_build_forecasting.sh"
fi

step 'Pipeline completed'
printf 'Season: %s\n' "$SEASON"
printf 'Data:   %s\n' "$DATA_DIR"
printf 'Results: %s/results\n' "$ARTIFACT_DIR"
printf 'Builds: %s/dist\n' "$PROJECT_DIR"
