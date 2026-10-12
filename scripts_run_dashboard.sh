#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/c/Users/brian/Downloads/cfb_forecast/cfb_forecast}"
VENV_DIR="${VENV_DIR:-$HOME/pythonlib1}"
PORT="${CFB_DASHBOARD_PORT:-8501}"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
export CFB_ARTIFACTS_DIR="${CFB_ARTIFACTS_DIR:-$PROJECT_DIR/artifacts}"
export CFB_DATA_DIR="${CFB_DATA_DIR:-$PROJECT_DIR/data}"

[[ -d "$CFB_DATA_DIR" ]] || { echo "Data directory not found: $CFB_DATA_DIR" >&2; exit 1; }
[[ -d "$CFB_ARTIFACTS_DIR" ]] || mkdir -p "$CFB_ARTIFACTS_DIR"

exec streamlit run dashboard/app.py \
  --server.address=127.0.0.1 \
  --server.port="$PORT" \
  --server.fileWatcherType=none \
  --browser.gatherUsageStats=false
