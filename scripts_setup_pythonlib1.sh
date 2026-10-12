#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/c/Users/brian/Downloads/cfb_forecast/cfb_forecast}"
VENV_DIR="${VENV_DIR:-$HOME/pythonlib1}"
KERNEL_NAME="pythonlib1"

command -v python3 >/dev/null || { echo 'python3 is required' >&2; exit 1; }
[[ -d "$PROJECT_DIR" ]] || { echo "Project directory not found: $PROJECT_DIR" >&2; exit 1; }

if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git python3-dev python3-venv build-essential
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel build
python -m pip install --upgrade jupyterlab ipykernel
python -m pip install -e "$PROJECT_DIR[test]"
python -m pip install -r "$PROJECT_DIR/dashboard/requirements.txt"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"

python -m pip check
python - <<'PY'
import enum, subprocess, sys
import IPython, jupyterlab, numpy, pandas, pyarrow, scipy, sklearn, streamlit
print(f"Python {sys.version.split()[0]} verified")
print("Standard library and forecasting/dashboard imports verified")
PY

printf 'Environment ready: %s\n' "$VENV_DIR"
printf 'Activate with: source %s/bin/activate\n' "$VENV_DIR"
