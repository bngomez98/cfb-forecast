#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/c/Users/brian/Downloads/cfb_forecast/cfb_forecast}"
VENV_DIR="${VENV_DIR:-$HOME/pythonlib1}"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

[[ "$(command -v python)" == "$VENV_DIR/bin/python" ]] || { echo 'Wrong Python interpreter is active' >&2; exit 1; }
python -m pip check
python -c 'import enum, subprocess, cfb_forecast, numpy, pandas, scipy, sklearn, pyarrow, streamlit; print("All imports passed")'
python -m pytest -q
python -m build --no-isolation
WHEEL="$(find dist -maxdepth 1 -type f -name '*.whl' -print -quit)"
[[ -n "$WHEEL" ]] || { echo 'No wheel found' >&2; exit 1; }
python -m pip install --force-reinstall "$WHEEL"
python -m pip check
cfb-forecast --help >/dev/null

echo "Verification passed"
echo "Python: $(python --version)"
echo "Venv: $VENV_DIR"
echo "Wheel: $WHEEL"
