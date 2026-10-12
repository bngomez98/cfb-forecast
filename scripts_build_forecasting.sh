#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/c/Users/brian/Downloads/cfb_forecast/cfb_forecast}"
VENV_DIR="${VENV_DIR:-$HOME/pythonlib1}"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

python -m pip check
python -m pytest -q
rm -rf build dist
python -m build

WHEEL="$(find dist -maxdepth 1 -type f -name '*.whl' -print -quit)"
[[ -n "$WHEEL" ]] || { echo 'Wheel was not created' >&2; exit 1; }
python -m pip install --force-reinstall "$WHEEL"
python -m pip check
python -c 'import cfb_forecast; print(cfb_forecast.__file__)'
cfb-forecast --help >/dev/null

printf '\nBuild verified: %s\n' "$PROJECT_DIR/dist"
find dist -maxdepth 1 -type f -printf '%f %s bytes\n'
