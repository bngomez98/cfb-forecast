#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/c/Users/brian/cfb-forecast}"
VENV_DIR="${VENV_DIR:-/home/brian/.venvs/cfb-forecast}"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
export CFB_PROJECT_ROOT="$PROJECT_DIR"

python -m pip check
cp -f sql/schema.sql src/cfb_forecast/schema.sql
python -m pytest -q
rm -rf build dist
python -m build

WHEEL="$(find dist -maxdepth 1 -type f -name '*.whl' -print -quit)"
[[ -n "$WHEEL" ]] || { echo 'Wheel was not created' >&2; exit 1; }
python -m pip install --force-reinstall "$WHEEL"
python -m pip check
python -c 'import cfb_forecast; print(cfb_forecast.__file__)'
cfb-forecast --help >/dev/null
cfb-forecast build-runtime --database artifacts/cfb_forecast.sqlite --artifacts artifacts --season 2026

printf '\nBuild verified: %s\n' "$PROJECT_DIR/dist"
find dist -maxdepth 1 -type f -printf '%f %s bytes\n'
