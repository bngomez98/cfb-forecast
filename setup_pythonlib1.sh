#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/c/Users/brian/Downloads/cfb_forecast/cfb_forecast}"
VENV_DIR="${VENV_DIR:-$HOME/pythonlib1}"
KERNEL_NAME="pythonlib1"

if [[ ! -d "$PROJECT_DIR" ]]; then
  printf 'Project directory not found: %s\n' "$PROJECT_DIR" >&2
  exit 1
fi

if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git python3-dev python3-venv build-essential
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel build
python -m pip install --upgrade jupyterlab ipykernel

python -m pip install -e "$PROJECT_DIR[test]"
python -m pip install -r "$PROJECT_DIR/dashboard/requirements.txt"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"

python - <<'PY'
import enum
import subprocess
import sys

import IPython
import jupyterlab
import numpy
import pandas
import pyarrow
import scipy
import sklearn
import streamlit

print(f"Python verified: {sys.version.split()[0]}")
print("Standard library verified: enum, subprocess")
print("Packages verified: IPython, JupyterLab, NumPy, pandas, PyArrow, SciPy, scikit-learn, Streamlit")
PY

cd "$PROJECT_DIR"
python -m pytest
python -m build
cfb-forecast --help >/dev/null

printf '\nSetup completed.\n'
printf 'Activate: source %q/bin/activate\n' "$VENV_DIR"
printf 'Project:  %s\n' "$PROJECT_DIR"
printf 'Builds:   %s/dist\n' "$PROJECT_DIR"
printf 'Jupyter:  jupyter lab\n'
