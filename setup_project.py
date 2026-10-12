#!/usr/bin/env python3
"""Bootstrap and validate the full-stack CFB forecasting application.

Usage:
    python setup_project.py
    python setup_project.py --start-stack
    python setup_project.py --skip-tests
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"
VENV_DIR = PROJECT_ROOT / ".venv"
DATA_DIR = Path(os.getenv("CFB_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACTS_DIR = Path(os.getenv("CFB_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
MODEL_DIR = Path(os.getenv("CFB_MODEL_DIR", ARTIFACTS_DIR / "backtest" / "model"))


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(command))
    return subprocess.run(command, cwd=PROJECT_ROOT, check=check)


def python_executable() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def read_project_config() -> dict:
    if not PYPROJECT_FILE.exists():
        raise SystemExit(f"Missing project configuration: {PYPROJECT_FILE}")
    with PYPROJECT_FILE.open("rb") as file:
        config = tomllib.load(file)
    project = config.get("project", {})
    if not project.get("name") or not project.get("dependencies"):
        raise SystemExit("pyproject.toml must define project.name and project.dependencies")
    print(f"Project: {project['name']} {project.get('version', '')}".strip())
    print(f"Python requirement: {project.get('requires-python', 'unspecified')}")
    print(f"Runtime dependencies: {len(project['dependencies'])}")
    print(f"Test extras: {list(project.get('optional-dependencies', {}).get('test', []))}")
    return config


def bootstrap_environment(skip_tests: bool, start_stack: bool) -> None:
    config = read_project_config()
    if not VENV_DIR.exists():
        run([sys.executable, "-m", "venv", str(VENV_DIR)])

    py = python_executable()
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    # pyproject.toml is the single source of truth for runtime and test dependencies.
    install_target = ".[test]" if "test" in config.get("project", {}).get("optional-dependencies", {}) else "."
    run([str(py), "-m", "pip", "install", "-e", install_target])

    if not DATA_DIR.is_dir():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "CFB_DATA_DIR": str(DATA_DIR),
        "CFB_ARTIFACTS_DIR": str(ARTIFACTS_DIR),
        "CFB_MODEL_DIR": str(MODEL_DIR),
        "CFB_DATABASE": str(ARTIFACTS_DIR / "cfb_forecast.sqlite3"),
    })

    print("Checking source data and established model")
    check_script = """
from pathlib import Path
import os
from cfb_forecast.data import load_data
from cfb_forecast.model import load_model
schedules, summaries, paths = load_data(Path(os.environ['CFB_DATA_DIR']))
model = load_model(Path(os.environ['CFB_MODEL_DIR']))
print(f'schedule rows: {len(schedules)}')
print(f'advanced summary rows: {len(summaries)}')
print(f'source files: {len(paths)}')
print(f'model variant: {model["variant"]}')
print(f'model trained through: {model["training_through_season"]}')
"""
    run([str(py), "-c", check_script])

    print("Running continuous-awareness check")
    awareness_script = """
import os
from cfb_forecast.awareness import awareness_report
report = awareness_report(os.environ['CFB_DATA_DIR'], os.environ['CFB_MODEL_DIR'], os.environ['CFB_DATABASE'])
print(report)
if report['overall'] != 'pass':
    raise SystemExit('awareness check failed')
"""
    run([str(py), "-c", awareness_script],)

    if not skip_tests:
        run([str(py), "-m", "pytest", "-q"])

    check_command(["docker", "info"], "Docker daemon check")
    check_command(["docker", "compose", "config"], "Docker Compose validation")

    if start_stack:
        run(["docker", "compose", "up", "--build", "-d"])
        wait_for_url("http://localhost:8000/ready")
        wait_for_url("http://localhost:8000/awareness")
        print("API: http://localhost:8000")
        print("Dashboard: http://localhost:8501")


def wait_for_url(url: str, attempts: int = 30) -> None:
    import time
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"Service did not become ready: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up and validate CFB Forecast")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--start-stack", action="store_true")
    args = parser.parse_args()
    bootstrap_environment(args.skip_tests, args.start_stack)
    print("Setup complete")


if __name__ == "__main__":
    main()
