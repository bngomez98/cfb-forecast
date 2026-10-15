"""The whole application in one process: the scheduler and the dashboard.

    cfb-forecast app                     # scheduler (daily forecast) + dashboard on :8501
    cfb-forecast app --no-dashboard      # scheduler only
    cfb-forecast app --no-scheduler      # dashboard only

The scheduler runs in this process (signal handlers, single-instance lock, heartbeat); the
dashboard is a child Streamlit process that is stopped when the application stops. The first
scheduler tick runs the forecast immediately when none exists for today, so a fresh start
produces predictions without any manual step.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from .logs import add_file, configure, get
from .scheduler import Scheduler

log = get("app")


def dashboard_script():
    """``dashboard/app.py`` next to the ``src`` tree (editable install or the container's /app)."""
    for root in (Path(__file__).resolve().parents[2], Path.cwd()):
        candidate = root / "dashboard" / "app.py"
        if candidate.exists():
            return candidate
    return None


def start_dashboard(data, artifacts, port, address="0.0.0.0"):
    """Launch Streamlit as a child process; returns it, or None when it cannot run here."""
    script = dashboard_script()
    if script is None:
        log.warning("dashboard/app.py not found; running without the dashboard")
        return None
    if importlib.util.find_spec("streamlit") is None:
        log.warning("streamlit is not installed (pip install 'cfb-forecast[dashboard]'); running without the dashboard")
        return None
    env = {**os.environ, "CFB_DATA_DIR": str(Path(data).resolve()), "CFB_ARTIFACTS_DIR": str(Path(artifacts).resolve())}
    command = [sys.executable, "-m", "streamlit", "run", str(script), f"--server.port={port}", f"--server.address={address}",
               "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
    log.info("dashboard starting on http://localhost:%s", port)
    return subprocess.Popen(command, env=env)  # noqa: S603 - fixed argument list, no shell


def stop(process, timeout=10):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()


def add_arguments(parser):
    env = os.environ.get
    parser.add_argument("--data", default=env("CFB_DATA_DIR", "data"))
    parser.add_argument("--artifacts", default=env("CFB_ARTIFACTS_DIR", "artifacts"))
    parser.add_argument("--season", type=int, default=None, help="Default: CFB_FORECAST_SEASON, or the current season")
    parser.add_argument("--port", type=int, default=int(env("CFB_DASHBOARD_PORT", "8501")))
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-scheduler", action="store_true")
    return parser


def run_from_args(args):
    configure()
    add_file(Path(args.artifacts) / "logs", "app", slot="cfb-app-file")
    if args.no_dashboard and args.no_scheduler:
        raise SystemExit("Nothing to run: both --no-dashboard and --no-scheduler were given")
    dashboard = None if args.no_dashboard else start_dashboard(args.data, args.artifacts, args.port)
    try:
        if args.no_scheduler:
            if dashboard is None:
                return 1
            return dashboard.wait()
        Scheduler(args.data, args.artifacts, args.season).forever()
        return 0
    finally:
        stop(dashboard)
