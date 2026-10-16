"""Nested command line: ``cfb <group> <action>``.

    cfb start [app|scheduler]            run the application (scheduler + dashboard) or just the scheduler
    cfb build  [model|players|data]      train the model, rebuild player aggregates, download season inputs
    cfb dashboard [api]                  open the dashboard (or the JSON API)
    cfb update [forecast|sunday|results|sources]
    cfb historical [forecasts|reproduce] point-in-time backtest of this season, or the 2025 held-out evaluation
    cfb validation [forecasts|status]    validate logged forecasts against stored results

Every action maps onto a ``cfb-forecast`` command; add ``--season`` to change the season.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

TREE = {
    "start": {"app": "Scheduler and dashboard together", "scheduler": "Scheduler only"},
    "build": {"model": "Train the production model if missing", "players": "Rebuild per-player aggregates", "data": "Download season inputs"},
    "dashboard": {"app": "Streamlit dashboard", "api": "JSON API on :8000"},
    "update": {"forecast": "Daily forecast of every upcoming game", "sunday": "Sunday service now", "results": "Final scores", "sources": "External web sources"},
    "historical": {"forecasts": "Point-in-time backtest of this season", "reproduce": "2025 held-out evaluation"},
    "validation": {"forecasts": "Validate logged forecasts against stored results", "status": "Scheduler jobs and last runs"},
}
DEFAULT = {"start": "app", "build": "model", "dashboard": "app", "update": "forecast", "historical": "forecasts", "validation": "forecasts"}


def season():
    return os.environ.get("CFB_FORECAST_SEASON", "2026")


def plan(group, action, year):
    """The command (argv list) that implements ``group action``."""
    s = str(year)
    cf = [sys.executable, "-m", "cfb_forecast"]
    table = {
        ("start", "app"): cf + ["app", "--season", s],
        ("start", "scheduler"): cf + ["scheduler", "--season", s],
        ("build", "model"): [sys.executable, "-c", "from cfb_forecast import paths, sunday; sunday.ensure_model(paths.data_root(), paths.model_root(), %s)" % s],
        ("build", "players"): cf + ["player-aggregates", "--seasons", s, "--refresh"],
        ("build", "data"): cf + ["fetch", "--seasons", s, "--refresh"],
        ("dashboard", "app"): [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
        ("dashboard", "api"): [sys.executable, "-m", "uvicorn", "cfb_forecast.api:app", "--host", "127.0.0.1", "--port", "8000"],
        ("update", "forecast"): cf + ["pipeline", "forecast", "--season", s, "--refresh-data"],
        ("update", "sunday"): cf + ["sunday-run", "--season", s],
        ("update", "results"): cf + ["pipeline", "results", "--season", s],
        ("update", "sources"): cf + ["sources-collect", "--season", s],
        ("historical", "forecasts"): cf + ["pipeline", "backtest", "--season", s],
        ("historical", "reproduce"): cf + ["backtest", "--data", "data", "--out", "artifacts/reproduced", "--archive", "data/archive/matched_predictions.csv"],
        ("validation", "forecasts"): [sys.executable, "-c", "from cfb_forecast import dbflow, paths; print(dbflow.run(paths.database_path(), paths.data_root(), %s))" % s],
        ("validation", "status"): cf + ["scheduler", "--status", "--season", s],
    }
    return table[(group, action)]


def parser():
    p = argparse.ArgumentParser(prog="cfb", description="CFB forecast: start, build, dashboard, update, historical, validation")
    p.add_argument("--season", type=int, default=int(season()))
    groups = p.add_subparsers(dest="group", required=True)
    for group, actions in TREE.items():
        g = groups.add_parser(group, help=", ".join(actions))
        g.add_argument("action", nargs="?", default=DEFAULT[group], choices=list(actions), help=" | ".join(f"{a}: {d}" for a, d in actions.items()))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    raise SystemExit(subprocess.call(plan(args.group, args.action, args.season)))


if __name__ == "__main__":
    main()
