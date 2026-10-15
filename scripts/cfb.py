"""Human-friendly command-line entry points for the CFB forecast pipeline."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def docker_run(image: str, command: list[str], writable_data: bool = False) -> None:
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError(
            "Docker is unavailable in this environment. Start Docker Desktop, enable "
            "Settings > Resources > WSL Integration for this distro, then reopen WSL."
        )
    data_mode = "rw" if writable_data else "ro"
    args = [
        docker, "run", "--rm",
        "-v", f"{DATA}:/app/data:{data_mode}",
        "-v", f"{ARTIFACTS}:/app/artifacts:rw",
        image,
        *command,
    ]
    print("Running:", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def ensure_data(image: str, season: int) -> None:
    schedule = DATA / f"schedule_{season}.csv"
    advanced = DATA / f"adv_team_{season}.parquet"
    if not schedule.exists() or not advanced.exists():
        print(f"Downloading missing {season} season data...", flush=True)
        docker_run(image, ["fetch", "--data", "/app/data", "--seasons", str(season)], writable_data=True)


def run_pipeline(args: argparse.Namespace) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ensure_data(args.image, args.season)
    docker_run(args.image, ["preflight", "--data", "/app/data", "--artifacts", "/app/artifacts", "--season", str(args.season)])
    docker_run(args.image, ["results-refresh", "--data", "/app/data", "--out", "/app/artifacts/results", "--season", str(args.season)])
    docker_run(args.image, ["weekly-forecast", "--data", "/app/data", "--model", "/app/artifacts/training/model", "--out", "/app/artifacts", "--season", str(args.season), "--draws", str(args.draws)], writable_data=True)
    docker_run(args.image, ["season-backtest", "--data", "/app/data", "--model", "/app/artifacts/training/model", "--out", "/app/artifacts/backtest", "--season", str(args.season), "--draws", str(args.draws), "--lead-hours", str(args.lead_hours)])
    print(f"Weekly pipeline finished for {args.season}.")
    print(f"Dashboard: {args.dashboard_url}")


def refresh_results(args: argparse.Namespace) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ensure_data(args.image, args.season)
    docker_run(args.image, ["results-refresh", "--data", "/app/data", "--out", "/app/artifacts/results", "--season", str(args.season), *(["--week", str(args.week)] if args.week else [])])
    print(f"Results refreshed for {args.season}.")


def backtest_season(args: argparse.Namespace) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ensure_data(args.image, args.season)
    docker_run(args.image, ["season-backtest", "--data", "/app/data", "--model", "/app/artifacts/training/model", "--out", "/app/artifacts/backtest", "--season", str(args.season), "--draws", str(args.draws), "--lead-hours", str(args.lead_hours)])
    print(f"Season backtest finished for {args.season}.")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.cfb", description="Run CFB forecasts, live results, and backtests.")
    p.add_argument("--image", default=os.environ.get("CFB_IMAGE", "cfb-forecast:local"), help="Docker image to run")
    p.add_argument("--season", type=int, default=int(os.environ.get("CFB_FORECAST_SEASON", "2026")), help="College football season")
    p.add_argument("--draws", type=int, default=int(os.environ.get("CFB_FORECAST_DRAWS", "50000")), help="Simulation draws")
    p.add_argument("--lead-hours", type=float, default=48, help="Forecast cutoff before kickoff")
    p.add_argument("--dashboard-url", default="http://localhost:8501", help="Dashboard URL")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("weekly", help="Refresh results, forecast the next week, and backtest the season")
    refresh = sub.add_parser("results", help="Refresh live ESPN game results")
    refresh.add_argument("--week", type=int, help="Only refresh one week")
    sub.add_parser("backtest", help="Run the season backtest")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.action == "weekly":
        run_pipeline(args)
    elif args.action == "results":
        refresh_results(args)
    elif args.action == "backtest":
        backtest_season(args)


if __name__ == "__main__":
    main()
