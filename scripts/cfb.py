"""Run the containerized pipeline from the host: python -m scripts.cfb daily|sunday|results|sources|backtest."""
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


def pipeline(args: argparse.Namespace, action: str, *extra: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    docker_run(args.image, ["pipeline", action, "--season", str(args.season), "--draws", str(args.draws),
                            "--lead-hours", str(args.lead_hours), "--data", "/app/data", "--artifacts", "/app/artifacts", *extra], writable_data=True)
    print(f"Pipeline {action} finished for {args.season}.")


def sunday(args: argparse.Namespace) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    docker_run(args.image, ["sunday-run", "--data", "/app/data", "--artifacts", "/app/artifacts", "--season", str(args.season)], writable_data=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.cfb", description="Run CFB forecasts, live results, and backtests.")
    p.add_argument("--image", default=os.environ.get("CFB_IMAGE", "cfb-forecast:local"), help="Docker image to run")
    p.add_argument("--season", type=int, default=int(os.environ.get("CFB_FORECAST_SEASON", "2026")), help="College football season")
    p.add_argument("--draws", type=int, default=int(os.environ.get("CFB_FORECAST_DRAWS", "50000")), help="Draws for the optional sampled outputs")
    p.add_argument("--lead-hours", type=float, default=48, help="Forecast cutoff before kickoff")
    p.add_argument("--dashboard-url", default="http://localhost:8501", help="Dashboard URL")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("daily", help="Daily pipeline: refresh data, results and players, forecast every upcoming game")
    sub.add_parser("sunday", help="Sunday service: sources, validation, learning, the weekend slate")
    sub.add_parser("results", help="Refresh final scores from ESPN and result sites")
    sub.add_parser("sources", help="Collect every external web source")
    sub.add_parser("backtest", help="Point-in-time backtest of the season so far")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.action == "sunday":
        sunday(args)
    else:
        pipeline(args, {"daily": "forecast"}.get(args.action, args.action))


if __name__ == "__main__":
    main()
