"""Run the complete Dockerized CFB forecast pipeline from Python."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def run(*args: str, read_only_data: bool = False) -> None:
    command = ["docker", "run", "--rm"]
    command += ["-v", f"{DATA}:/app/data:{'ro' if read_only_data else 'rw'}"]
    command += ["-v", f"{ARTIFACTS}:/app/artifacts:rw"]
    command += [os.environ.get("CFB_IMAGE", "cfb-forecast:local"), *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    season = os.environ.get("CFB_FORECAST_SEASON", "2026")
    draws = os.environ.get("CFB_FORECAST_DRAWS", "50000")
    image = os.environ.get("CFB_IMAGE", "cfb-forecast:local")
    DATA.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    schedule = DATA / f"schedule_{season}.csv"
    advanced = DATA / f"adv_team_{season}.parquet"
    if not schedule.exists() or not advanced.exists():
        run("fetch", "--data", "/app/data", "--seasons", season)

    run("preflight", "--data", "/app/data", "--artifacts", "/app/artifacts", "--season", season, read_only_data=True)
    run("results-refresh", "--data", "/app/data", "--out", "/app/artifacts/results", "--season", season, read_only_data=True)
    run("weekly-forecast", "--data", "/app/data", "--model", "/app/artifacts/backtest/model", "--out", "/app/artifacts", "--season", season, "--draws", draws)
    run("season-backtest", "--data", "/app/data", "--model", "/app/artifacts/backtest/model", "--out", "/app/artifacts/backtest", "--season", season, "--draws", draws, "--lead-hours", "48", read_only_data=True)

    expected = [
        ARTIFACTS / "latest-weekly" / "forecasts.csv",
        ARTIFACTS / "backtest" / f"season-{season}" / "summary.json",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Pipeline completed without expected artifacts: {missing}")
    print("Pipeline completed successfully.")
    for path in expected:
        print(path)


if __name__ == "__main__":
    main()
