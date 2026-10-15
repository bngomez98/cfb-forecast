#!/usr/bin/env python3
"""Refresh and validate the CFB forecasting data pipeline.

This script checks public providers, refreshes integrated datasets, optionally
fetches raw PBP, validates every local data layer, refreshes ESPN results,
runs tests, generates a weekly forecast, and builds distributions.

Run inside pythonlib1:
    python update_forecast.py --season 2026
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_DATA = ROOT / "data"
DEFAULT_ARTIFACTS = ROOT / "artifacts"

# Public endpoints and repositories referenced by the project catalog.
SOURCES = {
    "ESPN scoreboard API": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?limit=1",
    "ESPN site API": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
    "SportsDataverse data releases": "https://github.com/sportsdataverse/sportsdataverse-data",
    "SportsDataverse schedules": "https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_schedules",
    "SportsDataverse advanced team": "https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_team",
    "SportsDataverse PBP": "https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_pbp",
    "SportsDataverse betting": "https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_betting",
    "cfbfastR schedules": "https://github.com/sportsdataverse/cfbfastR-data",
    "cfbfastR data": "https://github.com/sportsdataverse/cfbfastR-cfb-data",
    "NCAA predictions": "https://github.com/ssmarathontrading/ncaa_predictions",
}

# Optional research layers listed in available_data_catalog.json. They are
# checked as sources but are not silently mixed into the production model.
OPTIONAL_LAYERS = {
    "rosters": "https://github.com/sportsdataverse/cfbfastR-data",
    "injuries": "https://github.com/sportsdataverse/cfbfastR-data",
    "drives": "https://github.com/sportsdataverse/sportsdataverse-data",
    "adv_situational": "https://github.com/sportsdataverse/sportsdataverse-data",
    "special_teams": "https://github.com/sportsdataverse/sportsdataverse-data",
    "player_box_scores": "https://github.com/sportsdataverse/cfbfastR-data",
    "power_index": "https://github.com/ssmarathontrading/ncaa_predictions",
}


def inventory_workspace(workspace: Path, output: Path) -> None:
    """Record every nested file while excluding generated/cache directories."""
    excluded = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}
    files = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        try:
            relative = path.relative_to(workspace)
            files.append({
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
            })
        except OSError:
            continue
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"workspace": str(workspace), "files": files}, indent=2))
    print(f"OK workspace inventory: {len(files)} files -> {output}", flush=True)


def check_sources() -> None:
    print("\n==> Verify public ESPN, GitHub, and SportsDataverse sources", flush=True)
    headers = {"User-Agent": "cfb-forecast-updater/0.3"}
    for label, url in {**SOURCES, **OPTIONAL_LAYERS}.items():
        last_error: Exception | None = None
        for attempt in range(1, 4):
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    # The scoreboard payload is larger than a short probe, so parse the
                    # whole body. Other sources only need a successful response.
                    body = response.read() if label == "ESPN scoreboard API" else response.read(1024)
                    if not 200 <= response.status < 400:
                        raise RuntimeError(f"{label} returned HTTP {response.status}")
                    if label == "ESPN scoreboard API":
                        json.loads(body.decode("utf-8"))
                print(f"OK {label}: HTTP {response.status}", flush=True)
                last_error = None
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                print(f"RETRY {label} attempt {attempt}: {exc}", flush=True)
                time.sleep(2 * attempt)
        if last_error is not None:
            raise last_error


def verify_manifest(data_dir: Path) -> None:
    manifest = data_dir / "source_manifest.json"
    if not manifest.exists():
        print("No source_manifest.json found; checksum validation skipped", flush=True)
        return
    payload = json.loads(manifest.read_text())
    for name, metadata in payload.get("files", {}).items():
        path = data_dir / name
        expected = metadata.get("sha256")
        if path.exists() and expected:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise RuntimeError(f"SHA-256 mismatch: {path}")
    print("OK source manifest checksums", flush=True)


def verify_data(data_dir: Path, season: int, raw_pbp: bool) -> None:
    from cfb_forecast.data import load_data

    print("\n==> Verify integrated datasets", flush=True)
    schedules, advanced, loaded = load_data(data_dir)
    if schedules.empty:
        raise RuntimeError("No schedule rows loaded")
    if advanced.empty:
        raise RuntimeError("No advanced-team rows loaded")
    schedule = data_dir / f"schedule_{season}.csv"
    advanced_path = data_dir / f"adv_team_{season}.parquet"
    for path in (schedule, advanced_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Required dataset missing: {path}")
    if raw_pbp:
        pbp = data_dir / f"pbp_{season}.parquet"
        if not pbp.exists() or pbp.stat().st_size == 0:
            raise RuntimeError(f"Requested raw PBP dataset missing: {pbp}")
    print(f"OK schedules: {len(schedules)} rows", flush=True)
    print(f"OK advanced-team rows: {len(advanced)}", flush=True)
    print(f"OK files loaded: {len(loaded)}", flush=True)
    for pattern in ("schedule_*", "adv_team_*", "pbp_*", "market*", "*roster*", "*injur*", "*drive*", "*situat*", "*special*", "*player*", "*power*"):
        for path in sorted(data_dir.glob(pattern)):
            if path.is_file() and path.stat().st_size:
                print(f"FOUND data layer: {path.name}", flush=True)
    verify_manifest(data_dir)


def run(label: str, *args: str) -> None:
    """Run one pipeline stage with this interpreter and stop the update if it fails."""
    print(f"\n== {label} ==", flush=True)
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode:
        raise SystemExit(f"FAILED: {label} (exit {result.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("season", nargs="?", type=int, default=2026, help="Season; default: 2026")
    parser.add_argument("draws", nargs="?", type=int, default=50000, help="Simulation draws; default: 50000")
    parser.add_argument("--seasons", nargs="+", type=int, default=None, help="Additional seasons to download")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--raw-pbp", "--pbp", action="store_true", help="Download raw play-by-play data")
    parser.add_argument("--no-sources", "--offline", dest="skip_source_check", action="store_true", help="Skip online source checks")
    parser.add_argument("--no-fetch", dest="skip_fetch", action="store_true", help="Do not download datasets")
    parser.add_argument("--no-results", dest="skip_results", action="store_true", help="Do not refresh ESPN results")
    parser.add_argument("--no-forecast", dest="skip_forecast", action="store_true", help="Do not generate a forecast")
    parser.add_argument("--no-tests", dest="skip_tests", action="store_true", help="Do not run tests")
    parser.add_argument("--no-build", dest="skip_build", action="store_true", help="Do not build distributions")
    parser.add_argument("--quick", action="store_true", help="Skip tests and package build")
    args = parser.parse_args()
    if args.quick:
        args.skip_tests = True
        args.skip_build = True
    data_dir = args.data.resolve()
    artifacts_dir = args.artifacts.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    seasons = args.seasons or [args.season]
    workspace = args.workspace.resolve()
    inventory_path = (args.inventory or (artifacts_dir / "workspace_inventory.json")).resolve()
    inventory_workspace(workspace, inventory_path)

    if not args.skip_source_check:
        check_sources()
    if not args.skip_fetch:
        fetch_args = ["-m", "cfb_forecast", "fetch", "--data", str(data_dir), "--seasons", *map(str, seasons), "--refresh"]
        if args.raw_pbp:
            fetch_args.append("--raw-pbp")
        run("Refresh integrated public datasets", *fetch_args)
    verify_data(data_dir, args.season, args.raw_pbp)
    if not args.skip_results:
        run("Refresh ESPN results", "-m", "cfb_forecast", "results-refresh", "--data", str(data_dir), "--out", str(artifacts_dir / "results"), "--season", str(args.season))
    if not args.skip_forecast:
        run("Generate weekly forecast", "-m", "cfb_forecast", "weekly-forecast", "--data", str(data_dir), "--model", str(artifacts_dir / "training" / "model"), "--out", str(artifacts_dir), "--season", str(args.season), "--draws", str(args.draws))
    if not args.skip_tests:
        run("Run tests", "-m", "pytest", "-q")
    if not args.skip_build:
        run("Build distributions", "-m", "build")
    print("\nUpdate completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
