"""Command-line entry point for independent weekly source collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingest.data import load_data, timestamp
from .ingest.external_sources import collect_extended_sources
from .ingest.sources import utc_now


def select_next_week(schedules, season: int, now):
    at = timestamp(now)
    eligible = schedules[
        schedules.fbs_game
        & (schedules.season == int(season))
        & (schedules.start_date > at)
    ].copy()
    if eligible.empty:
        return eligible
    week = int(eligible.sort_values("start_date").iloc[0].week)
    return eligible[eligible.week.astype(int) == week].copy()


def run(data_root="data", artifacts_root="artifacts/external-sources", season=2026):
    schedules, _, _ = load_data(data_root)
    now = utc_now()
    games = select_next_week(schedules, season, now)
    output = Path(artifacts_root) / f"{season}-week-{int(games.week.iloc[0]) if len(games) else 'none'}"
    output.mkdir(parents=True, exist_ok=True)
    report = collect_extended_sources(games, season, output)
    report["game_count"] = int(len(games))
    report["as_of"] = now
    (output / "run.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(output), "games": len(games), "sources": len(report["sources"])}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Collect independent CFB data sources and weekly schedule validation")
    parser.add_argument("--data", default="data")
    parser.add_argument("--artifacts", default="artifacts/external-sources")
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()
    run(args.data, args.artifacts, args.season)


if __name__ == "__main__":
    main()
