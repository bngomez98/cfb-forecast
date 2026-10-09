"""Forecast orchestration for upcoming FBS games.

``run_daily_forecast`` is the one ongoing entry point: as of now, it forecasts
every remaining upcoming FBS game in the season from the current state (built
from all completed games to date) and refreshes ``artifacts/latest``. It is
re-run each day; no per-game ledger or chronological tracking is kept.
"""
from pathlib import Path
from datetime import datetime, timezone
import shutil
import pandas as pd

from .data import load_data, timestamp
from .forecast import forecast


def run_daily_forecast(data_root, model_root, out_root, season, as_of=None, market_path=None, draws=50000):
    """Forecast the full remaining upcoming FBS slate from the current state."""
    data_root = Path(data_root)
    out_root = Path(out_root)
    as_of = timestamp(as_of or datetime.now(timezone.utc).isoformat())
    schedules, _, _ = load_data(data_root)
    upcoming = schedules[
        schedules.fbs_game
        & (schedules.season == int(season))
        & (schedules.start_date > as_of)
        & ~schedules.result_usable
    ].copy()
    if upcoming.empty:
        raise ValueError(f"No upcoming FBS games remain in season {season}")
    last_kickoff = upcoming.start_date.max()
    stamp = as_of.strftime("%Y%m%dT%H%M%SZ")
    destination = out_root / "forecasts" / stamp
    result = forecast(
        data_root=data_root, model_root=model_root, out_root=destination,
        as_of=as_of.isoformat(), through=last_kickoff.isoformat(), season=int(season),
        market_path=market_path, draws=draws,
        target_game_ids=upcoming.game_id.astype(int).tolist(),
    )
    weeks = sorted(upcoming.week.dropna().astype(int).unique())
    manifest = {
        "season": int(season), "as_of": as_of.isoformat(), "games": int(len(result)),
        "weeks": [int(w) for w in weeks], "next_week": int(weeks[0]) if weeks else None,
        "first_kickoff": upcoming.start_date.min().isoformat(), "last_kickoff": last_kickoff.isoformat(),
        "output": str(destination),
    }
    pd.DataFrame([manifest]).to_json(destination / "forecast_run_manifest.json", orient="records", indent=2)
    latest = out_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(destination, latest)
    return result, manifest


def run_weekly_forecast(
    data_root,
    model_root,
    out_root,
    season,
    week=None,
    as_of=None,
    market_path=None,
    draws=50000,
):
    """Forecast the next upcoming week, or an explicitly selected week."""
    data_root = Path(data_root)
    out_root = Path(out_root)
    as_of = timestamp(as_of or datetime.now(timezone.utc).isoformat())
    schedules, _, _ = load_data(data_root)
    games = schedules[
        schedules.fbs_game
        & (schedules.season == int(season))
        & (schedules.start_date > as_of)
        & ~schedules.result_usable
    ].copy()
    if week is None:
        if games.empty:
            raise ValueError(f"No upcoming FBS games remain in season {season}")
        week = int(games.sort_values("start_date").iloc[0].week)
    else:
        week = int(week)
    games = games[games.week == week].copy()
    if games.empty:
        raise ValueError(f"No upcoming FBS games found for season {season}, week {week}")

    first_kickoff = games.start_date.min()
    last_kickoff = games.start_date.max()
    if first_kickoff <= as_of:
        raise ValueError("Weekly forecast cutoff must precede every selected kickoff")

    destination = out_root / "weekly" / f"{int(season)}-week-{week}"
    result = forecast(
        data_root=data_root,
        model_root=model_root,
        out_root=destination,
        as_of=as_of.isoformat(),
        through=last_kickoff.isoformat(),
        season=int(season),
        market_path=market_path,
        draws=draws,
        target_game_ids=games.game_id.astype(int).tolist(),
    )
    manifest = {
        "season": int(season),
        "week": week,
        "as_of": as_of.isoformat(),
        "first_kickoff": first_kickoff.isoformat(),
        "last_kickoff": last_kickoff.isoformat(),
        "games": int(len(result)),
        "output": str(destination),
    }
    pd.DataFrame([manifest]).to_json(destination / "weekly_manifest.json", orient="records", indent=2)

    latest = out_root / "latest-weekly"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(destination, latest)
    return result, manifest
