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


def previous_snapshot(out_root, stamp):
    """Most recent earlier dated forecast snapshot directory, or None."""
    runs = sorted(d for d in (Path(out_root) / "forecasts").glob("*") if d.is_dir() and d.name < stamp and (d / "forecasts.csv").exists())
    return runs[-1] if runs else None


def forecast_changes(current, previous_dir):
    """What moved since the previous daily run, game by game.

    Returns the games present in both runs with prior and current win probability and
    projected margin, their differences, and the prior run's as-of time. Games that are new
    in this run (a schedule change) get blank prior columns.
    """
    keep = ["game_id", "away_team", "home_team", "kickoff", "as_of", "home_win_probability", "projected_home_margin", "projected_total"]
    cur = current[keep].copy()
    cols = ["game_id", "away_team", "home_team", "kickoff", "prior_as_of", "prior_home_win_probability", "home_win_probability",
            "win_probability_change", "prior_projected_home_margin", "projected_home_margin", "margin_change",
            "prior_projected_total", "projected_total", "total_change"]
    if previous_dir is None:
        nan = float("nan")
        out = cur.assign(prior_as_of=None, prior_home_win_probability=nan, win_probability_change=nan,
                         prior_projected_home_margin=nan, margin_change=nan, prior_projected_total=nan, total_change=nan)
        return out[cols]
    prev = pd.read_csv(Path(previous_dir) / "forecasts.csv", usecols=lambda c: c in keep)
    prev = prev.rename(columns={"as_of": "prior_as_of", "home_win_probability": "prior_home_win_probability",
                                "projected_home_margin": "prior_projected_home_margin", "projected_total": "prior_projected_total"})
    prev = prev[["game_id", "prior_as_of", "prior_home_win_probability", "prior_projected_home_margin", "prior_projected_total"]]
    out = cur.merge(prev, on="game_id", how="left")
    out["win_probability_change"] = out.home_win_probability - out.prior_home_win_probability
    out["margin_change"] = out.projected_home_margin - out.prior_projected_home_margin
    out["total_change"] = out.projected_total - out.prior_projected_total
    return out[cols].sort_values("win_probability_change", key=lambda c: c.abs(), ascending=False, na_position="last").reset_index(drop=True)


def prune_snapshots(out_root, keep):
    """Delete all but the newest ``keep`` dated forecast snapshots."""
    runs = sorted(d for d in (Path(out_root) / "forecasts").glob("*") if d.is_dir())
    removed = runs[:-keep] if keep and len(runs) > keep else []
    for d in removed:
        shutil.rmtree(d)
    return len(removed)


def run_daily_forecast(data_root, model_root, out_root, season, as_of=None, market_path=None, draws=50000, keep=14):
    """Forecast the full remaining upcoming FBS slate from the current state.

    Writes a dated snapshot under ``<out_root>/forecasts/`` and refreshes ``<out_root>/latest``.
    The snapshot also carries ``changes.csv`` (what moved since the previous daily run) and the
    named ``player_ratings.csv`` / ``team_ratings.csv`` behind the forecast.
    """
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
    changes = forecast_changes(result, previous_snapshot(out_root, stamp))
    changes.to_csv(destination / "changes.csv", index=False)
    names = {int(i): n for i, n in zip(pd.concat([schedules.home_id, schedules.away_id]), pd.concat([schedules.home_team, schedules.away_team]))}
    from .players import write_player_report
    report = write_player_report(data_root, destination, as_of, names)
    weeks = sorted(upcoming.week.dropna().astype(int).unique())
    manifest = {
        "season": int(season), "as_of": as_of.isoformat(), "games": int(len(result)),
        "weeks": [int(w) for w in weeks], "next_week": int(weeks[0]) if weeks else None,
        "first_kickoff": upcoming.start_date.min().isoformat(), "last_kickoff": last_kickoff.isoformat(),
        "changed_games": int(changes.win_probability_change.abs().gt(0.005).sum()),
        "rated_players": report["players"], "rated_teams": report["teams"], "output": str(destination),
    }
    pd.DataFrame([manifest]).to_json(destination / "forecast_run_manifest.json", orient="records", indent=2)
    latest = out_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(destination, latest)
    manifest["pruned_snapshots"] = prune_snapshots(out_root, keep)
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
