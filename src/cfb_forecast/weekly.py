"""Forecast orchestration for upcoming FBS games.

``run_daily_forecast`` is the one ongoing entry point: as of now, it forecasts every
remaining upcoming FBS game in the season from the current state (all completed games to date,
refit live) and refreshes ``<out>/latest``. ``run_weekly_forecast`` is the same model limited to
one week for the Sunday slate.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import pandas as pd

from .data import load_data, timestamp
from .forecast import forecast
from .evaluate import metrics
from .sources import SOURCE_REGISTRY


def validate_previous_week_forecast(forecast_dir, data_root, season, week):
    """Settle the prior weekly forecast against newly available official results.

    CFB TXT is the designated weekly validation source; the normalized schedule
    remains the scoring authority after its results are ingested.
    """
    forecast_dir = Path(forecast_dir)
    data_root = Path(data_root)
    prediction_path = forecast_dir / "forecasts.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    schedules, _, _ = load_data(data_root)
    predictions = pd.read_csv(prediction_path)
    outcomes = schedules[["game_id", "result_usable", "home_points", "away_points"]]
    settled = predictions.merge(outcomes, on="game_id", how="left", validate="one_to_one")
    settled = settled[settled.result_usable.fillna(False)].copy()
    if settled.empty:
        return {"season": int(season), "week": int(week), "status": "awaiting_results", "games": 0}
    settled["home_win"] = (settled.home_points > settled.away_points).astype(int)
    settled["actual_margin"] = settled.home_points - settled.away_points
    settled["actual_total"] = settled.home_points + settled.away_points
    result = metrics(
        settled.home_win,
        settled.home_win_probability,
        settled.projected_home_margin,
        settled.actual_margin,
        settled.projected_total,
        settled.actual_total,
    )
    result.update({
        "season": int(season),
        "week": int(week),
        "run_id": forecast_dir.name,
        "status": "settled",
        "games": int(len(settled)),
        "validation_source": SOURCE_REGISTRY["cfbtxt_validation"],
        "validated_at": datetime.now(timezone.utc).isoformat(),
    })
    if "projected_home_margin" in settled:
        result["spread_mae"] = float((settled.actual_margin - settled.projected_home_margin).abs().mean())
    (forecast_dir / "weekly_validation.json").write_text(json.dumps(result, indent=2))
    from contextlib import closing
    from .database import connect, record_weekly_validation
    from .paths import database_path
    with closing(connect(database_path())) as conn:
        record_weekly_validation(conn, result)
    return result


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


DAILY_EXTRAS = ("changes.csv", "player_ratings.csv", "team_ratings.csv", "forecast_run_manifest.json")


def run_daily_forecast(data_root, model_root, out_root, season, as_of=None, market_path=None, draws=50000, keep=14, artifacts=None, publish=True):
    """Forecast the full remaining upcoming FBS slate from the current state.

    Writes a dated snapshot under ``<out_root>/forecasts/`` (``out_root`` is normally
    ``<artifacts>/daily``). The snapshot also carries ``changes.csv`` (what moved since the previous
    daily run) and the named ``player_ratings.csv`` / ``team_ratings.csv`` behind the forecast.
    With ``publish`` the run is then published to ``<artifacts>/latest``, SQLite, the web board,
    logs and email through ``publish_forecast_update`` and the extras are copied next to it.
    """
    data_root = Path(data_root)
    out_root = Path(out_root)
    artifacts = Path(artifacts) if artifacts is not None else out_root.parent
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
    manifest["pruned_snapshots"] = prune_snapshots(out_root, keep)
    if publish:
        try:
            from .publisher import publish_forecast_update
            manifest["distribution"] = publish_forecast_update(destination, artifacts, run_id=f"daily-{stamp}", season=int(season))
        except Exception as exc:
            manifest["distribution"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        latest = artifacts / "latest"
        latest.mkdir(parents=True, exist_ok=True)
        for name in DAILY_EXTRAS:
            if (destination / name).exists():
                shutil.copy2(destination / name, latest / name)
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
        raise ValueError(
            f"No upcoming FBS games found for season {season}, week {week}"
        )

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
        "forecast_sources": {name: url.format(season=int(season)) for name, url in SOURCE_REGISTRY.items()},
        "validation_source": SOURCE_REGISTRY["cfbtxt_validation"],
    }
    pd.DataFrame([manifest]).to_json(
        destination / "weekly_manifest.json", orient="records", indent=2
    )

    latest = out_root / "latest-weekly"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(destination, latest)
    try:
        from .publisher import publish_forecast_update
        manifest["distribution"] = publish_forecast_update(
            destination, out_root, run_id=f"{int(season)}-week-{week}", season=int(season)
        )
    except Exception as exc:
        manifest["distribution"] = {"status": "failed", "error": type(exc).__name__}
    return result, manifest
