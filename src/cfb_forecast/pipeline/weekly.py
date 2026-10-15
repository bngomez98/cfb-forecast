"""Weekly forecast orchestration for upcoming FBS games."""

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import pandas as pd

from ..ingest.data import load_data, timestamp
from ..modeling.forecast import forecast
from ..evaluation.evaluate import metrics
from ..ingest.sources import SOURCE_REGISTRY


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
    from ..storage.database import connect, record_weekly_validation
    from ..paths import database_path
    with closing(connect(database_path())) as conn:
        record_weekly_validation(conn, result)
    return result


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
