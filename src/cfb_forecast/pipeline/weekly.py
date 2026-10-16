"""Weekly forecast orchestration for upcoming FBS games."""

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import os
import pandas as pd

from ..ingest.data import load_data, timestamp, sha256
from ..modeling.forecast import forecast
from ..evaluation.evaluate import metrics
from ..ingest.sources import SOURCE_REGISTRY


def validate_previous_week_forecast(forecast_dir, data_root, season, week, as_of=None):
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
    at = timestamp(as_of or datetime.now(timezone.utc).isoformat())
    manifest_path = forecast_dir / "run_manifest.json"
    if manifest_path.exists():
        from .integrity import verify_run_integrity
        verify_run_integrity(forecast_dir, json.loads(manifest_path.read_text()))
    predictions = pd.read_csv(prediction_path)
    if "as_of" in predictions and (pd.to_datetime(predictions.as_of, utc=True) >= pd.to_datetime(predictions.kickoff, utc=True)).any():
        raise ValueError("Saved forecast cutoff must precede kickoff")
    eligible = schedules[schedules.fbs_game & schedules.season.eq(int(season))
                         & schedules.result_usable & (schedules.result_available_at <= at)]
    outcomes = eligible[["game_id", "result_usable", "home_points", "away_points"]]
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
        "validated_at": at.isoformat(),
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
    adjustment_config=None,
    legacy_score=False,
):
    """Forecast the next upcoming week, or an explicitly selected week."""
    data_root = Path(data_root)
    out_root = Path(out_root)
    live = as_of is None
    as_of = timestamp(as_of or datetime.now(timezone.utc).isoformat())
    schedules, _, paths = load_data(data_root)
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
    if destination.exists():
        raise FileExistsError(f"Weekly forecast is immutable: {destination}; use publish-update to retry distribution")
    final = destination
    from uuid import uuid4
    destination = out_root / "work" / f"weekly-{season}-{week}-{uuid4().hex}"
    destination.mkdir(parents=True)
    context = None
    if live:
        from ..ingest.sources import collect_game_inputs
        from .sunday import capture_prices
        context, _ = collect_game_inputs(games, season, destination / "inputs")
        market_path = market_path or data_root / "market_history.csv"
        capture_prices(games, destination / "inputs", market_path)
        from ..ingest.cfbd_market import capture_cfbd_prices
        capture_cfbd_prices(games, season, destination / "inputs" / "raw", market_path)
        as_of = timestamp(datetime.now(timezone.utc).isoformat())
        if first_kickoff <= as_of:
            raise ValueError("A target game started while collecting pregame inputs")
    from ..evaluation.validation import validate_runs
    _, previous = validate_runs(out_root / "runs", schedules, as_of)
    (destination / "previous_week_validation.json").write_text(json.dumps(previous, indent=2))
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
        known_future=context,
        adjustment_config=adjustment_config,
        database=Path(os.environ.get("CFB_DATABASE", out_root / "cfb_forecast.sqlite")),
        legacy_score=legacy_score,
    )
    if live and timestamp(datetime.now(timezone.utc).isoformat()) >= first_kickoff:
        raise ValueError("A target game started during calculation")
    from .decisions import apply_decisions
    complete = context.set_index("game_id").reindex(result.game_id).inputs_complete.fillna(False).to_numpy() if context is not None else None
    result = apply_decisions(result, complete)
    result.to_csv(destination / "forecasts.csv", index=False)
    (destination / "forecasts.json").write_text(result.to_json(orient="records", indent=2))
    manifest = {
        "run_id": final.name,
        "status": "completed",
        "prediction_sha256": sha256(destination / "forecasts.csv"),
        "source_hashes": {p.name: sha256(p) for p in (paths or [])},
        "model_sha256": sha256(Path(model_root) / "model.json") if (Path(model_root) / "model.json").exists() else None,
        "validation": previous,
        "input_hashes": {p.relative_to(destination / "inputs").as_posix(): sha256(p)
                         for p in (destination / "inputs").rglob("*") if p.is_file()},
        "season": int(season),
        "week": week,
        "as_of": as_of.isoformat(),
        "first_kickoff": first_kickoff.isoformat(),
        "last_kickoff": last_kickoff.isoformat(),
        "games": int(len(result)),
        "output": str(final),
        "forecast_sources": {name: url.format(season=int(season)) for name, url in SOURCE_REGISTRY.items()},
        "validation_source": SOURCE_REGISTRY["cfbtxt_validation"],
    }
    pd.DataFrame([manifest]).to_json(
        destination / "weekly_manifest.json", orient="records", indent=2
    )

    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    final.parent.mkdir(parents=True, exist_ok=True)
    destination.rename(final)
    destination = final
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
        manifest["distribution"] = {"status": "failed", "error": type(exc).__name__, "message": str(exc)}
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    (latest / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    if manifest["distribution"].get("status") == "published":
        (out_root / "latest" / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return result, manifest
