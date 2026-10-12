"""Historical retraining and forecast orchestration.

Each invocation retrains from the available historical seasons. Source files are
read-only: no schedules, market history, or ledgers are appended.
"""
from pathlib import Path
import json
import pandas as pd

from .backtest import run_backtest
from .forecast import forecast


def run_retrained_forecast(
    data_root,
    out_root,
    season,
    warmup=2018,
    train_start=2019,
    validation=None,
    draws=50000,
    as_of=None,
    through=None,
):
    """Retrain on completed historical seasons before ``season`` and forecast it.

    Every invocation rebuilds the model from source data. The requested season is
    excluded from fitting, even when its schedule exists.
    """
    season = int(season)
    validation = int(validation if validation is not None else season - 2)
    test = season - 1
    if not warmup < train_start < validation < test < season:
        raise ValueError("Require warmup < train_start < validation < prior season < forecast season")

    root = Path(out_root)
    training_root = root / "pretraining"
    _, manifest = run_backtest(
        data_root=data_root,
        out_root=training_root,
        warmup=warmup,
        train_start=train_start,
        validation=validation,
        test=test,
    )

    data_root = Path(data_root)
    schedules = pd.concat(
        [pd.read_csv(path) for path in sorted(data_root.glob("schedule_*.csv"))],
        ignore_index=True,
    )
    schedules["start_date"] = pd.to_datetime(schedules["start_date"], utc=True)
    season_games = schedules[(schedules["season"] == season) & schedules["start_date"].notna()]
    if season_games.empty:
        raise ValueError(f"No schedule games found for forecast season {season}")

    first = season_games["start_date"].min()
    last = season_games["start_date"].max()
    forecast_as_of = pd.Timestamp(as_of) if as_of else first - pd.Timedelta(hours=1)
    if forecast_as_of.tzinfo is None:
        raise ValueError("as_of requires an explicit timezone")
    forecast_through = pd.Timestamp(through) if through else last + pd.Timedelta(minutes=1)
    if forecast_through.tzinfo is None:
        raise ValueError("through requires an explicit timezone")

    forecast_root = root / f"forecast-{season}"
    predictions = forecast(
        data_root=data_root,
        model_root=training_root / "model",
        out_root=forecast_root,
        as_of=forecast_as_of.isoformat(),
        through=forecast_through.isoformat(),
        season=season,
        draws=draws,
    )
    predictions["actual_home_win"] = pd.NA
    predictions["actual_home_margin"] = pd.NA
    predictions["forecast_type"] = "historical_pretrained_no_append"
    predictions.to_csv(forecast_root / "predictions.csv", index=False)
    summary = {
        "forecast_season": season,
        "training_through_season": test,
        "validation_season": validation,
        "games_forecast": int(len(predictions)),
        "as_of": forecast_as_of.isoformat(),
        "through": forecast_through.isoformat(),
        "selected_variant": manifest["selected_variant"],
        "source_policy": "read-only historical source files; retrain on every run; no appending",
    }
    (forecast_root / "historical_forecast_manifest.json").write_text(json.dumps(summary, indent=2))
    return predictions, summary
