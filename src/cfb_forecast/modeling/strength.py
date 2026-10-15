"""Opponent-adjusted, time-decayed strength baseline from real final scores."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.linear_model import LogisticRegression

from ..ingest.data import load_data, sha256, timestamp
from ..evaluation.evaluate import metrics, calibration
from .ratings import cutoff_for_games


def strength_margin(
    schedules, targets, at, half_life_days=365.0, ridge_games=8.0, home_advantage=3.0
):
    at = timestamp(at)
    past = schedules[
        schedules.fbs_game
        & schedules.result_usable
        & (schedules.result_available_at < at)
    ]
    population = schedules[schedules.fbs_game]
    teams = sorted(set(population.home_id) | set(population.away_id))
    index = {int(team): i for i, team in enumerate(teams)}
    x = np.zeros((len(past), len(teams)))
    if len(past):
        rows = np.arange(len(past))
        x[rows, past.home_id.map(index)] = 1.0
        x[rows, past.away_id.map(index)] = -1.0
        age = (at - past.result_available_at).dt.total_seconds().to_numpy() / 86400
        weights = 2.0 ** (-age / half_life_days)
        adjusted = (past.home_points - past.away_points).to_numpy() - home_advantage * (
            ~past.neutral_site
        ).to_numpy()
        coef = solve(
            x.T @ (x * weights[:, None]) + ridge_games * np.eye(len(teams)),
            x.T @ (adjusted * weights),
            assume_a="pos",
        )
    else:
        coef = np.zeros(len(teams))
    return np.array(
        [
            coef[index[int(r.home_id)]]
            - coef[index[int(r.away_id)]]
            + (0 if r.neutral_site else home_advantage)
            for r in targets.itertuples(index=False)
        ]
    )


def strength_probability(schedules, targets, at, spec):
    if int(spec["training_through_season"]) >= int(targets.season.min()):
        raise ValueError("Strength probability training must precede forecast season")
    from scipy.special import expit

    margin = strength_margin(schedules, targets, at, **spec["settings"])
    return expit(float(spec["coefficient"]) * margin + float(spec["intercept"]))


def train_strength(data_root, output_root, warmup=2021, validation=2024, test=2025):
    schedules, _, paths = load_data(data_root)
    games = schedules[
        schedules.fbs_game
        & schedules.result_usable
        & schedules.season.between(warmup + 1, test)
    ].copy()
    games["forecast_cutoff"] = cutoff_for_games(games)
    games["strength_margin"] = np.nan
    for at, batch in games.groupby("forecast_cutoff"):
        games.loc[batch.index, "strength_margin"] = strength_margin(
            schedules[schedules.season >= warmup], batch, at
        )
    games["home_win"] = (games.home_points > games.away_points).astype(int)
    train = games[games.season < validation]
    val = games[games.season == validation]
    heldout = games[games.season == test].copy()
    if len(train) < 100 or len(val) < 50 or len(heldout) < 50:
        raise ValueError(
            "Strength training needs real training, validation and test seasons"
        )
    trials = []
    for c in [0.01, 0.1, 1.0, 10.0]:
        head = LogisticRegression(C=c).fit(train[["strength_margin"]], train.home_win)
        p = head.predict_proba(val[["strength_margin"]])[:, 1]
        trials.append({"C": c, **metrics(val.home_win, p)})
    best = min(trials, key=lambda row: row["log_loss"])
    fit = games[games.season <= validation]
    head = LogisticRegression(C=best["C"]).fit(fit[["strength_margin"]], fit.home_win)
    heldout["home_win_probability"] = head.predict_proba(heldout[["strength_margin"]])[
        :, 1
    ]
    result = metrics(heldout.home_win, heldout.home_win_probability)
    prod = LogisticRegression(C=best["C"]).fit(
        games[["strength_margin"]], games.home_win
    )
    spec = {
        "schema_version": 1,
        "method": "time_decayed_opponent_adjusted_strength",
        "training_through_season": test,
        "selection_season": validation,
        "regularization_C": best["C"],
        "coefficient": float(prod.coef_[0, 0]),
        "intercept": float(prod.intercept_[0]),
        "settings": {
            "half_life_days": 365.0,
            "ridge_games": 8.0,
            "home_advantage": 3.0,
        },
        "source_hashes": {p.name: sha256(p) for p in paths},
        "evaluation": result,
    }
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(out / "predictions.csv", index=False)
    calibration(heldout.home_win, heldout.home_win_probability).to_csv(
        out / "calibration.csv", index=False
    )
    pd.DataFrame(trials).to_csv(out / "selection.csv", index=False)
    (out / "model.json").write_text(json.dumps(spec, indent=2))
    return spec
