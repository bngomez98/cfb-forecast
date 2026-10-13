"""One forecasting model for college football games.

The hierarchical state supplies the pregame score location. Known-future
information (the market line, the total, rest, weather, and reported
injuries) shifts that location through a Koopman map. Partial known-future
values are filled from the games that already have them. Games in the same
week are reconciled toward that week's slate. The shifted location is the
Gaussian mean and the Poisson rate. The win probability is the logistic of
that margin. Spread prices are Monte Carlo frequencies of integer scores
drawn from that Gaussian.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from .analytic import score_summary
from .simulate import checked_covariance, game_seed, win_probability

METHOD = (
    "hierarchical state, known-future Koopman map, Gaussian scores, "
    "Poisson rates, logistic win probability, Monte Carlo spread"
)
KNOWN_COLUMNS = (
    "home_spread",
    "total_line",
    "market_home_probability",
    "home_rest_days",
    "away_rest_days",
    "weather_temperature_f",
    "weather_precipitation_probability",
    "home_reported_injuries",
    "away_reported_injuries",
)
RIDGE = 0.1
MIN_GAMES = 30


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def rest_days(games, schedules):
    """Days since each team's previous kickoff. The current game is excluded."""
    history = schedules.loc[schedules.start_date.notna(), ["home_id", "away_id", "start_date"]]
    appearances = pd.concat(
        [
            history.rename(columns={"home_id": "team_id"})[["team_id", "start_date"]],
            history.rename(columns={"away_id": "team_id"})[["team_id", "start_date"]],
        ],
        ignore_index=True,
    ).sort_values("start_date")
    frame = games.reset_index(drop=True)
    result = pd.DataFrame({"game_id": frame.game_id.to_numpy()})
    for side, column in (("home", "home_id"), ("away", "away_id")):
        left = frame[["game_id", column, "start_date"]].rename(columns={column: "team_id"})
        left = left.sort_values("start_date")
        merged = pd.merge_asof(
            left,
            appearances.rename(columns={"start_date": "previous_start"}),
            left_on="start_date",
            right_on="previous_start",
            by="team_id",
            allow_exact_matches=False,
        )
        days = (merged.start_date - merged.previous_start).dt.total_seconds() / 86400.0
        result = result.merge(
            pd.DataFrame({"game_id": merged.game_id, f"{side}_rest_days": days}),
            on="game_id",
            how="left",
        )
    return result


def known_future_matrix(frame):
    """Align known-future columns. Missing inputs stay missing."""
    rows = []
    for record in frame.to_dict("records"):
        home = _finite(record.get("home_ml_decimal"))
        away = _finite(record.get("away_ml_decimal"))
        market_probability = None
        if home and away and home > 1 and away > 1:
            overround = 1.0 / home + 1.0 / away
            market_probability = (1.0 / home) / overround
        record = {**record, "market_home_probability": market_probability}
        rows.append([_finite(record.get(column)) for column in KNOWN_COLUMNS])
    if not rows:
        return np.zeros((0, len(KNOWN_COLUMNS)))
    return np.array(rows, dtype=float)


class ForecastModel:
    """Fitted parameters of the one forecasting model."""

    def __init__(self):
        self.ready = False
        self.columns = []
        self.center = np.zeros(0)
        self.scale = np.ones(0)
        self.koopman = np.zeros((0, 2))
        self.fill = np.zeros(0)
        self.reconcile = 0.0
        self.logistic_intercept = 0.0
        self.logistic_coef = 0.0
        self.logistic_ready = False
        self.score_low = np.array([0.0, 0.0])
        self.score_high = np.array([150.0, 150.0])

    def to_dict(self):
        return {
            "schema_version": 1,
            "method": METHOD,
            "ready": self.ready,
            "columns": self.columns,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "koopman": self.koopman.tolist(),
            "fill": self.fill.tolist(),
            "reconcile": self.reconcile,
            "logistic_intercept": self.logistic_intercept,
            "logistic_coef": self.logistic_coef,
            "logistic_ready": self.logistic_ready,
            "score_low": self.score_low.tolist(),
            "score_high": self.score_high.tolist(),
        }

    @classmethod
    def from_dict(cls, value):
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported forecasting model schema")
        model = cls()
        model.ready = bool(value["ready"])
        model.columns = list(value["columns"])
        model.center = np.asarray(value["center"], float)
        model.scale = np.asarray(value["scale"], float)
        model.koopman = np.asarray(value["koopman"], float)
        model.fill = np.asarray(value["fill"], float)
        model.reconcile = float(value["reconcile"])
        model.logistic_intercept = float(value["logistic_intercept"])
        model.logistic_coef = float(value["logistic_coef"])
        model.logistic_ready = bool(value["logistic_ready"])
        model.score_low = np.asarray(value["score_low"], float)
        model.score_high = np.asarray(value["score_high"], float)
        return model

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def fit(cls, location, actual, known_future, home_wins, weeks=None):
        location = np.asarray(location, float)
        actual = np.asarray(actual, float)
        known = np.asarray(known_future, float)
        wins = np.asarray(home_wins, int)
        if location.ndim != 2 or location.shape[1] != 2 or actual.shape != location.shape:
            raise ValueError("Score location and actual scores must be (games, 2)")
        if known.shape != (len(location), len(KNOWN_COLUMNS)):
            raise ValueError("Known-future features do not match the model columns")
        if wins.shape != (len(location),) or not np.isin(wins, [0, 1]).all():
            raise ValueError("Home wins must be a binary outcome per game")
        model = cls()
        model.score_low = np.maximum(np.quantile(actual, 0.01, axis=0), 0.0)
        model.score_high = np.maximum(np.quantile(actual, 0.99, axis=0), model.score_low + 1.0)
        usable = np.isfinite(known).mean(axis=0) >= 0.2
        model.columns = [name for name, keep in zip(KNOWN_COLUMNS, usable) if keep]
        if model.columns and len(location) >= MIN_GAMES:
            selected = known[:, usable]
            observed = np.isfinite(selected).any(axis=1)
            fill = np.nanmean(selected, axis=0)
            fill = np.where(np.isfinite(fill), fill, 0.0)
            completed = np.where(np.isfinite(selected), selected, fill)
            center = completed[observed].mean(axis=0)
            scale = completed[observed].std(axis=0)
            scale = np.where(scale < 1e-6, 1.0, scale)
            design = (completed[observed] - center) / scale
            residual = actual[observed] - location[observed]
            gram = design.T @ design + RIDGE * np.eye(design.shape[1])
            koopman = np.linalg.solve(gram, design.T @ residual)
            model.fill = fill
            model.center = center
            model.scale = scale
            model.koopman = koopman
            model.ready = True
            shifted = location.copy()
            shifted[observed] = location[observed] + design @ koopman
        else:
            shifted = location.copy()
        if weeks is not None:
            weeks = np.asarray(weeks)
            parent = np.zeros_like(location)
            for week in pd.unique(weeks):
                mask = weeks == week
                parent[mask] = shifted[mask].mean(axis=0)
            delta = parent - shifted
            denom = float(np.sum(delta * delta))
            if denom > 1.0:
                weight = float(np.sum(delta * (actual - shifted)) / denom)
                model.reconcile = float(np.clip(weight, 0.0, 1.0))
        margin = shifted[:, 0] - shifted[:, 1]
        if len(wins) >= MIN_GAMES and len(np.unique(wins)) == 2 and np.isfinite(margin).all():
            link = LogisticRegression(C=1.0)
            link.fit(margin.reshape(-1, 1), wins)
            model.logistic_intercept = float(link.intercept_[0])
            model.logistic_coef = float(link.coef_[0, 0])
            model.logistic_ready = True
        return model

    def _shift(self, location, known):
        location = np.asarray(location, float)
        known = np.asarray(known, float)
        if not self.ready or not self.columns:
            return location.copy()
        index = [KNOWN_COLUMNS.index(name) for name in self.columns]
        selected = known[:, index]
        completed = np.where(np.isfinite(selected), selected, self.fill)
        design = (completed - self.center) / self.scale
        return location + design @ self.koopman

    def apply(self, location, known_future, covariance, market_rows, game_ids, draws=4000):
        location = np.asarray(location, float)
        known = np.asarray(known_future, float)
        covariance = checked_covariance(covariance)
        if known.shape != (len(location), len(KNOWN_COLUMNS)):
            raise ValueError("Known-future features do not match the model columns")
        shifted = self._shift(location, known)
        if len(shifted) > 1 and self.reconcile:
            parent = shifted.mean(axis=0)
            shifted = (1.0 - self.reconcile) * shifted + self.reconcile * parent
        shifted = np.clip(shifted, self.score_low, self.score_high)
        rows = []
        for means, known_row, market, game_id in zip(shifted, known, market_rows, game_ids):
            summary = score_summary(means, covariance, **market)
            summary["gaussian_home_win_probability"] = summary["home_win_probability"]
            margin = float(means[0] - means[1])
            if self.logistic_ready:
                probability = float(expit(self.logistic_intercept + self.logistic_coef * margin))
            else:
                probability = float(win_probability(means, covariance))
            probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
            summary["home_win_probability"] = probability
            summary["expected_home_score"] = float(means[0])
            summary["expected_away_score"] = float(means[1])
            summary["projected_home_margin"] = margin
            summary["projected_total"] = float(means.sum())
            summary["model_spread"] = -margin
            summary["distribution_method"] = "forecast_model"
            summary["score_model"] = "gaussian_location_poisson_rate"
            summary["win_link"] = "logistic" if self.logistic_ready else "gaussian"
            summary["monte_carlo_draws"] = int(draws)
            if draws:
                seed = game_seed(int(game_id))
                samples = np.random.default_rng(seed).multivariate_normal(means, covariance, size=int(draws))
                integer_margin = np.rint(samples[:, 0]).clip(min=0) - np.rint(samples[:, 1]).clip(min=0)
                spread = market.get("home_spread")
                if spread is not None and _finite(spread) is not None:
                    threshold = -float(spread)
                    home = float(np.mean(integer_margin > threshold))
                    away = float(np.mean(integer_margin < threshold))
                    summary["home_cover_probability"] = home
                    summary["away_cover_probability"] = away
                    summary["spread_push_probability"] = 1.0 - home - away
                line = market.get("total_line")
                if line is not None and _finite(line) is not None:
                    total = np.rint(samples).clip(min=0).sum(axis=1)
                    over = float(np.mean(total > float(line)))
                    under = float(np.mean(total < float(line)))
                    summary["over_probability"] = over
                    summary["under_probability"] = under
                    summary["total_push_probability"] = 1.0 - over - under
            rows.append(summary)
        return rows
