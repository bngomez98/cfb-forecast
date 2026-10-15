"""Layered empirical-Bayes football model.

This module uses closed-form Gaussian updates for team offense/defense latent
states, Poisson score rates, a Gaussian margin layer, and a ridge regression
adjustment. It is intentionally deterministic and stores sufficient statistics
rather than pretending to run MCMC.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, poisson
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class LayeredConfig:
    prior_mean: float = 0.0
    prior_variance: float = 4.0
    observation_variance: float = 9.0
    ridge_alpha: float = 10.0
    home_advantage: float = 2.5
    poisson_floor: float = 0.5
    poisson_cap: float = 70.0
    margin_scale: float = 14.0


class LayeredBayesModel:
    """Empirical-Bayes layered score model trained only on prior games."""

    def __init__(self, config: LayeredConfig | None = None):
        self.config = config or LayeredConfig()
        self.teams: list[int] = []
        self.offense: dict[int, tuple[float, float]] = {}
        self.defense: dict[int, tuple[float, float]] = {}
        self.ridge_scaler = StandardScaler()
        self.ridge = Ridge(alpha=self.config.ridge_alpha)
        self.ridge_fitted = False
        self.training_through_season: int | None = None

    def _posterior(self, prior_mean: float, prior_variance: float, observations: list[float]):
        if not observations:
            return prior_mean, prior_variance
        precision = 1.0 / prior_variance + len(observations) / self.config.observation_variance
        mean = (
            prior_mean / prior_variance
            + sum(observations) / self.config.observation_variance
        ) / precision
        return float(mean), float(1.0 / precision)

    def fit(self, games, feature_frame=None):
        required = {"home_id", "away_id", "home_points", "away_points", "season", "result_available_at", "result_usable"}
        missing = required - set(games.columns)
        if missing:
            raise ValueError(f"Layered model missing columns: {sorted(missing)}")
        usable = games[games.result_usable].copy().sort_values("result_available_at")
        if usable.empty:
            raise ValueError("Layered model requires completed prior outcomes")
        self.training_through_season = int(usable.season.max()) if len(usable) else None
        self.teams = sorted(set(usable.home_id.astype(int)) | set(usable.away_id.astype(int)))
        off_obs = {team: [] for team in self.teams}
        def_obs = {team: [] for team in self.teams}
        self.ridge_fitted = False
        regression_x, regression_y = [], []
        if feature_frame is not None:
            if not feature_frame.index.equals(games.index) or "available_at" not in feature_frame:
                raise ValueError("Features must align with games and include available_at")
            import pandas as pd
            if not (pd.to_datetime(feature_frame.available_at, utc=True) < games.start_date).all():
                raise ValueError("Features must be available before kickoff")
            predictors = feature_frame.drop(columns="available_at")
            forbidden = {"home_points", "away_points", "home_win", "margin", "total", "result_usable", "result_available_at"}
            if forbidden.intersection(predictors.columns) or predictors.empty:
                raise ValueError("Only pregame predictor columns are permitted")
            regression_x = predictors.loc[usable.index].to_numpy(dtype=float)
            if not np.isfinite(regression_x).all():
                raise ValueError("Pregame predictors must be finite")
        for row in usable.itertuples(index=False):
            home, away = int(row.home_id), int(row.away_id)
            home_points, away_points = float(row.home_points), float(row.away_points)
            off_obs[home].append(home_points - (0.0 if row.neutral_site else self.config.home_advantage))
            off_obs[away].append(away_points)
            def_obs[home].append(-away_points)
            def_obs[away].append(-home_points)
            regression_y.append(home_points - away_points)
        self.offense = {team: self._posterior(self.config.prior_mean, self.config.prior_variance, values) for team, values in off_obs.items()}
        self.defense = {team: self._posterior(self.config.prior_mean, self.config.prior_variance, values) for team, values in def_obs.items()}
        if len(regression_x) >= 3:
            self.ridge.fit(self.ridge_scaler.fit_transform(np.asarray(regression_x)), regression_y)
            self.ridge_fitted = True
        return self

    def _team(self, values, team: int):
        return values.get(int(team), (self.config.prior_mean, self.config.prior_variance))

    def predict(self, home_id: int, away_id: int, neutral_site: bool = False, feature_row=None) -> dict:
        home_off, home_off_var = self._team(self.offense, home_id)
        away_off, away_off_var = self._team(self.offense, away_id)
        home_def, home_def_var = self._team(self.defense, home_id)
        away_def, away_def_var = self._team(self.defense, away_id)
        home_latent = home_off + away_def + (0.0 if neutral_site else self.config.home_advantage)
        away_latent = away_off + home_def
        home_score = max(self.config.poisson_floor, min(self.config.poisson_cap, 21.0 + 0.12 * home_latent))
        away_score = max(self.config.poisson_floor, min(self.config.poisson_cap, 21.0 + 0.12 * away_latent))
        if self.ridge_fitted and feature_row is not None:
            adjustment = float(self.ridge.predict(self.ridge_scaler.transform(np.asarray(feature_row).reshape(1, -1)))[0])
            home_score = max(self.config.poisson_floor, home_score + 0.04 * adjustment)
            away_score = max(self.config.poisson_floor, away_score - 0.04 * adjustment)
        margin = home_score - away_score
        variance = self.config.margin_scale**2 + home_off_var + away_off_var + home_def_var + away_def_var
        margin_sd = float(np.sqrt(max(variance, 1e-9)))
        return {
            "expected_home_score": float(home_score),
            "expected_away_score": float(away_score),
            "projected_home_margin": float(margin),
            "projected_total": float(home_score + away_score),
            "home_win_probability": float(norm.cdf(margin / margin_sd)),
            "home_score_poisson_pmf": poisson.pmf(round(home_score), home_score).item(),
            "away_score_poisson_pmf": poisson.pmf(round(away_score), away_score).item(),
            "margin_sd": margin_sd,
            "model_layers": "empirical_bayes_latent+poisson_score+gaussian_margin+ridge_adjustment",
        }

    def to_dict(self):
        return {
            "schema_version": 1,
            "config": self.config.__dict__,
            "teams": self.teams,
            "offense": {str(k): list(v) for k, v in self.offense.items()},
            "defense": {str(k): list(v) for k, v in self.defense.items()},
            "training_through_season": self.training_through_season,
            "ridge_fitted": self.ridge_fitted,
            "ridge": {"mean": self.ridge_scaler.mean_.tolist(), "scale": self.ridge_scaler.scale_.tolist(), "coef": self.ridge.coef_.tolist(), "intercept": float(self.ridge.intercept_)} if self.ridge_fitted else None,
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, value):
        model = cls(LayeredConfig(**value["config"]))
        model.teams = [int(team) for team in value["teams"]]
        model.offense = {int(k): tuple(v) for k, v in value["offense"].items()}
        model.defense = {int(k): tuple(v) for k, v in value["defense"].items()}
        model.training_through_season = value.get("training_through_season")
        ridge = value.get("ridge")
        model.ridge_fitted = ridge is not None
        if ridge:
            model.ridge_scaler.mean_ = np.asarray(ridge["mean"])
            model.ridge_scaler.scale_ = np.asarray(ridge["scale"])
            model.ridge_scaler.var_ = model.ridge_scaler.scale_ ** 2
            model.ridge_scaler.n_features_in_ = len(ridge["mean"])
            model.ridge.coef_ = np.asarray(ridge["coef"])
            model.ridge.intercept_ = ridge["intercept"]
            model.ridge.n_features_in_ = len(ridge["coef"])
        return model
