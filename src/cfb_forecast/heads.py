"""Conditional score heads: the production prediction layer.

This module replaces the rating-features head for *live* forecasting. Each
head consumes only the conditioning matrix built by
:func:`cfb_forecast.conditional.build_conditional_matrix` — rolling recent
scores, opponent-recent-performance, venue, weather, injuries and roster
production — and never touches a team rating object. That is the deliberate
design choice from the literature review: predict directly from game data
(South & Egros, 2020; Tsagris et al., 2024; Horvat et al., 2023), keeping an
outcome-only Elo as a control arm rather than the primary method.

Three experts are implemented, matching the blend in
:mod:`cfb_forecast.adaptive`:

``ridge``
    L2-penalized linear regression on standardized differences between the
    two sides' conditioning vectors (home minus away), predicting margin and
    total. Regularization is required because early-season samples are thin
    and several blocks (weather, injuries) arrive with missing indicators
    (South & Egros found lasso-class models strongest on exactly this data).

``logit``
    Logistic classification of the winner on the same features. The evidence
    is consistent that classification beats regression for binary targets
    (Soto-Valero, 2016; Wang, 2025), so this head supplies the win
    probability fed to the blend.

``gbm``
    Gradient-boosted trees, included because tree methods exceeded 70%
    accuracy alongside lasso in the college-football comparison and capture
    interactions (e.g. altitude x weather) the linear heads cannot.

Every head follows the same discipline:

* Training rows come only from games whose result was available strictly
  before the prediction cutoff; the conditioning features are lagged by
  construction inside ``build_conditional_matrix``.
* Missing conditioning values are imputed with the training median *and*
  flagged by the explicit ``*_missing`` indicator columns, so "unknown"
  never masquerades as zero effect.
* Margin predictions are converted to a home-win probability with a logistic
  link calibrated on training residuals (Fredrickson-style scale estimate),
  giving every head a comparable probability output for the adaptive blend.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge

#: Minimum settled games before a conditional head is trusted to fit.
_MIN_FIT_GAMES = 80

#: Columns excluded from differencing: pure per-side context that has no
#: meaningful home-minus-away interpretation is already handled upstream.
_PROBABILITY_SCALE = 14.0  # points of margin ~ one logit unit (calibrated below)


def _difference_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    """Collapse a paired home/away conditioning matrix into matchup terms.

    The canonical column layout emits ``home_X``/``away_X`` pairs plus
    precomputed ``diff_X`` and matchup columns. We keep every ``diff_`` and
    ``match_`` column verbatim, difference any remaining ``home_``/``away_``
    pair that lacks a ``diff_`` counterpart, and pass through neutral context
    columns (venue, weather) unchanged.
    """
    columns = list(matrix.columns)
    out = pd.DataFrame(index=matrix.index)
    consumed: set[str] = set()
    for name in columns:
        if name.startswith(("diff_", "match_", "expected_total")) or name in {
            "venue_altitude_m",
            "venue_altitude_missing",
            "weather_indoor",
            "weather_missing",
        }:
            out[name] = matrix[name].to_numpy(dtype=float)
            consumed.add(name)
    for name in columns:
        if not name.startswith("home_"):
            continue
        partner = "away_" + name[len("home_") :]
        diff = "diff_" + name[len("home_") :]
        if partner in columns and diff not in consumed and name not in consumed:
            out[f"delta_{name[len('home_') :]}"] = matrix[name].to_numpy(dtype=float) - matrix[
                partner
            ].to_numpy(dtype=float)
            consumed.add(name)
            consumed.add(partner)
    return out


class ConditionalHeads:
    """Ridge / logistic / boosted-tree experts over the conditioning matrix.

    Parameters
    ----------
    alphas:
        Ridge penalty for the margin and total heads.
    c_logit:
        Inverse regularization strength for the logistic winner head.
    n_estimators, learning_rate:
        Boosting hyper-parameters for the tree expert.
    """

    EXPERTS = ("ridge", "logit", "gbm")

    def __init__(
        self,
        *,
        alpha_margin: float = 12.0,
        alpha_total: float = 12.0,
        c_logit: float = 0.1,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
    ) -> None:
        self.alpha_margin = float(alpha_margin)
        self.alpha_total = float(alpha_total)
        self.c_logit = float(c_logit)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.imputer = SimpleImputer(strategy="median")
        self.margin_model = Ridge(alpha=self.alpha_margin)
        self.total_model = Ridge(alpha=self.alpha_total)
        self.logit_model = LogisticRegression(C=self.c_logit, max_iter=2000, solver="lbfgs")
        self.gbm_model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=4,
            l2_regularization=1.0,
            early_stopping=False,
        )
        self.feature_names_: list[str] = []
        self.probability_scale_ = _PROBABILITY_SCALE
        self.trained_at_games = 0

    # -- fitting -----------------------------------------------------------
    def fit(self, matrix: pd.DataFrame, outcomes: pd.DataFrame) -> ConditionalHeads:
        """Fit all experts on settled games.

        ``matrix`` is the conditioning frame from
        :func:`~cfb_forecast.conditional.build_conditional_matrix`;
        ``outcomes`` must carry ``home_points`` and ``away_points`` aligned
        row-for-row with ``matrix``. No smoothing is applied to the observed
        scores themselves; they enter exactly as recorded.
        """
        outcomes = outcomes.reset_index(drop=True)
        matrix = matrix.reset_index(drop=True)
        if len(outcomes) != len(matrix):
            raise ValueError("Conditioning matrix and outcomes must align row-for-row")
        decided = outcomes.home_points.notna() & outcomes.away_points.notna()
        decided &= outcomes.home_points != outcomes.away_points
        X = _difference_frame(matrix)[decided.to_numpy()]
        y_margin = (outcomes.home_points - outcomes.away_points)[decided].to_numpy(dtype=float)
        y_total = (outcomes.home_points + outcomes.away_points)[decided].to_numpy(dtype=float)
        y_win = (y_margin > 0).astype(int)
        if len(y_margin) < _MIN_FIT_GAMES:
            raise ValueError(
                f"Need at least {_MIN_FIT_GAMES} settled decided games to fit; got {len(y_margin)}"
            )
        Z = self.imputer.fit_transform(X.to_numpy(dtype=float))
        self.feature_names_ = list(X.columns)
        self.margin_model.fit(Z, y_margin)
        self.total_model.fit(Z, y_total)
        self.logit_model.fit(Z, y_win)
        self.gbm_model.fit(Z, y_margin)
        # Calibrate the margin->probability link on training residuals:
        # p = sigmoid(margin_hat / scale), scale from the residual spread so
        # one-sided blowouts do not produce probabilities pinned at 0/1.
        residual = y_margin - self.margin_model.predict(Z)
        spread = float(np.std(residual))
        self.probability_scale_ = max(4.0, min(40.0, spread / 2.0))
        self.trained_at_games = int(len(y_margin))
        return self

    # -- prediction --------------------------------------------------------
    def _prepare(self, matrix: pd.DataFrame) -> np.ndarray:
        X = _difference_frame(matrix)
        X = X.reindex(columns=self.feature_names_)
        return self.imputer.transform(X.to_numpy(dtype=float))

    def predict(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Return per-game expert outputs for upcoming fixtures."""
        Z = self._prepare(matrix)
        ridge_margin = self.margin_model.predict(Z)
        expected_total = self.total_model.predict(Z)
        gbm_margin = self.gbm_model.predict(Z)
        logit_p = self.logit_model.predict_proba(Z)[:, 1]
        ridge_p = 1.0 / (1.0 + np.exp(-ridge_margin / self.probability_scale_))
        gbm_p = 1.0 / (1.0 + np.exp(-gbm_margin / self.probability_scale_))
        return pd.DataFrame(
            {
                "ridge_margin": ridge_margin,
                "ridge_expected_total": expected_total,
                "ridge_p_home": ridge_p,
                "logit_p_home": logit_p,
                "gbm_margin": gbm_margin,
                "gbm_p_home": gbm_p,
            },
            index=matrix.index,
        )

    # -- persistence -------------------------------------------------------
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import pickle

        payload = {
            "imputer": self.imputer,
            "margin_model": self.margin_model,
            "total_model": self.total_model,
            "logit_model": self.logit_model,
            "gbm_model": self.gbm_model,
            "feature_names_": self.feature_names_,
            "probability_scale_": self.probability_scale_,
            "trained_at_games": self.trained_at_games,
            "hyperparameters": {
                "alpha_margin": self.alpha_margin,
                "alpha_total": self.alpha_total,
                "c_logit": self.c_logit,
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
            },
        }
        with open(path / "heads.pkl", "wb") as handle:
            pickle.dump(payload, handle)
        manifest = {
            "trained_at_games": self.trained_at_games,
            "probability_scale": self.probability_scale_,
            "features": len(self.feature_names_),
        }
        with open(path / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    @classmethod
    def load(cls, path) -> ConditionalHeads:
        path = Path(path)
        import pickle

        with open(path / "heads.pkl", "rb") as handle:
            payload = pickle.load(handle)
        heads = cls(**payload["hyperparameters"])
        heads.imputer = payload["imputer"]
        heads.margin_model = payload["margin_model"]
        heads.total_model = payload["total_model"]
        heads.logit_model = payload["logit_model"]
        heads.gbm_model = payload["gbm_model"]
        heads.feature_names_ = payload["feature_names_"]
        heads.probability_scale_ = payload["probability_scale_"]
        heads.trained_at_games = payload["trained_at_games"]
        return heads


def expert_probability_columns() -> dict[str, str]:
    """Map blend expert names onto head output columns."""
    return {
        "ridge": "ridge_p_home",
        "logit": "logit_p_home",
        "gbm": "gbm_p_home",
    }


def sigmoid(value: float, scale: float = _PROBABILITY_SCALE) -> float:
    """Logistic link used for margin->probability conversion (scalar form)."""
    return 1.0 / (1.0 + math.exp(-value / scale))
