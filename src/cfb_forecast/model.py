"""Score prediction head and residual-covariance calibration.

The head is a plain regularized linear model (:class:`sklearn.linear_model.Ridge`)
fitted on standardized pregame rating features. The same equation predicts
both the home and away expected score; which side the row applies to is
encoded by the sign of the venue column in the feature layout.

Residual covariance for the simulated score distribution is estimated from
an out-of-fit validation season; its eigenvalues are floored at 1 so the
simulation can never see a degenerate joint distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .features import feature_columns

#: Minimum number of fitted games before the model is willing to fit a head.
_MIN_FIT_GAMES = 100

#: Minimum number of out-of-fit games required to calibrate the residual
#: covariance.
_MIN_CALIBRATION_GAMES = 50


class ScoreHead:
    """Standardized ridge regression over the pregame feature vector.

    Parameters
    ----------
    variant:
        ``"score"`` for the restricted six-column head, ``"pbp"`` for the
        full 26-column head.
    alpha:
        Ridge regularization strength. Larger values shrink coefficients
        toward zero.
    """

    def __init__(self, variant: str = "score", alpha: float = 10.0) -> None:
        self.variant = variant
        self.alpha = float(alpha)
        self.cols = feature_columns(variant)
        self.scaler = StandardScaler()
        self.regressor = Ridge(alpha=self.alpha)

    def fit(self, F: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> ScoreHead:
        """Fit the head on games selected by ``mask``.

        Each selected game contributes two training rows (home and away) with
        the corresponding side's final score as the target.
        """
        x = F[mask][:, :, self.cols].reshape(-1, len(self.cols))
        y = Y[mask, :, 0].reshape(-1)
        if len(y) < _MIN_FIT_GAMES:
            raise ValueError(
                f"Need at least {_MIN_FIT_GAMES // 2} completed games with finite "
                "pregame features to fit"
            )
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Need at least 50 completed games with finite pregame features to fit")
        self.regressor.fit(self.scaler.fit_transform(x), y)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        """Predict ``(n, 2)`` expected scores, floored at zero.

        Accepts a single pregame feature pair shaped ``(2, FEATURE_WIDTH)`` or
        a stacked batch shaped ``(n, 2, FEATURE_WIDTH)``.
        """
        F = np.asarray(F, dtype=float)
        if F.ndim == 2:
            F = F[None, :, :]
        x = F[:, :, self.cols].reshape(-1, len(self.cols))
        if not np.isfinite(x).all():
            raise ValueError("Missing or invalid pregame features")
        predicted = self.regressor.predict(self.scaler.transform(x)).reshape(-1, 2)
        return np.maximum(predicted, 0.0)

    def to_dict(self) -> dict:
        """Serialize the fitted head to a JSON-compatible dict."""
        return {
            "variant": self.variant,
            "alpha": self.alpha,
            "cols": self.cols,
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "coef": self.regressor.coef_.tolist(),
            "intercept": float(self.regressor.intercept_),
        }

    @classmethod
    def from_dict(cls, value: dict) -> ScoreHead:
        """Restore a fitted head from :meth:`to_dict`'s output.

        Validates that the saved column layout matches this variant's
        expected columns and that the stored mean and scale vectors are
        length-consistent.
        """
        head = cls(value["variant"], value["alpha"])
        if head.cols != value["cols"]:
            raise ValueError("Feature schema incompatible with saved model")
        mean = np.array(value["mean"], dtype=float)
        scale = np.array(value["scale"], dtype=float)
        if mean.shape != (len(head.cols),) or scale.shape != (len(head.cols),):
            raise ValueError("Saved scaler shape does not match the feature column set")
        coef = np.array(value["coef"], dtype=float)
        if coef.shape != (len(head.cols),):
            raise ValueError("Saved regression coefficients do not match the feature column set")
        head.scaler.mean_ = mean
        head.scaler.scale_ = scale
        head.scaler.var_ = scale ** 2
        head.scaler.n_features_in_ = len(head.cols)
        head.regressor.coef_ = coef
        head.regressor.intercept_ = value["intercept"]
        head.regressor.n_features_in_ = len(head.cols)
        return head


def residual_covariance(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Estimate the joint-score residual covariance from an out-of-fit set.

    The empirical covariance's eigenvalues are floored at 1 so the matrix is
    strictly positive-definite and the simulated score distribution never
    collapses to a degenerate line.
    """
    residual = actual - predicted
    if len(residual) < _MIN_CALIBRATION_GAMES:
        raise ValueError(
            f"At least {_MIN_CALIBRATION_GAMES} out-of-fit calibration games are required"
        )
    cov = np.cov(residual.T)
    values, vectors = np.linalg.eigh(cov)
    return (vectors * np.maximum(values, 1.0)) @ vectors.T


def write_model(path, model: dict) -> None:
    """Write a model dict as ``model.json`` under ``path``, creating directories."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.json").write_text(json.dumps(model, indent=2))


def load_model(path) -> dict:
    """Load a model dict from ``path`` and verify its schema version."""
    value = json.loads((Path(path) / "model.json").read_text())
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported model schema")
    return value
