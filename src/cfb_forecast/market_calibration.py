"""Estimate how informative the market is relative to the football model.

The market-conditioned forecast treats the quoted spread and vig-free
moneyline as additional noisy measurements of the underlying future-game
relationship::

    Spread        ~ N(-E[margin]_football + delta_spread, sigma_spread^2)
    logit(p_mkt)  ~ N(logit(p_football) + delta_ML,       sigma_ML^2)

This module fits ``delta_*`` (systematic market bias) and ``sigma_*``
(measurement noise) by maximum likelihood on historical games where both a
pregame quote observed before kickoff and a final score exist.  The fitted
values are stored in the ledger's ``market_deltas`` table with their data
cutoff so every conditioned forecast records which calibration produced it.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def fit_margin_measurement(football_margins: np.ndarray, spreads: np.ndarray,
                           margin_variances: np.ndarray | None = None):
    """Fit ``delta_spread`` and ``sigma_spread`` by profile ML.

    Observation equation: ``-spread = football_margin + bias + noise``,
    where the residual variance combines the model's own predictive variance
    (when supplied) with the market's measurement noise.  Returns
    ``(delta_spread, sigma_spread)``.
    """
    obs = -np.asarray(spreads, dtype=float)
    mu = np.asarray(football_margins, dtype=float)
    resid = obs - mu
    keep = np.isfinite(resid)
    resid = resid[keep]
    if len(resid) < 30:
        return 0.0, 7.0
    var_pred = None
    if margin_variances is not None:
        var_pred = np.asarray(margin_variances, dtype=float)[keep]
    delta = float(np.median(resid))                      # systematic bias
    centered = resid - delta
    total_var = float(np.var(centered))
    if var_pred is not None:
        market_noise = max(total_var - float(np.mean(var_pred)), 1.0)
    else:
        market_noise = max(total_var, 1.0)
    return delta, float(np.sqrt(market_noise))


def fit_probability_measurement(p_football: np.ndarray, p_market: np.ndarray):
    """Fit ``delta_ML`` and ``sigma_ML`` for the moneyline-logit observation.

    Uses logistic regression of the *market* logit residual on nothing but an
    intercept (bias) plus the empirical dispersion of
    ``logit(p_market) - logit(p_football)`` (noise).  Returns
    ``(delta_ML, sigma_ML)``.
    """
    eps = 1e-6
    p_f = np.clip(np.asarray(p_football, dtype=float), eps, 1 - eps)
    p_m = np.clip(np.asarray(p_market, dtype=float), eps, 1 - eps)
    if len(p_f) < 30:
        return 0.0, 0.5
    def nll(delta):
        z = np.log(p_m / (1 - p_m)) - delta - np.log(p_f / (1 - p_f))
        scale = max(float(np.std(z)), 0.1)
        return float(np.sum((z / scale) ** 2) + len(z) * np.log(scale))
    res = minimize_scalar(nll, bounds=(-1.5, 1.5), method="bounded")
    delta_ml = float(res.x)
    z = np.log(p_m / (1 - p_m)) - delta_ml - np.log(p_f / (1 - p_f))
    sigma_ml = float(max(np.std(z), 0.1))
    return delta_ml, sigma_ml
