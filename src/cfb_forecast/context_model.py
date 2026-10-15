"""Learn context effects from recorded pregame inputs and later real outcomes."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from .evaluate import metrics


def feature_columns(frame):
    return sorted(
        c
        for c in frame
        if (
            c.startswith("weather_")
            or (
                c.startswith(("home_", "away_"))
                and ("." in c or c.endswith("_reported_injuries"))
            )
        )
        and not c.endswith("_status")
        and pd.to_numeric(frame[c], errors="coerce").notna().any()
    )


def design(frame, columns, center=None, scale=None):
    values = (
        frame.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    values[~np.isfinite(values)] = np.nan
    if center is None:
        center = np.nanmedian(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
    center, scale = np.asarray(center), np.asarray(scale)
    missing = np.isnan(values)
    normalized = (np.where(missing, center, values) - center) / scale
    return np.c_[normalized, missing.astype(float)], center, scale


def fit_context(frame, columns, max_logit_change=0.2, penalty=1.0):
    x, center, scale = design(frame, columns)
    offset = logit(np.clip(frame.base_probability.to_numpy(float), 1e-9, 1 - 1e-9))
    y = frame.home_win.to_numpy(float)
    if len(frame) < 50 or len(np.unique(y)) != 2:
        raise ValueError(
            "Context training needs at least 50 real outcomes and both winners"
        )

    def objective(beta):
        delta = max_logit_change * np.tanh(x @ beta)
        z = offset + delta
        loss = np.mean(np.logaddexp(0, z) - y * z) + penalty * beta @ beta
        dz = max_logit_change * (1 - np.tanh(x @ beta) ** 2)
        gradient = x.T @ ((expit(z) - y) * dz) / len(y) + 2 * penalty * beta
        return loss, gradient

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise ValueError(f"Context fit failed: {result.message}")
    return {
        "schema_version": 1,
        "feature_columns": columns,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coefficients": result.x.tolist(),
        "max_logit_change": max_logit_change,
        "penalty": penalty,
        "training_games": len(frame),
        "training_through": str(frame.as_of.max()),
    }


def predict_context(probabilities, features, spec):
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported context model")
    x, _, _ = design(features, spec["feature_columns"], spec["center"], spec["scale"])
    delta = float(spec["max_logit_change"]) * np.tanh(
        x @ np.asarray(spec["coefficients"])
    )
    p = np.asarray(probabilities, float)
    adjusted = expit(logit(np.clip(p, 1e-9, 1 - 1e-9)) + delta)
    return adjusted, delta


def select_context_model(history):
    """Reserve the latest recorded week for selection; refit only if it improves."""
    if history.empty or len(history) < 150:
        return None, {"status": "insufficient_validated_games", "games": len(history)}
    history = history.sort_values("as_of").copy()
    batches = sorted(history.run_id.unique())
    if len(batches) < 3:
        return None, {"status": "insufficient_validation_weeks", "weeks": len(batches)}
    train = history[history.run_id != batches[-1]]
    test = history[history.run_id == batches[-1]]
    columns = feature_columns(train.drop(columns=["home_win"], errors="ignore"))
    if (
        not columns
        or len(train) < 100
        or len(test) < 20
        or train.home_win.nunique() < 2
    ):
        return None, {"status": "insufficient_context_coverage"}
    spec = fit_context(train, columns)
    candidate, _ = predict_context(test.base_probability, test, spec)
    raw_metrics = metrics(test.home_win, test.base_probability)
    candidate_metrics = metrics(test.home_win, candidate)
    report = {
        "status": "evaluated",
        "validation_run": batches[-1],
        "training_games": len(train),
        "validation_games": len(test),
        "base": raw_metrics,
        "candidate": candidate_metrics,
        "adopted": candidate_metrics["log_loss"] < raw_metrics["log_loss"]
        and candidate_metrics["brier"] <= raw_metrics["brier"],
    }
    return (fit_context(history, columns) if report["adopted"] else None), report
