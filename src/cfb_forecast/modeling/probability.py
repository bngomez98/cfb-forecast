"""Prior-season probability calibration and chronological candidate evaluation."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from ..ingest.data import require_columns, sha256
from ..evaluation.evaluate import metrics, calibration, paired_bootstrap
from .model import load_model, write_model


def probabilities(values):
    p = np.asarray(values, dtype=float)
    if (
        p.ndim != 1
        or not len(p)
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Probabilities must be a nonempty finite vector in [0, 1]")
    return p


def apply_calibration(p, spec):
    p = probabilities(p)
    if spec.get("method") != "regularized_platt" or spec.get("schema_version") != 1:
        raise ValueError("Unsupported probability calibration schema")
    slope, intercept = float(spec["slope"]), float(spec["intercept"])
    if not np.isfinite([slope, intercept]).all() or slope < 0:
        raise ValueError("Invalid probability calibration coefficients")
    return expit(slope * logit(np.clip(p, 1e-9, 1 - 1e-9)) + intercept)


def fit_calibration(
    p, y, through_season, regularization=0.01, min_games=100, max_change=None
):
    p = probabilities(p)
    y = np.asarray(y, dtype=float)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Calibration outcomes must be binary and match probabilities")
    if min_games < 50 or len(p) < min_games or len(np.unique(y)) < 2:
        raise ValueError(
            "Calibration needs at least min_games >= 50 games and both outcomes"
        )
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Calibration regularization must be positive")
    x = logit(np.clip(p, 1e-9, 1 - 1e-9))

    def objective(theta):
        z = theta[0] * x + theta[1]
        penalty = theta - np.array([1.0, 0.0])
        loss = np.mean(np.logaddexp(0, z) - y * z) + regularization * penalty @ penalty
        residual = expit(z) - y
        gradient = (
            np.array([np.mean(residual * x), residual.mean()])
            + 2 * regularization * penalty
        )
        return loss, gradient

    if max_change is not None:
        if not np.isfinite(max_change) or not 0 < max_change < 1:
            raise ValueError("max_change must be between zero and one")
        clean = np.unique(np.r_[p, np.linspace(0.0001, 0.9999, 1001)])
        clean_x = logit(np.clip(clean, 1e-9, 1 - 1e-9))
        constraint = {
            "type": "ineq",
            "fun": lambda theta: (
                max_change - np.abs(expit(theta[0] * clean_x + theta[1]) - clean)
            ),
        }
        result = minimize(
            objective,
            [1.0, 0.0],
            jac=True,
            method="SLSQP",
            bounds=[(0, None), (None, None)],
            constraints=[constraint],
            options={"maxiter": 200, "ftol": 1e-10},
        )
    else:
        result = minimize(
            objective,
            [1.0, 0.0],
            jac=True,
            method="L-BFGS-B",
            bounds=[(0, None), (None, None)],
        )
    if not result.success:
        raise ValueError(f"Probability calibration failed: {result.message}")
    return {
        "schema_version": 1,
        "method": "regularized_platt",
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "training_through_season": int(through_season),
        "training_games": len(p),
        "regularization": float(regularization),
        "max_probability_change": max_change,
    }


def stress_calibration(p, y, spec, radius=0.1):
    """Adversarial coefficient sensitivity on labeled evaluation data, never deployment."""
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Stress radius must be nonnegative")
    p = probabilities(p)
    y = np.asarray(y, dtype=float)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Stress outcomes must be binary and match probabilities")
    x = logit(np.clip(p, 1e-9, 1 - 1e-9))
    theta = np.array([spec["slope"], spec["intercept"]], float)
    delta = np.zeros(2)

    def loss(d):
        z = (theta[0] + d[0]) * x + theta[1] + d[1]
        return float(np.mean(np.logaddexp(0, z) - y * z))

    best, best_loss = delta.copy(), loss(delta)
    for _ in range(100):
        z = (theta[0] + delta[0]) * x + theta[1] + delta[1]
        residual = expit(z) - y
        gradient = np.array([np.mean(residual * x), residual.mean()])
        norm = np.linalg.norm(gradient)
        if norm < 1e-12 or radius == 0:
            break
        delta += radius / 10 * gradient / norm
        delta[0] = max(delta[0], -theta[0])
        delta *= min(1.0, radius / max(np.linalg.norm(delta), 1e-12))
        value = loss(delta)
        if value > best_loss:
            best, best_loss = delta.copy(), value
    return {
        "radius": radius,
        "coefficient_delta": best.tolist(),
        "coefficient_delta_norm": float(np.linalg.norm(best)),
        "baseline_log_loss": loss(np.zeros(2)),
        "stress_log_loss": best_loss,
        "method": "projected_gradient_ascent",
        "interpretation": "Local sensitivity experiment, not a forecast adjustment or a certified worst-case bound",
    }


def forecast_probability(raw, model, season):
    """Calibrate winner probability only; preserve the score distribution separately."""
    spec = model.get("win_probability_calibration")
    if spec is None:
        return float(raw)
    if int(spec["training_through_season"]) >= int(season):
        raise ValueError(
            "Probability calibration training must end before the forecast season"
        )
    return float(apply_calibration([raw], spec)[0])


def audit_probabilities(
    input_path,
    out_root,
    probability_column="score_p",
    outcome_column="home_win",
    min_games=100,
    regularization=0.01,
    model_root=None,
    max_change=None,
    stress_radius=0.1,
):
    """Score each season with a calibrator fit exclusively on earlier seasons.

    The input probabilities must themselves be out-of-sample. Timestamp checks
    verify pregame cutoffs, but cannot certify the original head's training history.
    """
    f = pd.read_csv(input_path)
    require_columns(
        f,
        ["game_id", "season", probability_column, outcome_column],
        "probability audit",
    )
    if f.game_id.isna().any() or f.game_id.duplicated().any():
        raise ValueError("Probability audit needs unique nonmissing game IDs")
    seasons = pd.to_numeric(f.season, errors="raise")
    if seasons.isna().any() or not np.equal(seasons, np.floor(seasons)).all():
        raise ValueError("Season must be an integer")
    f["season"] = seasons.astype(int)
    f["raw_probability"] = probabilities(f[probability_column])
    y = pd.to_numeric(f[outcome_column], errors="raise")
    if not y.isin([0, 1]).all():
        raise ValueError("Audit outcomes must be binary")
    f["home_win"] = y.astype(int)
    kickoff_col = "kickoff" if "kickoff" in f else "start_date"
    cutoff_col = "as_of" if "as_of" in f else "forecast_cutoff"
    require_columns(f, [kickoff_col, cutoff_col], "pregame timestamps")
    kickoff = pd.to_datetime(f[kickoff_col], utc=True, errors="raise")
    cutoff = pd.to_datetime(f[cutoff_col], utc=True, errors="raise")
    if kickoff.isna().any() or cutoff.isna().any() or (cutoff >= kickoff).any():
        raise ValueError("Every probability must have a cutoff strictly before kickoff")
    f["start_date"] = kickoff
    for col in ["head_training_through_season", "calibration_training_through_season"]:
        if col in f and (pd.to_numeric(f[col], errors="raise") >= f.season).any():
            raise ValueError(f"{col} must precede every evaluation season")
    f["calibrated_probability"] = f.raw_probability
    f["calibration_status"] = "insufficient_prior_data"
    reports, bins, stress = [], [], []
    for year in sorted(f.season.unique()):
        train, test = f[f.season < year], f[f.season == year]
        eligible = len(train) >= min_games and train.home_win.nunique() == 2
        if eligible:
            spec = fit_calibration(
                train.raw_probability,
                train.home_win,
                int(train.season.max()),
                regularization,
                min_games,
                max_change,
            )
            f.loc[test.index, "calibrated_probability"] = apply_calibration(
                test.raw_probability, spec
            )
            f.loc[test.index, "calibration_status"] = "evaluated_on_later_season"
            stress.append(
                {
                    "season": int(year),
                    **stress_calibration(
                        test.raw_probability, test.home_win, spec, stress_radius
                    ),
                }
            )
        for name, column in [
            ("raw", "raw_probability"),
            ("calibrated", "calibrated_probability"),
        ]:
            test = f.loc[test.index]
            reports.append(
                {
                    "season": int(year),
                    "model": name,
                    "prior_training_games": len(train),
                    "candidate_evaluated": eligible,
                    **metrics(test.home_win, test[column]),
                }
            )
            table = calibration(test.home_win, test[column])
            table["season"], table["model"] = int(year), name
            bins.append(table)
    evaluated = f[f.calibration_status == "evaluated_on_later_season"]
    summary = {
        "schema_version": 1,
        "input_sha256": sha256(input_path),
        "probability_column": probability_column,
        "games": len(f),
        "evaluated_games": len(evaluated),
        "regularization": regularization,
        "min_games": min_games,
        "production_changed": False,
        "max_probability_change": max_change,
        "stress_tests": stress,
        "limitation": "Input probabilities must come from frozen out-of-sample forecasts. Timestamp validation does not reconstruct source vintages or certify upstream head training.",
    }
    if len(evaluated):
        summary["raw_metrics"] = metrics(evaluated.home_win, evaluated.raw_probability)
        summary["candidate_metrics"] = metrics(
            evaluated.home_win, evaluated.calibrated_probability
        )
        summary["delta_log_loss"] = (
            summary["candidate_metrics"]["log_loss"]
            - summary["raw_metrics"]["log_loss"]
        )
        if evaluated.start_date.dt.strftime("%G-%V").nunique() >= 2:
            summary["paired_comparison"] = paired_bootstrap(
                evaluated, "calibrated_probability", "raw_probability"
            )
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    f.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(reports).to_csv(out / "metrics.csv", index=False)
    pd.concat(bins, ignore_index=True).to_csv(out / "calibration.csv", index=False)
    if len(f) >= min_games and f.home_win.nunique() == 2:
        spec = fit_calibration(
            f.raw_probability,
            f.home_win,
            int(f.season.max()),
            regularization,
            min_games,
            max_change,
        )
        spec["input_sha256"], spec["probability_column"] = (
            summary["input_sha256"],
            probability_column,
        )
        (out / "candidate_calibrator.json").write_text(json.dumps(spec, indent=2))
        if model_root:
            model = load_model(model_root)
            if probability_column != f"{model['variant']}_p":
                raise ValueError(
                    "Candidate model variant must match its audit probability column"
                )
            if model.get("win_probability_calibration"):
                raise ValueError("Audit needs the original uncalibrated model")
            spec["base_model_sha256"] = sha256(Path(model_root) / "model.json")
            model["win_probability_calibration"] = spec
            write_model(out / "candidate_model", model)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
