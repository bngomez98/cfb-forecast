"""Joint possession-level football forecast simulator.

The simulator consumes already-estimated football quantities. It does not train
or invent player data. Every draw samples one internally consistent matchup,
shared game-clock possessions, availability, terminal outcomes, and scores.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import expit, logit, softmax


@dataclass(frozen=True)
class JointSimulationConfig:
    draws: int = 50000
    touchdown_points: float = 7.0
    field_goal_points: float = 3.0
    safety_points: float = 2.0
    min_possessions: int = 4
    max_possessions: int = 24
    seed: int | None = 0


def _no_vig(home_decimal: float, away_decimal: float):
    home_raw = 1.0 / float(home_decimal)
    away_raw = 1.0 / float(away_decimal)
    total = home_raw + away_raw
    return home_raw / total, away_raw / total


def _normal_pair(rng, means, covariance, draws):
    means = np.asarray(means, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    covariance = (vectors * np.maximum(values, 1e-9)) @ vectors.T
    return rng.multivariate_normal(means, covariance, size=draws)


def _required(payload, name, default):
    value = payload.get(name, default)
    return float(value) if np.isscalar(value) else np.asarray(value, dtype=float)


def simulate_joint_game(
    home: dict,
    away: dict,
    conditions: dict | None = None,
    covariance: np.ndarray | None = None,
    market: dict | None = None,
    config: JointSimulationConfig | None = None,
) -> dict:
    """Return empirical joint score and market-conditioned distributions.

    ``home`` and ``away`` contain estimated football states, not raw source rows.
    Availability probabilities are integrated with Bernoulli draws per player.
    """
    cfg = config or JointSimulationConfig()
    if cfg.draws < 1000:
        raise ValueError("At least 1000 joint draws are required")
    conditions = conditions or {}
    market = market or {}
    rng = np.random.default_rng(cfg.seed)
    draws = int(cfg.draws)

    h_avail = np.asarray(home.get("availability_probabilities", [1.0]), dtype=float)
    a_avail = np.asarray(away.get("availability_probabilities", [1.0]), dtype=float)
    h_participation = rng.binomial(1, np.clip(h_avail, 0, 1), size=(draws, len(h_avail))).mean(axis=1)
    a_participation = rng.binomial(1, np.clip(a_avail, 0, 1), size=(draws, len(a_avail))).mean(axis=1)

    h_states = np.array([
        _required(home, "offense", 0.0), _required(home, "defense", 0.0),
        _required(home, "tempo", 0.0), _required(home, "drive_sustainability", 0.0),
        _required(home, "special_teams", 0.0),
    ], dtype=float)
    a_states = np.array([
        _required(away, "offense", 0.0), _required(away, "defense", 0.0),
        _required(away, "tempo", 0.0), _required(away, "drive_sustainability", 0.0),
        _required(away, "special_teams", 0.0),
    ], dtype=float)
    latent_cov = np.asarray(covariance if covariance is not None else np.eye(10) * 0.04, dtype=float)
    latent = _normal_pair(rng, np.r_[h_states, a_states], latent_cov, draws)
    h = latent[:, :5]
    a = latent[:, 5:]

    wind = float(conditions.get("wind_speed", 0.0) or 0.0)
    precipitation = float(conditions.get("precipitation", 0.0) or 0.0)
    weather_penalty = 1.0 - min(0.25, 0.002 * wind + 0.001 * precipitation)
    tempo_h = np.exp(np.clip(3.0 + 0.04 * h[:, 2] + 0.03 * a[:, 2], 1.2, 3.5))
    tempo_a = np.exp(np.clip(3.0 + 0.04 * a[:, 2] + 0.03 * h[:, 2], 1.2, 3.5))
    total_possessions = np.clip(np.rint((tempo_h + tempo_a) / 2), cfg.min_possessions, cfg.max_possessions).astype(int)
    home_possessions = np.maximum(1, np.rint(total_possessions / 2 + 0.25 * (tempo_h - tempo_a)).astype(int))
    away_possessions = np.maximum(1, total_possessions - home_possessions)

    home_scores = np.zeros(draws)
    away_scores = np.zeros(draws)
    for index in range(draws):
        h_points = 0.0
        a_points = 0.0
        for possession in range(max(home_possessions[index], away_possessions[index])):
            if possession < home_possessions[index]:
                field_position = np.clip(rng.normal(50 - 3 * h[index, 4] + 3 * a[index, 4], 12), 1, 99)
                logit_td = -1.8 + 0.35 * (h[index, 0] - a[index, 1]) + 0.015 * (field_position - 50)
                logit_td += 0.2 * (h_participation[index] - 1) - 0.2 * (a_participation[index] - 1)
                logit_td += math.log(weather_penalty)
                probabilities = softmax([logit_td, -0.7 + 0.01 * (field_position - 50), -0.2, -0.5, -0.8])
                outcome = rng.choice(5, p=probabilities / probabilities.sum())
                h_points += [cfg.touchdown_points, cfg.field_goal_points, 0, 0, cfg.safety_points][outcome]
            if possession < away_possessions[index]:
                field_position = np.clip(rng.normal(50 - 3 * a[index, 4] + 3 * h[index, 4], 12), 1, 99)
                logit_td = -1.8 + 0.35 * (a[index, 0] - h[index, 1])
                logit_td += 0.015 * (field_position - 50) + 0.2 * (a_participation[index] - 1) - 0.2 * (h_participation[index] - 1)
                logit_td += math.log(weather_penalty)
                probabilities = softmax([logit_td, -0.7 + 0.01 * (field_position - 50), -0.2, -0.5, -0.8])
                outcome = rng.choice(5, p=probabilities / probabilities.sum())
                a_points += [cfg.touchdown_points, cfg.field_goal_points, 0, 0, cfg.safety_points][outcome]
        home_scores[index] = h_points
        away_scores[index] = a_points

    margin = home_scores - away_scores
    total = home_scores + away_scores
    football = {
        "home_win_probability": float(np.mean(margin > 0)),
        "home_tie_probability": float(np.mean(margin == 0)),
        "expected_home_score": float(home_scores.mean()),
        "expected_away_score": float(away_scores.mean()),
        "expected_margin": float(margin.mean()),
        "margin_variance": float(margin.var(ddof=1)),
        "expected_total": float(total.mean()),
        "total_variance": float(total.var(ddof=1)),
        "score_interval_80": [float(np.quantile(home_scores, .1)), float(np.quantile(home_scores, .9))],
        "away_score_interval_80": [float(np.quantile(away_scores, .1)), float(np.quantile(away_scores, .9))],
        "margin_interval_80": [float(np.quantile(margin, .1)), float(np.quantile(margin, .9))],
        "total_interval_80": [float(np.quantile(total, .1)), float(np.quantile(total, .9))],
        "distribution_method": "joint_possession_monte_carlo",
        "draws": draws,
    }
    conditioned = dict(football)
    if market.get("home_decimal") and market.get("away_decimal"):
        fair_home, _ = _no_vig(market["home_decimal"], market["away_decimal"])
        market_logit = logit(np.clip(fair_home, 1e-6, 1 - 1e-6))
        football_logit = logit(np.clip(football["home_win_probability"], 1e-6, 1 - 1e-6))
        combined_logit = 0.75 * football_logit + 0.25 * market_logit
        conditioned["home_win_probability"] = float(expit(combined_logit))
        conditioned["market_weight"] = 0.25
    return {"football_forecast": football, "market_conditioned_forecast": conditioned}
