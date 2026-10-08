"""Score-distribution simulation and summary statistics.

A game is modeled as a 2-D Gaussian over ``(home_score, away_score)`` with
mean from the regression head and covariance calibrated from a held-out
season. Draws are rounded to nonnegative integer scores. Draws that end in a
regulation tie are given an approximate overtime increment so every reported
simulated scoreline has a winner.

This is a score-distribution approximation, not a drive-level or
possession-level game simulator. The continuous Gaussian margin probability
(:func:`win_probability`) is used in historical probability evaluation; the
bounded simulation probability is a separate, inspectable number.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.stats import norm

from .distribution import distribution_summary

#: Possible scoring increments on a single overtime possession, with their
#: approximate relative frequencies. These are the standard touchdown and
#: field goal values; the mix reflects that most teams score something in
#: overtime but field goals are more common.
_OVERTIME_INCREMENT_VALUES = np.array([3, 7])
_OVERTIME_INCREMENT_PROBABILITIES = np.array([0.6, 0.4])

#: Standard deviation (in points) assumed for the overtime margin when
#: choosing which side wins a tied simulation. The Gaussian cumulative at the
#: regulation expected margin, divided by this sigma, gives the home-side win
#: probability for the overtime leg. The value is intentionally much smaller
#: than the regulation margin SD (~16): overtime is short and reflects the
#: strongest team only on a limited sample of possessions.
_OVERTIME_MARGIN_SIGMA = 12.0


def checked_covariance(cov) -> np.ndarray:
    """Return ``cov`` as a validated 2x2 positive-definite matrix."""
    cov = np.asarray(cov, dtype=float)
    if (
        cov.shape != (2, 2)
        or not np.isfinite(cov).all()
        or not np.allclose(cov, cov.T)
        or np.linalg.eigvalsh(cov).min() <= 0
    ):
        raise ValueError("Score covariance must be a finite positive-definite 2x2 matrix")
    return cov


def win_probability(mu, cov) -> np.ndarray:
    """Continuous home-team win probability from the Gaussian margin."""
    cov = checked_covariance(cov)
    sd = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    mu = np.asarray(mu, dtype=float)
    return np.clip(norm.cdf((mu[..., 0] - mu[..., 1]) / sd), 1e-9, 1 - 1e-9)


def game_seed(game_id, seed: int = 20261002) -> int:
    """Deterministic per-game simulation seed derived from ``game_id``."""
    digest = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()[:8]
    return int.from_bytes(digest, "little")


def _overtime_home_win_probability(mu: np.ndarray) -> float:
    """Probability the home team wins the overtime leg given regulation ``mu``.

    The home side's overtime win probability tracks the sign and magnitude of
    the regulation expected margin, but with a smaller standard deviation
    than regulation to reflect that overtime is a short, high-variance
    sudden-death period rather than a full game.
    """
    expected_margin = float(mu[0] - mu[1])
    return float(norm.cdf(expected_margin / _OVERTIME_MARGIN_SIGMA))


def draw_score_samples(
    mu,
    cov,
    game_id,
    draws: int = 50000,
    seed: int = 20261002,
) -> np.ndarray:
    """Draw ``draws`` integer scoreline samples for one game.

    Returns an ``(draws, 2)`` integer array. Regulation ties receive an
    overtime increment biased by the regulation expected margin, so the
    stronger team wins overtime more often than not without being
    deterministic.
    """
    if draws < 1000:
        raise ValueError("Use at least 1000 simulation draws")
    mu = np.asarray(mu, dtype=float)
    if mu.shape != (2,) or not np.isfinite(mu).all() or (mu < 0).any():
        raise ValueError("Expected scores must be two nonnegative finite values")
    cov = checked_covariance(cov)

    rng = np.random.default_rng(game_seed(game_id, seed))
    scores = np.maximum(
        np.rint(rng.multivariate_normal(mu, cov, size=draws)),
        0,
    ).astype(int)

    tied_idx = np.flatnonzero(scores[:, 0] == scores[:, 1])
    if len(tied_idx):
        home_wins_ot_probability = _overtime_home_win_probability(mu)
        home_wins_ot = rng.random(len(tied_idx)) < home_wins_ot_probability
        winners = np.where(home_wins_ot, 0, 1)
        increments = rng.choice(
            _OVERTIME_INCREMENT_VALUES,
            len(tied_idx),
            p=_OVERTIME_INCREMENT_PROBABILITIES,
        )
        scores[tied_idx, winners] += increments
    return scores


def _spread_terms(
    margin: np.ndarray,
    home_spread: float,
    home_decimal: float | None,
    away_decimal: float | None,
) -> dict:
    """Spread cover and expected-return numbers for one home-spread value."""
    adjusted = margin + float(home_spread)
    terms: dict = {
        "home_spread": float(home_spread),
        "home_cover_probability": float(np.mean(adjusted > 0)),
        "away_cover_probability": float(np.mean(adjusted < 0)),
        "spread_push_probability": float(np.mean(adjusted == 0)),
    }
    for side, odds in [("home", home_decimal), ("away", away_decimal)]:
        if odds is None:
            continue
        if not np.isfinite(odds) or odds <= 1:
            raise ValueError("Decimal odds must exceed 1")
        p_win = terms[f"{side}_cover_probability"]
        other = "away" if side == "home" else "home"
        p_lose = terms[f"{other}_cover_probability"]
        terms[f"{side}_spread_expected_net_per_unit"] = p_win * (odds - 1) - p_lose
    return terms


def summarize_score_samples(
    scores,
    mu,
    cov,
    game_id,
    seed: int = 20261002,
    home_spread: float | None = None,
    total_line: float | None = None,
    home_decimal: float | None = None,
    away_decimal: float | None = None,
) -> dict:
    """Summary statistics computed from a single game's score samples."""
    scores = np.asarray(scores, dtype=int)
    mu = np.asarray(mu, dtype=float)
    cov = checked_covariance(cov)
    margin = scores[:, 0] - scores[:, 1]
    total = scores.sum(axis=1)

    summary: dict = {
        "expected_home_score": float(mu[0]),
        "expected_away_score": float(mu[1]),
        "projected_home_margin": float(mu[0] - mu[1]),
        "projected_total": float(mu.sum()),
        "home_win_probability": float(win_probability(mu, cov)),
        "simulated_home_win_probability": float(np.mean(margin > 0)),
        "simulation_draws": int(len(scores)),
        "simulation_seed": game_seed(game_id, seed),
        "margin_interval_80": np.quantile(margin, [0.1, 0.9]).tolist(),
        "total_interval_80": np.quantile(total, [0.1, 0.9]).tolist(),
        "home_score_interval_80": np.quantile(scores[:, 0], [0.1, 0.9]).tolist(),
        "away_score_interval_80": np.quantile(scores[:, 1], [0.1, 0.9]).tolist(),
        "simulated_mean_home_score": float(scores[:, 0].mean()),
        "simulated_mean_away_score": float(scores[:, 1].mean()),
        **distribution_summary(scores),
    }
    if home_spread is not None:
        summary.update(_spread_terms(margin, home_spread, home_decimal, away_decimal))
    if total_line is not None:
        summary.update(
            total_line=float(total_line),
            over_probability=float(np.mean(total > total_line)),
            under_probability=float(np.mean(total < total_line)),
            total_push_probability=float(np.mean(total == total_line)),
        )
    return summary


def simulate_scores(
    mu,
    cov,
    game_id,
    draws: int = 50000,
    seed: int = 20261002,
    home_spread: float | None = None,
    total_line: float | None = None,
    home_decimal: float | None = None,
    away_decimal: float | None = None,
    return_samples: bool = False,
):
    """Draw score samples and return a summary (and optionally the raw draws)."""
    scores = draw_score_samples(mu, cov, game_id, draws=draws, seed=seed)
    result = summarize_score_samples(
        scores,
        mu,
        cov,
        game_id,
        seed,
        home_spread,
        total_line,
        home_decimal,
        away_decimal,
    )
    if return_samples:
        return result, scores
    return result
