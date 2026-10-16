"""Direct Gaussian forecast calculations; no generated score samples."""

import numpy as np
from scipy.stats import norm



def checked_covariance(cov):
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T) or np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("Score covariance must be a finite positive-definite 2x2 matrix")
    return cov


def win_probability(mu, cov):
    cov = checked_covariance(cov)
    sd = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    mu = np.asarray(mu, dtype=float)
    return np.clip(norm.cdf((mu[..., 0] - mu[..., 1]) / sd), 1e-9, 1 - 1e-9)


def score_summary(means, covariance, **market):
    mu = np.asarray(means, float)
    cov = checked_covariance(covariance)
    margin, total = float(mu[0] - mu[1]), float(mu.sum())
    margin_sd = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    total_sd = float(np.sqrt(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1]))
    z = norm.ppf(0.9)
    result = {
        "expected_home_score": float(mu[0]),
        "expected_away_score": float(mu[1]),
        "projected_home_margin": margin,
        "projected_total": total,
        "home_win_probability": float(win_probability(mu, cov)),
        "margin_interval_80": [margin - z * margin_sd, margin + z * margin_sd],
        "total_interval_80": [total - z * total_sd, total + z * total_sd],
        "distribution_method": "analytic_gaussian",
        "margin_standard_deviation": margin_sd,
        "total_standard_deviation": total_sd,
    }
    for side, index in (("home", 0), ("away", 1)):
        sd = float(np.sqrt(cov[index, index]))
        result[f"{side}_score_standard_deviation"] = sd
        result[f"{side}_score_interval_80"] = [float(mu[index] - z * sd), float(mu[index] + z * sd)]
        line = market.get(f"{side}_total_line")
        if line is not None:
            line = float(line)
            if not np.isfinite(line) or line < 0:
                raise ValueError("Team total lines must be finite and nonnegative")
            lower, upper = (line - .5, line + .5) if line.is_integer() else (line, line)
            over = float(norm.sf((upper - mu[index]) / sd))
            under = float(norm.cdf((lower - mu[index]) / sd))
            result.update({
                f"{side}_total_over_probability": over,
                f"{side}_total_under_probability": under,
                f"{side}_total_push_probability": max(0., 1 - over - under),
            })
    spread = market.get("home_spread")
    if spread is not None:
        threshold = -float(spread)
        # A tied game goes to overtime, so a pick'em line (0) cannot push.
        integer = float(threshold).is_integer() and threshold != 0
        lower, upper = (
            (threshold - 0.5, threshold + 0.5) if integer else (threshold, threshold)
        )
        home = float(norm.sf((upper - margin) / margin_sd))
        away = float(norm.cdf((lower - margin) / margin_sd))
        result.update(
            home_cover_probability=home,
            away_cover_probability=away,
            spread_push_probability=1 - home - away,
        )
        for side, prob, loss in [("home", home, away), ("away", away, home)]:
            price = market.get(f"{side}_decimal")
            if price is not None:
                result[f"{side}_spread_expected_net_per_unit"] = (
                    prob * (price - 1) - loss
                )
    line = market.get("total_line")
    if line is not None:
        integer = float(line).is_integer()
        lower, upper = (line - 0.5, line + 0.5) if integer else (line, line)
        over = float(norm.sf((upper - total) / total_sd))
        under = float(norm.cdf((lower - total) / total_sd))
        result.update(
            over_probability=over,
            under_probability=under,
            total_push_probability=1 - over - under,
        )
    return result
