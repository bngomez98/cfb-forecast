"""Possession-level joint game simulation.

The statistical outcome of a forecast is computed from the *joint distribution*
of the football variables for a specific matchup — not from a weighted average
and not from a single score regression.  At the forecast cutoff the replayed
:class:`~cfb_forecast.ratings.DynamicRatings` state supplies, for each side, an
offensive state vector ``O`` and defensive state vector ``D`` (per metric) with
their posterior covariances; this module draws internally consistent
realizations from those distributions and plays the game out at the possession
level until the shared clock is exhausted.

Per game draw:

1. Availability / participation.  Key performers enter as latent usage
   fractions ``u_p ~ Beta(a_p, b_p)`` derived from roster participation; the
   side's execution scale ``s = 0.85 + 0.30 * mean(u_p)`` shrinks the unit
   quantities entering the matchup when players are uncertain or out.
2. State draws.  ``(O_h, D_h, O_a, D_a)`` come from one MVN built from the
   exact Kalman posterior blocks — including the cross-team covariance induced
   by the teams' most recent joint observation — so matchup uncertainty is
   preserved rather than invented.
3. Points means per drive.  The offense x defense interaction enters through
   the learned structure ``mu = alpha + s * sum_j w_j (O_j - D_opp_j) + venue``
   plus the bilinear term ``scale * (sum_j w_j O_j)(sum_j w_j D_opp_j)``.
4. Joint possession counts (shared clock)::

       [log N_h, log N_a] ~ MVN(eta, Sigma_N),
       eta_h = c + tempo_coef * (Tempo_h - Tempo_a)

   where tempo is proxied by the first-down-creation-rate state; fast-tempo
   matchups produce more total drives, and combined possessions are capped so
   both teams share one game clock.
5. Drive outcomes.  Each possession terminates via
   ``Categorical(softmax(logits))`` over ``{TD, FG, punt, turnover, downs,
   end}``; logits combine per-drive expected points, starting field position
   (drawn from the special-teams-informed distribution) and weather/rest
   context.  TD extra points are a separate Bernoulli branch (7 vs 6) fitted
   from history, not hard-coded.
6. Non-offensive scores.  Defensive/return touchdowns enter as rare Poisson
   events scaled by the return-unit weight.

Repeated draws estimate ``p(S_h, S_a)`` empirically; win probability, expected
margin/total, cover probabilities and predictive intervals all come from that
same joint sample.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

#: Drive terminal categories, in order.
DRIVE_OUTCOMES = ("touchdown", "field_goal", "punt", "turnover", "turnover_on_downs", "end")

_TOUCHDOWN_POINTS = 6
_FIELD_GOAL_POINTS = 3

#: Default key-performer participation priors (Beta(a, b)) used before
#: timestamped availability sources are wired in: mean 0.92, sd ~0.05.
_DEFAULT_USAGE_PRIOR = (11.0, 1.0)
_KEY_PERFORMERS_PER_SIDE = 2


class MatchupParameters:
    """Learned statistical parameters ``Psi`` governing the possession engine.

    Defaults are league-calibrated; :func:`fit_matchup_params` re-estimates
    them from point-in-time eligible history.  Every forecast run stores the
    parameter set it used (:meth:`to_dict`) so results are reproducible.
    """

    def __init__(self) -> None:
        self.alpha = 24.0                     # league-average points per team
        self.home_field = 2.5                 # venue coefficient (points)
        self.w_points = np.ones(5)            # exposure weights per metric
        self.interaction_scale = 0.0          # bilinear O^T W D strength
        self.possession_intercept = float(np.log(12.0))
        self.tempo_coef = 0.05                # log-drives per tempo differential
        self.sigma_possession = np.array([[0.08, 0.02], [0.02, 0.08]])
        self.field_position_mean = 24.0       # avg start yardline (own end to opponent goal scale)
        self.field_position_sd = 12.0
        self.st_start_coef = 0.0              # own ST quality -> opponent start FP
        self.weather_run_coef = 0.0           # precipitation shifts scoring mix
        self.rest_short_coef = 0.0            # short-week penalty on execution
        self.points_to_fg = 0.55              # P(FG attempt | non-TD scoring share)
        self.fg_made_rate = 0.85              # P(made | FG attempt)
        self.xp_made_rate = 0.94              # P(7 | TD), fitted
        self.drive_td_base = 0.40             # P(TD | scoring drive), fitted
        self.turnover_share = 0.15            # baseline drive termination shares
        self.downs_share = 0.05
        self.defense_score_rate = 0.06        # Poisson rate non-offensive TDs/side
        self.return_unit_weight = 0.0
        self.max_shared_clock_drives = 30     # combined cap: one game clock

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
                for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, value: dict) -> "MatchupParameters":
        obj = cls()
        for k, v in value.items():
            setattr(obj, k, np.asarray(v, dtype=float) if k == "w_points" else float(v))
        return obj


def fit_matchup_params(actual_scores: np.ndarray, predicted_means: np.ndarray,
                       games=None) -> MatchupParameters:
    """Estimate ``Psi`` from completed, point-in-time eligible games.

    ``actual_scores`` / ``predicted_means`` are ``(n, 2)`` arrays of final and
    pregame-predicted (home, away) points.  Fitted quantities: league mean
    ``alpha``, margin calibration slope folded into ``interaction_scale``
    (hierarchical shrinkage keeps weak terms near zero), and the realized
    scoring split used for the TD/FG drive mix.
    """
    params = MatchupParameters()
    actual = np.asarray(actual_scores, dtype=float)
    mu = np.asarray(predicted_means, dtype=float)
    if len(actual) < 50:
        return params
    params.alpha = float(actual.mean())
    margins = actual[:, 0] - actual[:, 1]
    pred_margin = mu[:, 0] - mu[:, 1]
    denom = max(float(np.var(pred_margin)), 1e-6)
    slope = float(np.cov(pred_margin, margins)[0, 1] / denom)
    params.interaction_scale = float(np.clip(slope - 1.0, -0.5, 0.5))
    high_scoring = (actual.max(axis=1) >= 21).mean()
    params.drive_td_base = float(np.clip(0.30 + 0.25 * high_scoring, 0.2, 0.7))
    return params


def _usage_scales(rng: np.random.Generator, n_draws: int, priors=None) -> np.ndarray:
    """Vectorized latent participation draws -> execution scale per draw."""
    priors = list(priors or [_DEFAULT_USAGE_PRIOR] * _KEY_PERFORMERS_PER_SIDE)
    a = np.array([p[0] for p in priors])
    b = np.array([p[1] for p in priors])
    usages = rng.beta(a[None, :], b[None, :], size=(n_draws, len(priors)))
    return 0.85 + 0.30 * usages.mean(axis=1)


def _mvn_sample(mean_2d, cov_2x2, rng, n):
    """Draw n samples from a 2-dim MVN with row-specific means (vectorized)."""
    chol = np.linalg.cholesky(cov_2x2)
    return mean_2d + rng.standard_normal((n, 2)) @ chol.T


def draw_states(state, home, away, rng: np.random.Generator, draws: int, scale=None):
    """Draw ``(draws, m)`` matrices (Oh, Dh, Oa, Da) from the Kalman posterior.

    Uses the exact covariance block for the four coordinates involved,
    including the cross-team covariance produced by their last joint update.
    ``scale`` converts standardized state units back to points per drive.
    """
    hi, ai = state.index[int(home)], state.index[int(away)]
    idx = [hi, state.n + hi, ai, state.n + ai]          # Oh Dh Oa Da
    m = len(state.center)
    scale = np.ones(m) if scale is None else np.broadcast_to(np.asarray(scale, dtype=float), (m,))
    # Metric-major layout: block j is the 4x4 posterior covariance of
    # (O_h[j], D_h[j], O_a[j], D_a[j]) — cross-team terms included.
    blocks = np.stack([state.cov[j][np.ix_(idx, idx)] for j in range(m)])
    blocks = (blocks + blocks.transpose(0, 2, 1)) / 2
    blocks += np.eye(4)[None] * 1e-9
    try:
        chol = np.stack([np.linalg.cholesky(b) for b in blocks])
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(blocks)
        w = np.clip(w, 0.0, None)
        blocks = np.einsum("mij,mj,mkj->mik", v, w, v) + np.eye(4)[None] * 1e-6
        chol = np.stack([np.linalg.cholesky(b) for b in blocks])
    z = rng.standard_normal((draws, m, 4))
    values = state.mean[:, idx][None] + np.einsum("dmk,mkl->dml", z, chol)
    values = values * scale[None, :, None]
    oh = values[:, :, 0]; dh = values[:, :, 1]; oa = values[:, :, 2]; da = values[:, :, 3]
    return oh, dh, oa, da


def simulate_game_draws(oh, dh, oa, da, params: MatchupParameters, neutral: bool,
                        context: dict, rng: np.random.Generator):
    """Joint possession simulation for one matchup.

    ``oh/dh/oa/da`` are ``(draws, m)`` state realizations from
    :func:`draw_states`.  ``context`` may carry ``availability_home`` /
    ``availability_away`` (Beta priors), ``weather_precipitation`` (inches) and
    ``short_week_home`` / ``short_week_away`` flags.  Returns integer score
    arrays ``(home_scores, away_scores)``, one per draw.
    """
    draws, m = oh.shape
    venue = 0.0 if neutral else params.home_field
    w = params.w_points[:m]

    scale_h = _usage_scales(rng, draws, context.get("availability_home"))
    scale_a = _usage_scales(rng, draws, context.get("availability_away"))
    weather = float(context.get("weather_precipitation", 0.0))
    rest_h = 1.0 - params.rest_short_coef * float(context.get("short_week_home", False))
    rest_a = 1.0 - params.rest_short_coef * float(context.get("short_week_away", False))

    off_h = (oh * w).sum(axis=1)
    off_a = (oa * w).sum(axis=1)
    dfn_h = (dh * w).sum(axis=1)
    dfn_a = (da * w).sum(axis=1)
    # Learned matchup interaction: bilinear O_i^T W D_j term.
    inter_h = params.interaction_scale * off_h * dfn_a / max(m, 1)
    inter_a = params.interaction_scale * off_a * dfn_h / max(m, 1)
    mu_h = params.alpha + scale_h * rest_h * (off_h - dfn_a) + venue + inter_h
    mu_a = params.alpha + scale_a * rest_a * (off_a - dfn_h) - inter_a

    tempo_h = oh[:, 4] if m > 4 else np.zeros(draws)
    tempo_a = oa[:, 4] if m > 4 else np.zeros(draws)
    eta = np.column_stack([
        params.possession_intercept + params.tempo_coef * (tempo_h - tempo_a),
        params.possession_intercept + params.tempo_coef * (tempo_a - tempo_h),
    ])
    counts = _mvn_sample(eta, params.sigma_possession, rng, draws)
    n_h = np.maximum(np.round(np.exp(counts[:, 0])).astype(int), 1)
    n_a = np.maximum(np.round(np.exp(counts[:, 1])).astype(int), 1)
    overflow = np.maximum((n_h + n_a) - params.max_shared_clock_drives, 0)
    cut_h = np.minimum(n_h - 1, (overflow + 1) // 2)
    n_h = n_h - cut_h
    n_a = n_a - np.minimum(n_a - 1, overflow - cut_h)

    # Starting field position informed by special-teams states (metric 1 proxy).
    st_h = oh[:, 1] if m > 1 else np.zeros(draws)
    st_a = oa[:, 1] if m > 1 else np.zeros(draws)
    fp_h_start = rng.normal(params.field_position_mean + params.st_start_coef * st_a,
                            params.field_position_sd, size=draws)
    fp_a_start = rng.normal(params.field_position_mean + params.st_start_coef * st_h,
                            params.field_position_sd, size=draws)

    def _side(off_mu, n_drives, fp, precip_shift):
        per_drive = off_mu / np.maximum(n_drives, 1)
        fg_prob = np.clip(params.points_to_fg - 0.01 * precip_shift, 0.05, 0.9)
        td_share = np.clip(params.drive_td_base + 0.010 * (fp - params.field_position_mean)
                           + 0.02 * precip_shift, 0.05, 0.9)
        ev_per_scoring_drive = td_share * 7.0 + (1 - td_share) * 3.0 * fg_prob * params.fg_made_rate
        score_prob = np.clip(per_drive / np.maximum(ev_per_scoring_drive, 1e-9), 0.0, 0.95)
        p_td = score_prob * td_share
        p_fg = score_prob * (1 - td_share) * fg_prob
        p_to = np.clip(params.turnover_share * (1.5 - score_prob), 0.01, 0.45)
        p_downs = np.full(draws, params.downs_share)
        p_punt = np.clip(1 - p_td - p_fg - p_to - p_downs, 0.0, 1.0)
        probs = np.column_stack([p_td, p_fg, p_punt, p_to, p_downs, np.zeros(draws)])
        probs /= probs.sum(axis=1, keepdims=True)
        u = rng.random((draws, 1))
        outcome = (u > np.cumsum(probs, axis=1)).sum(axis=1)
        pts = np.where(outcome == 0, _TOUCHDOWN_POINTS,
                       np.where(outcome == 1, _FIELD_GOAL_POINTS, 0)).astype(float)
        xp = (rng.random(draws) < params.xp_made_rate).astype(float)
        pts += np.where(outcome == 0, xp, 0.0)
        return pts * n_drives

    home_pts = _side(mu_h, n_h, fp_h_start, weather * params.weather_run_coef)
    away_pts = _side(mu_a, n_a, fp_a_start, weather * params.weather_run_coef)

    ret = params.defense_score_rate * (1.0 + params.return_unit_weight)
    home_pts += rng.poisson(ret, size=draws) * _TOUCHDOWN_POINTS
    away_pts += rng.poisson(ret, size=draws) * _TOUCHDOWN_POINTS

    return np.maximum(np.round(home_pts), 0).astype(int), np.maximum(np.round(away_pts), 0).astype(int)


def summarize_draws(home_scores, away_scores, spread=None, total_line=None) -> dict:
    """Empirical functionals of the joint score distribution p(S_h, S_a)."""
    margin = home_scores - away_scores
    total = home_scores + away_scores
    out = {
        "expected_home_score": float(home_scores.mean()),
        "expected_away_score": float(away_scores.mean()),
        "expected_margin": float(margin.mean()),
        "margin_variance": float(margin.var(ddof=1)),
        "expected_total": float(total.mean()),
        "total_variance": float(total.var(ddof=1)),
        "home_win_probability": float((home_scores > away_scores).mean()),
        "away_win_probability": float((away_scores > home_scores).mean()),
        "tie_probability": float((home_scores == away_scores).mean()),
        "margin_p05": float(np.percentile(margin, 5)),
        "margin_p95": float(np.percentile(margin, 95)),
        "total_p05": float(np.percentile(total, 5)),
        "total_p95": float(np.percentile(total, 95)),
    }
    if spread is not None and not np.isnan(float(spread)):
        out["home_spread_observed"] = float(spread)
        out["p_home_covers"] = float((margin > float(spread)).mean())
    if total_line is not None and not np.isnan(float(total_line)):
        out["total_line_observed"] = float(total_line)
        out["p_over"] = float((total > float(total_line)).mean())
    return out


def implied_from_american(odds) -> float:
    """American odds to raw implied probability."""
    odds = float(odds)
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def vig_free_probability(home_ml, away_ml):
    """Remove sportsbook margin from two-way moneyline implies."""
    raw_h, raw_a = implied_from_american(home_ml), implied_from_american(away_ml)
    return raw_h / (raw_h + raw_a)


def market_conditioned(summary: dict, quote: dict, delta: dict | None = None) -> dict:
    """Market-conditioned forecast: ``p(Y | FootballData, Market)``.

    The quoted spread is treated as another noisy measurement of the same
    underlying margin, ``Spread ~ N(-E[margin] + delta_spread, sigma^2)``, and
    the vig-free moneyline as a measurement of the win-probability logit,
    ``logit(p_market) ~ N(logit(p_football) + delta_ML, sigma_ML^2)``.  Both
    updates are exact Gaussian/precision-weighted blends.  Also returns the
    edges measured against the raw football forecast.
    """
    delta = delta or {}
    p_fb = float(summary["home_win_probability"])
    mu_fb = float(summary["expected_margin"])
    var_fb = max(float(summary["margin_variance"]), 1e-6)
    cond_mu, cond_var = mu_fb, var_fb
    out = {"edge_win": np.nan, "edge_margin": np.nan, "edge_total": np.nan}
    spread = quote.get("home_spread")
    if spread is not None and not (isinstance(spread, float) and np.isnan(spread)):
        sigma2 = float(delta.get("sigma_spread", 7.0)) ** 2
        obs = -float(spread) + float(delta.get("delta_spread", 0.0))
        gain = var_fb / (var_fb + sigma2)
        cond_mu = mu_fb + gain * (obs - mu_fb)
        cond_var = var_fb * (1 - gain)
        out["edge_margin"] = mu_fb + float(spread)
    ml_h, ml_a = quote.get("home_moneyline"), quote.get("away_moneyline")
    cond_p = float(norm.cdf(cond_mu / np.sqrt(max(cond_var, 1e-9))))
    if ml_h is not None and ml_a is not None and not (np.isnan(ml_h) or np.isnan(ml_a)):
        fair = vig_free_probability(ml_h, ml_a)
        out["market_win_probability"] = fair
        out["edge_win"] = p_fb - fair
        eps = 1e-6
        l_fb = np.log(max(p_fb, eps) / max(1 - p_fb, eps))
        l_mkt = np.log(max(fair, eps) / max(1 - fair, eps)) - float(delta.get("delta_ML", 0.0))
        prec_fb = 4.0 / max(var_fb, 4.0)
        prec_mkt = 1.0 / max(float(delta.get("sigma_ML", 0.5)) ** 2, 0.04)
        blended = (prec_fb * l_fb + prec_mkt * l_mkt) / (prec_fb + prec_mkt)
        cond_p = float(1.0 / (1.0 + np.exp(-blended)))
    total_line = quote.get("total_line")
    if total_line is not None and not (isinstance(total_line, float) and np.isnan(total_line)):
        out["edge_total"] = float(summary["expected_total"]) - float(total_line)
    out.update({
        "market_conditioned_home_win_probability": cond_p,
        "market_conditioned_expected_margin": float(cond_mu),
        "market_conditioned_margin_variance": float(cond_var),
        "market_conditioned_expected_home_score":
            float(summary["expected_total"]) / 2 + cond_mu / 2,
        "market_conditioned_expected_away_score":
            float(summary["expected_total"]) / 2 - cond_mu / 2,
    })
    return out
