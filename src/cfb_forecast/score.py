"""Point-in-time joint forecasting: rating state -> possession-level distributions.

Given the data root, a production model root and an as-of cutoff,
:func:`score_games` replays only results available by the cutoff and produces
one row per target game.  The statistical outcome is computed from the *joint*
distribution of the football variables for each matchup via the possession
engine in :mod:`cfb_forecast.joint_sim` — repeated draws sample the
uncertainty already present in the hierarchical Bayesian state (team offense /
defense representations, posterior covariances, availability, tempo, shared
game clock), not noise bolted onto a regression point estimate.

When a market history file is supplied, each row additionally carries the raw
market quote at the cutoff, the vig-free market probability, the edges against
the football-only forecast, and a separate market-conditioned forecast
(``p(Y | FootballData, Market)``) so the three can be graded on identical
games.  No artifacts, no manifests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import load_data, timestamp
from .forecast_state import replay_state
from .joint_sim import (MatchupParameters, draw_states, fit_matchup_params,
                        market_conditioned, simulate_game_draws, summarize_draws,
                        vig_free_probability)
from .market import read_market_history
from .market_calibration import fit_margin_measurement, fit_probability_measurement
from .model import ScoreHead, load_model

DEFAULT_DRAWS = 50_000


def _spread_to_home_margin(spread):
    """Bookmaker home spread (home -spread = favored) -> expected home margin."""
    if spread is None or (isinstance(spread, float) and np.isnan(spread)):
        return np.nan
    return -float(spread)


def _latest_quote(market, game_id, at):
    """Last observed quote strictly before the cutoff (point-in-time)."""
    if market is None or market.empty:
        return {}
    rows = market[(market.game_id.astype(int) == int(game_id))
                  & (market.observed_at <= at)]
    if rows.empty:
        return {}
    row = rows.sort_values("observed_at").iloc[-1]
    out = {"home_spread": float(row.home_spread) if pd.notna(row.home_spread) else np.nan}
    for col in ("total_line", "home_moneyline", "away_moneyline"):
        if col in market.columns and pd.notna(getattr(row, col, None)):
            out[col] = float(getattr(row, col))
    out["quote_observed_at"] = str(row.observed_at)
    return out


def score_games(data_root, model_root, as_of, through, season, target_game_ids=None,
                market_path=None, draws=DEFAULT_DRAWS, seed=20261013):
    """Joint possession-level point-in-time forecasts for ``[as_of, through]``."""
    at, end = timestamp(as_of), timestamp(through)
    if at >= end:
        raise ValueError("Forecast window must end after as-of time")
    model = load_model(model_root)
    if model["training_through_season"] >= int(season):
        raise ValueError("Model training must end before the forecast season")
    schedules, summaries, _ = load_data(data_root)
    population = schedules[
        schedules.fbs_game & schedules.season.between(model["warmup_season"], int(season))
    ].copy().reset_index(drop=True)
    targets = schedules[schedules.fbs_game & (schedules.season == int(season))
                       & (schedules.start_date >= at) & (schedules.start_date <= end)].copy()
    if target_game_ids is not None:
        wanted = {int(g) for g in target_game_ids}
        missing = wanted - set(targets.game_id.astype(int))
        if missing:
            raise ValueError(f"Target game IDs outside the window: {sorted(missing)}")
        targets = targets[targets.game_id.isin(wanted)]
    if targets.empty:
        return pd.DataFrame()
    past = population[population.result_usable & (population.result_available_at < at)]
    past = past.copy().reset_index(drop=True)
    if not (past.season == model["warmup_season"]).any():
        raise ValueError("Historical warmup season missing")
    state = replay_state(population, past, summaries, model, at)
    head = ScoreHead.from_dict(model["head"])

    # --- Learned possession parameters Psi (fit on point-in-time history) ---
    params = MatchupParameters()
    if "matchup_parameters" in model:
        params = MatchupParameters.from_dict(model["matchup_parameters"])
    else:
        hist_done = population[population.result_usable
                               & (population.result_available_at < at)]
        if len(hist_done) >= 50:
            F_hist = np.stack([state.features(r.home_id, r.away_id,
                                              bool(r.neutral_site), int(r.week) <= 4)
                               for r in hist_done.itertuples(index=False)])
            mu_hist = head.predict(F_hist)
            actual = hist_done[["home_points", "away_points"]].to_numpy(dtype=float)
            # Per-metric pregame (offense - opponent defense) differentials —
            # feature columns [10:15] of each side's row (see ratings.features).
            diffs = F_hist[:, :, 10:15]
            params = fit_matchup_params(actual, mu_hist, state_diffs=diffs)

    # --- Market calibration from history available before the cutoff ---
    market = read_market_history(market_path) if market_path else pd.DataFrame()
    delta_params = {"delta_spread": 0.0, "sigma_spread": 7.0,
                    "delta_ML": 0.0, "sigma_ML": 0.5}
    if market is not None and not market.empty:
        mrows = market[market.observed_at < at]
        hist = population[population.result_usable
                          & (population.result_available_at < at)]
        if not mrows.empty and len(hist):
            latest = mrows.sort_values("observed_at").groupby("game_id").tail(1)
            merged = hist.merge(latest, on="game_id", how="inner")
            merged = merged[pd.notna(merged.home_spread)]
            if len(merged) >= 30:
                cov = np.asarray(model["score_covariance"], dtype=float)
                var_m = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
                d_s, s_s = fit_margin_measurement(
                    (merged.home_points - merged.away_points).to_numpy(),
                    merged.home_spread.to_numpy(),
                    np.full(len(merged), var_m))
                delta_params.update({"delta_spread": d_s, "sigma_spread": s_s})

    rng = np.random.default_rng(seed)
    rows = []
    for r in targets.itertuples(index=False):
        F = state.features(r.home_id, r.away_id, bool(r.neutral_site), int(r.week) <= 4)
        F = np.asarray(F, dtype=float)
        if F.ndim == 2:
            F = F.reshape(1, *F.shape)
        means = head.predict(F)[0]
        quote = _latest_quote(market, r.game_id, at)
        spread = quote.get("home_spread", np.nan)
        total_line = quote.get("total_line", np.nan)

        # Joint possession-level draws from the posterior football-state MVN.
        oh, dh, oa, da = draw_states(state, r.home_id, r.away_id, rng, int(draws))
        context = {
            "weather_precipitation": float(getattr(r, "weather_precipitation", 0.0) or 0.0),
            "short_week_home": False, "short_week_away": False,
        }
        home_scores, away_scores = simulate_game_draws(
            oh, dh, oa, da, params, bool(r.neutral_site), context, rng)
        summary = summarize_draws(home_scores, away_scores,
                                  spread=None if np.isnan(spread) else spread,
                                  total_line=None if (isinstance(total_line, float) and np.isnan(total_line)) else total_line)
        cond = market_conditioned(summary, quote, delta_params) if quote else {}

        row = {
            "game_id": int(r.game_id),
            "season": int(r.season),
            "week": int(r.week),
            "season_type": str(r.season_type),
            "home_team": str(r.home_team),
            "away_team": str(r.away_team),
            "kickoff": r.start_date.isoformat(),
            "as_of": at.isoformat(),
            "model_variant": model["variant"],
            "prediction_method": ("hierarchical Bayesian team-state posterior + "
                                  "possession-level joint simulation"),
            "draws": int(draws),
            "expected_home_score": summary["expected_home_score"],
            "expected_away_score": summary["expected_away_score"],
            "projected_home_margin": summary["expected_margin"],
            "projected_total": summary["expected_total"],
            "home_win_probability": summary["home_win_probability"],
            "margin_variance": summary["margin_variance"],
            "total_variance": summary["total_variance"],
            "tie_probability": summary["tie_probability"],
            "margin_p05": summary["margin_p05"],
            "margin_p95": summary["margin_p95"],
            "total_p05": summary["total_p05"],
            "total_p95": summary["total_p95"],
            "regression_expected_home_score": float(means[0]),
            "regression_expected_away_score": float(means[1]),
        }
        if "p_home_covers" in summary:
            row["p_home_covers"] = summary["p_home_covers"]
            row["home_spread_observed"] = summary["home_spread_observed"]
        if "p_over" in summary:
            row["p_over"] = summary["p_over"]
            row["total_line_observed"] = summary["total_line_observed"]
        if quote:
            row["market_quote_observed_at"] = quote.get("quote_observed_at")
            row["market_home_spread"] = spread
            if "market_win_probability" in cond:
                row["market_win_probability"] = cond["market_win_probability"]
            for key in ("edge_win", "edge_margin", "edge_total"):
                if key in cond and np.isfinite(cond[key]):
                    row[f"football_minus_market_{key.split('_')[1]}"] = float(cond[key])
            row["market_conditioned_home_win_probability"] = cond["market_conditioned_home_win_probability"]
            row["market_conditioned_projected_home_margin"] = cond["market_conditioned_expected_margin"]
            row["market_conditioned_expected_home_score"] = cond["market_conditioned_expected_home_score"]
            row["market_conditioned_expected_away_score"] = cond["market_conditioned_expected_away_score"]
        rows.append(row)
    return pd.DataFrame(rows)
