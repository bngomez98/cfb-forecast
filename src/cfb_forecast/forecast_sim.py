"""Per-game Monte Carlo simulation and output-row assembly for forecasts.

Shared by the full-replay path (:func:`cfb_forecast.forecast.forecast`) and
the exact-state advance path
(:func:`cfb_forecast.forecast.advance_forecast_snapshot`) so both modes emit
identically shaped ``forecasts.csv`` rows and distribution tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .distribution import probability_tables
from .market import market_at_cutoff, market_history_summary
from .simulate import simulate_scores

#: Keys of a per-game market snapshot that feed into the simulation anchoring.
SIM_MARKET_KEYS = ("home_spread", "total_line", "home_decimal", "away_decimal")

#: Distribution tables written alongside the point forecasts, in file order.
DISTRIBUTION_STEMS = ("score_distribution", "margin_distribution", "total_distribution")


def pregame_features(state, targets):
    """Stack pregame feature tensors for ``targets`` under ``state``."""
    return np.stack([
        state.features(r.home_id, r.away_id, r.neutral_site, int(r.week) <= 4)
        for r in targets.itertuples(index=False)
    ])


def market_context(market, game_id, at, kickoff, provider):
    """Snapshot plus history summary available for one game at cutoff ``at``.

    Returns ``({}, {"market_snapshot_count": 0})`` when no market history was
    supplied, mirroring the shape produced for games without eligible quotes.
    """
    if market is None or market.empty:
        return {}, {"market_snapshot_count": 0}
    snapshot = market_at_cutoff(market, game_id, at, kickoff, provider=provider)
    history = market_history_summary(market, game_id, at, kickoff, provider=provider)
    return snapshot, history


def forecast_rows(targets, mu, model, state, at, market, market_provider, draws,
                  prior_fbs_games, prediction_method, joint, margins, totals,
                  save_distribution):
    """Simulate every target game and assemble the per-game output records.

    ``joint``, ``margins`` and ``totals`` are accumulator lists appended to
    when ``save_distribution`` is true; callers concatenate them into the
    distribution files afterwards.
    """
    rows = []
    for r, means in zip(targets.itertuples(index=False), mu):
        m, hist = market_context(market, r.game_id, at, r.start_date, market_provider)
        sim, samples = simulate_scores(
            means, model["score_covariance"], r.game_id, draws=draws, return_samples=True,
            **{k: v for k, v in m.items() if k in SIM_MARKET_KEYS},
        )
        if save_distribution:
            tabs = probability_tables(samples, r.game_id)
            joint.append(tabs["joint"])
            margins.append(tabs["margin"])
            totals.append(tabs["total"])
        rows.append({
            "game_id": int(r.game_id),
            "season": int(r.season),
            "home_team": r.home_team,
            "away_team": r.away_team,
            "kickoff": r.start_date.isoformat(),
            "as_of": at.isoformat(),
            "latest_result_available_at": state.latest_update.isoformat() if state.latest_update is not None else None,
            "prior_fbs_games": prior_fbs_games,
            "model_variant": model["variant"],
            "head_training_through_season": model["training_through_season"],
            "prediction_method": prediction_method,
            **sim,
            **m,
            **hist,
        })
    return rows


def frame_from_rows(rows):
    """Coerce assembled row dicts back into the canonical forecast frame."""
    return pd.DataFrame(rows)
