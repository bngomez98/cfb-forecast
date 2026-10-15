"""Per-game observations and the feature-column layout for score heads.

This module turns the raw advanced-team summary rows into the standardized
observation tensor the dynamic-ratings model updates on, and tells the score
head which of the 26 feature columns it is allowed to see.

The clip ranges applied to each yardage metric are generous bounds, well
above the observed FBS maxima: they are only there to protect the state
updates from occasional corrupt rows in the ESPN feed (e.g. a pass yardage
value of several hundred that would be dominated by a stat correction after
the fact). Realistic game values are untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import METRICS
from .data import require_columns

#: Columns required on the raw play-by-play frame when the optional
#: deterministic cleaning step is used.
_RAW_PLAY_COLUMNS = [
    "game_id",
    "game_play_number",
    "pos_team_id",
    "def_pos_team_id",
    "down",
    "distance",
    "statYardage",
    "rush",
    "pass",
    "sack",
    "kneel_down",
    "penalty_no_play",
    "period",
    "start.pos_score_diff",
]
_RAW_PLAY_FLAG_COLUMNS = ("rush", "pass", "sack", "kneel_down", "penalty_no_play")

#: Observation clip ranges, keyed by position in :data:`METRICS` (index 1
#: through 4). Chosen as hard upper bounds on physically reasonable FBS values
#: so bad feed data cannot swing the state updates.
_METRIC_CLIP_RANGES = {
    1: (0.0, 15.0),  # yards per play
    2: (0.0, 20.0),  # yards per pass
    3: (-2.0, 15.0),  # yards per rush
    4: (0.0, 1.0),  # first-down creation rate
}


def clean_plays(plays: pd.DataFrame) -> pd.DataFrame:
    """Filter and annotate raw play-by-play rows.

    Keeps only rushing, passing and sack plays that were run from a valid
    down and distance in a reasonable game-state window; adds boolean
    ``pass_play``, ``success`` and ``explosive`` columns plus a yardage field
    clipped at the usual EPA-model bounds.
    """
    require_columns(plays, _RAW_PLAY_COLUMNS, "raw plays")
    plays = plays.drop_duplicates(["game_id", "game_play_number"]).copy()
    for column in _RAW_PLAY_FLAG_COLUMNS:
        if not plays[column].dropna().isin([True, False, 0, 1]).all():
            raise ValueError(f"Invalid raw-play flag {column}")
        plays[column] = plays[column].fillna(False).astype(bool)
    is_offensive_play = plays.rush | plays["pass"] | plays.sack
    is_live_ball = ~plays.kneel_down & ~plays.penalty_no_play
    plays = plays[
        is_offensive_play
        & is_live_ball
        & plays.down.between(1, 4)
        & plays.distance.between(1, 99)
        & plays.period.between(1, 4)
        & plays["start.pos_score_diff"].abs().le(21)
        & plays.statYardage.notna()
    ].copy()
    plays["pass_play"] = plays["pass"] | plays.sack
    threshold = plays.distance * np.select(
        [plays.down.eq(1), plays.down.eq(2)],
        [0.5, 0.7],
        default=1.0,
    )
    plays["success"] = plays.statYardage.ge(threshold).astype(float)
    plays["explosive"] = plays.statYardage.ge(np.where(plays.pass_play, 20, 10)).astype(float)
    plays["clipped_yards"] = plays.statYardage.clip(-20, 80)
    return plays


def aggregate_plays(plays: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned plays to per-game offense/defense summary rows."""
    cleaned = clean_plays(plays)
    return cleaned.groupby(["game_id", "pos_team_id", "def_pos_team_id"], as_index=False).agg(
        plays=("game_play_number", "size"),
        success_rate=("success", "mean"),
        explosive_rate=("explosive", "mean"),
        yards_per_play=("clipped_yards", "mean"),
    )


def observations(games: pd.DataFrame, summaries: pd.DataFrame) -> np.ndarray:
    """Build the ``(n_games, 2, n_metrics)`` observation tensor.

    The score metric is read from the schedule's final scores. The yardage
    metrics are read from the ESPN-derived per-game team summary. Provider
    EPA, win probability, Elo and betting-line fields are deliberately
    excluded from the inputs.
    """
    observation_tensor = np.full((len(games), 2, len(METRICS)), np.nan)
    observation_tensor[:, 0, 0] = games.home_points.to_numpy()
    observation_tensor[:, 1, 0] = games.away_points.to_numpy()
    if len(summaries):
        indexed_summaries = summaries.set_index(["game_id", "pos_team_id"])
        for side, team_column in enumerate(("home_id", "away_id")):
            index = pd.MultiIndex.from_arrays([games.game_id, games[team_column]])
            observation_tensor[:, side, 1:] = indexed_summaries.reindex(index)[list(METRICS[1:])].to_numpy(
                dtype=float
            )
    for metric_idx, (lower, upper) in _METRIC_CLIP_RANGES.items():
        observation_tensor[:, :, metric_idx] = np.clip(observation_tensor[:, :, metric_idx], lower, upper)
    return observation_tensor


def normalization(Y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-metric centering and scaling vectors from a warmup mask.

    The warmup slice must have finite data for every metric and non-zero
    spread; otherwise standardization is undefined.
    """
    center = np.nanmean(Y[mask], axis=(0, 1))
    scale = np.nanstd(Y[mask], axis=(0, 1))
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale < 1e-8).any():
        raise ValueError("Warmup data must contain variable observations for every metric")
    return center, scale


def feature_columns(variant: str) -> list[int]:
    """Return the column indices the given variant's score head is fit on.

    The 26 columns produced by :meth:`cfb_forecast.ratings.DynamicRatings.features`
    are laid out as:

    - Columns 0-4: offense means (per metric).
    - Columns 5-9: defense means.
    - Columns 10-14: offense minus defense.
    - Columns 15-19: offense times defense.
    - Column 20: variance of the points matchup.
    - Column 21: mean variance of the PBP matchups.
    - Column 22: signed venue indicator.
    - Column 23: early-season flag.
    - Column 24: offense pass-rush differential.
    - Column 25: defense pass-rush differential.

    The ``"score"`` variant restricts the head to a six-column subset:
    points offense, points defense, points O-D, points-matchup variance,
    signed venue, and the early-season flag. The ``"pbp"`` variant uses all
    26 columns.
    """
    if variant == "score":
        return [0, 5, 10, 20, 22, 23]
    if variant == "pbp":
        return list(range(26))
    raise ValueError(f"Unknown variant {variant}")
