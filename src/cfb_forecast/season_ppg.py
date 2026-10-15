"""Season-to-date points-per-game, strictly before a cutoff time.

This complements the opponent-adjusted points rating that the dynamic
ratings model already fits (metric index 0 in
:data:`cfb_forecast.config.METRICS`). The ratings model asks a Kalman
filter "how good is this team's offense and defense versus the schedule
they played"; this module answers the simpler question "what has this
team actually scored and allowed per game to date".

The two numbers are different signals. The Kalman rating adjusts for
opponent strength and shrinks noisy early-season data toward average,
which makes it the right input for a probability model but a lossy
summary for people reading the forecast. Raw PPG is noisy on tiny
samples and does not adjust for opponent, but it is the number any human
reader can check against the box score.

The module is strictly cutoff-safe: a game is only included in a team's
PPG when its final-score availability timestamp is strictly earlier than
the given ``as_of``. This is the same rule the dynamic ratings use so a
later game cannot leak into an earlier forecast.
"""

from __future__ import annotations

import logging

import pandas as pd

from .data import timestamp

log = logging.getLogger(__name__)

PPG_COLUMNS = [
    "team_id",
    "games_played_to_date",
    "points_scored_total",
    "points_allowed_total",
    "ppg_scored",
    "ppg_allowed",
    "ppg_margin",
    "as_of",
]


def compute_season_ppg(
    schedules: pd.DataFrame,
    *,
    season: int,
    as_of,
    fbs_only: bool = True,
) -> pd.DataFrame:
    """Return a per-team season-to-date PPG table at ``as_of``.

    Parameters
    ----------
    schedules:
        Normalized schedule frame from :func:`cfb_forecast.data.load_data`.
    season:
        Season to summarize. Only games in this season are included.
    as_of:
        Cutoff timestamp. A game only contributes once its
        ``result_available_at`` is strictly earlier than this time.
    fbs_only:
        When true (default), restrict both inclusion and the team roster
        to FBS-vs-FBS games.

    Returns
    -------
    pd.DataFrame
        One row per team that played at least one usable game before the
        cutoff. Columns are in :data:`PPG_COLUMNS`. Teams with no games
        before ``as_of`` do not appear; callers that want a complete
        roster should left-join against the schedule's team list.
    """
    cutoff = timestamp(as_of)
    frame = schedules.copy()
    frame = frame[frame.season.eq(int(season)) & frame.result_usable]
    if fbs_only:
        frame = frame[frame.fbs_game]
    frame = frame[frame.result_available_at < cutoff]
    if frame.empty:
        return pd.DataFrame(columns=PPG_COLUMNS)
    home = frame.rename(
        columns={
            "home_id": "team_id",
            "home_points": "points_scored",
            "away_points": "points_allowed",
        }
    )[["team_id", "points_scored", "points_allowed"]]
    away = frame.rename(
        columns={
            "away_id": "team_id",
            "away_points": "points_scored",
            "home_points": "points_allowed",
        }
    )[["team_id", "points_scored", "points_allowed"]]
    stacked = pd.concat([home, away], ignore_index=True)
    summed = stacked.groupby("team_id", as_index=False).agg(
        games_played_to_date=("points_scored", "size"),
        points_scored_total=("points_scored", "sum"),
        points_allowed_total=("points_allowed", "sum"),
    )
    summed["ppg_scored"] = summed.points_scored_total / summed.games_played_to_date
    summed["ppg_allowed"] = summed.points_allowed_total / summed.games_played_to_date
    summed["ppg_margin"] = summed.ppg_scored - summed.ppg_allowed
    summed["as_of"] = cutoff.isoformat()
    log.info(
        "season_ppg: season=%s teams=%d games=%d as_of=%s",
        season,
        len(summed),
        len(frame),
        cutoff.isoformat(),
    )
    return summed[PPG_COLUMNS].astype(
        {
            "team_id": "int64",
            "games_played_to_date": "int64",
            "points_scored_total": "int64",
            "points_allowed_total": "int64",
        }
    )


def attach_ppg_to_forecasts(
    forecasts: pd.DataFrame,
    ppg_table: pd.DataFrame,
) -> pd.DataFrame:
    """Attach season-PPG columns to each forecast row for both sides.

    The returned frame carries, for every row, the home side's and away
    side's PPG scored, allowed and margin, plus the number of games each
    side had played before the forecast cutoff. Teams absent from the
    PPG table (zero games played) receive NaN; downstream consumers
    treat that as "no sample yet" and never as zero.
    """
    required = {"home_id", "away_id"}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Forecast frame missing columns: {sorted(missing)}")
    if ppg_table.empty:
        enriched = forecasts.copy()
        for side in ("home", "away"):
            for suffix in (
                "ppg_scored",
                "ppg_allowed",
                "ppg_margin",
                "games_played_to_date",
            ):
                enriched[f"{side}_{suffix}"] = float("nan")
        return enriched
    renamed = ppg_table.rename(
        columns={
            "ppg_scored": "ppg_scored_val",
            "ppg_allowed": "ppg_allowed_val",
            "ppg_margin": "ppg_margin_val",
            "games_played_to_date": "games_played_val",
        }
    )[
        [
            "team_id",
            "ppg_scored_val",
            "ppg_allowed_val",
            "ppg_margin_val",
            "games_played_val",
        ]
    ]
    enriched = forecasts.copy()
    for side, column in (("home", "home_id"), ("away", "away_id")):
        merged = enriched.merge(renamed, left_on=column, right_on="team_id", how="left").drop(
            columns="team_id"
        )
        enriched[f"{side}_ppg_scored"] = merged.ppg_scored_val.to_numpy()
        enriched[f"{side}_ppg_allowed"] = merged.ppg_allowed_val.to_numpy()
        enriched[f"{side}_ppg_margin"] = merged.ppg_margin_val.to_numpy()
        enriched[f"{side}_games_played_to_date"] = merged.games_played_val.to_numpy()
    return enriched


__all__ = [
    "PPG_COLUMNS",
    "attach_ppg_to_forecasts",
    "compute_season_ppg",
]
