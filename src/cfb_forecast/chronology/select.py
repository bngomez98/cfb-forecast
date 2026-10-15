"""Picking which games are due to be forecast at a given wall-clock time."""

from __future__ import annotations

import pandas as pd

from ..data import timestamp


def select_due_games(
    schedules: pd.DataFrame,
    season: int,
    now,
    lead_hours: float,
    already_forecasted=(),
) -> pd.DataFrame:
    """Pick FBS games whose fixed lead-time cutoff has just arrived.

    A game is "due" when its ``start_date - lead_hours`` is at or before ``now``
    and kickoff is still in the future. Games already in
    ``already_forecasted`` are excluded, so repeated runs don't reforecast the
    same game.

    Parameters
    ----------
    schedules : pd.DataFrame
        Normalized schedule with ``fbs_game``, ``season``, ``game_id`` and
        ``start_date`` columns.
    season : int
        Season to restrict selection to.
    now : datetime-like
        Current wall-clock time. Parsed through ``data.timestamp``.
    lead_hours : float
        How many hours before kickoff the forecast cutoff falls.
    already_forecasted : iterable of int
        Game IDs that have already been written to the ledger.

    Returns
    -------
    pd.DataFrame
        Due games with an added ``scheduled_cutoff`` column, sorted by
        cutoff, then kickoff, then game ID.
    """
    at = timestamp(now)
    lead = pd.Timedelta(hours=float(lead_hours))
    f = schedules[(schedules.fbs_game) & (schedules.season == int(season))].copy()
    f["scheduled_cutoff"] = f.start_date - lead
    done = {int(x) for x in already_forecasted}
    due = f[(f.start_date > at) & (f.scheduled_cutoff <= at)]
    if done:
        due = due[~due.game_id.astype(int).isin(done)]
    return due.sort_values(["scheduled_cutoff", "start_date", "game_id"]).reset_index(drop=True)
