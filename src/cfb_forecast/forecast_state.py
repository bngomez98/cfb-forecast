"""As-of rating-state construction for prospective forecasts.

Two ways to obtain the frozen state that a forecast is scored against:

- :func:`replay_state` rebuilds it from scratch by replaying every offseason
  reset and usable result available before the cutoff, in event order. The
  simulation clock never crosses the cutoff, so no post-cutoff information can
  enter the state.
- :func:`restore_state` resumes an exact previously saved state (from a frozen
  forecast's ``state_snapshot.npz``) and lets it decay forward. Callers must
  verify independently that no new result became eligible between cutoffs;
  see :mod:`cfb_forecast.forecast` for those guards.
"""
from __future__ import annotations

import pandas as pd

from .config import StateConfig
from .data import timestamp
from .features import observations
from .ratings import DynamicRatings


def replay_state(population, past, summaries, model, at):
    """Rebuild the rating state by replaying every event available before ``at``.

    Offseason resets fire 21 days before each season's first game; usable
    results apply in availability order.

    Args:
        population: Normalized FBS schedule restricted to the model's warmup
            through forecast seasons. Events may extend past ``at``; only the
            ones dated before it are replayed.
        past: Rows of ``population`` whose final result was usable strictly
            before ``at``, with a ``result_available_at`` column.
        summaries: Advanced team-stat frames fed through :func:`observations`.
        model: Loaded model dictionary (center, scale, state_config).
        at: Cutoff timestamp; the state is frozen exactly here.

    Returns:
        The advanced :class:`DynamicRatings` state at ``at``.
    """
    teams = sorted(set(population.home_id) | set(population.away_id))
    state = DynamicRatings(teams, model["center"], model["scale"], StateConfig(**model["state_config"]))
    Y = observations(past, summaries)
    events = []
    for year, g in population.groupby("season"):
        reset = g.start_date.min() - pd.Timedelta(days=21)
        if reset < at:
            events.append((reset.value, 0, int(year)))
    records = list(past.itertuples(index=False))
    for k, r in enumerate(records):
        events.append((r.result_available_at.value, 1, k))
    for tm, typ, k in sorted(events):
        event_at = pd.Timestamp(tm, tz="UTC")
        if typ == 0:
            state.offseason(event_at)
        else:
            r = records[k]
            state.update(r.home_id, r.away_id, r.neutral_site, Y[k], event_at)
    state.advance(at)
    return state


def restore_state(snapshot, manifest, model, old_at, at):
    """Restore an exact saved state and let it decay forward to ``at``.

    Args:
        snapshot: Loaded ``state_snapshot.npz`` from a previous forecast run
            (keys ``team_ids``, ``mean``, ``cov``).
        manifest: The previous run's ``forecast_manifest.json`` dict; supplies
            ``latest_result_available_at`` so update recency carries over.
        model: Loaded model dictionary matching the saved state.
        old_at: Cutoff the saved state was frozen at.
        at: New cutoff to advance to; must be later than ``old_at``.
    """
    state = DynamicRatings(
        snapshot["team_ids"].astype(int).tolist(),
        model["center"],
        model["scale"],
        StateConfig(**model["state_config"]),
    )
    state.mean = snapshot["mean"].copy()
    state.cov = snapshot["cov"].copy()
    state.clock = old_at
    latest = manifest.get("latest_result_available_at")
    state.latest_update = timestamp(latest) if latest else None
    state.advance(at)
    return state
