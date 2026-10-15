"""Exact Kalman updates for additive offense and defense team states.

Each of the five metrics in :data:`cfb_forecast.config.METRICS` has an
independent Gaussian state model shared across all teams: every team has an
offense coefficient and a defense coefficient per metric, and games enter the
model as joint observations of ``home_offense - away_defense`` and
``away_offense - home_defense`` plus noise.

The implementation here keeps three invariants that make the model safe to
use for live forecasting:

- The internal clock never runs backward. Any attempt to advance to an
  earlier timestamp raises.
- The covariance matrices stay symmetric and positive semi-definite after
  every update, because the Kalman form is used explicitly and the result is
  symmetrized to defend against floating-point drift.
- A forecast cutoff cannot see a result whose availability timestamp is at or
  after the cutoff. :func:`build_features` asserts this.

Hyperparameters are fixed for an entire evaluated season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import METRICS, StateConfig
from .data import timestamp

#: Length of the feature row returned by :meth:`DynamicRatings.features`.
#: The layout is documented in :func:`cfb_forecast.features.feature_columns`.
FEATURE_WIDTH = 26

#: Maximum between-game gap, in weeks, that counts as a mid-season step. Gaps
#: longer than this are treated as offseasons by :meth:`DynamicRatings.advance`
#: so weekly diffusion does not accumulate across a dormant period.
MAX_WEEKLY_GAP_WEEKS = 15


class DynamicRatings:
    """Per-metric Gaussian offense/defense ratings for a fixed roster of teams.

    Parameters
    ----------
    teams:
        Iterable of integer team IDs. The order is retained for all internal
        indexing.
    center, scale:
        Vectors of length ``len(METRICS)`` used to standardize observations
        before they enter the state update.
    config:
        Hyperparameters, see :class:`cfb_forecast.config.StateConfig`.
    """

    def __init__(
        self,
        teams,
        center,
        scale,
        config: StateConfig,
    ) -> None:
        self.teams = [int(t) for t in teams]
        self.index = {t: i for i, t in enumerate(self.teams)}
        self.n = len(self.teams)
        self.dim = 2 * self.n
        self.center = np.asarray(center, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.config = config
        self.mean = np.zeros((len(METRICS), self.dim))
        self.cov = np.tile(
            np.eye(self.dim) * config.prior_variance,
            (len(METRICS), 1, 1),
        )
        self.clock: pd.Timestamp | None = None
        self.latest_update: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------
    def advance(self, at) -> None:
        """Advance the internal clock to ``at`` with weekly diffusion.

        A step longer than :data:`MAX_WEEKLY_GAP_WEEKS` is treated as an
        offseason by not applying weekly diffusion; the dedicated
        :meth:`offseason` call then handles the reset. Backward steps raise.
        """
        at = timestamp(at)
        if self.clock is not None:
            if at < self.clock:
                raise ValueError("State clock cannot run backward")
            weeks = (at - self.clock).total_seconds() / 604800.0
            if 0 < weeks < MAX_WEEKLY_GAP_WEEKS:
                carry = self.config.weekly_carry**weeks
                self.mean *= carry
                self.cov *= carry * carry
                diag = np.arange(self.dim)
                self.cov[:, diag, diag] += self.config.process_variance * weeks
        self.clock = at

    def offseason(self, at) -> None:
        """Apply the offseason shrinkage and variance bump at time ``at``."""
        self.advance(at)
        carry = self.config.offseason_carry
        self.mean *= carry
        self.cov *= carry * carry
        diag = np.arange(self.dim)
        self.cov[:, diag, diag] += self.config.offseason_variance

    # ------------------------------------------------------------------
    # Features for the scoring head
    # ------------------------------------------------------------------
    def features(self, home, away, neutral: bool, early: bool = False) -> np.ndarray:
        """Build a pair of pregame feature rows, one per side.

        The returned array has shape ``(2, FEATURE_WIDTH)``. Side 0 is the
        home team's view of the matchup; side 1 is the away team's view. The
        column layout is documented in
        :func:`cfb_forecast.features.feature_columns`.
        """
        hi = self.index[int(home)]
        ai = self.index[int(away)]
        pairs = [(hi, self.n + ai), (ai, self.n + hi)]
        venue = 0 if neutral else 1
        features = np.zeros((2, FEATURE_WIDTH))
        for side, (offense_idx, defense_idx) in enumerate(pairs):
            offense_mean = self.mean[:, offense_idx]
            defense_mean = self.mean[:, defense_idx]
            variances = (
                self.cov[:, offense_idx, offense_idx]
                + self.cov[:, defense_idx, defense_idx]
                - 2 * self.cov[:, offense_idx, defense_idx]
            )
            venue_sign = venue * (1 if side == 0 else -1)
            features[side] = np.r_[
                offense_mean,
                defense_mean,
                offense_mean - defense_mean,
                offense_mean * defense_mean,
                variances[0],
                variances[1:].mean(),
                venue_sign,
                int(early),
                offense_mean[2] - offense_mean[3],
                defense_mean[2] - defense_mean[3],
            ]
        return features

    # ------------------------------------------------------------------
    # Observation / Kalman update
    # ------------------------------------------------------------------
    def update(
        self,
        home,
        away,
        neutral: bool,
        values: np.ndarray,
        available_at,
    ) -> None:
        """Apply a joint Kalman update for one completed game.

        ``values`` has shape ``(2, len(METRICS))``; non-finite entries skip
        that metric's update for that side. Both sides' observations in a
        game are conditioned jointly, with the fixed observation correlation
        from :class:`StateConfig`.
        """
        at = timestamp(available_at)
        self.advance(at)
        hi = self.index[int(home)]
        ai = self.index[int(away)]
        pairs = [(hi, self.n + ai), (ai, self.n + hi)]
        venue = 0 if neutral else 1
        for j in range(len(METRICS)):
            sides_with_values = np.flatnonzero(np.isfinite(values[:, j]))
            if not len(sides_with_values):
                continue
            # Cross-covariance between every state coordinate and the
            # observation at each side.
            cross = np.stack(
                [self.cov[j, :, pairs[s][0]] - self.cov[j, :, pairs[s][1]] for s in sides_with_values],
                axis=1,
            )
            # Observation covariance for the sides actually observed.
            observation_cov = np.stack([cross[pairs[s][0]] - cross[pairs[s][1]] for s in sides_with_values])
            residual_cov = np.full_like(observation_cov, self.config.observation_correlation)
            np.fill_diagonal(residual_cov, 1.0)

            residual = []
            for s in sides_with_values:
                offense_idx, defense_idx = pairs[s]
                venue_adjustment = (
                    venue * self.config.home_margin / 2 / self.scale[0] * (1 if s == 0 else -1)
                    if j == 0
                    else 0.0
                )
                prediction = self.mean[j, offense_idx] - self.mean[j, defense_idx]
                standardized = (values[s, j] - self.center[j]) / self.scale[j]
                residual.append(standardized - venue_adjustment - prediction)

            gain = np.linalg.solve(observation_cov + residual_cov, cross.T).T
            self.mean[j] += gain @ np.asarray(residual)
            self.cov[j] -= gain @ cross.T
            # Symmetrize to defend against floating-point drift.
            self.cov[j] = (self.cov[j] + self.cov[j].T) / 2
        self.latest_update = at


def cutoff_for_games(games: pd.DataFrame, explicit: dict | None = None) -> pd.Series:
    """Compute the forecast cutoff for each game.

    The default cutoff is Monday 00:00 UTC of the week leading into kickoff.
    ``explicit`` optionally maps ``game_id`` to a verified archived forecast
    timestamp; any mapped game uses that explicit cutoff instead.

    Raises if any cutoff falls at or after its own kickoff.
    """
    dates = games.start_date
    cutoffs = dates.dt.normalize() - pd.to_timedelta(dates.dt.weekday, unit="D")
    cutoffs = cutoffs.where(cutoffs < dates, cutoffs - pd.Timedelta(days=7))
    if explicit:
        cutoffs = pd.Series(
            [
                timestamp(explicit[int(gid)]) if int(gid) in explicit else cut
                for gid, cut in zip(games.game_id, cutoffs, strict=True)
            ],
            index=games.index,
        )
    if (cutoffs >= games.start_date).any():
        raise ValueError("A forecast cutoff must be before kickoff")
    return cutoffs


def build_features(
    games: pd.DataFrame,
    Y: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    config: StateConfig,
    explicit: dict | None = None,
    warmup: int = 2021,
) -> tuple[np.ndarray, pd.Series, list]:
    """Replay the schedule in chronological order and return pregame features.

    For each game past the warmup season, the state is advanced to that
    game's cutoff and the feature row is read off before the game's own
    result is applied. The returned arrays contain, for each game: the two
    sides' feature rows, the cutoff timestamp, and the timestamp of the most
    recent result that had entered the state at that cutoff (``None`` when
    the game is the first to be forecast).

    The assertion ``latest_update < cutoff`` guards against a future result
    leaking into an earlier forecast; it is deliberately kept in production.
    """
    teams = sorted(set(games.home_id) | set(games.away_id))
    states = DynamicRatings(teams, center, scale, config)
    features = np.full((len(games), 2, FEATURE_WIDTH), np.nan)
    cutoffs = cutoff_for_games(games, explicit)
    events: list[tuple[int, int, int]] = []
    records = list(games.itertuples(index=False))
    for k, record in enumerate(records):
        if int(record.season) > warmup:
            events.append((cutoffs.iloc[k].value, 1, k))
        events.append((record.result_available_at.value, 2, k))
    for year, group in games.groupby("season"):
        events.append(((group.start_date.min() - pd.Timedelta(days=21)).value, 0, int(year)))
    latest: list[pd.Timestamp | None] = [None] * len(games)
    for ns, event_type, k in sorted(events):
        at = pd.Timestamp(ns, tz="UTC")
        if event_type == 0:
            states.offseason(at)
            continue
        record = records[k]
        states.advance(at)
        if event_type == 1:
            if states.latest_update is not None and states.latest_update >= at:
                raise AssertionError("Result at or after forecast cutoff entered ratings")
            features[k] = states.features(
                record.home_id,
                record.away_id,
                record.neutral_site,
                int(record.week) <= 4,
            )
            latest[k] = states.latest_update
        else:
            states.update(
                record.home_id,
                record.away_id,
                record.neutral_site,
                Y[k],
                at,
            )
    return features, cutoffs, latest
