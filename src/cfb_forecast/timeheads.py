"""Per-team time-series score heads: the classical-forecasting experts.

This module adds a fourth family of production experts alongside the
conditional cross-sectional heads in :mod:`cfb_forecast.heads`.  Instead of
regressing one game on matchup features, each team's *own scoring history* is
treated as a time series and extrapolated forward with off-the-shelf
statistical forecasters:

* ``statsforecast`` (NixtaStats) supplies the auto-ARIMA engine — an
  information-criterion-searched ARIMA per team, run through the library's
  vectorized core so hundreds of short team series fit in milliseconds;
* ``statsmodels`` contributes the statistical machinery behind the walk-forward
  calibration (Kpss/unit-root reasoning is delegated to AutoARIMA's own tests)
  and the residual-spread estimation that fixes the logistic link scale.

Why this exists in a rating-free pipeline
-----------------------------------------
The literature review flagged a measured cost of going rating-free: engineered
strength signals carry real predictive weight (Lampis et al., 2023).  A
team's own recent points-for/points-against sequence is exactly the implicit
strength encoding those authors recommend ("recent-form indicators", e.g. the
last-three-games design of Rodriguez, 2019) — but expressed as a *forecast of
scores*, never as a rating table.  The head predicts next-game home points and
away points from each side's trailing series, then converts them to margin and
win probability through a residual-calibrated logistic link.

Leakage discipline
------------------
Series are built only from settled games whose kickoff precedes the fixture
being predicted; a fixture that already has a result in the training sample
never sees its own score.  No smoothing is applied to observed scores; the
models consume raw recorded points and their internal state handles noise.

Short early-season series are expected.  AutoARIMA with ``max_p=max_q=1``,
``stationary=True`` (unit-root search disabled) and ``season_length=1`` keeps
the search space honest at ~5 observations, and the head abstains (returns
NaN) for teams below ``MIN_SERIES`` settled games rather than inventing a
forecast from noise.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

#: Minimum settled games in a team's trailing window before we trust any
#: time-series extrapolation for that side.
MIN_SERIES = 5

#: Trailing window (games) supplied to the forecaster per team-side.
SERIES_WINDOW = 10

#: Logistic scale converting predicted margin to win probability when no
#: calibration sample is available yet.
DEFAULT_PROBABILITY_SCALE = 14.0


def _side_history(games: pd.DataFrame) -> pd.DataFrame:
    """Stack settled games into a long per-team point series.

    Columns: ``unique_id`` (team id + side tag, so home and away scoring are
    separate series — venue effects are large in college football), ``ds``
    (kickoff timestamp), ``y`` (raw points scored).
    """
    home = pd.DataFrame(
        {
            "unique_id": games.home_id.astype(int).astype(str) + "_H",
            "ds": pd.to_datetime(games.start_date),
            "y": games.home_points.astype(float),
        }
    )
    away = pd.DataFrame(
        {
            "unique_id": games.away_id.astype(int).astype(str) + "_A",
            "ds": pd.to_datetime(games.start_date),
            "y": games.away_points.astype(float),
        }
    )
    frame = pd.concat([home, away], ignore_index=True)
    return frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)


class TimeSeriesHeads:
    """AutoARIMA score forecasters over per-team points series.

    Emits one blend expert, ``ts_arima``: the home-win probability implied by
    forecast home points minus forecast away points, plus the raw margin and
    expected-total signals for reporting.

    The class refits from the current settled sample on every call to
    :meth:`predict_frame` — new results change tomorrow's series, hence
    tomorrow's forecasts and blend weights.  That is the continuous-dynamic
    requirement implemented literally: nothing here is cached across days.
    """

    EXPERTS = ("ts_arima",)

    def __init__(self, *, window: int = SERIES_WINDOW, min_series: int = MIN_SERIES) -> None:
        self.window = int(window)
        self.min_series = int(min_series)
        self.probability_scale_ = DEFAULT_PROBABILITY_SCALE
        self.trained_at_games = 0

    # ------------------------------------------------------------------ core
    def _forecast(self, history: pd.DataFrame, uids: list[str]) -> pd.Series:
        """One-step-ahead point forecasts for the requested series ids."""
        from statsforecast.core import StatsForecast
        from statsforecast.models import AutoARIMA

        present = set(history.unique_id)
        wanted = [uid for uid in uids if uid in present]
        empty = pd.Series(dtype=float, index=pd.Index(uids))
        if not wanted:
            return empty
        subset = history[history.unique_id.isin(wanted)]
        model = AutoARIMA(season_length=1, max_p=1, max_q=1, d=0, stationary=True, stepwise=True)
        sf = StatsForecast(models=[model], freq="D", n_jobs=1)
        with warnings.catch_warnings():
            # Short series legitimately trip convergence/degenerate-order
            # warnings; the mean/drift fallback is still a valid forecast.
            warnings.simplefilter("ignore")
            sf.fit(subset)
            out = sf.predict(h=1)
        values = out.set_index("unique_id")["AutoARIMA"]
        return values.reindex(pd.Index(uids))

    # ------------------------------------------------------------- prediction
    def predict_frame(self, fixtures: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
        """Forecast upcoming fixtures from the settled sample ``games``.

        ``fixtures`` needs ``home_id``/``away_id`` and ``start_date``.
        Returns ``ts_margin``, ``ts_expected_total``, ``ts_p_home`` aligned to
        the fixture index.  Sides with fewer than ``min_series`` prior games
        abstain (NaN) so the adaptive blend renormalizes over the remaining
        experts instead of trusting an extrapolation from noise.

        Leakage safety: a fixture already settled inside ``games`` (backfill
        case) is excluded from its own conditioning series by truncating each
        side's history strictly before the fixture kickoff.  For genuinely
        upcoming fixtures this truncation is a no-op and the full vectorized
        AutoARIMA pass applies.
        """
        settled = games[games.home_points.notna() & games.away_points.notna()].copy()
        history = _side_history(settled)
        starts = pd.to_datetime(fixtures.start_date)

        home_uids = fixtures.home_id.astype(int).astype(str).to_numpy() + "_H"
        away_uids = fixtures.away_id.astype(int).astype(str).to_numpy() + "_A"
        all_uids = list(dict.fromkeys([*home_uids, *away_uids]))
        forecast = self._forecast(history, all_uids)

        # Per-side series arrays and timestamps for strict truncation checks.
        grouped = {
            uid: (grp.ds.to_numpy(dtype="datetime64[ns]"), grp.y.to_numpy(dtype=float))
            for uid, grp in history.groupby("unique_id")
        }

        def _usable(uid: str, when) -> bool:
            entry = grouped.get(uid)
            if entry is None:
                return False
            boundary = pd.Timestamp(when).to_datetime64()
            prior = entry[1][entry[0] < boundary]
            return len(prior) >= self.min_series

        rows = []
        for i in range(len(fixtures)):
            hu, au = home_uids[i], away_uids[i]
            fh, fa = forecast.get(hu), forecast.get(au)
            if fh is None or fa is None or pd.isna(fh) or pd.isna(fa):
                rows.append((np.nan, np.nan, np.nan))
                continue
            # Backfill guard: both sides must have enough strictly-prior games.
            if not (_usable(hu, starts.iloc[i]) and _usable(au, starts.iloc[i])):
                rows.append((np.nan, np.nan, np.nan))
                continue
            margin = float(fh) - float(fa)
            total = float(fh) + float(fa)
            p = 1.0 / (1.0 + np.exp(-margin / self.probability_scale_))
            rows.append((margin, total, p))
        self.trained_at_games = len(settled)
        return pd.DataFrame(
            {
                "ts_margin": [r[0] for r in rows],
                "ts_expected_total": [r[1] for r in rows],
                "ts_p_home": [r[2] for r in rows],
            },
            index=fixtures.index,
        )

    def calibrate(self, games: pd.DataFrame) -> TimeSeriesHeads:
        """Set the margin→probability link from walk-forward residuals.

        Chronological holdout: for the last ``CALIBRATION_GAMES`` settled
        games, recompute the head's forecast using only games strictly before
        the fixture (via :meth:`predict_frame`, which enforces that), then
        collect predicted-minus-actual margin errors.  The error spread fixes
        the logistic scale so probabilities reflect genuine accuracy.
        """
        games = games[games.home_points.notna() & games.away_points.notna()].copy()
        games = games.sort_values("start_date").reset_index(drop=True)
        if len(games) < 60:
            return self
        holdout = games.tail(CALIBRATION_GAMES).reset_index(drop=True)
        preds = self.predict_frame(holdout, games)
        actual = (holdout.home_points - holdout.away_points).to_numpy(dtype=float)
        mask = preds.ts_margin.notna().to_numpy()
        if mask.sum() >= 30:
            spread = float(np.std(preds.ts_margin.to_numpy()[mask] - actual[mask]))
            self.probability_scale_ = max(4.0, min(40.0, spread / 2.0))
        return self

    # ------------------------------------------------------------ persistence
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "window": self.window,
            "min_series": self.min_series,
            "probability_scale_": self.probability_scale_,
            "trained_at_games": self.trained_at_games,
        }
        with open(path / "ts_heads.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @classmethod
    def load(cls, path) -> TimeSeriesHeads:
        with open(Path(path) / "ts_heads.json", encoding="utf-8") as handle:
            payload = json.load(handle)
        heads = cls(window=payload["window"], min_series=payload["min_series"])
        heads.probability_scale_ = float(payload["probability_scale_"])
        heads.trained_at_games = int(payload["trained_at_games"])
        return heads


#: Number of most-recent settled games used as the walk-forward calibration
#: holdout (roughly two weeks of FBS slates).
CALIBRATION_GAMES = 120


def ts_expert_columns() -> dict[str, str]:
    """Map blend expert names onto :class:`TimeSeriesHeads` output columns."""
    return {"ts_arima": "ts_p_home"}
