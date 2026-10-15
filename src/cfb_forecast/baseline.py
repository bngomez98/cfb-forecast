"""Independent chronological winner baselines: outcome-only and margin-scaled Elo.

Two Elo variants are provided so the analysis can answer two different
questions from the same score file:

``winner`` (outcome-only)
    Updates ratings from the win/loss indicator only, with a fixed K factor.
    This is the deliberately conservative benchmark: a one-point win and a
    fifty-point win move the rating by exactly the same amount, which is how
    outcome-only ranking systems treat results without victory margins.

``margin`` (margin-of-victory Elo)
    Multiplies the update by ``log(|margin| + 1)`` and divides it by the square
    root of the absolute point total. The logarithm keeps decisive wins
    informative without letting blowouts dominate; the total-points divisor
    shrinks a narrow win in a high-scoring shootout relative to an equally
    narrow win in a low-scoring game, because those two results carry different
    evidence about separation between teams.

Both variants share the event-time machinery in :func:`_replay_ratings`: each
game's pregame ratings are read at its forecast cutoff, and a result may only
update the ratings once its availability timestamp has arrived, so a later
score can never change an earlier forecast. Ratings regress toward the 1500
center across the offseason, matching the shrinkage logic the dynamic Gaussian
ratings use for the same reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import metrics
from .ratings import cutoff_for_games

#: Rating center every team starts from, and regresses toward between seasons.
ELO_CENTER = 1500.0

#: Base rating exchange per game for the winner-only variant.
WINNER_K_FACTOR = 20.0

#: Base rating exchange per game for the margin-scaled variant. The margin and
#: total adjustments scale this value up or down for each result.
MARGIN_K_FACTOR = 20.0

#: Multiplicative home-field advantage, in Elo points, used when predicting a
#: non-neutral game.
HOME_ADVANTAGE_POINTS = 55.0

#: Divisor converting an Elo difference into the logistic-style scale that is
#: fed to the calibration layer.
RATING_SCALE_POINTS = 400.0

#: Fraction of last season's rating deviation carried into the next season.
OFFSEASON_CARRY = 0.67

#: Regularization grid for the logistic calibration layer. Smaller values keep
#: the mapping from rating differences to probabilities closer to uniform.
_CALIBRATION_GRID = (0.01, 0.1, 1.0, 10.0)

#: Feature columns returned by :func:`elo_features`: signed venue indicator and
#: scaled rating difference.
ELO_FEATURE_NAMES = ("venue", "rating_difference")


def _margin_multiplier(margin: float) -> float:
    """Bounded log multiplier on the Elo update for a given winning margin."""
    return float(np.log(abs(float(margin)) + 1.0))


def _total_divisor(total: float) -> float:
    """Point-total adjustment keeping low-scoring games from dominating.

    A 3-0 result and a 45-42 result both have small margins but very different
    scoring environments. The sqrt(2)/sqrt(total) factor normalizes the margin
    contribution so updates stay comparable across scoring levels; it is
    floored so a zero-total placeholder cannot divide by zero.
    """
    return float(np.sqrt(2.0) / max(np.sqrt(float(total)), 1e-8))


def elo_update_size(home_points, away_points, style: str = "winner") -> float:
    """Return the multiplicative update applied to the base K factor.

    ``"winner"`` always returns 1.0, so binary outcomes alone decide how far
    ratings move. ``"margin"`` returns the log-margin multiplier divided by the
    point-total adjustment, which is the standard margin-of-victory correction.
    """
    if style == "winner":
        return 1.0
    if style == "margin":
        margin = abs(float(home_points) - float(away_points))
        total = float(home_points) + float(away_points)
        return _margin_multiplier(margin) * _total_divisor(total)
    raise ValueError(f"Unknown Elo update style {style!r}")


def _validate_style(style: str) -> str:
    if style not in ("winner", "margin"):
        raise ValueError("Elo style must be 'winner' or 'margin'")
    return style


def _replay_ratings(games: pd.DataFrame, explicit=None, *, style: str = "winner") -> np.ndarray:
    """Replay the schedule in event time, returning pregame ratings per side.

    The returned array has shape ``(n_games, 2)`` with the home team's rating
    and the away team's rating as they stood at that game's forecast cutoff.
    Results are applied only at their availability timestamps, and ratings are
    shrunk toward :data:`ELO_CENTER` at each season boundary.
    """
    style = _validate_style(style)
    rating = {int(team): ELO_CENTER for team in set(games.home_id) | set(games.away_id)}
    cutoffs = cutoff_for_games(games, explicit)
    records = list(games.itertuples(index=False))
    events: list[tuple[int, int, int]] = []
    for k, record in enumerate(records):
        events.append((cutoffs.iloc[k].value, 1, k))
        events.append((record.result_available_at.value, 2, k))
    for year, group in games.groupby("season"):
        reset = group.start_date.min() - pd.Timedelta(days=21)
        events.append((reset.value, 0, int(year)))
    pregame = np.full((len(games), 2), np.nan)
    for _, event_type, k in sorted(events):
        if event_type == 0:
            rating = {
                team: ELO_CENTER + OFFSEASON_CARRY * (value - ELO_CENTER) for team, value in rating.items()
            }
            continue
        record = records[k]
        home, away = int(record.home_id), int(record.away_id)
        if event_type == 1:
            pregame[k] = (rating[home], rating[away])
            continue
        venue = float(not record.neutral_site)
        expected = 1.0 / (
            1.0
            + 10.0 ** (-(rating[home] - rating[away] + HOME_ADVANTAGE_POINTS * venue) / RATING_SCALE_POINTS)
        )
        home_win = float(record.home_points > record.away_points)
        base = WINNER_K_FACTOR if style == "winner" else MARGIN_K_FACTOR
        delta = base * elo_update_size(record.home_points, record.away_points, style) * (home_win - expected)
        rating[home] += delta
        rating[away] -= delta
    return pregame


def elo_features(games: pd.DataFrame, explicit=None, *, style: str = "winner") -> np.ndarray:
    """Build the two-column pregame design matrix for one Elo variant.

    Column 0 is the signed venue indicator (1 for a home game, 0 for neutral);
    column 1 is the home-minus-away rating difference rescaled to unit-ish
    magnitude. Keeping the rescaling here means the calibration layer sees a
    stationary covariate regardless of the K factor used by the state replay.
    """
    ratings = _replay_ratings(games, explicit, style=style)
    venue = (~games.neutral_site.to_numpy(dtype=bool)).astype(float)[:, None]
    difference = (ratings[:, 0] - ratings[:, 1]) / RATING_SCALE_POINTS
    return np.column_stack([venue, difference])


def _calibrate(features: np.ndarray, outcome: np.ndarray, C: float):
    """Fit the symmetric-logistic calibration layer on one regularization value.

    Each game contributes two rows, one per side, so the fitted coefficients
    stay antisymmetric: swapping the two teams inverts the predicted probability
    rather than shifting it by an intercept.
    """
    stacked = np.vstack([features, -features])
    labels = np.concatenate([outcome, 1 - outcome])
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, fit_intercept=False, max_iter=1000),
    ).fit(stacked, labels)


def evaluate_elo(
    games: pd.DataFrame,
    train: np.ndarray,
    val: np.ndarray,
    final_train: np.ndarray,
    test: np.ndarray,
    explicit=None,
    *,
    style: str = "winner",
):
    """Select the calibration strength on validation and grade the test slice.

    Returns the test-set win probabilities, the per-``C`` validation trials and
    the selected trial record.
    """
    features = elo_features(games, explicit, style=style)
    outcome = (games.home_points > games.away_points).to_numpy(dtype=int)
    trials = []
    for C in _CALIBRATION_GRID:
        model = _calibrate(features[train], outcome[train], C)
        p = model.predict_proba(features[val])[:, 1]
        trials.append({"C": C, "style": style, **metrics(outcome[val], p)})
    chosen = min(trials, key=lambda record: record["log_loss"])
    production = _calibrate(features[final_train], outcome[final_train], chosen["C"])
    return production.predict_proba(features[test])[:, 1], trials, chosen
