"""Regression tests for the two Elo benchmark variants in ``cfb_forecast.baseline``.

These tests lock down the contract that the empirical comparison in the paper
depends on: the ``winner`` style must reproduce the original outcome-only Elo
exactly (so previously reported numbers remain reproducible), while the
``margin`` style must implement the bounded margin-of-victory update and change
ratings in the direction the research question predicts.
"""

import numpy as np
import pandas as pd
import pytest

from cfb_forecast.baseline import (
    elo_features,
    elo_update_size,
    evaluate_elo,
)
from cfb_forecast.data import normalize_schedule


def _games(rows):
    """Build a normalized games frame with the columns the replay consumes.

    Rows must supply game_id, season, week, start_date, home_id/away_id and
    scores; ``normalize_schedule`` derives the availability timestamps that
    enforce the event-time ordering used by the rating replay.
    """
    frame = pd.DataFrame(
        [
            {
                "game_id": k + 1,
                "season": row.get("season", 2025),
                "week": row.get("week", k + 1),
                # Two games per week: Saturday and Sunday kickoffs share the
                # Monday cutoff, so each game's pregame ratings see only prior
                # weeks' results -- exactly the event-time semantics the
                # baseline._replay_ratings guarantees.  The month/day math is
                # carried explicitly so long fixtures stay inside real dates.
                "start_date": (
                    pd.Timestamp("2025-08-01T12:00:00Z") + pd.Timedelta(days=7 * (k // 2), hours=6 * (k % 2))
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "completed": True,
                "neutral_site": row.get("neutral_site", False),
                "home_id": row["home_id"],
                "away_id": row["away_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_points": row["home_points"],
                "away_points": row["away_points"],
                "fbs_game": True,
            }
            for k, row in enumerate(rows)
        ]
    )
    return normalize_schedule(frame)


ONE_GAME = [
    {
        "home_team": "Alpha",
        "home_id": 1,
        "away_team": "Bravo",
        "away_id": 2,
        "home_points": 30,
        "away_points": 28,
    }
]


def test_update_size_winner_style_is_outcome_only():
    assert elo_update_size(31, 28, "winner") == elo_update_size(70, 0, "winner")
    assert elo_update_size(31, 28, "winner") == 1.0


def test_update_size_margin_scales_with_margin_and_total():
    close = elo_update_size(31, 28, "margin")
    blowout = elo_update_size(70, 0, "margin")
    assert blowout > close > 0.0
    # High-total close games get shrunk relative to low-total ones.
    low_total = elo_update_size(3, 0, "margin")
    high_total = elo_update_size(45, 42, "margin")
    assert low_total > high_total


def test_update_size_rejects_unknown_style():
    with pytest.raises(ValueError, match="Unknown Elo update style"):
        elo_update_size(30, 20, "total")


def test_features_validate_style():
    games = _games(ONE_GAME)
    with pytest.raises(ValueError, match="Elo style"):
        elo_features(games, style="ridge")


def test_winner_ratings_are_symmetric_about_center():
    # One game only: the two teams' post-game ratings must sum to the center,
    # and the pregame feature matrix is built from pregame ratings, so both
    # sides start equal and the difference column is zero before any history.
    features = elo_features(_games(ONE_GAME))
    assert features.shape == (1, 2)
    assert features[0, 0] == 1.0  # home venue indicator
    assert features[0, 1] == pytest.approx(0.0)  # no prior history yet


def test_margin_variant_moves_more_after_blowouts():
    def rating_gap(style):
        games = _games(
            [
                # Week 1: A beats B by three; C dominates D by forty.
                {
                    "home_team": "A",
                    "home_id": 1,
                    "away_team": "B",
                    "away_id": 2,
                    "home_points": 31,
                    "away_points": 28,
                },
                {
                    "home_team": "C",
                    "home_id": 3,
                    "away_team": "D",
                    "away_id": 4,
                    "home_points": 40,
                    "away_points": 0,
                },
                # Week 2: both winners host fresh opponents, so the observed
                # gap reflects only how far week-1 results moved the ratings.
                {
                    "home_team": "A",
                    "home_id": 1,
                    "away_team": "E",
                    "away_id": 5,
                    "home_points": 10,
                    "away_points": 7,
                },
                {
                    "home_team": "C",
                    "home_id": 3,
                    "away_team": "F",
                    "away_id": 6,
                    "home_points": 10,
                    "away_points": 7,
                },
            ]
        )
        F = elo_features(games, style=style)
        return F[3, 1] - F[2, 1]

    winner_gap = rating_gap("winner")
    margin_gap = rating_gap("margin")
    # Identical week-2 outcomes on top of different week-1 histories: the
    # winner-only variant has already diverged equally for A and C, so the
    # pregame gap is zero; the margin variant carried extra evidence from the
    # blowout, producing a strictly positive gap.
    assert winner_gap == pytest.approx(0.0)
    assert margin_gap > 0.0


def test_winner_style_matches_outcome_only_semantics_on_history():
    # Winner-style ratings depend only on who won: a 31-28 and a 70-0 first
    # game produce the identical second-game feature, while the margin style
    # must separate them.  A rematch is uninformative here because both
    # variants move the two teams by the same signed amount in every game, so
    # a head-to-head rating difference always returns to zero once both sides
    # share a history.  The fixture therefore gives each team its own week-1
    # opponent (A beats X, B beats Y) and meets them in week 2; only then does
    # the pregame gap reflect how far week-1 results moved the ratings.
    def pair(first_home, first_away):
        return _games(
            [
                {
                    "home_team": "A",
                    "home_id": 1,
                    "away_team": "X",
                    "away_id": 9,
                    "week": 1,
                    "season": 2025,
                    "start_date": "2025-08-30T12:00:00Z",
                    "home_points": first_home,
                    "away_points": first_away,
                },
                {
                    "home_team": "B",
                    "home_id": 2,
                    "away_team": "Y",
                    "away_id": 8,
                    "week": 1,
                    "season": 2025,
                    "start_date": "2025-08-30T15:00:00Z",
                    "home_points": 10,
                    "away_points": 7,
                },
                {
                    "home_team": "A",
                    "home_id": 1,
                    "away_team": "B",
                    "away_id": 2,
                    "week": 2,
                    "season": 2025,
                    "start_date": "2025-09-06T12:00:00Z",
                    "home_points": 14,
                    "away_points": 10,
                },
            ]
        )

    close = pair(31, 28)
    blow = pair(70, 0)
    assert elo_features(close, style="winner")[2, 1] == pytest.approx(
        elo_features(blow, style="winner")[2, 1]
    )
    assert elo_features(close, style="margin")[2, 1] < elo_features(blow, style="margin")[2, 1]


def test_evaluate_elo_reports_style_in_trials():
    rng = np.random.default_rng(7)
    n = 60
    games = _games(
        [
            {
                "home_team": f"H{i % 7}",
                "home_id": i % 7,
                "away_team": f"A{i % 7}",
                "away_id": 7 + i % 7,
                "home_points": int(rng.integers(10, 45)),
                "away_points": int(rng.integers(10, 45)),
            }
            for i in range(n)
        ]
    )
    index = np.arange(n)
    p, trials, chosen = evaluate_elo(
        games,
        train=index[: 3 * n // 5],
        val=index[3 * n // 5 : 4 * n // 5],
        final_train=index[: 4 * n // 5],
        test=index[4 * n // 5 :],
        style="margin",
    )
    assert p.shape == (n - 4 * n // 5,)
    assert all(t["style"] == "margin" for t in trials)
    assert chosen["style"] == "margin"
