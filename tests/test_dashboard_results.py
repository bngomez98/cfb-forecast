"""The dashboard's results page grades saved forecasts against final scores."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"


def _game(week, home_chance, home_points, away_points, margin):
    return {
        "game_id": f"{week}-{home_chance}",
        "season": 2026,
        "week": week,
        "season_type": "regular",
        "home_team": f"Home {week}",
        "away_team": f"Away {week}",
        "kickoff": "2026-09-05T23:00:00Z",
        "home_win_probability": home_chance,
        "projected_home_margin": margin,
        "expected_home_score": 28,
        "expected_away_score": 28 - margin,
        "home_points": home_points,
        "away_points": away_points,
    }


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    games = [
        _game(1, 0.80, 31, 20, 7.0),   # right, missed the margin by 4
        _game(1, 0.30, 14, 24, -6.0),  # right (away pick), missed by 4
        _game(2, 0.65, 10, 17, 3.0),   # wrong, missed by 10
        _game(2, 0.55, 21, 20, 1.0),   # right, exact margin
    ]
    folder = tmp_path / "backtest" / "season-2026"
    folder.mkdir(parents=True)
    pd.DataFrame(games).to_csv(folder / "graded_games.csv", index=False)
    monkeypatch.setenv("CFB_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(DASHBOARD))
    yield tmp_path
    for name in ("results", "board"):
        sys.modules.pop(name, None)


def test_results_grade_each_pick(artifacts):
    import results

    frame, source, season = results.load_results()
    assert season == 2026
    assert source == results.BACKTEST
    assert frame.correct.tolist() == [True, True, False, True]
    assert frame.margin_miss.tolist() == pytest.approx([4.0, 4.0, 10.0, 0.0])
    assert frame.pick_chance.tolist() == pytest.approx([0.80, 0.70, 0.65, 0.55])
    assert frame.log_loss.tolist() == pytest.approx([-np.log(0.8), -np.log(0.7), -np.log(0.35), -np.log(0.55)])
    assert results.season_accuracy() == (pytest.approx(0.75), 4)


def test_sunday_runs_are_preferred_over_rebuilt_weeks(artifacts):
    run = artifacts / "runs" / "2026-week-03"
    run.mkdir(parents=True)
    pd.DataFrame([_game(3, 0.9, 7, 35, 10.0)]).assign(run_id="2026-week-03").to_csv(run / "validation.csv", index=False)
    import results

    frame, source, _ = results.load_results()
    assert source == results.SUNDAY
    assert frame.week_label.tolist() == ["Week 3"]
    assert frame.correct.tolist() == [False]


def test_results_page_shows_accuracy(artifacts):
    app = AppTest.from_string(
        "import results\nresults.results_page()", default_timeout=30
    ).run()
    assert not app.exception
    text = " ".join(block.proto.body for block in app.get("html"))
    assert "2026 results and accuracy" in text
    assert "75.0%" in text and "3 of 4 games" in text
    assert "4.5 pts" in text
    assert "Win-chance scores by week" in text
    bands = app.dataframe[1].value
    assert list(bands["Holds up"]) == ["Yes"] * 4
    assert bands["Games"].sum() == 4
