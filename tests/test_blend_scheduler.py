"""Final-forecast blend, results overlay and scheduler timing."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from cfb_forecast import blend
from cfb_forecast.data import apply_results_overlay, normalize_schedule
from cfb_forecast.scheduler import Calendar, Scheduler, build_jobs

SPEC = {"weights": {"model": 0.3, "market": 0.3, "external": 0.4}, "sigma": 13.6, "market_max_weight": 0.35,
        "cross_validated": {"blend": {"margin_mae": 11.9, "log_loss": 0.52}, "model_only": {"margin_mae": 12.4, "log_loss": 0.54}}}


def write_forecast(folder, as_of="2026-10-10T12:00:00Z"):
    folder.mkdir(parents=True)
    pd.DataFrame({
        "game_id": [1, 2, 3], "kickoff": ["2026-10-11T19:00:00Z"] * 3, "as_of": [as_of] * 3,
        "away_team": ["A", "C", "E"], "home_team": ["B", "D", "F"],
        "expected_home_score": [30.0, 24.0, 20.0], "expected_away_score": [20.0, 24.0, 27.0],
        "projected_home_margin": [10.0, 0.0, -7.0], "projected_total": [50.0, 48.0, 47.0],
        "home_win_probability": [0.76, 0.5, 0.3],
    }).to_csv(folder / "forecasts.csv", index=False)


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / "data"
    (data / "sources").mkdir(parents=True)
    forecast = tmp_path / "fc"
    write_forecast(forecast)
    spec_path = tmp_path / "blend.json"
    spec_path.write_text(json.dumps(SPEC))
    pd.DataFrame({"game_id": [1, 2, 3], "provider": ["book"] * 3,
                  "observed_at": ["2026-10-10T10:00:00Z", "2026-10-10T10:00:00Z", "2026-10-11T20:00:00Z"],  # game 3: after kickoff
                  "home_spread": [-4.0, -3.0, 7.0]}).to_csv(data / "market_history.csv", index=False)
    return data, forecast, spec_path


def test_market_informs_but_never_decides(setup):
    data, forecast, spec_path = setup
    out = blend.apply_blend(forecast, data, spec_path, data / "market_history.csv", as_of="2026-10-10T12:00:00Z").set_index("game_id")
    # Game 1: model +10, market +4, no external -> weights renormalize to 0.5/0.5, then the cap pulls market to 0.35.
    assert out.loc[1, "weight_market"] == pytest.approx(0.35)
    assert out.loc[1, "weight_model"] == pytest.approx(0.65)
    assert out.loc[1, "projected_home_margin"] == pytest.approx(0.65 * 10 + 0.35 * 4)
    assert out.loc[1, "market_shift_points"] == pytest.approx(0.35 * (4 - 10))
    assert out.loc[1, "expected_home_score"] - out.loc[1, "expected_away_score"] == pytest.approx(out.loc[1, "projected_home_margin"])
    # A line observed after kickoff is never used: game 3 stays the model's own forecast.
    assert out.loc[3, "blend_inputs"] == "model"
    assert out.loc[3, "home_win_probability"] == pytest.approx(0.3)
    assert (out.weight_market <= np.minimum(0.35, out.weight_model) + 1e-12).all()


def test_external_predictions_enter_with_coverage_shrink(setup, monkeypatch):
    data, forecast, spec_path = setup
    monkeypatch.setenv("CFB_EXTERNAL_FULL_SOURCES", "2")
    pd.DataFrame({
        "source": ["s1", "s2", "s1"], "game_id": [2, 2, 1], "kickoff": ["2026-10-11T19:00:00Z"] * 3,
        "retrieved_at": ["2026-10-09T12:00:00Z", "2026-10-10T11:00:00Z", "2026-10-10T13:00:00Z"],  # last one after cutoff
        "pick_home_score": [31, None, 40], "pick_away_score": [24, None, 10], "pick_home_spread": [None, -5.0, None],
        "pick_home_win_prob": [None, None, None], "home_team": ["D", "D", "B"], "away_team": ["C", "C", "A"],
    }).to_csv(data / "sources" / "picks.csv", index=False)
    out = blend.apply_blend(forecast, data, spec_path, data / "market_history.csv", as_of="2026-10-10T12:00:00Z").set_index("game_id")
    assert out.loc[2, "external_home_margin"] == pytest.approx((7 + 5) / 2)
    assert out.loc[2, "blend_inputs"] == "model+market+external"
    assert out.loc[2, ["weight_model", "weight_market", "weight_external"]].sum() == pytest.approx(1)
    assert np.isnan(out.loc[1, "external_home_margin"])  # published after the cutoff
    assert (out.weight_market <= out.weight_model + 1e-12).all()


def test_fit_weights_respects_market_cap():
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 15, 2000)
    X = np.column_stack([truth + rng.normal(0, 14, 2000), truth + rng.normal(0, 6, 2000), truth + rng.normal(0, 12, 2000)])
    w = blend.fit_weights(X, truth + rng.normal(0, 3, 2000), cap=0.35)
    assert w.sum() == pytest.approx(1) and w[1] <= 0.35 + 1e-6 and w[1] <= w[0] + 1e-6


def test_committed_blend_spec_is_valid():
    spec = blend.load_blend()
    assert spec is not None, "artifacts/backtest/blend/blend.json must be committed"
    w = spec["weights"]
    assert sum(w.values()) == pytest.approx(1)
    assert w["market"] <= min(spec["market_max_weight"], w["model"]) + 1e-6


def test_results_overlay_fills_only_unfinished_games():
    raw = pd.DataFrame({
        "game_id": [1, 2], "season": 2026, "week": 6, "start_date": ["2026-10-03T19:00:00Z", "2026-10-03T19:00:00Z"],
        "completed": [False, True], "neutral_site": False, "home_id": [1, 3], "away_id": [2, 4],
        "home_team": ["B", "D"], "away_team": ["A", "C"], "home_points": [None, 21.0], "away_points": [None, 14.0],
        "home_division": "fbs", "away_division": "fbs"})
    overlay = pd.DataFrame({"game_id": [1, 2], "home_points": [28, 99], "away_points": [17, 0],
                            "completed_at": ["2026-10-03T23:10:00Z", "2026-10-03T23:00:00Z"]})
    s = normalize_schedule(apply_results_overlay(raw, overlay)).set_index("game_id")
    assert (s.loc[1, "home_points"], s.loc[1, "away_points"], bool(s.loc[1, "result_usable"])) == (28, 17, True)
    assert s.loc[1, "result_available_at"] == pd.Timestamp("2026-10-03T23:10:00Z")  # when sources agreed, not the 8h proxy
    assert (s.loc[2, "home_points"], s.loc[2, "away_points"]) == (21, 14)              # schedule scores are never overwritten


def _calendar(tmp_path, kickoffs):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    pd.DataFrame({"start_date": kickoffs, "home_division": "fbs", "away_division": "fbs"}).to_csv(data / "schedule_2026.csv", index=False)
    cal = Calendar(data, 2026)
    cal.refresh()
    return cal


def test_scheduler_job_timing(tmp_path, monkeypatch):
    monkeypatch.setenv("CFB_WEEKLY_DAY", "mon")
    monkeypatch.setenv("CFB_WEEKLY_HOUR", "13")
    now = datetime(2026, 10, 10, 20, 0, tzinfo=timezone.utc)  # a Saturday
    cal = _calendar(tmp_path, [(now - timedelta(hours=1)).isoformat(), (now + timedelta(days=2)).isoformat()])
    jobs = {j.name: j for j in build_jobs()}
    assert jobs["live-results"].due(now, {}, cal)
    assert not jobs["live-results"].due(now, {"live-results": {"last_started": (now - timedelta(minutes=5)).isoformat()}}, cal)
    assert jobs["pregame"].due(now, {}, cal)
    monday = datetime(2026, 10, 5, 13, 0, tzinfo=timezone.utc)
    assert not jobs["weekly"].due(now, {"weekly": {"last_started": (monday + timedelta(minutes=1)).isoformat()}}, cal)
    assert jobs["weekly"].due(now, {"weekly": {"last_started": (monday - timedelta(days=1)).isoformat()}}, cal)
    quiet = now + timedelta(hours=8)
    assert not jobs["live-results"].due(quiet, {}, cal)


def test_scheduler_once_records_state(tmp_path, monkeypatch):
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    _calendar(tmp_path, ["2026-03-01T00:00:00Z"])  # off-season calendar: only weekly/data jobs eligible
    calls = []
    monkeypatch.setattr("cfb_forecast.pipeline.run", lambda action, ctx: calls.append(action) or {"status": "success"})
    sched = Scheduler(data, artifacts, 2026, jobs=["data", "live-results"])
    assert sched.tick() == ["data"]
    state = json.loads((artifacts / "scheduler" / "state.json").read_text())
    assert state["data"]["last_status"] == "success" and calls == ["data"]
    assert (artifacts / "scheduler" / "heartbeat").exists()
    assert sched.tick() == []  # already ran in this daily slot
