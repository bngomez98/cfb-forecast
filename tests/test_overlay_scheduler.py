"""Results overlay and scheduler timing."""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from cfb_forecast.data import apply_results_overlay, normalize_schedule
from cfb_forecast.scheduler import Calendar, Scheduler, build_jobs

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
    monkeypatch.setenv("CFB_FORECAST_HOURS", "24")
    monkeypatch.setenv("CFB_FORECAST_HOUR", "9")
    monkeypatch.setenv("CFB_BACKTEST_DAY", "tue")
    monkeypatch.setenv("CFB_BACKTEST_HOUR", "11")
    now = datetime(2026, 10, 10, 20, 0, tzinfo=timezone.utc)  # a Saturday, in season
    cal = _calendar(tmp_path, [(now - timedelta(hours=1)).isoformat(), (now + timedelta(days=2)).isoformat()])
    jobs = {j.name: j for j in build_jobs()}
    assert set(jobs) == {"sunday", "forecast", "backtest"}  # Sunday slate + the ongoing daily pipeline + a weekly backtest
    # forecast is due once every 24h; a run 25h ago is due, 1h ago is not
    assert jobs["forecast"].due(now, {"forecast": {"last_started": (now - timedelta(hours=25)).isoformat()}}, cal)
    assert not jobs["forecast"].due(now, {"forecast": {"last_started": (now - timedelta(hours=1)).isoformat()}}, cal)
    # backtest only after its weekly Tuesday slot
    last_tue = datetime(2026, 10, 6, 11, 0, tzinfo=timezone.utc)
    assert not jobs["backtest"].due(now, {"backtest": {"last_started": (last_tue + timedelta(minutes=1)).isoformat()}}, cal)
    assert jobs["backtest"].due(now, {"backtest": {"last_started": (last_tue - timedelta(days=1)).isoformat()}}, cal)
    # the Sunday service: due from 10:00 Chicago time on Sunday, once per Sunday
    monkeypatch.setenv("CFB_SUNDAY_HOUR", "10")
    sunday_jobs = {j.name: j for j in build_jobs()}
    sun_cal = _calendar(tmp_path, [(datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc) + timedelta(days=5)).isoformat()])
    before = datetime(2026, 10, 11, 14, 59, tzinfo=timezone.utc)   # 09:59 CDT
    after = datetime(2026, 10, 11, 15, 1, tzinfo=timezone.utc)     # 10:01 CDT
    assert not sunday_jobs["sunday"].due(before, {}, sun_cal)
    assert sunday_jobs["sunday"].due(after, {}, sun_cal)
    assert not sunday_jobs["sunday"].due(after, {"sunday": {"last_started": (after - timedelta(seconds=30)).isoformat()}}, sun_cal)
    assert not sunday_jobs["sunday"].due(now, {}, sun_cal)  # a Saturday
    # off-season: nothing is due
    off = _calendar(tmp_path, ["2026-03-01T00:00:00Z"])
    assert not jobs["forecast"].due(now, {}, off)
    assert not jobs["backtest"].due(now, {}, off)


def test_scheduler_once_records_state(tmp_path, monkeypatch):
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    now = datetime.now(timezone.utc)
    _calendar(tmp_path, [(now + timedelta(days=3)).isoformat()])  # in-season calendar
    calls = []
    monkeypatch.setattr("cfb_forecast.pipeline.run", lambda action, ctx: calls.append(action) or {"status": "success"})
    sched = Scheduler(data, artifacts, 2026, jobs=["forecast"])
    assert sched.tick() == ["forecast"]
    state = json.loads((artifacts / "scheduler" / "state.json").read_text())
    assert state["forecast"]["last_status"] == "success" and calls == ["forecast"]
    assert (artifacts / "scheduler" / "heartbeat").exists()
    assert sched.tick() == []  # already ran this cycle
