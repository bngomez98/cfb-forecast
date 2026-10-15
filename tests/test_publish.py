import json
import sqlite3

import pandas as pd

from cfb_forecast.publish import board_rows, publish_forecast_update


def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    pd.DataFrame([
        dict(game_id=2, season=2026, kickoff="2026-10-17T19:00Z", home_team="B", away_team="A",
             expected_home_score=20.0, expected_away_score=27.0, home_win_probability=0.3),
        dict(game_id=1, season=2026, kickoff="2026-10-17T16:00Z", home_team="C", away_team="D",
             expected_home_score=31.0, expected_away_score=17.0, home_win_probability=0.8),
    ]).to_csv(d / "forecasts.csv", index=False)
    return d


def test_board_picks_winner_and_confidence(tmp_path):
    rows = board_rows(pd.read_csv(run_dir(tmp_path) / "forecasts.csv").to_dict("records"))
    assert [r["game_id"] for r in rows] == [1, 2]
    assert rows[0]["predicted_winner"] == "C" and abs(rows[0]["confidence"] - 0.8) < 1e-9
    assert rows[1]["predicted_winner"] == "A" and abs(rows[1]["confidence"] - 0.7) < 1e-9


def test_publish_writes_every_consumer_and_is_idempotent(tmp_path, monkeypatch):
    for key in ("DD_API_KEY", "CFB_SMTP_HOST", "CFB_SMTP_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    art, web, d = tmp_path / "artifacts", tmp_path / "web", run_dir(tmp_path)
    for _ in range(2):
        receipt = publish_forecast_update(d, "2026-week-7", art, web_public=web)
    assert receipt["datadog"] == "log_only" and receipt["email"] == "outbox_only"
    assert json.loads((web / "forecasts.json").read_text())["games"][0]["game_id"] == 1
    assert (art / "latest" / "forecast_board.json").exists()
    assert (art / "outbox" / "2026-week-7.eml").exists()
    assert len((art / "logs" / "datadog.jsonl").read_text().splitlines()) == 2
    with sqlite3.connect(art / "cfb_forecast.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM forecast_log").fetchone()[0] == 2
