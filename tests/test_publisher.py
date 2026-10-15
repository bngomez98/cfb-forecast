import json
from pathlib import Path

import pandas as pd

from cfb_forecast.publisher import board_row, publish_forecast_update
from cfb_forecast.sources import source_registry


def test_board_row_names_the_winner_and_confidence():
    home = board_row({
        "game_id": 1, "season": 2026, "home_team": "Home", "away_team": "Away",
        "kickoff": "2026-10-17T19:00:00Z", "expected_home_score": 31.2,
        "expected_away_score": 24.4, "home_win_probability": 0.64,
        "projected_home_margin": 6.8,
    })
    assert home["winner"] == "Home"
    assert home["confidence"] == 0.64
    away = board_row({**{
        "game_id": 2, "season": 2026, "home_team": "Home", "away_team": "Away",
        "kickoff": "2026-10-17T19:00:00Z", "expected_home_score": 20,
        "expected_away_score": 27, "home_win_probability": 0.41,
    }})
    assert away["winner"] == "Away"
    assert away["confidence"] == 0.59


def test_source_catalog_includes_the_public_football_sites():
    catalog = source_registry(2026)
    for name in ("collegefootballdb", "cfbtracker", "cfbtrack", "cfbresults", "collegefootballdata", "football_db"):
        assert name in catalog
        assert catalog[name].startswith("https://")


def test_publish_writes_sqlite_react_board_and_email(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CFB_SMTP_HOST", raising=False)
    monkeypatch.delenv("DD_API_KEY", raising=False)
    run = tmp_path / "2026-week-7"
    run.mkdir()
    pd.DataFrame([{
        "game_id": 7, "season": 2026, "home_team": "Texas", "away_team": "Oklahoma",
        "kickoff": "2026-10-17T19:00:00Z", "expected_home_score": 31.8,
        "expected_away_score": 24.7, "projected_home_margin": 7.1,
        "home_win_probability": 0.64, "prediction_method": "score head",
    }]).to_csv(run / "forecasts.csv", index=False)
    web = tmp_path / "web" / "forecasts.json"
    report = publish_forecast_update(
        run,
        artifacts=tmp_path / "artifacts",
        database=tmp_path / "cfb.sqlite",
        web_json=web,
        run_id="2026-week-7",
    )
    assert report["status"] == "published"
    assert report["email"]["status"] == "saved"
    assert report["email"]["to"] == "bngomez@ieee.org"
    assert report["datadog"]["status"] == "local_only"
    board = json.loads(web.read_text())
    assert board["forecasts"][0]["winner"] == "Texas"
    assert board["forecasts"][0]["confidence"] == 0.64
    assert any(source["name"] == "cfbresults" for source in board["sources"])
    assert (tmp_path / "artifacts" / "latest" / "forecasts.csv").exists()
    assert (tmp_path / "artifacts" / "logs" / "datadog.jsonl").exists()
    assert list((tmp_path / "artifacts" / "logs" / "email_outbox").glob("*.txt"))
