import json
from pathlib import Path

import pandas as pd
import pytest

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
    assert home["game_date"] == "2026-10-17"
    away = board_row({**{
        "game_id": 2, "season": 2026, "home_team": "Home", "away_team": "Away",
        "kickoff": "2026-10-17T19:00:00Z", "expected_home_score": 20,
        "expected_away_score": 27, "home_win_probability": 0.41,
    }})
    assert away["winner"] == "Away"
    assert away["confidence"] == 0.59


def test_board_row_uses_the_chicago_calendar_date():
    late = board_row({
        "game_id": 3, "season": 2026, "home_team": "Home", "away_team": "Away",
        "kickoff": "2026-10-18T03:30:00Z", "expected_home_score": 24,
        "expected_away_score": 21, "home_win_probability": 0.55,
    })
    assert late["game_date"] == "2026-10-17"
    assert late["kickoff"] == "2026-10-18T03:30:00Z"


def test_board_row_rejects_a_forecast_without_a_date():
    with pytest.raises(ValueError, match="game date"):
        board_row({
            "game_id": 4, "season": 2026, "home_team": "Home", "away_team": "Away",
            "expected_home_score": 24, "expected_away_score": 21,
            "home_win_probability": 0.55,
        })


def test_source_catalog_includes_the_public_football_sites():
    catalog = source_registry(2026)
    for name in ("collegefootballdb", "cfbtracker", "cfbtrack", "cfbresults", "collegefootballdata", "football_db"):
        assert name in catalog
        assert catalog[name].startswith("https://")


def test_publish_writes_sqlite_react_board_and_email(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CFB_SMTP_HOST", raising=False)
    monkeypatch.delenv("DD_API_KEY", raising=False)
    monkeypatch.setenv("CFB_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
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
    assert board["forecasts"][0]["game_date"] == "2026-10-17"
    assert board["forecasts"][0]["confidence"] == 0.64
    assert any(source["name"] == "cfbresults" for source in board["sources"])
    assert (tmp_path / "artifacts" / "latest" / "forecasts.csv").exists()
    assert (tmp_path / "artifacts" / "logs" / "datadog.jsonl").exists()
    email = next((tmp_path / "artifacts" / "logs" / "email_outbox").glob("*.txt"))
    assert "2026-10-17" in email.read_text(encoding="utf-8")


def test_publish_reaches_streamlit_and_both_react_copies(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CFB_SMTP_HOST", raising=False)
    monkeypatch.delenv("DD_API_KEY", raising=False)
    project = tmp_path / "project"
    monkeypatch.setenv("CFB_PROJECT_ROOT", str(project))
    monkeypatch.setenv("CFB_ARTIFACTS_DIR", str(project / "artifacts"))
    run_artifacts = project / "runtime" / "artifacts"
    run = run_artifacts / "runs" / "2026-week-8"
    run.mkdir(parents=True)
    pd.DataFrame([{
        "game_id": 8, "season": 2026, "home_team": "Oregon", "away_team": "Nebraska",
        "kickoff": "2026-10-18T02:30:00Z", "expected_home_score": 27.0,
        "expected_away_score": 20.0, "projected_home_margin": 7.0,
        "home_win_probability": 0.66, "prediction_method": "score head",
    }]).to_csv(run / "forecasts.csv", index=False)
    (run / "run_manifest.json").write_text(json.dumps({"as_of": "2026-10-14T00:00:00Z", "run_id": "2026-week-8"}), encoding="utf-8")
    report = publish_forecast_update(
        run,
        artifacts=run_artifacts,
        database=tmp_path / "cfb.sqlite",
        run_id="2026-week-8",
    )
    assert report["status"] == "published"
    assert report["apps"]["streamlit"]
    for path in (
        project / "web" / "public" / "forecasts.json",
        project / "web" / "dist" / "forecasts.json",
        project / "artifacts" / "latest" / "forecast_board.json",
        run_artifacts / "latest" / "forecast_board.json",
    ):
        board = json.loads(path.read_text(encoding="utf-8"))
        assert board["run_id"] == "2026-week-8"
        assert board["games"] == 1
        assert board["forecasts"][0]["game_date"] == "2026-10-17"
    repeated = publish_forecast_update(
        run,
        artifacts=run_artifacts,
        database=tmp_path / "cfb.sqlite",
        run_id="2026-week-8",
        notify=False,
    )
    assert repeated["email"]["status"] == "skipped"
    assert json.loads((project / "web" / "public" / "forecasts.json").read_text(encoding="utf-8"))["games"] == 1
