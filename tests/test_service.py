import json

import pandas as pd
from fastapi.testclient import TestClient

from cfb_forecast.service import app


def test_service_exposes_saved_python_forecast_only(tmp_path, monkeypatch):
    data = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    latest = artifacts / "latest"
    inputs = latest / "inputs"
    data.mkdir()
    inputs.mkdir(parents=True)

    pd.DataFrame([
        {
            "game_id": 1,
            "season": 2026,
            "week": 7,
            "start_date": "2026-10-17T17:00:00Z",
            "completed": False,
            "neutral_site": False,
            "home_id": 10,
            "away_id": 20,
            "home_team": "Home U",
            "away_team": "Away U",
            "home_points": None,
            "away_points": None,
            "fbs_game": True,
        }
    ]).to_csv(data / "schedule_2026.csv", index=False)

    forecast = {
        "game_id": 1,
        "kickoff": "2026-10-17T17:00:00+00:00",
        "as_of": "2026-10-12T15:00:00+00:00",
        "home_team": "Home U",
        "away_team": "Away U",
        "home_win_probability": 0.63,
        "expected_home_score": 28.1,
        "expected_away_score": 22.4,
        "projected_home_margin": 5.7,
        "projected_total": 50.5,
        "prior_fbs_games": 350,
        "prediction_method": "python-test-model",
        "inputs_complete": True,
    }
    (latest / "forecasts.json").write_text(json.dumps([forecast]))
    (latest / "run_manifest.json").write_text(json.dumps({"as_of": forecast["as_of"]}))
    (inputs / "game_inputs.json").write_text(json.dumps([
        {"game_id": 1, "venue": {"fullName": "Test Stadium"}}
    ]))

    monkeypatch.setenv("CFB_DATA_DIR", str(data))
    monkeypatch.setenv("CFB_ARTIFACTS_DIR", str(artifacts))

    client = TestClient(app)
    response = client.get("/v1/forecast?season=2026&week=7")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "authoritative saved Python forecast run"
    assert body["fittedGames"] == 350
    assert len(body["forecasts"]) == 1
    game = body["forecasts"][0]
    assert game["homeWin"] == 0.63
    assert game["margin"] == 5.7
    assert game["homeExpectedPoints"] == 28.1
    assert game["venue"] == "Test Stadium"
    assert game["model"] == "python-test-model"
