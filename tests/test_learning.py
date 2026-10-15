import json
import os
from contextlib import closing

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cfb_forecast.config import StateConfig
from cfb_forecast.storage.database import connect
from cfb_forecast.ingest.features import feature_columns
from cfb_forecast.modeling.learning import learn, validate_in_r
from cfb_forecast.modeling.model import write_model


@pytest.fixture
def training_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    schedules, summaries = [], []
    for year in (2021, 2022, 2023):
        for number in range(80):
            game = year * 1000 + number
            schedules.append(dict(game_id=game, season=year, week=number // 5 + 1, start_date=(pd.Timestamp(f"{year}-08-01", tz="UTC") + pd.Timedelta(days=number)).isoformat(), completed=True, neutral_site=False, home_id=1, away_id=2, home_team="A", away_team="B", home_points=35 + number % 7, away_points=14 + number % 5, fbs_game=True))
            for team in (1, 2):
                summaries.append(dict(game_id=game, pos_team_id=team, yards_per_play=5 + team, yards_per_pass=7 + team, yards_per_rush=4 + team, first_downs_created_rate=.2 + team / 20))
    pd.DataFrame(schedules).to_csv(data / "schedule_test.csv", index=False)
    pd.DataFrame(summaries).to_parquet(data / "adv_team_test.parquet", index=False)
    (data / "source_manifest.json").write_text("{}")
    cols = feature_columns("score")
    model = dict(schema_version=1, variant="score", warmup_season=2021, train_start_season=2022, training_through_season=2022, center=[24, 6, 8, 5, .3], scale=[10, 1, 1, 1, .1], state_config=StateConfig().to_dict(), score_covariance=[[144, 0], [0, 144]], head=dict(variant="score", alpha=10, cols=cols, mean=[0] * len(cols), scale=[1] * len(cols), coef=[0] * len(cols), intercept=24))
    base = tmp_path / "base"
    write_model(base, model)
    return data, base, tmp_path / "learning", tmp_path / "model.sqlite3"


def test_sql_training_promotes_and_preserves_temporal_evidence(training_files):
    report = learn(*training_files, "2023-12-01T00:00Z")
    assert report["status"] == "promoted"
    assert report["candidate"]["log_loss"] < report["incumbent"]["log_loss"]
    with closing(connect(training_files[-1])) as conn:
        rows = conn.execute("SELECT * FROM training_examples").fetchall()
    assert len(rows) == 160
    assert all(pd.Timestamp(row["latest_feature_result"]) < pd.Timestamp(row["forecast_cutoff"]) for row in rows)
    assert all(pd.Timestamp(row["result_available_at"]) < pd.Timestamp(report["as_of"]) for row in rows)
    repeated = learn(*training_files, "2023-12-02T00:00Z")
    assert repeated["model_path"] == report["model_path"]
    with closing(connect(training_files[-1])) as conn:
        assert conn.execute("SELECT count(*) FROM learning_versions").fetchone()[0] == 1


def test_insufficient_results_do_not_create_candidate(training_files):
    report = learn(*training_files, "2023-08-10T00:00Z")
    assert report["status"] == "insufficient_new_results"
    assert report["model_path"] == str(training_files[1].resolve())


def test_missing_parquet_blocks_training(training_files):
    (training_files[0] / "adv_team_test.parquet").unlink()
    with pytest.raises(ValueError, match="watchdog"):
        learn(*training_files, "2023-12-01T00:00Z")


def test_future_outcomes_do_not_enter_sql(training_files):
    report = learn(*training_files, "2023-10-01T00:00Z")
    with closing(connect(training_files[-1])) as conn:
        dates = conn.execute("SELECT result_available_at FROM training_examples").fetchall()
    assert dates
    assert all(pd.Timestamp(row[0]) < pd.Timestamp("2023-10-01T00:00Z") for row in dates)


def test_r_python_metrics_agree(tmp_path):
    import shutil
    if not shutil.which("Rscript"):
        if os.environ.get("CFB_REQUIRE_R") == "1":
            pytest.fail("CFB_REQUIRE_R=1 but Rscript is not installed")
        pytest.skip("R is installed in the production image and scheduled runner")
    frame = pd.DataFrame({"game_id": range(40), "home_win": [0, 1] * 20, "candidate_p": [.2, .7] * 20})
    assert validate_in_r(frame, tmp_path)["status"] == "verified"


def test_adaptive_model_cannot_predict_its_past(training_files, tmp_path):
    from cfb_forecast.modeling.forecast import forecast
    report = learn(*training_files, "2023-12-01T00:00Z")
    with pytest.raises(ValueError, match="trained after"):
        forecast(training_files[0], report["model_path"], tmp_path / "predictions", "2023-11-01T00:00Z", "2023-11-02T00:00Z", 2023)


def test_learning_api_and_path_guards(training_files, monkeypatch):
    import cfb_forecast.api as api
    learn(*training_files, "2023-12-01T00:00Z")
    monkeypatch.setattr(api, "DATABASE_PATH", training_files[-1])
    client = TestClient(api.app)
    assert client.get("/api/learning").json()["versions"][0]["status"] == "promoted"
    assert client.get("/api/watchdog").json()["files_checked"] == 3
    response = client.post("/api/forecasts", json={"season": 2023, "as_of": "2023-12-01T00:00Z", "through": "2023-12-02T00:00Z", "model_path": "/etc"})
    assert response.status_code == 400
