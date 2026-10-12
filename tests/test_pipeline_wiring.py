import json
import sys

import pandas as pd
import pytest

from cfb_forecast import paths
from cfb_forecast.data import load_data
from cfb_forecast.pipeline_status import PipelineJournal
from cfb_forecast.training_audit import training_audit


def archive(tmp_path):
    games = pd.DataFrame([dict(game_id=year, season=year, week=1, start_date=f"{year}-09-01T12:00Z", completed=True, neutral_site=False, home_id=1, away_id=2, home_team="A", away_team="B", home_points=28, away_points=21, fbs_game=True) for year in range(2021, 2026)])
    summaries = pd.DataFrame([dict(game_id=year, pos_team_id=team, yards_per_play=5.5, yards_per_pass=7., yards_per_rush=4., first_downs_created_rate=.3) for year in range(2021, 2026) for team in (1, 2)])
    games.to_csv(tmp_path / "schedule_history.csv", index=False)
    summaries.to_parquet(tmp_path / "adv_team_history.parquet", index=False)
    return load_data(tmp_path)


def test_paths_ignore_calling_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("CFB_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CFB_DATA_DIR", "runtime/data")
    monkeypatch.setenv("CFB_ARTIFACTS_DIR", "runtime/artifacts")
    monkeypatch.delenv("CFB_MODEL_DIR", raising=False)
    monkeypatch.chdir(tmp_path.parent)
    assert paths.data_root() == tmp_path / "runtime/data"
    assert paths.model_root() == tmp_path / "runtime/artifacts/training/model"
    assert paths.cli_path("data/market_history.csv") == tmp_path / "runtime/data/market_history.csv"


def test_scheduled_cli_calls_production_service(monkeypatch, tmp_path):
    from cfb_forecast.cli import main
    import cfb_forecast.sunday as service
    captured = []
    monkeypatch.setattr(service, "sunday_run", lambda *args: captured.append(args) or {"status": "ok"})
    monkeypatch.setenv("CFB_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["cfb-forecast", "sunday-run", "--data", "runtime/data", "--artifacts", "runtime/artifacts", "--season", "2026", "--scheduled", "--no-refresh"])
    main()
    args = captured[0]
    assert args[:3] == (str(tmp_path / "runtime/data"), str(tmp_path / "runtime/artifacts"), 2026)
    assert args[5:] == (True, False)


def test_parquet_observations_enter_training_audit(tmp_path):
    games, summaries, files = archive(tmp_path)
    report = training_audit(games, summaries, files, 2021, 2022, 2024, 2025)
    assert len(report["seasons"]) == 5
    assert all(row["complete_advanced_rows"] == 2 for row in report["seasons"])
    assert any(row["format"] == "parquet" for row in report["source_files"])
    assert report["seasons"][-1]["role"].startswith("held-out")
    with pytest.raises(ValueError, match="no matched advanced"):
        training_audit(games, summaries[summaries.game_id.ne(2023)], files, 2021, 2022, 2024, 2025)


def test_advanced_team_identity_guard(tmp_path):
    archive(tmp_path)
    path = tmp_path / "adv_team_history.parquet"
    rows = pd.read_parquet(path)
    rows.loc[0, "pos_team_id"] = 999
    rows.to_parquet(path)
    with pytest.raises(ValueError, match="team does not belong"):
        load_data(tmp_path)


def test_execution_journal_preserves_failure(tmp_path):
    path = tmp_path / "pipeline.json"
    journal = PipelineJournal(path)
    journal.stage("sources")
    journal.stage("context")
    journal.fail(ValueError("Source unavailable"))
    report = json.loads(path.read_text())
    assert report["status"] == "failed"
    assert report["stages"][0]["status"] == "completed"
    assert report["stages"][1]["status"] == "failed"
    assert report["stages"][2]["status"] == "pending"


def test_empty_slate_publication_replaces_latest(tmp_path):
    from cfb_forecast.sunday import publish_latest
    old = tmp_path / "latest"
    old.mkdir()
    (old / "forecasts.csv").write_text("old predictions")
    final = tmp_path / "runs" / "empty"
    final.mkdir(parents=True)
    (final / "run_manifest.json").write_text('{"games": 0}')
    publish_latest(tmp_path, final)
    assert not (old / "forecasts.csv").exists()
    assert json.loads((old / "run_manifest.json").read_text())["games"] == 0
