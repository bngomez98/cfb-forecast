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


def test_database_loop_ingests_validates_and_cross_checks(tmp_path, monkeypatch):
    import sqlite3
    from contextlib import closing

    from cfb_forecast import dbflow
    from cfb_forecast.database import connect

    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({
        "game_id": [1, 2, 3], "season": [2026] * 3, "week": [1, 1, 2], "start_date": ["2026-09-05T18:00:00Z", "2026-09-05T22:00:00Z", "2026-09-12T18:00:00Z"],
        "completed": [True, True, False], "neutral_site": [False] * 3, "home_id": [10, 12, 14], "away_id": [11, 13, 15],
        "home_team": ["A", "C", "E"], "away_team": ["B", "D", "F"], "home_points": [30, 10, None], "away_points": [20, 24, None],
        "home_division": ["fbs"] * 3, "away_division": ["fbs"] * 3,
    }).to_csv(data / "schedule_2026.csv", index=False)
    (data / "sources").mkdir()
    pd.DataFrame({"game_id": [1], "sources_agree": [True], "schedule_agrees": [True], "source_scores": ["cfbtrack 20-30"], "schedule_score": ["20-30"]}).to_csv(
        data / "sources" / "results_crosscheck.csv", index=False)
    db = tmp_path / "t.sqlite"
    with closing(connect(db)) as conn:
        for run, p1, p2 in (("daily-1", 0.7, 0.3), ("daily-2", 0.6, 0.2)):
            for gid, p, home in ((1, p1, "A"), (2, p2, "C"), (3, 0.5, "E")):
                conn.execute("""INSERT INTO forecast_log(run_id, season, game_id, kickoff, home_team, away_team, expected_home_score, expected_away_score,
                    winner, confidence, home_win_probability, logged_at, projected_home_margin) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (run, 2026, gid, {1: "2026-09-05T18:00:00Z", 2: "2026-09-05T22:00:00Z", 3: "2026-09-12T18:00:00Z"}[gid], home, "X", 28, 21, home, max(p, 1 - p), p,
                     "2026-09-01T00:00:00Z", 7.0))
        # a forecast logged after kickoff must never count
        conn.execute("""INSERT INTO forecast_log(run_id, season, game_id, kickoff, home_team, away_team, expected_home_score, expected_away_score,
            winner, confidence, home_win_probability, logged_at) VALUES('late', 2026, 1, '2026-09-05T18:00:00Z', 'A', 'B', 1, 1, 'A', .9, .9, '2026-09-06T00:00:00Z')""")
        conn.commit()
    report = dbflow.run(db, data, 2026)
    assert report["ingest"]["results"] == 2 and report["ingest"]["external_checks"] == 1
    assert report["validation"] == {"runs": 2, "games": 4}
    assert report["recomputed"] == {"status": "verified", "runs": 2}
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM forecast_validation WHERE run_id='late'").fetchone()[0] == 0
        wins = dict(conn.execute("SELECT game_id, home_win FROM forecast_validation WHERE run_id='daily-1'").fetchall())
        assert wins == {1: 1, 2: 0}
        assert conn.execute("SELECT COUNT(*) FROM weekly_validation").fetchone()[0] == 2
    dbflow.run(db, data, 2026)  # idempotent
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM weekly_validation").fetchone()[0] == 2
