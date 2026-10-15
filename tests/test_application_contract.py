"""Integration contracts across publication, APIs, and the weekly ledger."""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cfb_forecast import api
from cfb_forecast.pipeline.publisher import publish_forecast_update, board_frame
from cfb_forecast.ingest.data import sha256, normalize_schedule
from cfb_forecast.storage.database import connect


@pytest.fixture
def publication(tmp_path, monkeypatch):
    monkeypatch.delenv("CFB_SMTP_HOST", raising=False)
    monkeypatch.delenv("DD_API_KEY", raising=False)
    monkeypatch.delenv("CFB_DATABASE", raising=False)
    run = tmp_path / "runs" / "2026-week-7"
    run.mkdir(parents=True)
    frame = pd.DataFrame([dict(game_id=7, season=2026, home_team="Home", away_team="Away",
        kickoff="2026-10-17T19:00:00Z", as_of="2026-10-11T15:00:00Z",
        expected_home_score=28., expected_away_score=24., home_win_probability=.6,
        home_ml_decimal=1.8, away_ml_decimal=2.2, market_provider="recorded book",
        market_observed_at="2026-10-11T14:00:00Z")])
    frame.to_csv(run / "forecasts.csv", index=False)
    manifest = dict(run_id=run.name, season=2026, as_of=frame.as_of.iloc[0],
        prediction_sha256=sha256(run / "forecasts.csv"), source_hashes={"source": "original"},
        validation={"settled_games": 4, "metrics": {"brier": .2, "margin_mae": 5.}})
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    kwargs = dict(artifacts=tmp_path / "artifacts", web_json=tmp_path / "public" / "forecasts.json")
    return run, frame, manifest, kwargs


def test_publication_keeps_cutoff_audit_metrics_and_distinct_book_prices(publication):
    run, _, manifest, kwargs = publication
    publish_forecast_update(run, **kwargs)
    board = json.loads(kwargs["web_json"].read_text())
    assert board["as_of"] == manifest["as_of"]
    assert board["validation"] == manifest["validation"]
    row = board["forecasts"][0]
    assert row["home_moneyline"] == -150
    assert row["home_book_moneyline"] == -125
    assert row["away_book_moneyline"] == 120
    latest = json.loads((kwargs["artifacts"] / "latest" / "run_manifest.json").read_text())
    assert latest["prediction_sha256"] == manifest["prediction_sha256"]
    assert latest["source_hashes"] == manifest["source_hashes"]
    assert (kwargs["artifacts"] / "cfb_forecast.sqlite").exists()


def test_republication_is_idempotent_and_cannot_change_predictions(publication):
    run, frame, manifest, kwargs = publication
    publish_forecast_update(run, **kwargs)
    original = kwargs["web_json"].read_bytes()
    publish_forecast_update(run, **kwargs)
    frame.loc[0, "home_win_probability"] = .8
    frame.to_csv(run / "forecasts.csv", index=False)
    # Even changing the local receipt cannot overwrite a previously published row.
    manifest["prediction_sha256"] = sha256(run / "forecasts.csv")
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="immutable"):
        publish_forecast_update(run, **kwargs)
    assert json.loads(kwargs["web_json"].read_text())["forecasts"] == json.loads(original)["forecasts"]
    conn = connect(kwargs["artifacts"] / "cfb_forecast.sqlite")
    assert conn.execute("SELECT COUNT(*) FROM forecast_log").fetchone()[0] == 1
    assert conn.execute("SELECT home_win_probability FROM forecast_log").fetchone()[0] == .6
    conn.close()


def test_tampered_forecast_is_rejected_before_publication(publication):
    run, frame, _, kwargs = publication
    frame.loc[0, "expected_home_score"] = 50.
    frame.to_csv(run / "forecasts.csv", index=False)
    with pytest.raises(ValueError, match="changed"):
        publish_forecast_update(run, **kwargs)
    assert not kwargs["web_json"].exists()


def test_empty_slate_clears_board_without_falling_back_to_old_games(publication):
    run, frame, _, kwargs = publication
    publish_forecast_update(run, **kwargs)
    empty = run.parent / "empty"
    empty.mkdir()
    frame.iloc[:0].to_csv(empty / "forecasts.csv", index=False)
    report = publish_forecast_update(empty, season=2026, **kwargs)
    assert report["status"] == "published"
    assert json.loads(kwargs["web_json"].read_text())["forecasts"] == []


@pytest.mark.parametrize("column,value", [("home_win_probability", 1.2), ("expected_home_score", float('inf')), ("expected_away_score", -1), ("as_of", "2026-10-18T00:00:00Z")])
def test_invalid_board_rows_are_rejected(publication, column, value):
    _, frame, _, _ = publication
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        board_frame(frame)


def test_python_api_serves_identical_published_snapshot(publication, monkeypatch):
    run, _, _, kwargs = publication
    publish_forecast_update(run, **kwargs)
    monkeypatch.setattr(api, "ARTIFACT_ROOT", kwargs["artifacts"])
    client = TestClient(api.app)
    response = client.get("/api/season?season=2026")
    assert response.status_code == 200
    assert response.json() == json.loads(kwargs["web_json"].read_text())
    assert client.get("/api/season?season=2025").status_code == 404
    assert client.get("/api/live-results?start=2026-10-01T00:00:00Z&end=2026-10-20T00:00:00Z").status_code == 400


def test_live_api_normalizes_status_without_updating_saved_forecast(monkeypatch):
    import cfb_forecast.ingest.results as results
    monkeypatch.setattr(results, "fetch_scoreboard", lambda *a: pd.DataFrame([
        dict(game_id=7, status="STATUS_IN_PROGRESS", is_final=False, home_points=7., away_points=0.),
        dict(game_id=8, status="STATUS_FINAL", is_final=True, home_points=21., away_points=14.),
    ]))
    response = TestClient(api.app).get("/api/live-results")
    assert response.status_code == 200
    assert [row["status"] for row in response.json()["results"]] == ["live", "final"]


@pytest.mark.parametrize("score", [2.5, float('inf'), -1])
def test_invalid_final_scores_are_quarantined(score):
    frame = pd.DataFrame([dict(game_id=1, season=2026, week=1, start_date="2026-09-01T12:00Z",
        completed=True, neutral_site=False, home_id=1, away_id=2, home_team="A", away_team="B",
        home_points=score, away_points=7, fbs_game=True)])
    result = normalize_schedule(frame)
    assert result.invalid_completed_outcome.iloc[0]
    assert not result.result_usable.iloc[0]


def test_weekly_retry_cannot_overwrite_existing_run(tmp_path, monkeypatch):
    import cfb_forecast.pipeline.weekly as weekly
    row = dict(game_id=1, season=2026, week=7, start_date="2026-10-17T19:00Z", completed=False,
               neutral_site=False, home_id=1, away_id=2, home_team="Home", away_team="Away",
               home_points=None, away_points=None, fbs_game=True)
    games = normalize_schedule(pd.DataFrame([row]))
    monkeypatch.setattr(weekly, "load_data", lambda _: (games, None, None))
    destination = tmp_path / "weekly" / "2026-week-7"
    destination.mkdir(parents=True)
    saved = destination / "forecasts.csv"
    saved.write_text("immutable original")
    with pytest.raises(FileExistsError, match="immutable"):
        weekly.run_weekly_forecast(tmp_path, tmp_path, tmp_path, 2026, week=7, as_of="2026-10-11T15:00Z")
    assert saved.read_text() == "immutable original"


def test_cfbd_prices_require_matching_teams_kickoff_and_pregame_receipt(tmp_path, monkeypatch):
    import cfb_forecast.ingest.cfbd_market as cfbd
    monkeypatch.setenv("CFBD_API_KEY", "test-only")
    games = pd.DataFrame([dict(game_id=1, week=7, home_team="Home", away_team="Away", start_date=pd.Timestamp("2026-10-17T19:00Z"))])
    event = dict(homeTeam="Home", awayTeam="Away", startDate="2026-10-17T19:00Z", lines=[dict(provider="test", spread=-3, overUnder=51, homeMoneyline=-120, awayMoneyline=110)])
    receipt = dict(status="ok", retrieved_at="2026-10-11T15:00Z", url="https://example.org/lines")
    monkeypatch.setattr(cfbd, "_cfbd_get", lambda *a, **kw: ([event], receipt))
    store = tmp_path / "market.csv"
    assert cfbd.capture_cfbd_prices(games, 2026, tmp_path, store) == 1
    saved = pd.read_csv(store)
    assert saved.home_spread.iloc[0] == -3
    assert saved.home_ml_decimal.iloc[0] == pytest.approx(1 + 100 / 120)
    event["homeTeam"] = "Different team"
    assert cfbd.capture_cfbd_prices(games, 2026, tmp_path, store) == 0
    event["homeTeam"] = "Home"
    receipt["retrieved_at"] = "2026-10-17T19:00Z"
    assert cfbd.capture_cfbd_prices(games, 2026, tmp_path, store) == 0


def test_database_update_archives_source_rows_with_retrieval_provenance(tmp_path, monkeypatch):
    import cfb_forecast.pipeline.database_update as update
    games = normalize_schedule(pd.DataFrame([dict(game_id=1, season=2026, week=7,
        start_date="2026-10-17T19:00Z", completed=False, neutral_site=False, home_id=1,
        away_id=2, home_team="Home", away_team="Away", home_points=None, away_points=None, fbs_game=True)]))
    monkeypatch.setattr(update, "load_data", lambda _: (games, None, []))
    def collect(season, destination):
        destination.mkdir(parents=True)
        (destination / "test.bin").write_text('[{"recorded": 42}]')
        return [dict(name="test", status="ok", url="https://example.org/real-source")]
    monkeypatch.setattr(update, "ingest_catalog", collect)
    database = tmp_path / "db.sqlite"
    report = update.update_database(tmp_path, database, 2026, tmp_path / "artifacts", fetch_sources=True)
    assert report["games"] == 1
    conn = connect(database)
    row = conn.execute("SELECT payload_json, source_url FROM source_rows JOIN source_retrievals USING(retrieval_id) JOIN source_registry USING(source_id)").fetchone()
    assert json.loads(row[0]) == {"recorded": 42}
    assert row[1] == "https://example.org/real-source"
    conn.close()
