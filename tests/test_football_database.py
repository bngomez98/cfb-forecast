from pathlib import Path

import pandas as pd

from cfb_forecast.ingest.data import normalize_schedule
from cfb_forecast.storage.database import connect
from cfb_forecast.storage.football_database import (
    create_forecast_run,
    persist_forecast,
    record_parameter_set,
    register_source,
    upsert_schedule,
)


def test_football_database_records_provenance_schedule_and_forecast(tmp_path: Path):
    db = tmp_path / "football.sqlite"
    conn = connect(db)
    source_id = register_source(conn, "test-schedule", "fixture", "https://example.test/schedule", "1")
    raw = pd.DataFrame([{
        "game_id": 1, "season": 2026, "week": 1, "start_date": "2026-09-05T16:00Z",
        "completed": False, "neutral_site": False, "home_id": 10, "away_id": 20,
        "home_team": "Home", "away_team": "Away", "home_points": None, "away_points": None,
        "fbs_game": True,
    }])
    schedule = normalize_schedule(raw)
    assert upsert_schedule(conn, schedule, source_id) == 1
    parameter_id = record_parameter_set(conn, "test", "1", "2026-09-01T00:00Z", {"alpha": 1}, {}, {"source": "test"})
    run_id = create_forecast_run(conn, parameter_id, 2026, "2026-09-04T12:00Z", {"source": "test"})
    forecast_id = persist_forecast(conn, run_id, 1, {
        "expected_home_score": 28, "expected_away_score": 21,
        "home_win_probability": 0.7, "projected_home_margin": 7,
        "projected_total": 49, "margin_sd": 10,
    })
    assert forecast_id > 0
    assert conn.execute("SELECT COUNT(*) FROM source_retrievals").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1
    conn.close()
