import json
from pathlib import Path

import pandas as pd

from cfb_forecast.database import connect, list_forecasts
from cfb_forecast.watchdog import sync_artifacts


def test_watchdog_persists_forecasts_without_duplicates(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    run = artifacts / "runs" / "2026-week-1"
    run.mkdir(parents=True)
    frame = pd.DataFrame([
        {
            "game_id": 1,
            "season": 2026,
            "home_team": "A",
            "away_team": "B",
            "kickoff": "2026-09-05T16:00:00Z",
            "as_of": "2026-09-04T16:00:00Z",
            "expected_home_score": 28,
            "expected_away_score": 21,
            "projected_home_margin": 7,
            "projected_total": 49,
            "home_win_probability": 0.7,
        }
    ])
    frame.to_csv(run / "forecasts.csv", index=False)
    (run / "run_manifest.json").write_text(json.dumps({"run_id": "2026-week-1"}))
    database = artifacts / "cfb_forecast.sqlite"
    first = sync_artifacts(artifacts, database)
    second = sync_artifacts(artifacts, database)
    assert first["rows_inserted"] == 1
    assert second["rows_inserted"] == 0
    conn = connect(database)
    assert len(list_forecasts(conn, 2026)) == 1
    conn.close()
