from pathlib import Path

from cfb_forecast.monitoring.data_watchdog import check_data
from cfb_forecast.storage.database import connect


def test_data_watchdog_records_integrity_and_changes(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "schedule_2026.csv").write_text("game_id\n1\n")
    database = tmp_path / "artifacts" / "cfb.sqlite"
    first = check_data(data, database, ["schedule_*.csv"])
    assert first["status"] == "ok"
    assert first["files_checked"] == 1

    (data / "schedule_2026.csv").write_text("game_id\n2\n")
    second = check_data(data, database, ["schedule_*.csv"])
    assert second["status"] == "warning"
    assert "schedule_2026.csv" in second["changed_files"]

    (data / "schedule_2026.csv").unlink()
    third = check_data(data, database, ["schedule_*.csv"])
    assert third["status"] == "error"
    conn = connect(database)
    assert conn.execute("SELECT COUNT(*) FROM watchdog_events").fetchone()[0] >= 3
    conn.close()
