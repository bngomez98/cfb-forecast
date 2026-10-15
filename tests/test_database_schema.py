import sqlite3

from cfb_forecast import database

DESIGN_TABLES = """sources source_retrievals source_files teams players coaches venues games game_teams plays drives
team_game_statistics player_game_statistics player_usage rosters player_team_membership transfers recruiting
returning_production player_availability weather schedule_conditions external_ratings market_quotes
derived_play_observations derived_drive_observations derived_team_observations derived_player_observations
statistical_parameter_sets team_state_estimates player_state_estimates unit_state_estimates availability_estimates
forecast_runs forecast_inputs forecasts game_results forecast_evaluations""".split()


def names(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_connect_creates_every_designed_table(tmp_path):
    conn = database.connect(tmp_path / "x.sqlite3")
    assert set(DESIGN_TABLES) <= names(conn)
    assert {"data_files", "watchdog_events", "run_sources", "run_forecasts"} <= names(conn)


def test_legacy_database_is_migrated_without_losing_rows(tmp_path):
    path = tmp_path / "old.sqlite3"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE runs(id INTEGER PRIMARY KEY, run_key TEXT NOT NULL UNIQUE, season INTEGER NOT NULL, started_at TEXT,
            completed_at TEXT, status TEXT NOT NULL DEFAULT 'completed', manifest_json TEXT NOT NULL DEFAULT '{}', created_at TEXT);
        CREATE TABLE sources(id INTEGER PRIMARY KEY, run_id INTEGER, provider TEXT NOT NULL, url TEXT, retrieved_at TEXT,
            sha256 TEXT, status TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE forecasts(id INTEGER PRIMARY KEY, run_id INTEGER, model_id INTEGER NOT NULL, game_id INTEGER NOT NULL);
        CREATE TABLE games(game_id INTEGER PRIMARY KEY, season INTEGER NOT NULL, week INTEGER, start_date TEXT NOT NULL,
            completed INTEGER NOT NULL, neutral_site INTEGER NOT NULL, fbs_game INTEGER NOT NULL, result_usable INTEGER NOT NULL,
            result_available_at TEXT, home_id INTEGER NOT NULL, away_id INTEGER NOT NULL, home_team TEXT, away_team TEXT,
            home_points REAL, away_points REAL);
        INSERT INTO sources(provider,status) VALUES('espn','ok');
        INSERT INTO forecasts(model_id,game_id) VALUES(1,9);
        INSERT INTO games VALUES(9,2026,1,'2026-09-05T16:00:00+00:00',0,0,1,0,NULL,1,2,'A','B',NULL,NULL);
    """)
    old.commit(); old.close()
    conn = database.connect(path)
    assert conn.execute("SELECT provider FROM run_sources").fetchone()[0] == "espn"
    assert conn.execute("SELECT game_id FROM run_forecasts").fetchone()[0] == 9
    assert "venue_id" in {r[1] for r in conn.execute("PRAGMA table_info(games)")}
    assert "home_expected_points" in {r[1] for r in conn.execute("PRAGMA table_info(forecasts)")}
    assert conn.execute("SELECT home_id FROM games").fetchone()[0] == 1
