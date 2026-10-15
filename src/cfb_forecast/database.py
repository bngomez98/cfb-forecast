"""SQLite schema and persistence for forecast runs, games, sources, and models."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    run_key TEXT NOT NULL UNIQUE,
    season INTEGER NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'completed',
    manifest_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    training_through_season INTEGER NOT NULL,
    variant TEXT NOT NULL,
    created_at TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    url TEXT,
    retrieved_at TEXT,
    sha256 TEXT,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL REFERENCES models(id),
    game_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    kickoff TEXT NOT NULL,
    as_of TEXT NOT NULL,
    expected_home_score REAL NOT NULL,
    expected_away_score REAL NOT NULL,
    projected_home_margin REAL NOT NULL,
    projected_total REAL NOT NULL,
    home_win_probability REAL NOT NULL,
    market_home_probability_no_vig REAL,
    best_bet_side TEXT,
    best_bet_edge REAL,
    best_bet_ev_per_unit REAL,
    recommended_stake_fraction REAL,
    bet_status TEXT,
    payload_json TEXT NOT NULL,
    UNIQUE(model_id, game_id, as_of)
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    game_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    home_points REAL,
    away_points REAL,
    completed INTEGER NOT NULL DEFAULT 0,
    result_available_at TEXT,
    source TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(game_id, season)
);

CREATE INDEX IF NOT EXISTS idx_runs_season ON runs(season);
CREATE INDEX IF NOT EXISTS idx_sources_run ON sources(run_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_season_kickoff ON forecasts(season, kickoff);
CREATE INDEX IF NOT EXISTS idx_forecasts_game ON forecasts(game_id);
CREATE INDEX IF NOT EXISTS idx_results_season ON results(season);

CREATE TABLE IF NOT EXISTS forecast_log (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    game_id INTEGER NOT NULL,
    season INTEGER,
    kickoff TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    expected_home_score REAL,
    expected_away_score REAL,
    home_win_probability REAL,
    predicted_winner TEXT,
    confidence REAL,
    published_at TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(run_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_forecast_log_run ON forecast_log(run_id);

CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER,
    start_date TEXT NOT NULL,
    completed INTEGER NOT NULL,
    neutral_site INTEGER NOT NULL,
    fbs_game INTEGER NOT NULL,
    result_usable INTEGER NOT NULL,
    result_available_at TEXT,
    home_id INTEGER NOT NULL,
    away_id INTEGER NOT NULL,
    home_team TEXT,
    away_team TEXT,
    home_points REAL,
    away_points REAL
);

CREATE TABLE IF NOT EXISTS game_teams (
    game_id INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    team_id INTEGER NOT NULL,
    team_name TEXT,
    is_home INTEGER NOT NULL,
    points REAL,
    PRIMARY KEY(game_id, team_id)
);

CREATE TABLE IF NOT EXISTS team_game_facts (
    game_id INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    team_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    yards_per_play REAL,
    yards_per_pass REAL,
    yards_per_rush REAL,
    first_downs_created_rate REAL,
    PRIMARY KEY(game_id, team_id)
);

CREATE TABLE IF NOT EXISTS source_registry (
    name TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    role TEXT
);

CREATE TABLE IF NOT EXISTS source_retrievals (
    id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    name TEXT NOT NULL,
    url TEXT,
    status TEXT,
    retrieved_at TEXT,
    sha256 TEXT,
    bytes INTEGER,
    receipt_json TEXT NOT NULL,
    UNIQUE(season, name, retrieved_at)
);

CREATE TABLE IF NOT EXISTS source_rows (
    id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    source TEXT NOT NULL,
    game_id INTEGER,
    row_json TEXT NOT NULL,
    collected_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_games_season_start ON games(season, start_date);
CREATE INDEX IF NOT EXISTS idx_team_game_facts_team ON team_game_facts(team_id, season);
CREATE INDEX IF NOT EXISTS idx_source_rows_source ON source_rows(season, source);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    schema_path = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"
    conn.executescript(schema_path.read_text() if schema_path.exists() else SCHEMA)
    return conn


def initialize(path: str | Path) -> Path:
    conn = connect(path)
    conn.close()
    return Path(path)


def register_run(conn, run_key: str, season: int, manifest: dict) -> int:
    conn.execute(
        """INSERT INTO runs(run_key, season, started_at, completed_at, status, manifest_json)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(run_key) DO UPDATE SET
             completed_at=excluded.completed_at,
             status=excluded.status,
             manifest_json=excluded.manifest_json""",
        (run_key, int(season), manifest.get("started_at"), manifest.get("completed_at"),
         manifest.get("status", "completed"), json.dumps(manifest)),
    )
    conn.commit()
    return int(conn.execute("SELECT id FROM runs WHERE run_key=?", (run_key,)).fetchone()[0])


def register_model(conn, name: str, path: str, manifest: dict) -> int:
    model = json.loads((Path(path) / "model.json").read_text())
    conn.execute(
        """INSERT INTO models(name,path,training_through_season,variant,created_at,manifest_json)
           VALUES(?,?,?,?,datetime('now'),?)
           ON CONFLICT(name) DO UPDATE SET path=excluded.path,
           training_through_season=excluded.training_through_season,
           variant=excluded.variant, manifest_json=excluded.manifest_json""",
        (name, str(path), int(model["training_through_season"]), model["variant"], json.dumps(manifest)),
    )
    conn.commit()
    return int(conn.execute("SELECT id FROM models WHERE name=?", (name,)).fetchone()[0])


def insert_forecasts(conn, model_id: int, rows: Iterable[dict], run_id: int | None = None) -> int:
    count = 0
    for row in rows:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO forecasts(
            run_id,model_id,game_id,season,home_team,away_team,kickoff,as_of,
            expected_home_score,expected_away_score,projected_home_margin,
            projected_total,home_win_probability,market_home_probability_no_vig,
            best_bet_side,best_bet_edge,best_bet_ev_per_unit,recommended_stake_fraction,
            bet_status,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, model_id, int(row["game_id"]), int(row["season"]), row["home_team"], row["away_team"],
             row["kickoff"], row["as_of"], float(row["expected_home_score"]), float(row["expected_away_score"]),
             float(row["projected_home_margin"]), float(row["projected_total"]), float(row["home_win_probability"]),
             row.get("market_home_probability_no_vig"), row.get("best_bet_side"), row.get("best_bet_edge"),
             row.get("best_bet_ev_per_unit"), row.get("recommended_stake_fraction"), row.get("bet_status"), json.dumps(row)),
        )
        count += max(cursor.rowcount, 0)
    conn.commit()
    return count



def list_forecasts(conn: sqlite3.Connection, season: int | None = None):
    if season is None:
        return conn.execute("SELECT * FROM forecasts ORDER BY kickoff").fetchall()
    return conn.execute("SELECT * FROM forecasts WHERE season=? ORDER BY kickoff", (int(season),)).fetchall()


def persist_completed_run(database, model_path, records, receipt, schedules, at):
    from contextlib import closing
    from .data import sha256
    with closing(connect(database)) as conn:
        run_id = register_run(conn, receipt["run_id"], receipt["season"], receipt)
        model_id = register_model(conn, sha256(Path(model_path) / "model.json"), str(model_path), receipt.get("learning", {}))
        insert_forecasts(conn, model_id, records, run_id)
        completed = schedules[schedules.result_usable & (schedules.result_available_at < at)]
        conn.executemany("""INSERT INTO results(game_id,season,home_points,away_points,completed,result_available_at,source)
            VALUES(?,?,?,?,1,?,?) ON CONFLICT(game_id,season) DO UPDATE SET
            home_points=excluded.home_points,away_points=excluded.away_points,
            result_available_at=excluded.result_available_at,completed=1,source=excluded.source""",
            [(int(r.game_id), int(r.season), float(r.home_points), float(r.away_points), r.result_available_at.isoformat(), "validated_schedule") for r in completed.itertuples(index=False)])
        conn.execute("DELETE FROM sources WHERE run_id=?", (run_id,))
        conn.executemany("INSERT INTO sources(run_id,provider,sha256,status,metadata_json) VALUES(?,?,?,?,?)",
            [(run_id, name, digest, "snapshotted", json.dumps({"as_of": receipt["as_of"]})) for name, digest in receipt["source_hashes"].items()])
        conn.commit()


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Initialize the CFB Forecast SQLite database")
    parser.add_argument("--database", default="artifacts/cfb_forecast.sqlite")
    args = parser.parse_args()
    initialize(args.database)
    print(args.database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
