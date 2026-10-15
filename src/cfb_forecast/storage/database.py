"""SQLite schema and persistence for forecast runs, games, sources, and models."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

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

CREATE TABLE IF NOT EXISTS source_registry (
    source_id INTEGER PRIMARY KEY,
    source_key TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    source_url TEXT,
    schema_version TEXT
);

CREATE TABLE IF NOT EXISTS source_retrievals (
    retrieval_id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES source_registry(source_id),
    retrieved_at TEXT NOT NULL,
    cutoff_at TEXT,
    http_status INTEGER,
    content_sha256 TEXT,
    byte_count INTEGER,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS source_files (
    file_id INTEGER PRIMARY KEY,
    retrieval_id INTEGER NOT NULL REFERENCES source_retrievals(retrieval_id),
    path TEXT NOT NULL UNIQUE,
    content_sha256 TEXT,
    byte_count INTEGER,
    format TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    kickoff_datetime TEXT NOT NULL,
    venue_id INTEGER,
    neutral_site INTEGER NOT NULL,
    status TEXT NOT NULL,
    source_id INTEGER REFERENCES source_registry(source_id),
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS game_teams (
    game_id INTEGER NOT NULL REFERENCES games(game_id),
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    side TEXT NOT NULL,
    points REAL,
    PRIMARY KEY (game_id, side)
);

CREATE TABLE IF NOT EXISTS statistical_parameter_sets (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    data_cutoff TEXT NOT NULL,
    source_manifest_json TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    uncertainty_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS forecast_runs (
    id INTEGER PRIMARY KEY,
    parameter_set_id INTEGER NOT NULL REFERENCES statistical_parameter_sets(id),
    season INTEGER NOT NULL,
    information_cutoff TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    source_manifest_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    modified_ns INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS watchdog_events (
    id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    path TEXT,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS learning_versions (
    version TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    cutoff TEXT NOT NULL,
    status TEXT NOT NULL,
    model_path TEXT NOT NULL,
    report_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_examples (
    version TEXT NOT NULL REFERENCES learning_versions(version),
    game_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    forecast_cutoff TEXT NOT NULL,
    result_available_at TEXT NOT NULL,
    latest_feature_result TEXT,
    features_json TEXT NOT NULL,
    observations_json TEXT NOT NULL,
    PRIMARY KEY(version, game_id)
);

CREATE INDEX IF NOT EXISTS training_examples_season ON training_examples(season);
CREATE INDEX IF NOT EXISTS idx_runs_season ON runs(season);
CREATE INDEX IF NOT EXISTS idx_sources_run ON sources(run_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_season_kickoff ON forecasts(season, kickoff);
CREATE INDEX IF NOT EXISTS idx_forecasts_game ON forecasts(game_id);
CREATE INDEX IF NOT EXISTS idx_results_season ON results(season);
CREATE INDEX IF NOT EXISTS idx_data_files_status ON data_files(status);
CREATE INDEX IF NOT EXISTS idx_watchdog_events_time ON watchdog_events(observed_at);
CREATE TABLE IF NOT EXISTS forecast_log (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    kickoff TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    expected_home_score REAL NOT NULL,
    expected_away_score REAL NOT NULL,
    winner TEXT NOT NULL,
    confidence REAL NOT NULL,
    home_win_probability REAL NOT NULL,
    logged_at TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    projected_home_margin REAL,
    model_spread REAL,
    market_spread REAL,
    home_cover_probability REAL,
    home_moneyline REAL,
    away_moneyline REAL,
    UNIQUE(run_id, game_id)
);

CREATE TABLE IF NOT EXISTS weekly_validation (
    id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER,
    run_id TEXT NOT NULL,
    validated_at TEXT NOT NULL,
    games INTEGER NOT NULL,
    status TEXT NOT NULL,
    log_loss REAL,
    brier REAL,
    margin_mae REAL,
    spread_mae REAL,
    report_json TEXT NOT NULL,
    UNIQUE(season, week, run_id)
);

CREATE TABLE IF NOT EXISTS team_game_facts (
    game_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    season INTEGER,
    week INTEGER,
    plays INTEGER,
    points_for REAL,
    points_against REAL,
    yards_per_play REAL,
    success_rate REAL,
    explosive_rate REAL,
    turnover_rate REAL,
    pass_yards REAL,
    rush_yards REAL,
    receiving_yards REAL,
    epa REAL,
    PRIMARY KEY (game_id, team_id)
);

CREATE TABLE IF NOT EXISTS player_game_facts (
    game_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    position_group TEXT NOT NULL,
    season INTEGER,
    plays INTEGER,
    yards REAL,
    epa REAL,
    PRIMARY KEY (game_id, team_id, player_name, position_group)
);

CREATE INDEX IF NOT EXISTS idx_team_game_facts_team ON team_game_facts(team_id, season);
CREATE INDEX IF NOT EXISTS idx_player_game_facts_team ON player_game_facts(team_id, season);
CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week);
CREATE INDEX IF NOT EXISTS idx_forecast_runs_season ON forecast_runs(season);
CREATE INDEX IF NOT EXISTS idx_forecast_log_season_kickoff ON forecast_log(season, kickoff);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    schema_path = Path(__file__).resolve().parents[3] / "sql" / "schema.sql"
    conn.executescript(schema_path.read_text() if schema_path.exists() else SCHEMA)
    _ensure_forecast_columns(conn)
    return conn


def _ensure_forecast_columns(conn: sqlite3.Connection) -> None:
    """Add forecast columns to a database created before the spread fields existed."""
    present = {row[1] for row in conn.execute("PRAGMA table_info(forecast_log)")}
    for name in (
        "projected_home_margin", "model_spread", "market_spread", "home_cover_probability",
        "home_moneyline", "away_moneyline",
    ):
        if name not in present:
            conn.execute(f"ALTER TABLE forecast_log ADD COLUMN {name} REAL")
    conn.commit()


def record_history_facts(conn: sqlite3.Connection, teams: pd.DataFrame, players: pd.DataFrame | None = None) -> None:
    """Store team and player facts read from historical parquet files."""
    team_columns = [
        "game_id", "team_id", "season", "week", "plays", "points_for", "points_against",
        "yards_per_play", "success_rate", "explosive_rate", "turnover_rate",
        "pass_yards", "rush_yards", "receiving_yards", "epa",
    ]
    if teams is not None and len(teams):
        frame = teams.copy()
        for column in team_columns:
            if column not in frame:
                frame[column] = None
        frame = frame[team_columns].drop_duplicates(["game_id", "team_id"], keep="last")
        conn.executemany(
            """INSERT INTO team_game_facts(
                 game_id, team_id, season, week, plays, points_for, points_against,
                 yards_per_play, success_rate, explosive_rate, turnover_rate,
                 pass_yards, rush_yards, receiving_yards, epa)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(game_id, team_id) DO UPDATE SET
                 season=excluded.season, week=excluded.week, plays=excluded.plays,
                 points_for=excluded.points_for, points_against=excluded.points_against,
                 yards_per_play=excluded.yards_per_play, success_rate=excluded.success_rate,
                 explosive_rate=excluded.explosive_rate, turnover_rate=excluded.turnover_rate,
                 pass_yards=excluded.pass_yards, rush_yards=excluded.rush_yards,
                 receiving_yards=excluded.receiving_yards, epa=excluded.epa""",
            frame.itertuples(index=False, name=None),
        )
    player_columns = ["game_id", "team_id", "player_name", "position_group", "season", "plays", "yards", "epa"]
    if players is not None and len(players):
        frame = players.copy()
        for column in player_columns:
            if column not in frame:
                frame[column] = None
        frame = frame[player_columns].dropna(subset=["player_name", "position_group"])
        frame = frame.drop_duplicates(["game_id", "team_id", "player_name", "position_group"], keep="last")
        conn.executemany(
            """INSERT INTO player_game_facts(
                 game_id, team_id, player_name, position_group, season, plays, yards, epa)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(game_id, team_id, player_name, position_group) DO UPDATE SET
                 season=excluded.season, plays=excluded.plays, yards=excluded.yards, epa=excluded.epa""",
            frame.itertuples(index=False, name=None),
        )
    conn.commit()


def record_weekly_validation(conn: sqlite3.Connection, report: dict) -> None:
    conn.execute(
        """INSERT INTO weekly_validation(
             season, week, run_id, validated_at, games, status, log_loss, brier,
             margin_mae, spread_mae, report_json)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(season, week, run_id) DO UPDATE SET
             validated_at=excluded.validated_at,
             games=excluded.games,
             status=excluded.status,
             log_loss=excluded.log_loss,
             brier=excluded.brier,
             margin_mae=excluded.margin_mae,
             spread_mae=excluded.spread_mae,
             report_json=excluded.report_json""",
        (
            int(report["season"]),
            None if report.get("week") is None else int(report["week"]),
            str(report["run_id"]),
            report["validated_at"],
            int(report.get("games", report.get("settled_games", 0))),
            report.get("status", "settled"),
            report.get("log_loss"),
            report.get("brier"),
            report.get("margin_mae"),
            report.get("spread_mae"),
            json.dumps(report),
        ),
    )
    conn.commit()


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


def list_published_board(conn: sqlite3.Connection, season: int | None = None):
    """Return the published forecast log, which carries winner and confidence."""
    sql = """SELECT run_id, season, game_id, kickoff, away_team, home_team,
        expected_away_score, expected_home_score, projected_home_margin,
        model_spread, market_spread, home_cover_probability, home_moneyline, away_moneyline,
        winner, confidence, home_win_probability, logged_at FROM forecast_log"""
    params: tuple = ()
    if season is not None:
        sql += " WHERE season = ?"
        params = (int(season),)
    rows = conn.execute(sql + " ORDER BY kickoff, game_id", params).fetchall()
    if rows:
        return rows
    return list_forecasts(conn, season)


def persist_completed_run(database, model_path, records, receipt, schedules, at):
    from contextlib import closing
    from ..ingest.data import sha256
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
    parser.add_argument("--database", default="data/cfb_forecast.sqlite")
    args = parser.parse_args()
    initialize(args.database)
    print(args.database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
