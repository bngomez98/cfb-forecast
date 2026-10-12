"""SQLite persistence for forecasts and established model metadata."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    training_through_season INTEGER NOT NULL,
    variant TEXT NOT NULL,
    created_at TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY,
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
CREATE INDEX IF NOT EXISTS idx_forecasts_season ON forecasts(season);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


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


def insert_forecasts(conn, model_id: int, rows: Iterable[dict]) -> int:
    count = 0
    for row in rows:
        conn.execute(
            """INSERT OR IGNORE INTO forecasts(
            model_id,game_id,season,home_team,away_team,kickoff,as_of,
            expected_home_score,expected_away_score,projected_home_margin,
            projected_total,home_win_probability,market_home_probability_no_vig,
            best_bet_side,best_bet_edge,best_bet_ev_per_unit,recommended_stake_fraction,
            bet_status,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (model_id, int(row["game_id"]), int(row["season"]), row["home_team"], row["away_team"],
             row["kickoff"], row["as_of"], float(row["expected_home_score"]), float(row["expected_away_score"]),
             float(row["projected_home_margin"]), float(row["projected_total"]), float(row["home_win_probability"]),
             row.get("market_home_probability_no_vig"), row.get("best_bet_side"), row.get("best_bet_edge"),
             row.get("best_bet_ev_per_unit"), row.get("recommended_stake_fraction"), row.get("bet_status"), json.dumps(row)),
        )
        count += 1
    conn.commit()
    return count


def list_forecasts(conn, season: int | None = None):
    if season is None:
        return conn.execute("SELECT * FROM forecasts ORDER BY kickoff").fetchall()
    return conn.execute("SELECT * FROM forecasts WHERE season=? ORDER BY kickoff", (int(season),)).fetchall()
