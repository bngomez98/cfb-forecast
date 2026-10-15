"""SQLite schema and persistence for forecast runs, games, sources, and models."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

def schema_text() -> str:
    """The canonical schema, shipped inside the package so every install has it."""
    from importlib import resources
    return resources.files("cfb_forecast").joinpath("schema.sql").read_text()


# Tables renamed when the football-data design reused their names.
LEGACY_RENAMES = (
    ("sources", "run_sources", "run_id"),
    ("forecasts", "run_forecasts", "model_id"),
    ("team_game_facts", "team_game_statistics", None),
    ("source_registry", None, None),
)


def _columns(conn, table):
    return {r[1]: r[2] for r in conn.execute(f"PRAGMA table_info({table})")}


def migrate(conn) -> None:
    """Bring an existing database up to the current schema without losing rows."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for old, new, marker in LEGACY_RENAMES:
        if old not in tables or (marker and marker not in _columns(conn, old)):
            continue
        if new and new not in tables:
            conn.execute(f"ALTER TABLE {old} RENAME TO {new}")
        elif new is None:
            conn.execute(f"DROP TABLE {old}")
    # Add columns introduced after a table was first created (nullable, no
    # constraints) before the schema runs, so its indexes can reference them.
    reference = sqlite3.connect(":memory:")
    reference.executescript(schema_text())
    existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for (table,) in reference.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        if table not in existing:
            continue
        have = _columns(conn, table)
        for name, kind in _columns(reference, table).items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
    reference.close()
    conn.executescript(schema_text())
    conn.commit()


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
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
            """INSERT OR IGNORE INTO run_forecasts(
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
        return conn.execute("SELECT * FROM run_forecasts ORDER BY kickoff").fetchall()
    return conn.execute("SELECT * FROM run_forecasts WHERE season=? ORDER BY kickoff", (int(season),)).fetchall()


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
        conn.execute("DELETE FROM run_sources WHERE run_id=?", (run_id,))
        conn.executemany("INSERT INTO run_sources(run_id,provider,sha256,status,metadata_json) VALUES(?,?,?,?,?)",
            [(run_id, name, digest, "snapshotted", json.dumps({"as_of": receipt["as_of"]})) for name, digest in receipt["source_hashes"].items()])
        conn.commit()


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Initialize the CFB Forecast SQLite database")
    parser.add_argument("--database", default="artifacts/cfb_forecast.sqlite3")
    args = parser.parse_args()
    initialize(args.database)
    print(args.database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
