"""Database-facing football-system ingestion and forecast persistence.

Provenance tables (source registry, games, parameter sets, forecast runs) live
beside the canonical forecast table. Persisted forecasts are written into
``forecasts`` so one database holds both the run ledger and the published rows.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_source(conn: sqlite3.Connection, source_key: str, provider: str, url: str | None = None, schema_version: str | None = None) -> int:
    conn.execute(
        """INSERT INTO source_registry(source_key, provider, source_url, schema_version)
           VALUES(?,?,?,?) ON CONFLICT(source_key) DO UPDATE SET
           provider=excluded.provider, source_url=excluded.source_url,
           schema_version=excluded.schema_version""",
        (source_key, provider, url, schema_version),
    )
    conn.commit()
    return int(conn.execute("SELECT source_id FROM source_registry WHERE source_key=?", (source_key,)).fetchone()[0])


def record_retrieval(conn, source_id: int, path: str | Path, status: str = "ok", cutoff_at: str | None = None, http_status: int | None = None, metadata: dict | None = None) -> int:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
    size = path.stat().st_size if path.exists() else 0
    now = utc_now()
    conn.execute(
        """INSERT INTO source_retrievals(source_id,retrieved_at,cutoff_at,http_status,content_sha256,byte_count,status,metadata_json)
           VALUES(?,?,?,?,?,?,?,?)""",
        (source_id, now, cutoff_at, http_status, digest, size, status, json.dumps(metadata or {})),
    )
    retrieval_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.execute(
        """INSERT INTO source_files(retrieval_id,path,content_sha256,byte_count,format,metadata_json)
           VALUES(?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET
           retrieval_id=excluded.retrieval_id, content_sha256=excluded.content_sha256,
           byte_count=excluded.byte_count, format=excluded.format,
           metadata_json=excluded.metadata_json""",
        (retrieval_id, str(path), digest, size, path.suffix.lstrip("."), json.dumps(metadata or {})),
    )
    conn.commit()
    return retrieval_id


def upsert_schedule(conn, schedule: pd.DataFrame, source_id: int | None = None) -> int:
    required = {"game_id", "season", "week", "start_date", "home_id", "away_id", "neutral_site"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule missing columns: {sorted(missing)}")
    count = 0
    for row in schedule.itertuples(index=False):
        conn.execute(
            """INSERT INTO games(game_id,season,week,kickoff_datetime,venue_id,neutral_site,status,source_id,metadata_json)
               VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET
               season=excluded.season, week=excluded.week, kickoff_datetime=excluded.kickoff_datetime,
               neutral_site=excluded.neutral_site, status=excluded.status, source_id=excluded.source_id""",
            (int(row.game_id), int(row.season), int(row.week), str(row.start_date), None,
             int(bool(row.neutral_site)), "completed" if bool(row.completed) else "scheduled", source_id, "{}"),
        )
        for side in ("home", "away"):
            team_id = int(getattr(row, f"{side}_id"))
            points = getattr(row, f"{side}_points", None)
            name = getattr(row, f"{side}_team", str(team_id))
            conn.execute("INSERT OR IGNORE INTO teams(team_id,name) VALUES(?,?)", (team_id, name))
            conn.execute(
                """INSERT INTO game_teams(game_id,team_id,side,points) VALUES(?,?,?,?)
                   ON CONFLICT(game_id,side) DO UPDATE SET team_id=excluded.team_id,points=excluded.points""",
                (int(row.game_id), team_id, side, None if pd.isna(points) else float(points)),
            )
        count += 1
    conn.commit()
    return count


def record_parameter_set(conn, name: str, version: str, cutoff: str, parameters: dict, uncertainty: dict, source_manifest: dict) -> int:
    conn.execute(
        """INSERT INTO statistical_parameter_sets(name,version,data_cutoff,source_manifest_json,parameters_json,uncertainty_json)
           VALUES(?,?,?,?,?,?)""",
        (name, version, cutoff, json.dumps(source_manifest), json.dumps(parameters), json.dumps(uncertainty)),
    )
    conn.commit()
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def create_forecast_run(conn, parameter_set_id: int, season: int, cutoff: str, source_manifest: dict, status: str = "completed") -> int:
    conn.execute(
        """INSERT INTO forecast_runs(parameter_set_id,season,information_cutoff,created_at,status,source_manifest_json)
           VALUES(?,?,?,?,?,?)""",
        (parameter_set_id, season, cutoff, utc_now(), status, json.dumps(source_manifest)),
    )
    conn.commit()
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _team_name(conn, game_id: int, side: str) -> str:
    row = conn.execute(
        """SELECT teams.name FROM game_teams
           JOIN teams ON teams.team_id = game_teams.team_id
           WHERE game_teams.game_id=? AND game_teams.side=?""",
        (game_id, side),
    ).fetchone()
    return str(row["name"]) if row else side


def persist_forecast(conn, run_id: int, game_id: int, values: dict) -> int:
    game = conn.execute("SELECT season, kickoff_datetime FROM games WHERE game_id=?", (int(game_id),)).fetchone()
    forecast_run = conn.execute(
        "SELECT season, information_cutoff FROM forecast_runs WHERE id=?", (int(run_id),)
    ).fetchone()
    season = int(game["season"]) if game else int(forecast_run["season"] if forecast_run else values.get("season", 0))
    kickoff = str(game["kickoff_datetime"]) if game else str(values.get("kickoff") or utc_now())
    as_of = str(forecast_run["information_cutoff"]) if forecast_run else utc_now()
    home_score = float(values.get("expected_home_score", 0.0))
    away_score = float(values.get("expected_away_score", 0.0))
    margin = values.get("projected_home_margin", home_score - away_score)
    total = values.get("projected_total", home_score + away_score)
    home_win = float(values.get("home_win_probability", 0.5))
    now = utc_now()
    conn.execute(
        """INSERT INTO models(name,path,training_through_season,variant,created_at,manifest_json)
           VALUES('joint-live','',?,'joint',?,'{}')
           ON CONFLICT(name) DO NOTHING""",
        (season, now),
    )
    model_id = int(conn.execute("SELECT id FROM models WHERE name='joint-live'").fetchone()[0])
    run_key = f"joint-run-{int(run_id)}"
    conn.execute(
        """INSERT INTO runs(run_key,season,started_at,completed_at,status,manifest_json)
           VALUES(?,?,?,?,'completed',?)
           ON CONFLICT(run_key) DO UPDATE SET
             completed_at=excluded.completed_at,
             manifest_json=excluded.manifest_json""",
        (run_key, season, now, now, json.dumps({"forecast_run_id": int(run_id)})),
    )
    canonical_run_id = int(conn.execute("SELECT id FROM runs WHERE run_key=?", (run_key,)).fetchone()[0])
    payload = dict(values)
    if "margin_variance" not in payload and values.get("margin_sd") is not None:
        payload["margin_variance"] = float(values["margin_sd"]) ** 2
    payload.setdefault("away_win_probability", 1 - home_win)
    from .data import game_date
    dated = game_date(values.get("game_date") or kickoff)
    conn.execute(
        """INSERT INTO forecasts(
            run_id,model_id,game_id,season,home_team,away_team,game_date,kickoff,as_of,
            expected_home_score,expected_away_score,projected_home_margin,
            projected_total,home_win_probability,payload_json)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(model_id, game_id, as_of) DO UPDATE SET
            game_date=excluded.game_date,
            run_id=excluded.run_id,
            expected_home_score=excluded.expected_home_score,
            expected_away_score=excluded.expected_away_score,
            projected_home_margin=excluded.projected_home_margin,
            projected_total=excluded.projected_total,
            home_win_probability=excluded.home_win_probability,
            payload_json=excluded.payload_json""",
        (
            canonical_run_id, model_id, int(game_id), season,
            values.get("home_team") or _team_name(conn, int(game_id), "home"),
            values.get("away_team") or _team_name(conn, int(game_id), "away"),
            dated, kickoff, as_of, home_score, away_score, float(margin), float(total),
            home_win, json.dumps(payload),
        ),
    )
    forecast_id = int(conn.execute(
        "SELECT id FROM forecasts WHERE model_id=? AND game_id=? AND as_of=?",
        (model_id, int(game_id), as_of),
    ).fetchone()[0])
    conn.commit()
    return forecast_id
