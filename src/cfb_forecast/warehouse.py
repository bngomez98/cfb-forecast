"""Load schedules, team game rates and source receipts into the forecast database."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import pandas as pd

from .data import load_data
from .database import connect
from .external_cli import select_next_week
from .external_sources import PUBLIC_SOURCES, collect_extended_sources
from .sources import SOURCE_REGISTRY, source_registry, utc_now

RATE_FIELDS = ("yards_per_play", "yards_per_pass", "yards_per_rush", "first_downs_created_rate")


def _none(value):
    """SQLite NULL for missing values; a missing rate is never filled in."""
    return None if pd.isna(value) else value


def load_games(conn, schedules, season):
    games = schedules[schedules.season == int(season)]
    game_rows, team_rows = [], []
    for g in games.itertuples(index=False):
        points = (_none(g.home_points), _none(g.away_points))
        game_rows.append((int(g.game_id), int(g.season), int(g.week), g.start_date.isoformat(), int(g.completed),
                          int(g.neutral_site), int(g.fbs_game), int(g.result_usable), g.result_available_at.isoformat(),
                          int(g.home_id), int(g.away_id), g.home_team, g.away_team, *points))
        team_rows.append((int(g.game_id), int(g.home_id), g.home_team, 1, points[0]))
        team_rows.append((int(g.game_id), int(g.away_id), g.away_team, 0, points[1]))
    conn.executemany("""INSERT INTO games VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(game_id) DO UPDATE SET week=excluded.week,start_date=excluded.start_date,
        completed=excluded.completed,neutral_site=excluded.neutral_site,fbs_game=excluded.fbs_game,
        result_usable=excluded.result_usable,result_available_at=excluded.result_available_at,
        home_team=excluded.home_team,away_team=excluded.away_team,
        home_points=excluded.home_points,away_points=excluded.away_points""", game_rows)
    conn.executemany("""INSERT INTO game_teams VALUES(?,?,?,?,?) ON CONFLICT(game_id,team_id) DO UPDATE SET
        team_name=excluded.team_name,points=excluded.points""", team_rows)
    return len(game_rows)


def load_team_facts(conn, schedules, adv, season):
    seasons = schedules.loc[schedules.season == int(season), ["game_id", "season"]]
    facts = adv.drop(columns=["season"], errors="ignore").merge(seasons, on="game_id", how="inner")
    conn.executemany(f"""INSERT INTO team_game_facts(game_id,team_id,season,{','.join(RATE_FIELDS)})
        VALUES(?,?,?,?,?,?,?) ON CONFLICT(game_id,team_id) DO UPDATE SET
        {','.join(f'{f}=excluded.{f}' for f in RATE_FIELDS)}""",
        [(int(r.game_id), int(r.pos_team_id), int(r.season), *(_none(getattr(r, f)) for f in RATE_FIELDS))
         for r in facts.itertuples(index=False)])
    return len(facts)


def load_sources(conn, season, report):
    now = report.get("collected_at") or utc_now()
    roles = {s["name"]: s.get("role") for s in PUBLIC_SOURCES}
    registry = {**source_registry(season), **{s["name"]: s["url"] for s in report["sources"] if s.get("url")}}
    conn.executemany("INSERT INTO source_registry VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET url=excluded.url,role=excluded.role",
                     [(n, u, roles.get(n)) for n, u in registry.items()])
    for s in report["sources"]:
        conn.execute("""INSERT OR IGNORE INTO source_retrievals(season,name,url,status,retrieved_at,sha256,bytes,receipt_json)
            VALUES(?,?,?,?,?,?,?,?)""", (int(season), s["name"], s.get("url"), s.get("status"), s.get("retrieved_at") or now,
                                         s.get("sha256"), s.get("bytes"), json.dumps(s, default=str)))
    rows = report.get("schedule_validation", [])
    conn.executemany("INSERT INTO source_rows(season,source,game_id,row_json,collected_at) VALUES(?,?,?,?,?)",
                     [(int(season), r.get("provider", "schedule_validation"), r.get("game_id"), json.dumps(r, default=str), now) for r in rows])
    return len(report["sources"]), len(rows)


def update_database(data_root, artifacts, season, database=None, collect=True):
    """Write one season into SQLite; a failed collection is reported, never faked."""
    artifacts = Path(artifacts)
    database = database or artifacts / "cfb_forecast.sqlite"
    schedules, adv, _ = load_data(data_root)
    summary = {"season": int(season), "database": str(database)}
    with closing(connect(database)) as conn:
        summary["games"] = load_games(conn, schedules, season)
        summary["team_game_facts"] = load_team_facts(conn, schedules, adv, season)
        conn.commit()
        if collect:
            upcoming = select_next_week(schedules, season, utc_now())
            weekly = upcoming if len(upcoming) else schedules[schedules.season == int(season)].head(0)
            report = collect_extended_sources(weekly, season, artifacts / "external-sources" / f"{int(season)}-database")
            summary["sources"], summary["source_rows"] = load_sources(conn, season, report)
            conn.commit()
    return summary


def load_data_from_database(database, seasons=None):
    """Rebuild the (schedules, adv) frames the forecast consumes from SQLite.

    Returns raw schedule columns (``data.load_data`` normalizes them, deriving result availability), so a forecast can read
    the database instead of the downloaded files.
    """
    where, args = "", []
    if seasons is not None:
        where = f"WHERE season IN ({','.join('?' * len(seasons))})"
        args = [int(s) for s in seasons]
    with closing(connect(database)) as conn:
        games = pd.read_sql_query(f"SELECT * FROM games {where} ORDER BY start_date, game_id", conn, params=args)
        adv = pd.read_sql_query(f"SELECT game_id, team_id AS pos_team_id, {','.join(RATE_FIELDS)} FROM team_game_facts "
                                f"{where} ORDER BY game_id, team_id", conn, params=args)
    if games.empty:
        raise ValueError(f"No games in {database}; run update-database first")
    for c in ("completed", "neutral_site", "fbs_game"):
        games[c] = games[c].astype(bool)
    games = games.drop(columns=["result_usable", "result_available_at"])
    games["start_date"] = pd.to_datetime(games.start_date, utc=True, format="ISO8601")
    return games, adv
