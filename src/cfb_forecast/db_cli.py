"""Entry point for the forecast SQLite database: init, update, status and SQL queries."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from .database import connect
from .paths import artifacts_root

TABLES = ("games", "game_teams", "team_game_facts", "source_registry", "source_retrievals", "source_rows",
          "runs", "models", "forecasts", "forecast_log", "results", "learning_versions", "training_examples")


def default_database():
    return artifacts_root() / "cfb_forecast.sqlite"


def status(database):
    with closing(connect(database)) as conn:
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES if t in existing}


def query(database, sql):
    """Run one statement on a read-only connection; writes are refused by SQLite itself."""
    uri = f"file:{Path(database).resolve()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        cursor = conn.execute(sql)
        columns = [c[0] for c in cursor.description or []]
        return columns, cursor.fetchall()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cfb-db", description="Work directly with the forecast SQLite database")
    parser.add_argument("--database", default=None, help="default: artifacts/cfb_forecast.sqlite")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create the database and apply sql/schema.sql")
    sub.add_parser("status", help="Row counts for every table")
    up = sub.add_parser("update", help="Load a season's schedule, team rates and sources")
    up.add_argument("--season", type=int, required=True)
    up.add_argument("--data", default=None)
    up.add_argument("--no-collect", action="store_true")
    q = sub.add_parser("query", help="Run a read-only SQL statement")
    q.add_argument("sql")
    args = parser.parse_args(argv)
    database = Path(args.database) if args.database else default_database()

    if args.command == "init":
        connect(database).close()
        print(database)
    elif args.command == "status":
        print(json.dumps(status(database), indent=2))
    elif args.command == "update":
        from .paths import data_root
        from .warehouse import update_database
        print(json.dumps(update_database(args.data or data_root(), database.parent, args.season, database, not args.no_collect), indent=2))
    else:
        try:
            columns, rows = query(database, args.sql)
        except sqlite3.Error as error:
            print(f"SQL error: {error}", file=sys.stderr)
            return 1
        print("\t".join(columns))
        for row in rows:
            print("\t".join("" if v is None else str(v) for v in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
