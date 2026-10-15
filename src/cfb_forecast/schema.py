"""SQLite run ledger for the ongoing weekly forecast/validation cycle."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    season_type TEXT,
    as_of TEXT NOT NULL,
    validated_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT
);
CREATE TABLE IF NOT EXISTS source_checks (
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    checked_at TEXT NOT NULL,
    r_ok INTEGER NOT NULL,
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecasts (
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    week INTEGER,
    game_id INTEGER NOT NULL,
    issued_at TEXT NOT NULL,
    kickoff TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    expected_home_score REAL NOT NULL,
    expected_away_score REAL NOT NULL,
    projected_home_margin REAL NOT NULL,
    projected_total REAL NOT NULL,
    home_win_probability REAL NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS validations (
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    game_id INTEGER NOT NULL,
    validated_at TEXT NOT NULL,
    home_points INTEGER NOT NULL,
    away_points INTEGER NOT NULL,
    actual_home_win INTEGER NOT NULL,
    margin_error REAL NOT NULL,
    total_error REAL NOT NULL,
    brier REAL NOT NULL,
    correct INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sent_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    recipient TEXT NOT NULL,
    subject TEXT NOT NULL,
    delivered INTEGER NOT NULL,
    transport TEXT NOT NULL,
    detail TEXT
);
"""


class RunLog:
    def __init__(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(SCHEMA)
        # migration: older databases predate the forecasts.week column
        cols = [c[1] for c in self.conn.execute("PRAGMA table_info(forecasts)")]
        if "week" not in cols:
            self.conn.execute("ALTER TABLE forecasts ADD COLUMN week INTEGER")
            self.conn.commit()

    @staticmethod
    def _now():
        return datetime.now(UTC).isoformat()

    def start_run(self, season, week, season_type, as_of, boundary):
        cur = self.conn.execute(
            "INSERT INTO runs(started_at,season,week,season_type,as_of,validated_at)"
            " VALUES(?,?,?,?,?,?)",
            (self._now(), int(season), int(week), season_type, str(as_of),
             boundary.isoformat() if boundary is not None else None),
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id, status="completed", error=None):
        self.conn.execute(
            "UPDATE runs SET finished_at=?,status=?,error=? WHERE run_id=?",
            (self._now(), status, error, run_id),
        )
        self.conn.commit()

    def log_source_check(self, run_id, r_ok, detail):
        self.conn.execute(
            "INSERT INTO source_checks(run_id,checked_at,r_ok,detail) VALUES(?,?,?,?)",
            (run_id, self._now(), int(bool(r_ok)), json.dumps(detail)),
        )
        self.conn.commit()

    def log_forecasts(self, run_id, frame):
        rows = []
        for r in frame.itertuples(index=False):
            payload = {k: v for k, v in r._asdict().items()}
            rows.append((
                run_id, int(getattr(r, "week", 0)), int(r.game_id), str(r.as_of), str(r.kickoff),
                str(r.home_team), str(r.away_team),
                float(r.expected_home_score), float(r.expected_away_score),
                float(r.projected_home_margin), float(r.projected_total),
                float(r.home_win_probability), json.dumps(payload, default=str),
            ))
        self.conn.executemany(
            "INSERT INTO forecasts(run_id,week,game_id,issued_at,kickoff,home_team,away_team,"
            "expected_home_score,expected_away_score,projected_home_margin,projected_total,"
            "home_win_probability,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", rows,
        )
        self.conn.commit()
        return len(rows)

    def log_validations(self, run_id, frame):
        rows = [(run_id, int(r.game_id), str(r.validated_at), int(r.home_points),
                 int(r.away_points), int(r.actual_home_win), float(r.margin_error),
                 float(r.total_error), float(r.brier), int(r.correct))
                for r in frame.itertuples(index=False)]
        self.conn.executemany(
            "INSERT INTO validations(run_id,game_id,validated_at,home_points,away_points,"
            "actual_home_win,margin_error,total_error,brier,correct) VALUES(?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def log_notification(self, kind, recipient, subject, delivered, transport, detail=None):
        self.conn.execute(
            "INSERT INTO notifications(sent_at,kind,recipient,subject,delivered,transport,detail)"
            " VALUES(?,?,?,?,?,?,?)",
            (self._now(), kind, recipient, subject, int(bool(delivered)), transport, detail),
        )
        self.conn.commit()

    def pending_validation_weeks(self, season):
        q = ("SELECT run_id,season,week,season_type,as_of FROM runs"
             " WHERE season=? AND validated_at IS NULL AND status='completed'")
        return self.conn.execute(q, (int(season),)).fetchall()

    def week_forecasts(self, run_id):
        frame = pd_read(self.conn, "SELECT * FROM forecasts WHERE run_id=?", (run_id,))
        if frame.empty:
            return frame
        meta = self.conn.execute(
            "SELECT season_type, as_of FROM runs WHERE run_id=?", (int(run_id),)
        ).fetchone()
        frame["season_type"] = meta[0] if meta else None
        frame["as_of"] = meta[1] if meta else frame["issued_at"]
        return frame

    def close(self):
        self.conn.close()


def pd_read(conn, query, params=()):
    import pandas as pd
    return pd.read_sql_query(query, conn, params=params)
