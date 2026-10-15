"""The database loop around the forecast.

    sources -> SQLite -> Python -> SQLite -> forecast -> SQLite (forecast_log) -> validation

1. ``ingest`` stores the schedule's outcomes (including finals that two or more web sources agree
   on, e.g. cfbtrack.com) and the source cross-check in SQLite.
2. ``validate_logged_forecasts`` joins every pregame forecast ever logged (Sunday and daily) to the
   stored outcomes and writes one ``forecast_validation`` row per settled forecast, plus a
   ``weekly_validation`` summary per run.
3. ``recompute_from_database`` reads those rows back out of the database, recomputes log loss, Brier
   score and margin error independently, stores them in ``db_validation`` and refuses any disagreement.

The forecast itself then reads the processed history from the same database (team and player facts),
and logs its output to ``forecast_log``, which the next run validates.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_data
from .database import connect, record_weekly_validation
from .evaluate import metrics
from .logs import get

log = get("dbflow")


def _now():
    return datetime.now(timezone.utc).isoformat()


def ingest(database, data_root, season):
    """Store outcomes and the web-source cross-check in SQLite."""
    schedules, _, _ = load_data(data_root)
    done = schedules[(schedules.season == int(season)) & schedules.result_usable]
    overlay = Path(data_root) / "results_overlay.csv"
    overlay_ids = set(pd.read_csv(overlay).game_id.astype(int)) if overlay.exists() else set()
    rows = [(int(r.game_id), int(r.season), float(r.home_points), float(r.away_points), 1, r.result_available_at.isoformat(),
             "web_sources" if int(r.game_id) in overlay_ids else "schedule") for r in done.itertuples(index=False)]
    external = 0
    with closing(connect(database)) as conn:
        conn.executemany(
            """INSERT INTO results(game_id, season, home_points, away_points, completed, result_available_at, source)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(game_id, season) DO UPDATE SET home_points=excluded.home_points, away_points=excluded.away_points,
                 completed=excluded.completed, result_available_at=excluded.result_available_at, source=excluded.source""", rows)
        cross = Path(data_root) / "sources" / "results_crosscheck.csv"
        if cross.exists():
            table = pd.read_csv(cross)
            observed = _now()
            ext = [(int(season), int(r.game_id), observed, _flag(r.sources_agree), _flag(r.schedule_agrees),
                    str(r.source_scores), str(r.schedule_score)) for r in table.itertuples(index=False)]
            conn.executemany("INSERT OR IGNORE INTO external_results(season, game_id, observed_at, sources_agree, schedule_agrees, source_scores, schedule_score) VALUES(?,?,?,?,?,?,?)", ext)
            external = len(ext)
        conn.commit()
    return {"results": len(rows), "from_web_sources": sum(1 for r in rows if r[-1] == "web_sources"), "external_checks": external}


def _flag(value):
    return None if pd.isna(value) else int(bool(value))


def validate_logged_forecasts(database, season):
    """Settle every logged pregame forecast against stored outcomes; returns per-run metrics."""
    with closing(connect(database)) as conn:
        frame = pd.read_sql_query(
            """SELECT f.run_id, f.game_id, f.season, f.logged_at, f.kickoff, f.home_win_probability, f.projected_home_margin,
                      r.home_points, r.away_points, r.source AS result_source
               FROM forecast_log f JOIN results r ON r.game_id = f.game_id AND r.season = f.season
               WHERE f.season = ? AND r.completed = 1""", conn, params=(int(season),))
        if frame.empty:
            return {"runs": 0, "games": 0}
        # Only genuine pregame forecasts count.
        frame = frame[pd.to_datetime(frame.logged_at, utc=True, format="ISO8601") < pd.to_datetime(frame.kickoff, utc=True, format="ISO8601")].copy()
        if frame.empty:
            return {"runs": 0, "games": 0}
        frame["home_win"] = (frame.home_points > frame.away_points).astype(int)
        frame["actual_margin"] = frame.home_points - frame.away_points
        stamp = _now()
        conn.executemany(
            """INSERT OR REPLACE INTO forecast_validation(run_id, game_id, season, week, logged_at, kickoff, home_win_probability,
                 projected_home_margin, home_win, actual_margin, result_source, validated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(r.run_id, int(r.game_id), int(r.season), None, r.logged_at, r.kickoff, float(r.home_win_probability),
              None if pd.isna(r.projected_home_margin) else float(r.projected_home_margin), int(r.home_win), float(r.actual_margin),
              r.result_source, stamp) for r in frame.itertuples(index=False)])
        conn.commit()
        reports = {}
        for run_id, d in frame.groupby("run_id"):
            have = d.projected_home_margin.notna()
            m = metrics(d.home_win, d.home_win_probability,
                        d.projected_home_margin[have] if have.all() else None, d.actual_margin[have] if have.all() else None)
            reports[run_id] = m
            record_weekly_validation(conn, {**m, "season": int(season), "week": 0, "run_id": run_id, "status": "settled",
                                            "validated_at": stamp, "games": int(len(d))})
    log.info("validated %d games across %d logged forecast runs", len(frame), len(reports))
    return {"runs": len(reports), "games": int(len(frame)), "metrics": reports}


def recompute_from_database(database, python_metrics):
    """Recompute log loss, Brier score and margin error from the stored rows alone and compare.

    This reads ``forecast_validation`` back out of SQLite with plain SQL and recomputes each run's
    metrics with a separate implementation (no shared code with ``evaluate.metrics``), stores them in
    ``db_validation``, and refuses any disagreement with the metrics computed at validation time.
    """
    with closing(connect(database)) as conn:
        rows = conn.execute("SELECT run_id, home_win_probability, home_win, projected_home_margin, actual_margin FROM forecast_validation").fetchall()
        by_run = {}
        for run_id, p, y, pm, am in rows:
            by_run.setdefault(run_id, []).append((float(p), int(y), pm, float(am)))
        stamp = _now()
        out = []
        for run_id, items in by_run.items():
            p = np.clip(np.array([i[0] for i in items]), 1e-9, 1 - 1e-9)
            y = np.array([i[1] for i in items])
            ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            br = float(np.mean((p - y) ** 2))
            err = [abs(i[2] - i[3]) for i in items if i[2] is not None]
            mae = float(np.mean(err)) if len(err) == len(items) else None
            out.append((run_id, stamp, len(items), ll, br, mae))
            expected = python_metrics.get(run_id)
            if expected:
                for name, value in (("log_loss", ll), ("brier", br)):
                    if not np.isclose(value, expected[name], atol=1e-6):
                        raise ValueError(f"Database recomputation mismatch for {run_id}: {name} {value} vs {expected[name]}")
        conn.execute("DELETE FROM db_validation")
        conn.executemany("INSERT INTO db_validation(run_id, computed_at, games, log_loss, brier, margin_mae) VALUES(?,?,?,?,?,?)", out)
        conn.commit()
    return {"status": "verified", "runs": len(out)}


def run(database, data_root, season):
    """Ingest, validate past forecasts, and cross-check in R."""
    report = {"ingest": ingest(database, data_root, season)}
    validation = validate_logged_forecasts(database, season)
    report["validation"] = {k: v for k, v in validation.items() if k != "metrics"}
    report["recomputed"] = recompute_from_database(database, validation.get("metrics", {})) if validation.get("runs") else {"status": "nothing_to_validate"}
    return report
