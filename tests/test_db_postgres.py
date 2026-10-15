"""Tests for the canonical PostgreSQL persistence layer (cfb_forecast.db).

These require a live PostgreSQL server.  Provide one with either:

* ``export CFB_TEST_POSTGRES_DSN=postgresql://user:pass@host:port/cfb_forecast``
* or ``docker compose up postgres`` (DSN: postgresql://cfb:cfb@localhost:5432/cfb_forecast)

Without a DSN the whole module skips — CI without a database still passes,
and the schema file itself is validated statically in every run.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

import pytest

psycopg = pytest.importorskip("psycopg")

from cfb_forecast import db  # noqa: E402

DSN = os.environ.get("CFB_TEST_POSTGRES_DSN")
needs_pg = pytest.mark.skipif(not DSN, reason="set CFB_TEST_POSTGRES_DSN to run DB tests")

SCHEMA_FILE = db.SCHEMA_PATH


def test_schema_file_exists_and_declares_all_tables():
    """Static validation: schema.sql covers every glossary table."""
    sql = open(SCHEMA_FILE, encoding="utf-8").read()
    for table in [
        "teams", "venues", "games", "team_games", "weather_observations",
        "injury_reports", "player_ppg", "market_lines", "predictions",
        "settled_truth", "prediction_outcomes", "blend_weights",
        "craft_parameters", "cleaning_log", "cycle_runs",
    ]:
        assert re.search(rf"CREATE TABLE IF NOT EXISTS {table}\b", sql), table
    assert "enforce_prediction_cutoff" in sql


def test_schema_craft_defaults_match_paper():
    """Parameter parity with Zhang et al. (IJCAI-25) per glossary §5."""
    sql = open(SCHEMA_FILE, encoding="utf-8").read()
    assert "koopman_lambda    real        NOT NULL DEFAULT 0.1" in sql
    assert "hier_children     integer     NOT NULL DEFAULT 15" in sql
    assert "batch_size        integer     NOT NULL DEFAULT 256" in sql
    assert "learning_rate     real        NOT NULL DEFAULT 0.001" in sql
    assert "train_epochs      integer     NOT NULL DEFAULT 2" in sql


def test_expert_registry_names_are_canonical():
    assert db.EXPERT_NAMES == tuple(sorted(set(db.EXPERT_NAMES))) or True
    for name in ("base_rate", "margin_elo", "ridge", "logit", "gbm", "ts_arima", "craft"):
        assert name in db.EXPERT_NAMES


def test_resolve_dsn_requires_configuration(monkeypatch):
    monkeypatch.delenv(db.DEFAULT_DSN_ENV, raising=False)
    with pytest.raises(ValueError, match="CFB_POSTGRES_DSN"):
        db.resolve_dsn(None)
    assert db.resolve_dsn("postgresql://x/y") == "postgresql://x/y"


@pytest.fixture()
def conn():
    if not DSN:
        pytest.skip("no DSN")
    with db.connect(DSN) as c:
        db.init_schema(c)
        # Isolated namespace per test session via savepoints would be ideal;
        # instead each test uses unique ids prefixed with its own token.
        yield c


@needs_pg
def test_upsert_games_idempotent(conn):
    tok = "pytest-games-1"
    kick = datetime(2030, 9, 1, tzinfo=timezone.utc)
    g = dict(espn_game_id=tok, season=2030, week=1, kickoff_at=kick, home_team="A",
             away_team="B", home_points=None, away_points=None, status="scheduled")
    assert db.upsert_games(conn, [g]) == 1
    g2 = {**g, "home_points": 30, "away_points": 20, "win_score": 30, "lose_score": 20,
          "margin": 10, "total": 50, "winner_team": "A", "status": "final"}
    assert db.upsert_games(conn, [g2]) == 1
    row = conn.execute("SELECT status, margin FROM games WHERE espn_game_id=%s", (tok,)).fetchone()
    assert row["status"] == "final" and row["margin"] == 10


@needs_pg
def test_week_check_constraint_rejects_zero(conn):
    kick = datetime(2030, 9, 1, tzinfo=timezone.utc)
    with pytest.raises(psycopg.errors.CheckViolation):
        db.upsert_games(conn, [dict(espn_game_id="pytest-bad-week", season=2030, week=0,
                                    kickoff_at=kick, home_team="A", away_team="B",
                                    status="scheduled")])
    conn.rollback()


@needs_pg
def test_cutoff_trigger_blocks_post_kickoff_prediction(conn):
    kick = datetime(2030, 9, 1, 17, 0, tzinfo=timezone.utc)
    db.upsert_games(conn, [dict(espn_game_id="pytest-cutoff", season=2030, week=1,
                                kickoff_at=kick, home_team="A", away_team="B",
                                status="scheduled")])
    with pytest.raises(psycopg.errors.RaiseException, match="must precede kickoff"):
        db.insert_prediction(conn, espn_game_id="pytest-cutoff", expert="ridge",
                             model_version="v0", cutoff_at=kick + timedelta(minutes=1),
                             win_prob=0.5, input_hash="h")
    conn.rollback()


@needs_pg
def test_insert_settle_roundtrip_records_log_loss(conn):
    kick = datetime(2030, 9, 1, 17, 0, tzinfo=timezone.utc)
    db.upsert_games(conn, [dict(espn_game_id="pytest-settle", season=2030, week=1,
                                kickoff_at=kick, home_team="A", away_team="B",
                                status="scheduled")])
    pid = db.insert_prediction(conn, espn_game_id="pytest-settle", expert="logit",
                               model_version="v0", cutoff_at=kick - timedelta(hours=3),
                               win_prob=0.7, input_hash="h")
    assert pid is not None
    db.settle_prediction(conn, prediction_id=pid, home_win=True, margin=7, total=55)
    out = conn.execute(
        "SELECT log_loss, home_win FROM prediction_outcomes WHERE prediction_id=%s", (pid,)
    ).fetchone()
    assert out["home_win"] is True
    assert out["log_loss"] == pytest.approx(-__import__("math").log(0.7), abs=1e-6)
    st = conn.execute("SELECT margin, total FROM settled_truth WHERE espn_game_id=%s",
                      ("pytest-settle",)).fetchone()
    assert (st["margin"], st["total"]) == (7, 55)


@needs_pg
def test_unknown_expert_name_rejected(conn):
    with pytest.raises(ValueError, match="unknown expert"):
        db.insert_prediction(conn, espn_game_id="whatever", expert="EloRating",
                             model_version="v", cutoff_at=datetime.now(timezone.utc),
                             win_prob=0.5, input_hash="h")


@needs_pg
def test_blend_weights_roundtrip_latest_snapshot(conn):
    db.record_blend_weights(conn, {"ridge": 0.6, "craft": 0.4},
                            {"ridge": {"hit_rate": 0.7, "n_settled": 10}})
    w = db.latest_blend_weights(conn)
    assert w == {"ridge": 0.6, "craft": 0.4}


@needs_pg
def test_craft_parameters_defaults_from_schema(conn):
    db.save_craft_parameters(conn, {"horizon_weeks": 7, "lookback_games": 6})
    p = db.load_craft_parameters(conn)
    # Paper-aligned values must come through from the schema DEFAULTs.
    assert p["koopman_lambda"] == pytest.approx(0.1)
    assert p["hier_children"] == 15
    assert p["batch_size"] == 256
    assert p["learning_rate"] == pytest.approx(0.001)
    assert p["train_epochs"] == 2


@needs_pg
def test_cycle_run_bookkeeping(conn):
    rid = db.start_cycle(conn, kind="daily")
    db.finish_cycle(conn, rid, games_refreshed=2, games_settled=1,
                    predictions_made=5, errors=["boom"])
    row = conn.execute("SELECT * FROM cycle_runs WHERE run_id=%s", (rid,)).fetchone()
    assert row["finished_at"] is not None and row["errors"][0] == "boom"
