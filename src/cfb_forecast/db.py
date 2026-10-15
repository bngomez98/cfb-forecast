"""PostgreSQL persistence layer — the single canonical database.

Vocabulary contract: ``docs/GLOSSARY.md``.  Table/column names here match
``schema/schema.sql`` exactly; do not introduce synonyms (glossary §7).

Connection string resolution (in order):

1. explicit ``dsn`` argument,
2. ``CFB_POSTGRES_DSN`` environment variable
   (e.g. ``postgresql://cfb:cfb@localhost:5432/cfb_forecast``),
3. ``postgres://cfb:cfb@localhost:5432/cfb_forecast`` from docker-compose.

The app uses psycopg 3 with a plain connection per operation; the daily cycle
is low-frequency and correctness (transactions, CHECK constraints, the
cutoff trigger) matters more than pooling throughput.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Mapping, Sequence

import psycopg
from psycopg.rows import dict_row

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "schema",
    "schema.sql",
)
DEFAULT_DSN_ENV = "CFB_POSTGRES_DSN"

#: Canonical expert registry names (glossary §4).  Kept in sync with the
#: Python registries in heads.py / timeheads.py / craft_model.py.
EXPERT_NAMES = (
    "base_rate",
    "winner_elo",
    "margin_elo",
    "ridge",
    "logit",
    "gbm",
    "ts_arima",
    "craft",
)


def resolve_dsn(dsn: str | None = None) -> str:
    """Return the DSN for the single canonical database."""
    resolved = dsn or os.environ.get(DEFAULT_DSN_ENV)
    if not resolved:
        raise ValueError(
            "No PostgreSQL DSN configured. Set CFB_POSTGRES_DSN or pass dsn=... "
            "(see docs/GLOSSARY.md §2 'Single database')."
        )
    return resolved


@contextmanager
def connect(dsn: str | None = None) -> Iterator[psycopg.Connection]:
    """Context-managed connection with autocommit off and dict rows on."""
    with psycopg.connect(resolve_dsn(dsn), row_factory=dict_row) as conn:
        yield conn


def init_schema(conn: psycopg.Connection) -> None:
    """Apply the declarative schema (idempotent: CREATE ... IF NOT EXISTS)."""
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        conn.execute(fh.read())
    conn.commit()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Ingestion upserts
# ---------------------------------------------------------------------------


def upsert_games(conn: psycopg.Connection, games: Iterable[Mapping[str, Any]]) -> int:
    """Insert/update game rows. Raw scores are stored untouched (no smoothing)."""
    sql = """
        INSERT INTO games (espn_game_id, season, week, kickoff_at, home_team, away_team,
                           venue, site_type, home_points, away_points, win_score, lose_score,
                           margin, total, winner_team, status, source, captured_at)
        VALUES (%(espn_game_id)s, %(season)s, %(week)s, %(kickoff_at)s, %(home_team)s,
                %(away_team)s, %(venue)s, %(site_type)s, %(home_points)s, %(away_points)s,
                %(win_score)s, %(lose_score)s, %(margin)s, %(total)s, %(winner_team)s,
                %(status)s, %(source)s, %(captured_at)s)
        ON CONFLICT (espn_game_id) DO UPDATE SET
            status = EXCLUDED.status,
            home_points = EXCLUDED.home_points,
            away_points = EXCLUDED.away_points,
            win_score = EXCLUDED.win_score,
            lose_score = EXCLUDED.lose_score,
            margin = EXCLUDED.margin,
            total = EXCLUDED.total,
            winner_team = EXCLUDED.winner_team,
            captured_at = EXCLUDED.captured_at
    """
    n = 0
    for g in games:
        row = {
            "espn_game_id": str(g["espn_game_id"]),
            "season": int(g["season"]),
            "week": int(g["week"]),
            "kickoff_at": g.get("kickoff_at"),
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "venue": g.get("venue"),
            "site_type": g.get("site_type", "home"),
            "home_points": g.get("home_points"),
            "away_points": g.get("away_points"),
            "win_score": g.get("win_score"),
            "lose_score": g.get("lose_score"),
            "margin": g.get("margin"),
            "total": g.get("total"),
            "winner_team": g.get("winner_team"),
            "status": g.get("status", "scheduled"),
            "source": g.get("source", "espn"),
            "captured_at": g.get("captured_at", _utcnow()),
        }
        conn.execute(sql, row)
        n += 1
    conn.commit()
    return n


def upsert_team_games(conn: psycopg.Connection, rows: Iterable[Mapping[str, Any]]) -> int:
    sql = """
        INSERT INTO team_games (season, team, week, espn_game_id, side, opponent,
                                points_for, points_against, margin, total, won)
        VALUES (%(season)s, %(team)s, %(week)s, %(espn_game_id)s, %(side)s, %(opponent)s,
                %(points_for)s, %(points_against)s, %(margin)s, %(total)s, %(won)s)
        ON CONFLICT DO NOTHING
    """
    n = 0
    for r in rows:
        conn.execute(sql, {k: r[k] for k in (
            "season", "team", "week", "espn_game_id", "side", "opponent",
            "points_for", "points_against", "margin", "total", "won")})
        n += 1
    conn.commit()
    return n


def upsert_weather(conn: psycopg.Connection, rows: Iterable[Mapping[str, Any]]) -> int:
    sql = """
        INSERT INTO weather_observations (espn_game_id, temperature_f, wind_mph,
                                          precipitation_in, conditions, source, captured_at)
        VALUES (%(espn_game_id)s, %(temperature_f)s, %(wind_mph)s, %(precipitation_in)s,
                %(conditions)s, %(source)s, %(captured_at)s)
        ON CONFLICT (espn_game_id) DO UPDATE SET
            temperature_f = EXCLUDED.temperature_f,
            wind_mph = EXCLUDED.wind_mph,
            precipitation_in = EXCLUDED.precipitation_in,
            conditions = EXCLUDED.conditions,
            captured_at = EXCLUDED.captured_at
    """
    n = 0
    for r in rows:
        payload = {
            "espn_game_id": str(r["espn_game_id"]),
            "temperature_f": r.get("temperature_f"),
            "wind_mph": r.get("wind_mph"),
            "precipitation_in": r.get("precipitation_in"),
            "conditions": r.get("conditions"),
            "source": r.get("source", "open_meteo"),
            "captured_at": r.get("captured_at", _utcnow()),
        }
        conn.execute(sql, payload)
        n += 1
    conn.commit()
    return n


def upsert_injuries(conn: psycopg.Connection, rows: Iterable[Mapping[str, Any]]) -> int:
    sql = """
        INSERT INTO injury_reports (season, week, team, player, position, status,
                                    injury, captured_at)
        VALUES (%(season)s, %(week)s, %(team)s, %(player)s, %(position)s, %(status)s,
                %(injury)s, %(captured_at)s)
        ON CONFLICT (season, week, team, player) DO UPDATE SET
            status = EXCLUDED.status,
            injury = EXCLUDED.injury,
            captured_at = EXCLUDED.captured_at
    """
    n = 0
    for r in rows:
        conn.execute(sql, {
            "season": int(r["season"]), "week": int(r["week"]), "team": r["team"],
            "player": r["player"], "position": r.get("position"),
            "status": r["status"], "injury": r.get("injury"),
            "captured_at": r.get("captured_at", _utcnow()),
        })
        n += 1
    conn.commit()
    return n


# ---------------------------------------------------------------------------
# Prediction ledger
# ---------------------------------------------------------------------------


def insert_prediction(
    conn: psycopg.Connection,
    *,
    espn_game_id: str,
    expert: str,
    model_version: str,
    cutoff_at: datetime,
    win_prob: float,
    input_hash: str,
    margin_points: float | None = None,
    total_points: float | None = None,
) -> int | None:
    """Seal one prospective prediction. The DB trigger rejects cutoff >= kickoff."""
    if expert not in EXPERT_NAMES:
        raise ValueError(f"unknown expert {expert!r}; registry is {EXPERT_NAMES} (glossary §4)")
    cur = conn.execute(
        """
        INSERT INTO predictions (espn_game_id, expert, model_version, cutoff_at,
                                 win_prob, margin_points, total_points, input_hash)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (espn_game_id, expert, cutoff_at) DO NOTHING
        RETURNING prediction_id
        """,
        (str(espn_game_id), expert, model_version, cutoff_at,
         float(win_prob), margin_points, total_points, input_hash),
    )
    row = cur.fetchone()
    conn.commit()
    return row["prediction_id"] if row else None


def settle_prediction(
    conn: psycopg.Connection,
    *,
    prediction_id: int,
    home_win: bool,
    margin: int,
    total: int,
    abs_margin_error: float | None = None,
    abs_total_error: float | None = None,
) -> None:
    """Join one sealed prediction to its truth and append to the learning ledger."""
    pred = conn.execute(
        "SELECT p.*, g.home_team FROM predictions p JOIN games g USING (espn_game_id) "
        "WHERE p.prediction_id = %s",
        (prediction_id,),
    ).fetchone()
    if pred is None:
        raise ValueError(f"prediction_id {prediction_id} not found")
    eps = 1e-9
    p = min(max(float(pred["win_prob"]), eps), 1 - eps)
    log_loss = -(float(home_win) * _ln(p) + (1 - float(home_win)) * _ln(1 - p))
    conn.execute(
        """
        INSERT INTO settled_truth (espn_game_id, settled_at, home_win, margin, total)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (espn_game_id) DO NOTHING
        """,
        (pred["espn_game_id"], _utcnow(), home_win, margin, total),
    )
    conn.execute(
        """
        INSERT INTO prediction_outcomes (prediction_id, espn_game_id, expert, win_prob,
                                         home_win, abs_margin_error, abs_total_error,
                                         log_loss, settled_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (prediction_id) DO NOTHING
        """,
        (prediction_id, pred["espn_game_id"], pred["expert"], p, home_win,
         abs_margin_error, abs_total_error, log_loss, _utcnow()),
    )
    conn.commit()


def _ln(x: float) -> float:
    import math

    return math.log(x)


# ---------------------------------------------------------------------------
# Blend weights & CRAFT parameters
# ---------------------------------------------------------------------------


def record_blend_weights(
    conn: psycopg.Connection,
    weights: Mapping[str, float],
    diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Snapshot learned blend weights (glossary §4) after a settle run."""
    stamp = _utcnow()
    diagnostics = diagnostics or {}
    for expert, weight in weights.items():
        d = diagnostics.get(expert, {})
        conn.execute(
            """
            INSERT INTO blend_weights (updated_at, expert, weight, hit_rate, brier_score,
                                       log_loss, calibration_slope, n_settled)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (stamp, expert, float(weight), d.get("hit_rate"), d.get("brier_score"),
             d.get("log_loss"), d.get("calibration_slope"), int(d.get("n_settled", 0))),
        )
    conn.commit()


def latest_blend_weights(conn: psycopg.Connection) -> dict[str, float]:
    row = conn.execute("SELECT max(updated_at) AS t FROM blend_weights").fetchone()
    if row is None or row["t"] is None:
        return {}
    rows = conn.execute(
        "SELECT expert, weight FROM blend_weights WHERE updated_at = %s", (row["t"],)
    ).fetchall()
    return {r["expert"]: float(r["weight"]) for r in rows}


def save_craft_parameters(conn: psycopg.Connection, params: Mapping[str, Any]) -> int:
    """Persist one CRAFT fit's hyperparameters (defaults aligned with the paper)."""
    cols = [
        "horizon_weeks", "lookback_games", "embed_dim", "koopman_lambda",
        "hier_children", "demand_beta", "alpha1", "alpha2", "alpha3",
        "batch_size", "learning_rate", "train_epochs", "wmape", "notes",
    ]
    # Omitted keys fall through to the schema DEFAULTs (paper-aligned values).
    present = [c for c in cols if params.get(c) is not None]
    vals = [params[c] for c in present]
    row = conn.execute(
        f"INSERT INTO craft_parameters ({', '.join(present)}) "
        f"VALUES ({', '.join(['%s'] * len(present))}) RETURNING fit_id",
        vals,
    ).fetchone()
    conn.commit()
    return int(row["fit_id"])


def load_craft_parameters(conn: psycopg.Connection) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM craft_parameters ORDER BY fitted_at DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Cycle bookkeeping
# ---------------------------------------------------------------------------


def start_cycle(conn: psycopg.Connection, kind: str = "daily") -> int:
    row = conn.execute(
        "INSERT INTO cycle_runs (kind) VALUES (%s) RETURNING run_id", (kind,)
    ).fetchone()
    conn.commit()
    return int(row["run_id"])


def finish_cycle(
    conn: psycopg.Connection,
    run_id: int,
    *,
    games_refreshed: int = 0,
    games_settled: int = 0,
    predictions_made: int = 0,
    errors: Sequence[str] = (),
    ok: bool = True,
) -> None:
    conn.execute(
        """
        UPDATE cycle_runs SET finished_at = %s, games_refreshed = %s, games_settled = %s,
               predictions_made = %s, errors = %s::jsonb, ok = %s
        WHERE run_id = %s
        """,
        (_utcnow(), games_refreshed, games_settled, predictions_made,
         json.dumps(list(errors)), ok, run_id),
    )
    conn.commit()


def record_cleaning_run(
    conn: psycopg.Connection,
    *,
    source: str,
    rows_in: int,
    rows_clean: int,
    reason_breakdown: Mapping[str, int],
) -> None:
    conn.execute(
        """
        INSERT INTO cleaning_log (source, rows_in, rows_clean, rows_quarantined,
                                  reason_breakdown)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        """,
        (source, rows_in, rows_clean, rows_in - rows_clean,
         json.dumps(dict(reason_breakdown))),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Reads used by forecasting / dashboard
# ---------------------------------------------------------------------------


def fetch_team_history(
    conn: psycopg.Connection, team: str, through_week: int, season: int, limit: int = 30
) -> list[dict[str, Any]]:
    """Completed team-games strictly before/at through_week (cutoff-respecting)."""
    rows = conn.execute(
        """
        SELECT week, opponent, points_for, points_against, margin, total, won
        FROM team_games
        WHERE team = %s AND season = %s AND week <= %s
        ORDER BY week DESC, espn_game_id
        LIMIT %s
        """,
        (team, season, through_week, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def fetch_upcoming(
    conn: psycopg.Connection, season: int, after_week: int, limit: int = 200
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT espn_game_id, week, kickoff_at, home_team, away_team, venue, site_type
        FROM games
        WHERE season = %s AND week > %s AND status = 'scheduled'
        ORDER BY kickoff_at NULLS LAST, week
        LIMIT %s
        """,
        (season, after_week, limit),
    ).fetchall()
    return [dict(r) for r in rows]
