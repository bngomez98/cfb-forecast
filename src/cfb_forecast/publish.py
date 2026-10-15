"""Publish a completed forecast board to SQLite, the web apps, Datadog and email."""
from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from contextlib import closing
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from .database import connect
from .paths import artifacts_root, project_root

RECIPIENT = "bngomez@ieee.org"
DATADOG_URL = "https://http-intake.logs.datadoghq.com/api/v2/logs"


def _records(run_dir):
    run_dir = Path(run_dir)
    for name in ("forecasts.json", "forecasts.csv"):
        path = run_dir / name
        if path.exists():
            if name.endswith(".json"):
                return json.loads(path.read_text())
            frame = pd.read_csv(path)
            return frame.astype(object).where(pd.notna(frame), None).to_dict("records")
    raise FileNotFoundError(f"No forecasts.json or forecasts.csv in {run_dir}")


def board_rows(records):
    """Kickoff, teams, expected score, winner and confidence for each game."""
    rows = []
    for r in records:
        p = float(r["home_win_probability"])
        home_wins = p >= 0.5
        rows.append({
            "game_id": int(r["game_id"]),
            "season": int(r["season"]) if r.get("season") is not None else None,
            "kickoff": r.get("kickoff") or r.get("start_date"),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "expected_home_score": float(r["expected_home_score"]),
            "expected_away_score": float(r["expected_away_score"]),
            "home_win_probability": p,
            "predicted_winner": r["home_team"] if home_wins else r["away_team"],
            "confidence": p if home_wins else 1.0 - p,
        })
    return sorted(rows, key=lambda row: (str(row["kickoff"]), row["game_id"]))


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    tmp.replace(path)


def _log_to_database(database, run_id, rows, records, published_at):
    with closing(connect(database)) as conn:
        conn.executemany(
            """INSERT INTO forecast_log(run_id,game_id,season,kickoff,home_team,away_team,
               expected_home_score,expected_away_score,home_win_probability,predicted_winner,
               confidence,published_at,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(run_id,game_id) DO UPDATE SET
               expected_home_score=excluded.expected_home_score,
               expected_away_score=excluded.expected_away_score,
               home_win_probability=excluded.home_win_probability,
               predicted_winner=excluded.predicted_winner,confidence=excluded.confidence,
               published_at=excluded.published_at,payload_json=excluded.payload_json""",
            [(run_id, r["game_id"], r["season"], r["kickoff"], r["home_team"], r["away_team"],
              r["expected_home_score"], r["expected_away_score"], r["home_win_probability"],
              r["predicted_winner"], r["confidence"], published_at, json.dumps(rec, default=str))
             for r, rec in zip(rows, records)],
        )
        conn.commit()


def _datadog_event(root, run_id, board, published_at):
    event = {"ddsource": "cfb-forecast", "service": "cfb-forecast", "status": "info",
             "message": f"Published forecast board {run_id} ({len(board)} games)",
             "run_id": run_id, "games": len(board), "published_at": published_at}
    log = root / "logs" / "datadog.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(event) + "\n")
    key = os.environ.get("DD_API_KEY")
    if not key:
        return "log_only"
    request = urllib.request.Request(DATADOG_URL, data=json.dumps([event]).encode(),
        headers={"Content-Type": "application/json", "DD-API-KEY": key}, method="POST")
    try:
        urllib.request.urlopen(request, timeout=30).close()
    except Exception as error:  # intake failure must not lose the published board
        return f"intake_failed: {error}"
    return "sent"


def _email(root, run_id, board, send):
    lines = [f"{b['kickoff']}  {b['away_team']} at {b['home_team']}: "
             f"{b['expected_away_score']:.1f}-{b['expected_home_score']:.1f}, "
             f"{b['predicted_winner']} ({b['confidence']:.0%})" for b in board]
    message = EmailMessage()
    message["To"] = RECIPIENT
    message["From"] = os.environ.get("CFB_SMTP_FROM", RECIPIENT)
    message["Subject"] = f"CFB forecast {run_id}"
    message.set_content("\n".join(lines) or "No games on this slate.")
    outbox = root / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    (outbox / f"{run_id}.eml").write_bytes(bytes(message))
    host, password = os.environ.get("CFB_SMTP_HOST"), os.environ.get("CFB_SMTP_PASSWORD")
    if not (send and host and password):
        return "outbox_only"
    try:
        with smtplib.SMTP(host, int(os.environ.get("CFB_SMTP_PORT", "587")), timeout=30) as smtp:
            smtp.starttls()
            smtp.login(os.environ.get("CFB_SMTP_USER", message["From"]), password)
            smtp.send_message(message)
    except Exception as error:
        return f"send_failed: {error}"
    return "sent"


def publish_forecast_update(run_dir, run_id, artifacts=None, database=None, web_public=None, send_email=True):
    """Store one calculated board in SQLite and fan it out to every consumer."""
    root = Path(artifacts) if artifacts else artifacts_root()
    database = database or os.environ.get("CFB_DATABASE") or root / "cfb_forecast.sqlite"
    web_public = Path(web_public) if web_public else project_root() / "web" / "public"
    records = _records(run_dir)
    board = board_rows(records)
    published_at = datetime.now(timezone.utc).isoformat()
    by_id = {int(r["game_id"]): r for r in records}
    _log_to_database(database, run_id, board, [by_id[b["game_id"]] for b in board], published_at)
    payload = {"run_id": run_id, "published_at": published_at, "games": board}
    _write_json(root / "latest" / "forecast_board.json", payload)
    _write_json(web_public / "forecasts.json", payload)
    return {"run_id": run_id, "games": len(board), "database": str(database),
            "datadog": _datadog_event(root, run_id, board, published_at),
            "email": _email(root, run_id, board, send_email)}
