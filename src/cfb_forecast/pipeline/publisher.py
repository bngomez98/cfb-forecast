"""Publish one forecast run to SQLite, the apps, the log, Datadog, and email.

Every completed calculation uses this module. The board records are the same
rows the React app, the Streamlit desk, and the email all read.
"""
from __future__ import annotations

import json
import ast
from contextlib import closing
import logging
import os
import smtplib
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from ..storage.database import connect
from ..paths import artifacts_root, database_path, project_root
from ..ingest.sources import source_registry

LOG = logging.getLogger("cfb_forecast.pipeline.publisher")
REQUIRED = (
    "game_id",
    "season",
    "home_team",
    "away_team",
    "kickoff",
    "expected_home_score",
    "expected_away_score",
    "home_win_probability",
)
RECIPIENT = "bngomez@ieee.org"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _structured(value):
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        # pandas CSVs of nested existing records use Python literal syntax.
        return ast.literal_eval(value)


def board_row(record: dict) -> dict:
    probability = float(record["home_win_probability"])
    home = str(record["home_team"])
    away = str(record["away_team"])
    winner = home if probability >= 0.5 else away
    margin = _number(record.get("projected_home_margin"))
    if margin is None:
        margin = float(record["expected_home_score"]) - float(record["expected_away_score"])
    model_spread = _number(record.get("model_spread"))
    if model_spread is None:
        model_spread = -margin
    try:
        from ..odds.decision import probability_to_american
        home_moneyline = probability_to_american(probability)
        away_moneyline = probability_to_american(1.0 - probability)
    except ValueError:
        home_moneyline = None
        away_moneyline = None
    return {
        "game_id": int(record["game_id"]),
        "as_of": record.get("as_of"),
        "forecast_mode": record.get("forecast_mode", "legacy_unspecified"),
        "season": int(record["season"]),
        "kickoff": str(record["kickoff"]),
        "away_team": away,
        "home_team": home,
        "expected_away_score": round(float(record["expected_away_score"]), 2),
        "expected_home_score": round(float(record["expected_home_score"]), 2),
        "projected_home_margin": round(float(margin), 2),
        "model_spread": round(float(model_spread), 2),
        "market_spread": _number(record.get("market_spread", record.get("home_spread"))),
        "home_cover_probability": _number(record.get("home_cover_probability")),
        "home_moneyline": home_moneyline,
        "away_moneyline": away_moneyline,
        "winner": winner,
        "confidence": round(max(probability, 1.0 - probability), 4),
        "home_win_probability": round(probability, 4),
        **({"model_variant": record["model_variant"]} if record.get("model_variant") else {}),
        **({"player_forecasts": _structured(record["player_forecasts"])} if isinstance(record.get("player_forecasts"), (dict, str)) else {}),
    }


def board_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"Forecast file is missing columns: {missing}")
    rows = [board_row(record) for record in frame.to_dict("records")]
    return pd.DataFrame(rows).sort_values(["kickoff", "game_id"]).reset_index(drop=True)


def _configure_log(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "forecast.log"
    if any(isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == str(path) for handler in LOG.handlers):
        return
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.setLevel(logging.INFO)
    LOG.addHandler(handler)


def register_sources(conn, season: int) -> int:
    count = 0
    for name, url in source_registry(season).items():
        conn.execute(
            """INSERT INTO source_registry(source_key, provider, source_url, schema_version)
               VALUES(?,?,?,?)
               ON CONFLICT(source_key) DO UPDATE SET
                 provider=excluded.provider,
                 source_url=excluded.source_url,
                 schema_version=excluded.schema_version""",
            (name, name, url, "catalog"),
        )
        count += 1
    conn.commit()
    return count


def _validate_published_rows(conn, run_id, rows):
    from ..storage.evidence import canonical_json
    for row in rows.to_dict("records"):
        old = conn.execute("SELECT payload_json FROM forecast_log WHERE run_id=? AND game_id=?", (run_id, int(row["game_id"]))).fetchone()
        if old:
            original = json.loads(old[0])
            comparable = {key: row.get(key) for key in original}
            if canonical_json(original) != canonical_json(comparable):
                raise ValueError(f"Published forecast changed for {run_id}/{row['game_id']}; use a new run ID")


def write_forecast_log(conn, run_id: str, rows: pd.DataFrame, logged_at: str) -> int:
    _validate_published_rows(conn, run_id, rows)
    for row in rows.to_dict("records"):
        conn.execute(
            """INSERT INTO forecast_log(
                 run_id, season, game_id, kickoff, home_team, away_team,
                 expected_home_score, expected_away_score, winner, confidence,
                 home_win_probability, logged_at, payload_json,
                 projected_home_margin, model_spread, market_spread, home_cover_probability,
                 home_moneyline, away_moneyline)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(run_id, game_id) DO NOTHING""",
            (
                run_id, int(row["season"]), int(row["game_id"]), row["kickoff"],
                row["home_team"], row["away_team"], row["expected_home_score"],
                row["expected_away_score"], row["winner"], row["confidence"],
                row["home_win_probability"], logged_at, json.dumps(row),
                row.get("projected_home_margin"), row.get("model_spread"),
                row.get("market_spread"), row.get("home_cover_probability"),
                row.get("home_moneyline"), row.get("away_moneyline"),
            ),
        )
    conn.commit()
    return len(rows)


def ship_datadog(events: list[dict], log_dir: Path) -> dict:
    """Keep a Datadog-shaped JSON log locally, and ship it when DD_API_KEY is set."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "datadog.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    key = os.environ.get("DD_API_KEY")
    if not key:
        return {"status": "local_only", "path": str(path), "events": len(events)}
    site = os.environ.get("DD_SITE", "datadoghq.com")
    body = json.dumps(events).encode()
    request = urllib.request.Request(
        f"https://http-intake.logs.{site}/api/v2/logs",
        data=body,
        headers={
            "DD-API-KEY": key,
            "Content-Type": "application/json",
            "User-Agent": "cfb-forecast/0.4",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {"status": "shipped", "http_status": response.status, "events": len(events)}
    except Exception as exc:
        LOG.warning("Datadog intake failed: %s", exc)
        return {"status": "failed", "error": type(exc).__name__, "path": str(path)}


def _email_body(board: dict) -> str:
    lines = [
        f"CFB forecast update {board['run_id']}",
        f"Season {board['season']} · {board['games']} games · as of {board['as_of']}",
        f"Model: {board.get('model') or 'repository score head'}",
        "",
        "Kickoff | Away | Home | Score | Winner | Confidence",
    ]
    for game in board["forecasts"][:40]:
        lines.append(
            f"{game['kickoff']} | {game['away_team']} {game['expected_away_score']:.1f} @ "
            f"{game['home_team']} {game['expected_home_score']:.1f} | "
            f"{game['winner']} | {game['confidence']:.0%}"
        )
    if board["games"] > 40:
        lines.append(f"... {board['games'] - 40} more games are on the forecast board.")
    lines.append("")
    lines.append("These are model predictions, not final results.")
    return "\n".join(lines)


def send_forecast_email(board: dict, outbox: Path) -> dict:
    """Send the update to the forecast mailbox. Without SMTP settings, keep the message."""
    outbox.mkdir(parents=True, exist_ok=True)
    body = _email_body(board)
    stamp = board["logged_at"].replace(":", "").replace("-", "")
    message_path = outbox / f"{stamp}.txt"
    message_path.write_text(body, encoding="utf-8")
    host = os.environ.get("CFB_SMTP_HOST")
    user = os.environ.get("CFB_SMTP_USER")
    password = os.environ.get("CFB_SMTP_PASSWORD")
    recipient = os.environ.get("CFB_FORECAST_EMAIL_TO", RECIPIENT)
    sender = os.environ.get("CFB_SMTP_FROM", user or RECIPIENT)
    if not host or not password:
        LOG.info("Forecast email saved for %s; SMTP is not configured", recipient)
        return {"status": "saved", "to": recipient, "path": str(message_path)}
    message = EmailMessage()
    message["Subject"] = f"CFB forecast update {board['run_id']}"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)
    port = int(os.environ.get("CFB_SMTP_PORT", "587"))
    try:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            if user:
                smtp.login(user, password)
            smtp.send_message(message)
    except Exception as exc:
        LOG.warning("Forecast email was not sent: %s", exc)
        return {"status": "failed", "to": recipient, "error": type(exc).__name__, "path": str(message_path)}
    LOG.info("Forecast email sent to %s", recipient)
    return {"status": "sent", "to": recipient, "path": str(message_path)}


def publish_forecast_update(
    run_dir: str | Path,
    artifacts: str | Path | None = None,
    database: str | Path | None = None,
    web_json: str | Path | None = None,
    run_id: str | None = None,
    season: int | None = None,
    fetch_sources: bool = False,
) -> dict:
    """Write the forecast board everywhere a completed run is shown."""
    run_dir = Path(run_dir)
    artifacts = Path(artifacts) if artifacts is not None else artifacts_root()
    database = Path(database) if database is not None else database_path()
    web_json = Path(web_json) if web_json is not None else project_root() / "web" / "public" / "forecasts.json"
    prediction_path = run_dir / "forecasts.csv"
    if not prediction_path.exists():
        return {"status": "skipped", "reason": "forecasts.csv missing"}
    frame = pd.read_csv(prediction_path)
    if any(column not in frame.columns for column in REQUIRED):
        return {"status": "skipped", "reason": "forecast file has no score board"}
    if frame.empty:
        return {"status": "skipped", "reason": "forecast file is empty"}
    rows = board_frame(frame)
    logged_at = utc_now()
    run_id = run_id or run_dir.name
    season = int(season or rows.season.iloc[0])
    manifest_path = run_dir / "weekly_manifest.json"
    manifest = {}
    if not manifest_path.exists():
        manifest_path = run_dir / "forecast_manifest.json"
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = loaded[0] if isinstance(loaded, list) and loaded else loaded
    model = None
    if "prediction_method" in frame.columns and frame.prediction_method.notna().any():
        model = str(frame.prediction_method.dropna().iloc[0])
    sources = [{"name": name, "url": url} for name, url in source_registry(season).items()]
    board = {
        "run_id": run_id,
        "season": season,
        "week": manifest.get("week"),
        "as_of": manifest.get("as_of") or logged_at,
        "logged_at": logged_at,
        "games": int(len(rows)),
        "model": model,
        "sources": sources,
        "forecasts": rows.to_dict("records"),
    }
    # Validate and freeze before changing any public/latest surface.
    from ..storage.evidence import freeze_forecast
    with closing(connect(database)) as evidence_conn:
        _validate_published_rows(evidence_conn, run_id, rows)
        freeze_forecast(evidence_conn, run_id, frame, run_dir, artifacts / "evidence")
    log_dir = artifacts / "logs"
    _configure_log(log_dir)
    latest = artifacts / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(latest / "forecasts.csv", index=False)
    (latest / "forecast_board.json").write_text(json.dumps(board, indent=2), encoding="utf-8")
    (latest / "run_manifest.json").write_text(json.dumps({
        "status": "completed",
        "run_id": run_id,
        "season": season,
        "games": board["games"],
        "as_of": board["as_of"],
        "completed_at": logged_at,
    }, indent=2), encoding="utf-8")
    web_json.parent.mkdir(parents=True, exist_ok=True)
    web_json.write_text(json.dumps(board, indent=2), encoding="utf-8")
    with (log_dir / "forecast_updates.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "logged_at": logged_at,
            "run_id": run_id,
            "season": season,
            "games": board["games"],
        }) + "\n")
    conn = connect(database)
    try:
        registered = register_sources(conn, season)
        stored = write_forecast_log(conn, run_id, rows, logged_at)
    finally:
        conn.close()
    source_ingest = None
    if fetch_sources:
        from ..ingest.sources import ingest_catalog
        source_ingest = ingest_catalog(season, artifacts / "source_ingest")
        (log_dir / "source_ingest.json").write_text(json.dumps(source_ingest, indent=2), encoding="utf-8")
    datadog = ship_datadog([{
        "ddsource": "cfb-forecast",
        "service": "cfb-forecast",
        "status": "info",
        "message": f"Published {board['games']} forecasts for {run_id}",
        "run_id": run_id,
        "season": season,
        "games": board["games"],
        "logged_at": logged_at,
    }], log_dir)
    email = send_forecast_email(board, log_dir / "email_outbox")
    LOG.info("Published %s games for %s", board["games"], run_id)
    return {
        "status": "published",
        "run_id": run_id,
        "games": board["games"],
        "sources": registered,
        "logged": stored,
        "web": str(web_json),
        "database": str(database),
        "datadog": datadog,
        "email": email,
        "source_ingest": source_ingest,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Publish a completed forecast run to every surface")
    parser.add_argument("--run", required=True, help="Directory containing forecasts.csv")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--web", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    report = publish_forecast_update(args.run, args.artifacts, args.database, args.web, args.run_id)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"published", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
