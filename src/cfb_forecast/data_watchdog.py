"""Data integrity watchdog for CFB forecast runtime files."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .database import connect

DEFAULT_PATTERNS = (
    "schedule_*.csv",
    "adv_team_*.parquet",
    "pbp_*.parquet",
    "source_manifest.json",
)

WATCHDOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS data_files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    modified_ns INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS watchdog_events (
    id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    path TEXT,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_data_files_status ON data_files(status);
CREATE INDEX IF NOT EXISTS idx_watchdog_events_time ON watchdog_events(observed_at);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_files(data_root: str | Path, patterns: Iterable[str] = DEFAULT_PATTERNS) -> list[Path]:
    root = Path(data_root)
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(found)


def record_event(conn: sqlite3.Connection, event_type: str, path: str | None, status: str, message: str, details: dict | None = None) -> None:
    conn.execute(
        "INSERT INTO watchdog_events(event_type,path,status,message,observed_at,details_json) VALUES(?,?,?,?,?,?)",
        (event_type, path, status, message, utc_now(), json.dumps(details or {})),
    )


def check_data(data_root: str | Path, database: str | Path, required: Iterable[str] = DEFAULT_PATTERNS) -> dict:
    root = Path(data_root)
    conn = connect(database)
    conn.executescript(WATCHDOG_SCHEMA)
    files = discover_files(root, required)
    expected = {pattern for pattern in required}
    missing = [pattern for pattern in expected if not any(root.glob(pattern))]
    changed = []
    healthy = []
    current_paths = set()

    for path in files:
        relative = str(path.relative_to(root))
        current_paths.add(relative)
        stat = path.stat()
        digest = sha256(path)
        previous = conn.execute("SELECT sha256 FROM data_files WHERE path=?", (relative,)).fetchone()
        status = "changed" if previous and previous[0] != digest else "ok"
        if status == "changed":
            changed.append(relative)
            record_event(conn, "file_changed", relative, "warning", "Dataset checksum changed", {"sha256": digest})
        else:
            healthy.append(relative)
        conn.execute(
            """INSERT INTO data_files(path,size_bytes,modified_ns,sha256,status,checked_at,metadata_json)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET size_bytes=excluded.size_bytes,
               modified_ns=excluded.modified_ns, sha256=excluded.sha256,
               status=excluded.status, checked_at=excluded.checked_at,
               metadata_json=excluded.metadata_json""",
            (relative, stat.st_size, stat.st_mtime_ns, digest, status, utc_now(), json.dumps({"root": str(root)})),
        )

    if not root.exists():
        record_event(conn, "root_missing", str(root), "error", "Data root does not exist")
    for pattern in missing:
        record_event(conn, "file_missing", pattern, "error", "Required dataset pattern has no matching file")
    stale = [row[0] for row in conn.execute("SELECT path FROM data_files") if row[0] not in current_paths]
    for path in stale:
        record_event(conn, "file_missing", path, "error", "Previously observed dataset is missing")

    status = "error" if missing or stale else ("warning" if changed else "ok")
    summary = {
        "status": status,
        "data_root": str(root),
        "database": str(database),
        "files_checked": len(files),
        "healthy_files": len(healthy),
        "changed_files": changed,
        "missing_patterns": missing,
        "missing_files": stale,
        "checked_at": utc_now(),
    }
    record_event(conn, "check", str(root), status, "Data watchdog check completed", summary)
    conn.commit()
    conn.close()
    return summary


def watch(data_root: str | Path, database: str | Path, interval: float) -> None:
    while True:
        print(json.dumps(check_data(data_root, database)), flush=True)
        time.sleep(interval)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor CFB forecast datasets for missing or changed files")
    parser.add_argument("--data", default="data")
    parser.add_argument("--database", default="artifacts/cfb_forecast.sqlite")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.watch:
        watch(args.data, args.database, args.interval)
    else:
        result = check_data(args.data, args.database)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] != "error" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
