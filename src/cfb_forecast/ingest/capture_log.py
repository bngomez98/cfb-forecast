"""Unified point-in-time capture log for every external data source.

Each enrichment module that pulls data from the outside world (weather,
injury reports, NFL scoreboards, market lines) writes a row to the capture
log. Each row records when the capture happened, what source was called,
what the status was (captured, skipped, or failed), and the request URL if
one applies. The log makes the audit trail for a saved forecast a single
file instead of a scattering of fields.

The log is append-only JSON Lines so it is cheap to tail and safe to merge
across runs. Each row is self-contained and does not refer to other rows.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

CAPTURE_LOG_FILENAME = "capture_log.jsonl"


def append_capture(
    log_path,
    *,
    source: str,
    status: str,
    observed_at: str | None = None,
    game_id: int | None = None,
    request_url: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Append one capture row to the log file. Returns the row that was written.

    Parameters
    ----------
    log_path:
        File path to write to. Parent directories are created when missing.
    source:
        Short name of the data source, e.g. ``"weather.open-meteo.forecast"``.
    status:
        One of ``"captured"``, ``"skipped"``, ``"failed"``, or a module-
        specific sentinel. The log never requires a value here to be from a
        fixed vocabulary so new sources can add their own without blocking.
    observed_at:
        The timestamp at which the capture was taken. When absent, falls
        back to the current UTC time.
    game_id:
        Optional game ID this capture applies to; omitted for schedule-wide
        or team-wide captures.
    request_url:
        Optional URL that was called, so the capture can be replayed if
        needed.
    extra:
        Any additional fields the module wants on the row.
    """
    row = {
        "logged_at": datetime.now(UTC).isoformat(),
        "source": source,
        "status": status,
    }
    if observed_at is not None:
        row["observed_at"] = observed_at
    if game_id is not None:
        row["game_id"] = int(game_id)
    if request_url is not None:
        row["request_url"] = request_url
    if extra:
        row.update(extra)
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    log.debug("capture_log: %s/%s -> %s", source, status, path)
    return row


def read_capture_log(log_path) -> list[dict]:
    """Read a capture log back into a list of rows. Empty when the file is absent."""
    path = Path(log_path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_capture_log(rows: list[dict]) -> dict:
    """Produce the summary that goes next to the forecast manifest."""
    by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        source = row.get("source", "unknown")
        status = row.get("status", "unknown")
        by_source.setdefault(source, {})
        by_source[source][status] = by_source[source].get(status, 0) + 1
    return {
        "total_captures": len(rows),
        "sources": sorted(by_source),
        "status_counts_by_source": by_source,
    }


__all__ = [
    "CAPTURE_LOG_FILENAME",
    "append_capture",
    "read_capture_log",
    "summarize_capture_log",
]
