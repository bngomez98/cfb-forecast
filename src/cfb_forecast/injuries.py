"""Pregame team injury reports via ESPN's public college-football endpoint.

ESPN publishes an injury list per team. The list is public, requires no
key, and is read-only from this module's perspective. We only read what the
provider publishes and never fabricate a status.

The data is captured as a point-in-time snapshot. Each snapshot records
when the request was made, which team it was for, how many players were on
the list, and a short per-player breakdown with name, position, status
(out, doubtful, questionable, probable, day-to-day) and the published
details string. We do not try to resolve a player's expected production;
that is a separate modeling step and not something a scraping module
should decide.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .data import timestamp

log = logging.getLogger(__name__)

_ESPN_INJURIES_ENDPOINT = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team_id}/injuries"
)

#: Severity labels we normalize ESPN's free-form text into. Any status that
#: does not match an entry here is kept as its original string in the row.
_STATUS_NORMALIZATION = {
    "out": "out",
    "doubtful": "doubtful",
    "questionable": "questionable",
    "probable": "probable",
    "day-to-day": "day_to_day",
    "day to day": "day_to_day",
    "injured reserve": "out",
    "ir": "out",
    "suspension": "suspended",
    "suspended": "suspended",
    "no status": "unknown",
    "": "unknown",
}

#: Capture statuses this module may write to the capture log.
STATUS_CAPTURED = "captured"
STATUS_NO_INJURIES_LISTED = "no_injuries_listed"
STATUS_NETWORK_ERROR = "network_or_parse_error"


@dataclass(frozen=True)
class PlayerInjury:
    """A single player's entry on the injury report."""

    player_name: str
    position: str | None
    status: str
    status_raw: str
    details: str | None


@dataclass(frozen=True)
class TeamInjuryReport:
    """A capture of one team's current injury list at ``observed_at``."""

    team_id: int
    status: str
    observed_at: str
    request_url: str
    players: list[PlayerInjury] = field(default_factory=list)
    note: str | None = None

    @property
    def count(self) -> int:
        return len(self.players)

    def to_record(self) -> dict:
        return {
            "team_id": int(self.team_id),
            "injury_capture_status": self.status,
            "injury_observed_at": self.observed_at,
            "injury_request_url": self.request_url,
            "injury_count": self.count,
            "injury_out_count": sum(1 for p in self.players if p.status == "out"),
            "injury_questionable_count": sum(1 for p in self.players if p.status == "questionable"),
            "injury_note": self.note,
        }

    def to_rows(self) -> list[dict]:
        """Return one row per player on the report, suitable for a CSV log."""
        return [
            {
                "team_id": int(self.team_id),
                "observed_at": self.observed_at,
                "player_name": player.player_name,
                "position": player.position,
                "status": player.status,
                "status_raw": player.status_raw,
                "details": player.details,
            }
            for player in self.players
        ]


def _normalize_status(status: str | None) -> tuple[str, str]:
    """Return ``(normalized_status, original_status)`` from a provider string."""
    original = (status or "").strip()
    key = original.lower()
    return _STATUS_NORMALIZATION.get(key, original.lower() or "unknown"), original


def _request_json(url: str, timeout: float = 15.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "cfb-forecast/0.3"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_players(payload: dict) -> list[PlayerInjury]:
    """Parse the ESPN response into our player dataclass list.

    ESPN returns a nested structure where each injury has an ``athlete``
    object with a display name and position short name, plus a top-level
    ``status`` string and ``details`` string. The parser is tolerant of
    missing fields: we never fabricate a value, so an athlete with no
    position becomes ``position=None`` rather than guessing a position.
    """
    players: list[PlayerInjury] = []
    for entry in payload.get("items", payload.get("injuries", [])) or []:
        athlete = entry.get("athlete", {}) if isinstance(entry, dict) else {}
        position_block = athlete.get("position") or {}
        position = position_block.get("abbreviation") or position_block.get("name")
        status_raw = entry.get("status", {})
        if isinstance(status_raw, dict):
            status_raw = status_raw.get("type") or status_raw.get("name") or ""
        normalized, original = _normalize_status(status_raw)
        details = entry.get("details") if isinstance(entry.get("details"), str) else None
        if not details:
            short = entry.get("shortComment") or entry.get("longComment")
            if isinstance(short, str) and short:
                details = short
        players.append(
            PlayerInjury(
                player_name=str(athlete.get("displayName") or "unknown"),
                position=str(position) if position else None,
                status=normalized,
                status_raw=str(original),
                details=details,
            )
        )
    return players


def fetch_team_injuries(
    team_id: int,
    *,
    observed_at=None,
    timeout: float = 15.0,
) -> TeamInjuryReport:
    """Fetch the current injury list for one team.

    Parameters
    ----------
    team_id:
        ESPN numeric team ID.
    observed_at:
        Real wall-clock time of the capture. Defaults to now. Carried on
        the returned snapshot for provenance.
    timeout:
        HTTP timeout in seconds. The call never retries; a failure returns
        a sentinel status with the reason in ``note``.
    """
    url = _ESPN_INJURIES_ENDPOINT.format(team_id=int(team_id))
    observed_ts = timestamp(observed_at) if observed_at is not None else datetime.now(UTC)
    observed = observed_ts.isoformat() if hasattr(observed_ts, "isoformat") else str(observed_ts)
    try:
        payload = _request_json(url, timeout=timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        log.warning("injury fetch failed for team %s: %s", team_id, exc)
        return TeamInjuryReport(
            team_id=int(team_id),
            status=STATUS_NETWORK_ERROR,
            observed_at=observed,
            request_url=url,
            players=[],
            note=f"{type(exc).__name__}: {exc}",
        )
    players = _extract_players(payload)
    if not players:
        return TeamInjuryReport(
            team_id=int(team_id),
            status=STATUS_NO_INJURIES_LISTED,
            observed_at=observed,
            request_url=url,
            players=[],
            note="ESPN returned no injury entries for this team at this time.",
        )
    return TeamInjuryReport(
        team_id=int(team_id),
        status=STATUS_CAPTURED,
        observed_at=observed,
        request_url=url,
        players=players,
    )


def fetch_injuries_for_games(
    team_ids,
    *,
    observed_at=None,
    timeout: float = 15.0,
) -> list[TeamInjuryReport]:
    """Fetch injury reports for a list of teams in one pass.

    Duplicate IDs are only fetched once; the output follows the first
    appearance of each ID in ``team_ids``.
    """
    seen: set[int] = set()
    reports: list[TeamInjuryReport] = []
    for team_id in team_ids:
        key = int(team_id)
        if key in seen:
            continue
        seen.add(key)
        reports.append(fetch_team_injuries(key, observed_at=observed_at, timeout=timeout))
    return reports


def summarize_reports(reports: list[TeamInjuryReport]) -> dict:
    """Simple summary suitable for the forecast manifest."""
    counts: dict[str, int] = {}
    for report in reports:
        counts[report.status] = counts.get(report.status, 0) + 1
    return {
        "teams": len(reports),
        "status_counts": counts,
        "total_players_listed": sum(report.count for report in reports),
    }


__all__ = [
    "PlayerInjury",
    "TeamInjuryReport",
    "fetch_team_injuries",
    "fetch_injuries_for_games",
    "summarize_reports",
    "STATUS_CAPTURED",
    "STATUS_NO_INJURIES_LISTED",
    "STATUS_NETWORK_ERROR",
]
