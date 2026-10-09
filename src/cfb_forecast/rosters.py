"""Team roster data: current from ESPN, historical from SportsDataverse.

Rosters serve two needs. For live forecasting we want the current roster
for a team the day before a game, keyed on team ID. For historical
backtesting we want the full roster for every FBS team at the start of a
given season, so a model can know which players were on which team at a
point in time.

Current rosters come from ESPN's public per-team roster endpoint. Each
athlete row carries their display name, position, jersey number, height,
weight, class and ESPN athlete ID. Historical rosters come from the free
SportsDataverse releases, which publish one Parquet file per season under
the ``cfb_rosters`` tag; the layout is similar enough that this module
normalizes both sources to the same column set before returning.

Every call records its source URL, status, and observed-at timestamp. The
module never fabricates a roster when the call fails: it returns a
sentinel status with the reason, and the caller decides whether that is
blocking.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .data import timestamp

log = logging.getLogger(__name__)

_ESPN_ROSTER_ENDPOINT = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team_id}/roster"
)
_SPORTSDATAVERSE_ROSTER_RELEASE = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_rosters/cfb_rosters_{year}.parquet"
)

STATUS_CAPTURED = "captured"
STATUS_EMPTY = "no_athletes_listed"
STATUS_NETWORK_ERROR = "network_or_parse_error"

#: Normalized columns the module always emits. Any additional columns the
#: provider has are kept on the frame but not promised in the contract.
ROSTER_COLUMNS = [
    "season",
    "team_id",
    "athlete_id",
    "player_name",
    "position",
    "jersey",
    "class",
    "height_inches",
    "weight_lbs",
    "source",
    "observed_at",
]


@dataclass(frozen=True)
class RosterCapture:
    """One team's roster snapshot at a point in time."""

    team_id: int
    season: int | None
    status: str
    observed_at: str
    request_url: str
    rows: list[dict] = field(default_factory=list)
    note: str | None = None

    @property
    def count(self) -> int:
        return len(self.rows)

    def to_record(self) -> dict:
        return {
            "team_id": int(self.team_id),
            "season": self.season,
            "roster_capture_status": self.status,
            "roster_observed_at": self.observed_at,
            "roster_request_url": self.request_url,
            "roster_player_count": self.count,
            "roster_capture_note": self.note,
        }

    def to_frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=ROSTER_COLUMNS)
        return pd.DataFrame(self.rows, columns=ROSTER_COLUMNS)


def _request_json(url: str, timeout: float = 15.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "cfb-forecast/0.3"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # drop NaN


def _parse_espn_athlete(athlete: dict, team_id: int, observed_at: str) -> dict:
    position_block = athlete.get("position") or {}
    status_block = athlete.get("status") or {}
    experience_block = athlete.get("experience") or {}
    return {
        "season": None,
        "team_id": int(team_id),
        "athlete_id": str(athlete.get("id") or ""),
        "player_name": str(athlete.get("displayName") or athlete.get("fullName") or ""),
        "position": str(position_block.get("abbreviation") or position_block.get("name") or ""),
        "jersey": str(athlete.get("jersey") or ""),
        "class": str(experience_block.get("displayValue") or status_block.get("name") or ""),
        "height_inches": _as_float(athlete.get("height")),
        "weight_lbs": _as_float(athlete.get("weight")),
        "source": "espn.site.cfb.roster",
        "observed_at": observed_at,
    }


def _extract_espn_athletes(payload: dict) -> list[dict]:
    """ESPN groups athletes by position; flatten into one list."""
    groups = payload.get("athletes") or []
    athletes: list[dict] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        items = group.get("items") or []
        for athlete in items:
            if isinstance(athlete, dict):
                athletes.append(athlete)
    return athletes


def fetch_current_roster(
    team_id: int,
    *,
    observed_at=None,
    timeout: float = 15.0,
) -> RosterCapture:
    """Fetch one team's current roster from ESPN."""
    url = _ESPN_ROSTER_ENDPOINT.format(team_id=int(team_id))
    observed_ts = (
        timestamp(observed_at)
        if observed_at is not None
        else datetime.now(UTC)
    )
    observed = observed_ts.isoformat()
    try:
        payload = _request_json(url, timeout=timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        log.warning("roster fetch failed for team %s: %s", team_id, exc)
        return RosterCapture(
            team_id=int(team_id),
            season=None,
            status=STATUS_NETWORK_ERROR,
            observed_at=observed,
            request_url=url,
            rows=[],
            note=f"{type(exc).__name__}: {exc}",
        )
    athletes = _extract_espn_athletes(payload)
    if not athletes:
        return RosterCapture(
            team_id=int(team_id),
            season=None,
            status=STATUS_EMPTY,
            observed_at=observed,
            request_url=url,
            rows=[],
            note="ESPN returned no athletes for this team.",
        )
    rows = [_parse_espn_athlete(athlete, team_id, observed) for athlete in athletes]
    return RosterCapture(
        team_id=int(team_id),
        season=None,
        status=STATUS_CAPTURED,
        observed_at=observed,
        request_url=url,
        rows=rows,
    )


def fetch_current_rosters(team_ids, *, observed_at=None, timeout: float = 15.0) -> list[RosterCapture]:
    """Fetch current rosters for a list of team IDs, deduplicating IDs."""
    seen: set[int] = set()
    out: list[RosterCapture] = []
    for team_id in team_ids:
        key = int(team_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(fetch_current_roster(key, observed_at=observed_at, timeout=timeout))
    return out


def _download_file(url: str, target: Path, timeout: float = 60.0) -> None:
    """Stream a URL to ``target``. Raises on any HTTP or network error."""
    request = urllib.request.Request(url, headers={"User-Agent": "cfb-forecast/0.3"})
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=timeout) as response, target.open("wb") as handle:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(chunk)


def fetch_historical_rosters(
    season: int,
    *,
    cache_dir=None,
    observed_at=None,
    timeout: float = 60.0,
) -> RosterCapture:
    """Download one season's full-league rosters from SportsDataverse.

    The downloaded Parquet file is cached under ``cache_dir`` (default:
    ``data/rosters``). Repeated calls for the same season reuse the cached
    file; pass a fresh ``cache_dir`` to force a redownload.
    """
    season = int(season)
    url = _SPORTSDATAVERSE_ROSTER_RELEASE.format(year=season)
    cache_root = Path(cache_dir) if cache_dir is not None else Path("data/rosters")
    cached_path = cache_root / f"cfb_rosters_{season}.parquet"
    observed_ts = (
        timestamp(observed_at)
        if observed_at is not None
        else datetime.now(UTC)
    )
    observed = observed_ts.isoformat()
    try:
        if not cached_path.exists():
            _download_file(url, cached_path, timeout=timeout)
        frame = pd.read_parquet(cached_path)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, ImportError) as exc:
        log.warning("historical roster fetch failed for season %s: %s", season, exc)
        return RosterCapture(
            team_id=0,
            season=season,
            status=STATUS_NETWORK_ERROR,
            observed_at=observed,
            request_url=url,
            rows=[],
            note=f"{type(exc).__name__}: {exc}",
        )
    if frame.empty:
        return RosterCapture(
            team_id=0,
            season=season,
            status=STATUS_EMPTY,
            observed_at=observed,
            request_url=url,
            rows=[],
            note="SportsDataverse returned an empty roster frame.",
        )
    rows = _normalize_sportsdataverse_frame(frame, season=season, observed_at=observed)
    return RosterCapture(
        team_id=0,  # multi-team capture
        season=season,
        status=STATUS_CAPTURED,
        observed_at=observed,
        request_url=url,
        rows=rows,
    )


def _normalize_sportsdataverse_frame(
    frame: pd.DataFrame, *, season: int, observed_at: str
) -> list[dict]:
    """Normalize a SportsDataverse roster parquet to :data:`ROSTER_COLUMNS`."""
    # SportsDataverse column names vary slightly year to year; look up the
    # most common names and fall back to None when a column is missing.
    def pick(row: dict, *candidates: str):
        for name in candidates:
            if name in row and row[name] is not None and str(row[name]) != "nan":
                return row[name]
        return None

    rows: list[dict] = []
    for record in frame.to_dict("records"):
        rows.append(
            {
                "season": int(pick(record, "season", "year") or season),
                "team_id": int(pick(record, "team_id", "teamId", "team") or 0),
                "athlete_id": str(pick(record, "athlete_id", "id", "athleteId") or ""),
                "player_name": str(
                    pick(record, "full_name", "displayName", "name") or ""
                ),
                "position": str(pick(record, "position", "pos") or ""),
                "jersey": str(pick(record, "jersey", "number") or ""),
                "class": str(pick(record, "class", "year_class", "experience") or ""),
                "height_inches": _as_float(pick(record, "height", "height_inches")),
                "weight_lbs": _as_float(pick(record, "weight", "weight_lbs")),
                "source": "sportsdataverse.cfb_rosters",
                "observed_at": observed_at,
            }
        )
    return rows


def fetch_historical_rosters_span(
    start_season: int,
    end_season: int,
    *,
    cache_dir=None,
    observed_at=None,
    timeout: float = 60.0,
) -> list[RosterCapture]:
    """Fetch historical rosters for every season in ``[start_season, end_season]``.

    Seasons are fetched in chronological order. A failure on one season
    does not stop the others: each returns its own capture (``captured``
    or a sentinel status), and the caller can see per-season status in the
    returned list. Repeat calls reuse the cache; pass a fresh
    ``cache_dir`` to force redownload.
    """
    start = int(start_season)
    end = int(end_season)
    if end < start:
        raise ValueError(
            f"end_season ({end}) must be >= start_season ({start})"
        )
    captures: list[RosterCapture] = []
    for season in range(start, end + 1):
        capture = fetch_historical_rosters(
            season,
            cache_dir=cache_dir,
            observed_at=observed_at,
            timeout=timeout,
        )
        captures.append(capture)
    log.info(
        "historical_rosters_span: start=%d end=%d captures=%d",
        start,
        end,
        len(captures),
    )
    return captures


def summarize_captures(captures: list[RosterCapture]) -> dict:
    counts: dict[str, int] = {}
    for capture in captures:
        counts[capture.status] = counts.get(capture.status, 0) + 1
    return {
        "captures": len(captures),
        "status_counts": counts,
        "total_players": sum(capture.count for capture in captures),
    }


__all__ = [
    "ROSTER_COLUMNS",
    "RosterCapture",
    "fetch_current_roster",
    "fetch_current_rosters",
    "fetch_historical_rosters",
    "fetch_historical_rosters_span",
    "summarize_captures",
    "STATUS_CAPTURED",
    "STATUS_EMPTY",
    "STATUS_NETWORK_ERROR",
]
