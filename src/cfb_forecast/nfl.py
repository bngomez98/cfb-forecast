"""NFL scoreboard capture via ESPN's public endpoint.

The college-football engine does not forecast NFL games, but having NFL
schedules and final scores alongside the college pipeline is useful for
cross-sport analysis and for making sure the same pipeline conventions
(observed-at timestamps, source URLs, no fabrication on failure) apply to
every data source we pull in.

The public ESPN NFL scoreboard returns the slate for a given date, with
each game's home and away teams, kickoff time, venue and current status.
This module exposes two functions: one to fetch a date's scoreboard, and
one to fetch the current week's upcoming games. Both return normalized
rows; failures return an empty list with a reason, never a guess.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime

from .data import timestamp

log = logging.getLogger(__name__)

_NFL_SCOREBOARD_ENDPOINT = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

STATUS_CAPTURED = "captured"
STATUS_NETWORK_ERROR = "network_or_parse_error"
STATUS_EMPTY = "no_games_listed"


@dataclass(frozen=True)
class NFLGame:
    """One NFL game's scoreboard row."""

    game_id: str
    kickoff: str
    home_team: str
    home_team_id: str
    away_team: str
    away_team_id: str
    venue: str | None
    status: str
    home_score: int | None
    away_score: int | None


@dataclass(frozen=True)
class NFLScoreboardCapture:
    """One capture of the ESPN NFL scoreboard."""

    status: str
    observed_at: str
    request_url: str
    date: str
    games: list[NFLGame]
    note: str | None = None

    def to_record(self) -> dict:
        return {
            "nfl_capture_status": self.status,
            "nfl_observed_at": self.observed_at,
            "nfl_request_url": self.request_url,
            "nfl_scoreboard_date": self.date,
            "nfl_games_listed": len(self.games),
            "nfl_capture_note": self.note,
        }

    def to_rows(self) -> list[dict]:
        return [
            {
                "scoreboard_date": self.date,
                "observed_at": self.observed_at,
                **game.__dict__,
            }
            for game in self.games
        ]


def _request_json(url: str, timeout: float = 15.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "cfb-forecast/0.3"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_game(entry: dict) -> NFLGame | None:
    competitions = entry.get("competitions") or []
    if not competitions:
        return None
    competition = competitions[0]
    competitors = competition.get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if home is None or away is None:
        return None

    def _team_name(competitor: dict) -> str:
        team = competitor.get("team") or {}
        return str(team.get("displayName") or team.get("name") or team.get("abbreviation") or "unknown")

    def _team_id(competitor: dict) -> str:
        team = competitor.get("team") or {}
        return str(team.get("id") or "unknown")

    def _score(competitor: dict) -> int | None:
        value = competitor.get("score")
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    venue_block = competition.get("venue") or {}
    status_block = (competition.get("status") or {}).get("type") or {}

    return NFLGame(
        game_id=str(entry.get("id") or competition.get("id") or ""),
        kickoff=str(entry.get("date") or competition.get("date") or ""),
        home_team=_team_name(home),
        home_team_id=_team_id(home),
        away_team=_team_name(away),
        away_team_id=_team_id(away),
        venue=str(venue_block.get("fullName")) if venue_block.get("fullName") else None,
        status=str(status_block.get("name") or "unknown"),
        home_score=_score(home),
        away_score=_score(away),
    )


def fetch_nfl_scoreboard(
    scoreboard_date=None,
    *,
    observed_at=None,
    timeout: float = 15.0,
) -> NFLScoreboardCapture:
    """Fetch the NFL scoreboard for one date (default: today, UTC)."""
    if scoreboard_date is None:
        scoreboard_date = datetime.now(UTC).date()
    elif isinstance(scoreboard_date, str):
        scoreboard_date = date.fromisoformat(scoreboard_date)
    elif isinstance(scoreboard_date, datetime):
        scoreboard_date = scoreboard_date.date()
    date_str = scoreboard_date.strftime("%Y%m%d")
    query = urllib.parse.urlencode({"dates": date_str})
    url = f"{_NFL_SCOREBOARD_ENDPOINT}?{query}"
    observed_ts = timestamp(observed_at) if observed_at is not None else datetime.now(UTC)
    observed = observed_ts.isoformat() if hasattr(observed_ts, "isoformat") else str(observed_ts)
    try:
        payload = _request_json(url, timeout=timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        log.warning("nfl scoreboard fetch failed: %s", exc)
        return NFLScoreboardCapture(
            status=STATUS_NETWORK_ERROR,
            observed_at=observed,
            request_url=url,
            date=scoreboard_date.isoformat(),
            games=[],
            note=f"{type(exc).__name__}: {exc}",
        )
    events = payload.get("events") or []
    games: list[NFLGame] = []
    for entry in events:
        game = _parse_game(entry)
        if game is not None:
            games.append(game)
    if not games:
        return NFLScoreboardCapture(
            status=STATUS_EMPTY,
            observed_at=observed,
            request_url=url,
            date=scoreboard_date.isoformat(),
            games=[],
            note="ESPN returned no games on this date.",
        )
    return NFLScoreboardCapture(
        status=STATUS_CAPTURED,
        observed_at=observed,
        request_url=url,
        date=scoreboard_date.isoformat(),
        games=games,
    )


__all__ = [
    "NFLGame",
    "NFLScoreboardCapture",
    "fetch_nfl_scoreboard",
    "STATUS_CAPTURED",
    "STATUS_NETWORK_ERROR",
    "STATUS_EMPTY",
]
