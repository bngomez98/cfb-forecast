"""Sports news feed for player injury and team updates (ESPN public).

The forecast benefits from knowing when something has happened to a key
player since the last game was played: a mid-week injury announcement, a
suspension, a transfer-portal entry, a quarterback change. ESPN publishes
a public news feed for college football that carries these items with
timestamps, headlines and descriptions, and does not require an API key.

This module captures that feed as a point-in-time snapshot and labels each
item as injury-related, roster-change-related, or other, using a small
keyword rule set. We never fabricate items the feed does not carry, and
we keep the raw headline and description on the row so a reader can see
what the model saw.

The intent is to feed two downstream consumers. The injuries module
already pulls the per-team pregame report; the news feed is a wider net
that catches the pre-report chatter and the items that never land on an
official injury report (transfers, suspensions, depth-chart changes).
Together they go on the forecast row and into the capture log.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from .data import timestamp

log = logging.getLogger(__name__)

_ESPN_NEWS_ENDPOINT = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/news"
_ESPN_TEAM_NEWS_ENDPOINT = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/news?team={team_id}"
)

STATUS_CAPTURED = "captured"
STATUS_EMPTY = "no_items_returned"
STATUS_NETWORK_ERROR = "network_or_parse_error"

NEWS_COLUMNS = [
    "item_id",
    "team_id",
    "published_at",
    "headline",
    "description",
    "category",
    "injury_mentioned",
    "roster_change_mentioned",
    "source_url",
    "observed_at",
]

_INJURY_PATTERNS = re.compile(
    r"\b(injur\w*|injured|hurt|strain|sprain|concussion|surgery|out for|miss\w* "
    r"the game|will not play|questionable|doubtful|probable|day[- ]to[- ]day|"
    r"knee|ankle|shoulder|hamstring|acl|mcl|ir|injured reserve|limited|"
    r"ruled out)\b",
    re.IGNORECASE,
)
_ROSTER_PATTERNS = re.compile(
    r"\b(transfer\w*|portal|suspend\w*|dismiss\w*|arrest\w*|opt[- ]out|"
    r"declar\w* for|will start|benched|starting (?:qb|quarterback)|"
    r"depth chart|retir\w*|leaves program)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NewsItem:
    """One normalized news item."""

    item_id: str
    team_id: int | None
    published_at: str
    headline: str
    description: str
    category: str
    injury_mentioned: bool
    roster_change_mentioned: bool
    source_url: str

    def to_row(self, observed_at: str) -> dict:
        return {
            "item_id": self.item_id,
            "team_id": self.team_id,
            "published_at": self.published_at,
            "headline": self.headline,
            "description": self.description,
            "category": self.category,
            "injury_mentioned": self.injury_mentioned,
            "roster_change_mentioned": self.roster_change_mentioned,
            "source_url": self.source_url,
            "observed_at": observed_at,
        }


@dataclass(frozen=True)
class NewsCapture:
    """Result of one news feed call."""

    status: str
    observed_at: str
    request_url: str
    team_id: int | None
    items: list[NewsItem] = field(default_factory=list)
    note: str | None = None

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def injury_count(self) -> int:
        return sum(1 for item in self.items if item.injury_mentioned)

    @property
    def roster_change_count(self) -> int:
        return sum(1 for item in self.items if item.roster_change_mentioned)

    def to_frame(self) -> pd.DataFrame:
        if not self.items:
            return pd.DataFrame(columns=NEWS_COLUMNS)
        return pd.DataFrame(
            [item.to_row(self.observed_at) for item in self.items],
            columns=NEWS_COLUMNS,
        )

    def to_record(self) -> dict:
        return {
            "news_capture_status": self.status,
            "news_observed_at": self.observed_at,
            "news_request_url": self.request_url,
            "news_team_id": self.team_id,
            "news_item_count": self.count,
            "news_injury_count": self.injury_count,
            "news_roster_change_count": self.roster_change_count,
            "news_capture_note": self.note,
        }


def _request_json(url: str, timeout: float = 15.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "cfb-forecast/0.4"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _classify(headline: str, description: str) -> tuple[str, bool, bool]:
    """Return ``(category, injury_mentioned, roster_change_mentioned)``.

    Category is one of: ``injury``, ``roster_change``, ``mixed``, ``other``.
    Mixed means both injury and roster-change language appear.
    """
    text = f"{headline}\n{description}"
    has_injury = bool(_INJURY_PATTERNS.search(text))
    has_roster = bool(_ROSTER_PATTERNS.search(text))
    if has_injury and has_roster:
        return "mixed", True, True
    if has_injury:
        return "injury", True, False
    if has_roster:
        return "roster_change", False, True
    return "other", False, False


def _parse_item(item: dict, fallback_team_id: int | None) -> NewsItem | None:
    headline = str(item.get("headline") or item.get("title") or "").strip()
    if not headline:
        return None
    description = str(item.get("description") or item.get("summary") or "").strip()
    item_id = str(item.get("id") or item.get("dataSourceIdentifier") or headline)
    published = str(item.get("published") or item.get("lastModified") or "")
    team_id = fallback_team_id
    categories = item.get("categories") or []
    for category in categories:
        if not isinstance(category, dict):
            continue
        team_block = category.get("team") or {}
        candidate = team_block.get("id")
        if candidate is not None:
            try:
                team_id = int(candidate)
                break
            except (TypeError, ValueError):
                continue
    links = item.get("links") or {}
    web = links.get("web") or {}
    source_url = str(web.get("href") or "")
    category, injury_flag, roster_flag = _classify(headline, description)
    return NewsItem(
        item_id=item_id,
        team_id=team_id,
        published_at=published,
        headline=headline,
        description=description,
        category=category,
        injury_mentioned=injury_flag,
        roster_change_mentioned=roster_flag,
        source_url=source_url,
    )


def fetch_news(
    *,
    team_id: int | None = None,
    observed_at=None,
    timeout: float = 15.0,
) -> NewsCapture:
    """Fetch the current news feed, optionally filtered to one team.

    When ``team_id`` is given, ESPN scopes the feed to that team; otherwise
    the general college-football feed is returned. The feed carries about
    the last day's items in practice; call it on your normal forecast
    cadence to catch new injuries and roster changes.
    """
    url = (
        _ESPN_TEAM_NEWS_ENDPOINT.format(team_id=int(team_id)) if team_id is not None else _ESPN_NEWS_ENDPOINT
    )
    observed_ts = timestamp(observed_at) if observed_at is not None else datetime.now(UTC)
    observed = observed_ts.isoformat()
    try:
        payload = _request_json(url, timeout=timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        log.warning("news fetch failed for team %s: %s", team_id, exc)
        return NewsCapture(
            status=STATUS_NETWORK_ERROR,
            observed_at=observed,
            request_url=url,
            team_id=team_id,
            items=[],
            note=f"{type(exc).__name__}: {exc}",
        )
    raw_items = payload.get("articles") or payload.get("items") or []
    items: list[NewsItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        parsed = _parse_item(raw, team_id)
        if parsed is not None:
            items.append(parsed)
    if not items:
        return NewsCapture(
            status=STATUS_EMPTY,
            observed_at=observed,
            request_url=url,
            team_id=team_id,
            items=[],
            note="ESPN returned no news items.",
        )
    return NewsCapture(
        status=STATUS_CAPTURED,
        observed_at=observed,
        request_url=url,
        team_id=team_id,
        items=items,
    )


def fetch_news_for_teams(team_ids, *, observed_at=None, timeout: float = 15.0) -> list[NewsCapture]:
    seen: set[int] = set()
    out: list[NewsCapture] = []
    for team_id in team_ids:
        key = int(team_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(fetch_news(team_id=key, observed_at=observed_at, timeout=timeout))
    return out


def summarize_captures(captures: list[NewsCapture]) -> dict:
    counts: dict[str, int] = {}
    injury_total = 0
    roster_total = 0
    for capture in captures:
        counts[capture.status] = counts.get(capture.status, 0) + 1
        injury_total += capture.injury_count
        roster_total += capture.roster_change_count
    return {
        "captures": len(captures),
        "status_counts": counts,
        "injury_items": injury_total,
        "roster_change_items": roster_total,
    }


def filter_injury_items(captures: list[NewsCapture]) -> pd.DataFrame:
    """Return one combined frame of just the injury-flagged items."""
    frames: list[pd.DataFrame] = []
    for capture in captures:
        frame = capture.to_frame()
        if frame.empty:
            continue
        frames.append(frame[frame.injury_mentioned])
    if not frames:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "NEWS_COLUMNS",
    "NewsCapture",
    "NewsItem",
    "STATUS_CAPTURED",
    "STATUS_EMPTY",
    "STATUS_NETWORK_ERROR",
    "fetch_news",
    "fetch_news_for_teams",
    "filter_injury_items",
    "summarize_captures",
]
