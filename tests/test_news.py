"""Tests for the sports news module.

HTTP calls are mocked; classification and injury filtering are checked
against a canned ESPN-style payload.
"""
from __future__ import annotations

from cfb_forecast.ingest import news as news_module
from cfb_forecast.ingest.news import (
    STATUS_CAPTURED,
    STATUS_EMPTY,
    STATUS_NETWORK_ERROR,
    fetch_news,
    fetch_news_for_teams,
    filter_injury_items,
    summarize_captures,
)


def _canned_payload() -> dict:
    return {
        "articles": [
            {
                "id": 1001,
                "headline": "Star QB ruled out for Saturday with knee injury",
                "description": "The starting quarterback suffered an MCL sprain in practice.",
                "published": "2026-10-08T14:00:00Z",
                "links": {"web": {"href": "https://espn.com/story/1001"}},
                "categories": [{"team": {"id": 213}}],
            },
            {
                "id": 1002,
                "headline": "Backup running back enters transfer portal",
                "description": "The former four-star is looking for a new program.",
                "published": "2026-10-08T15:00:00Z",
                "links": {"web": {"href": "https://espn.com/story/1002"}},
                "categories": [{"team": {"id": 213}}],
            },
            {
                "id": 1003,
                "headline": "Coach shuffles staff ahead of bye week",
                "description": "A new offensive coordinator takes play-calling duties.",
                "published": "2026-10-08T16:00:00Z",
                "links": {"web": {"href": "https://espn.com/story/1003"}},
                "categories": [{"team": {"id": 213}}],
            },
            {
                "id": 1004,
                "headline": "Starting WR questionable with ankle, QB suspended one game",
                "description": "Two major items hit the depth chart in one afternoon.",
                "published": "2026-10-08T17:00:00Z",
                "links": {"web": {"href": "https://espn.com/story/1004"}},
                "categories": [{"team": {"id": 213}}],
            },
        ]
    }


def test_fetch_news_classifies_items(monkeypatch):
    monkeypatch.setattr(news_module, "_request_json", lambda url, timeout: _canned_payload())
    capture = fetch_news(team_id=213, observed_at="2026-10-08T18:00:00Z")
    assert capture.status == STATUS_CAPTURED
    assert capture.count == 4
    categories = [item.category for item in capture.items]
    # Item 1 mentions "starting quarterback" (roster signal) and injury words -> mixed.
    # Item 2 is a transfer portal item -> roster_change.
    # Item 3 has no matching language -> other.
    # Item 4 pairs "questionable with ankle" and "suspended" -> mixed.
    assert categories == ["mixed", "roster_change", "other", "mixed"]
    assert capture.injury_count == 2  # both mixed items
    assert capture.roster_change_count == 3  # two mixed + one roster_change


def test_fetch_news_empty_payload(monkeypatch):
    monkeypatch.setattr(news_module, "_request_json", lambda url, timeout: {"articles": []})
    capture = fetch_news()
    assert capture.status == STATUS_EMPTY
    assert capture.count == 0


def test_fetch_news_network_error(monkeypatch):
    def always_fails(url, timeout):
        raise OSError("no net")

    monkeypatch.setattr(news_module, "_request_json", always_fails)
    capture = fetch_news(team_id=213)
    assert capture.status == STATUS_NETWORK_ERROR
    assert "no net" in (capture.note or "")


def test_fetch_news_for_teams_dedupes(monkeypatch):
    seen: list[str] = []

    def record(url, timeout):
        seen.append(url)
        return _canned_payload()

    monkeypatch.setattr(news_module, "_request_json", record)
    captures = fetch_news_for_teams([213, 194, 213])
    assert len(captures) == 2
    assert len(seen) == 2


def test_filter_injury_items_combines_frames(monkeypatch):
    monkeypatch.setattr(news_module, "_request_json", lambda url, timeout: _canned_payload())
    captures = fetch_news_for_teams([213, 194])
    injury_frame = filter_injury_items(captures)
    # Four items * 2 teams = 8 rows, 2 of 4 are injury-flagged (injury + mixed)
    assert len(injury_frame) == 4
    assert injury_frame.injury_mentioned.all()


def test_summarize_captures(monkeypatch):
    monkeypatch.setattr(news_module, "_request_json", lambda url, timeout: _canned_payload())
    captures = fetch_news_for_teams([213, 194])
    summary = summarize_captures(captures)
    assert summary["captures"] == 2
    assert summary["status_counts"][STATUS_CAPTURED] == 2
    assert summary["injury_items"] == 4  # 2 mixed items * 2 captures
    assert summary["roster_change_items"] == 6  # 3 roster-flagged items * 2 captures


def test_to_frame_round_trip(monkeypatch):
    monkeypatch.setattr(news_module, "_request_json", lambda url, timeout: _canned_payload())
    capture = fetch_news(team_id=213)
    frame = capture.to_frame()
    assert len(frame) == 4
    assert set(frame.columns) >= {"item_id", "headline", "category", "injury_mentioned"}
    assert frame.iloc[0]["source_url"].startswith("https://")
