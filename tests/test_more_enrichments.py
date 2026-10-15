"""Tests for the capture log, injuries, and NFL scoreboard modules.

Network access is never used. Each module's HTTP entry point is
monkeypatched to return canned payloads so behavior is verified offline.
"""

from __future__ import annotations

from cfb_forecast import injuries as injuries_module
from cfb_forecast import nfl as nfl_module
from cfb_forecast.capture_log import (
    append_capture,
    read_capture_log,
    summarize_capture_log,
)
from cfb_forecast.injuries import (
    STATUS_CAPTURED as INJURIES_CAPTURED,
)
from cfb_forecast.injuries import (
    STATUS_NETWORK_ERROR as INJURIES_NET_ERROR,
)
from cfb_forecast.injuries import (
    STATUS_NO_INJURIES_LISTED,
    fetch_injuries_for_games,
    fetch_team_injuries,
    summarize_reports,
)
from cfb_forecast.nfl import (
    STATUS_CAPTURED as NFL_CAPTURED,
)
from cfb_forecast.nfl import (
    STATUS_EMPTY as NFL_EMPTY,
)
from cfb_forecast.nfl import (
    fetch_nfl_scoreboard,
)

# ------------------------------------------------------------------
# Capture log
# ------------------------------------------------------------------


def test_append_capture_writes_jsonl_rows(tmp_path):
    log_path = tmp_path / "captures.jsonl"
    append_capture(
        log_path,
        source="weather.open-meteo.forecast",
        status="captured",
        observed_at="2026-10-10T19:00:00+00:00",
        game_id=42,
        request_url="https://api.open-meteo.com/v1/forecast?q=example",
        extra={"temperature_c": 11.0},
    )
    append_capture(
        log_path,
        source="injuries.espn.cfb",
        status="no_injuries_listed",
        observed_at="2026-10-10T19:00:00+00:00",
        extra={"team_id": 194},
    )
    rows = read_capture_log(log_path)
    assert len(rows) == 2
    assert rows[0]["source"] == "weather.open-meteo.forecast"
    assert rows[0]["temperature_c"] == 11.0
    assert rows[1]["source"] == "injuries.espn.cfb"


def test_read_capture_log_handles_missing_file(tmp_path):
    assert read_capture_log(tmp_path / "nope.jsonl") == []


def test_summarize_capture_log_counts_by_source_and_status(tmp_path):
    rows = [
        {"source": "weather", "status": "captured"},
        {"source": "weather", "status": "captured"},
        {"source": "weather", "status": "network_or_parse_error"},
        {"source": "injuries", "status": "captured"},
    ]
    summary = summarize_capture_log(rows)
    assert summary["total_captures"] == 4
    assert summary["sources"] == ["injuries", "weather"]
    assert summary["status_counts_by_source"]["weather"] == {
        "captured": 2,
        "network_or_parse_error": 1,
    }
    assert summary["status_counts_by_source"]["injuries"] == {"captured": 1}


# ------------------------------------------------------------------
# Injuries
# ------------------------------------------------------------------


def _canned_injury_payload() -> dict:
    return {
        "items": [
            {
                "athlete": {
                    "displayName": "John Smith",
                    "position": {"abbreviation": "QB", "name": "Quarterback"},
                },
                "status": {"name": "Out", "type": "Out"},
                "details": "Lower-body injury; ruled out for the week.",
            },
            {
                "athlete": {
                    "displayName": "Michael Jones",
                    "position": {"abbreviation": "RB"},
                },
                "status": "Questionable",
                "shortComment": "Will be a game-time decision.",
            },
            {
                "athlete": {"displayName": "Chris Brown"},
                "status": {"name": "Day-to-Day"},
            },
        ]
    }


def test_fetch_team_injuries_parses_and_normalizes_statuses(monkeypatch):
    calls: list[str] = []

    def fake_fetch(url, timeout):
        calls.append(url)
        return _canned_injury_payload()

    monkeypatch.setattr(injuries_module, "_request_json", fake_fetch)
    report = fetch_team_injuries(194, observed_at="2026-10-10T19:00:00Z")
    assert report.status == INJURIES_CAPTURED
    assert report.count == 3
    assert [player.status for player in report.players] == [
        "out",
        "questionable",
        "day_to_day",
    ]
    assert report.players[0].position == "QB"
    assert report.players[2].position is None
    assert "espn.com" in calls[0]


def test_fetch_team_injuries_handles_empty_payload(monkeypatch):
    def fake_fetch(url, timeout):
        return {"items": []}

    monkeypatch.setattr(injuries_module, "_request_json", fake_fetch)
    report = fetch_team_injuries(194)
    assert report.status == STATUS_NO_INJURIES_LISTED
    assert report.count == 0


def test_fetch_team_injuries_returns_sentinel_on_network_failure(monkeypatch):
    def always_fails(url, timeout):
        raise OSError("boom")

    monkeypatch.setattr(injuries_module, "_request_json", always_fails)
    report = fetch_team_injuries(194)
    assert report.status == INJURIES_NET_ERROR
    assert "boom" in (report.note or "")
    assert report.count == 0


def test_fetch_injuries_for_games_dedupes_team_ids(monkeypatch):
    seen: list[int] = []

    def fake_fetch(url, timeout):
        team_id = int(url.rsplit("/", 2)[1])
        seen.append(team_id)
        return _canned_injury_payload()

    monkeypatch.setattr(injuries_module, "_request_json", fake_fetch)
    reports = fetch_injuries_for_games([194, 213, 194])
    assert [report.team_id for report in reports] == [194, 213]
    assert seen == [194, 213]


def test_injury_report_record_and_rows_round_trip(monkeypatch):
    def fake_fetch(url, timeout):
        return _canned_injury_payload()

    monkeypatch.setattr(injuries_module, "_request_json", fake_fetch)
    report = fetch_team_injuries(194, observed_at="2026-10-10T19:00:00Z")
    record = report.to_record()
    rows = report.to_rows()
    assert record["injury_count"] == 3
    assert record["injury_out_count"] == 1
    assert record["injury_questionable_count"] == 1
    assert len(rows) == 3
    assert rows[0]["player_name"] == "John Smith"


def test_summarize_reports(monkeypatch):
    def fake_fetch(url, timeout):
        return _canned_injury_payload()

    monkeypatch.setattr(injuries_module, "_request_json", fake_fetch)
    reports = fetch_injuries_for_games([194, 213])
    summary = summarize_reports(reports)
    assert summary["teams"] == 2
    assert summary["status_counts"][INJURIES_CAPTURED] == 2
    assert summary["total_players_listed"] == 6


# ------------------------------------------------------------------
# NFL
# ------------------------------------------------------------------


def _canned_nfl_payload() -> dict:
    return {
        "events": [
            {
                "id": "401671801",
                "date": "2026-09-14T17:00Z",
                "competitions": [
                    {
                        "id": "401671801",
                        "date": "2026-09-14T17:00Z",
                        "venue": {"fullName": "Lincoln Financial Field"},
                        "status": {"type": {"name": "STATUS_SCHEDULED"}},
                        "competitors": [
                            {
                                "homeAway": "home",
                                "score": None,
                                "team": {
                                    "id": "21",
                                    "displayName": "Philadelphia Eagles",
                                    "abbreviation": "PHI",
                                },
                            },
                            {
                                "homeAway": "away",
                                "score": None,
                                "team": {
                                    "id": "6",
                                    "displayName": "Dallas Cowboys",
                                    "abbreviation": "DAL",
                                },
                            },
                        ],
                    }
                ],
            }
        ]
    }


def test_fetch_nfl_scoreboard_parses_games(monkeypatch):
    calls: list[str] = []

    def fake_fetch(url, timeout):
        calls.append(url)
        return _canned_nfl_payload()

    monkeypatch.setattr(nfl_module, "_request_json", fake_fetch)
    capture = fetch_nfl_scoreboard("2026-09-14", observed_at="2026-09-12T12:00:00Z")
    assert capture.status == NFL_CAPTURED
    assert len(capture.games) == 1
    assert capture.games[0].home_team == "Philadelphia Eagles"
    assert capture.games[0].away_team == "Dallas Cowboys"
    assert capture.games[0].venue == "Lincoln Financial Field"
    assert capture.date == "2026-09-14"
    assert "dates=20260914" in calls[0]


def test_fetch_nfl_scoreboard_handles_empty_events(monkeypatch):
    def fake_fetch(url, timeout):
        return {"events": []}

    monkeypatch.setattr(nfl_module, "_request_json", fake_fetch)
    capture = fetch_nfl_scoreboard("2026-09-14")
    assert capture.status == NFL_EMPTY
    assert capture.games == []


def test_fetch_nfl_scoreboard_handles_network_failure(monkeypatch):
    def always_fails(url, timeout):
        raise OSError("unreachable")

    monkeypatch.setattr(nfl_module, "_request_json", always_fails)
    capture = fetch_nfl_scoreboard("2026-09-14")
    assert capture.status == nfl_module.STATUS_NETWORK_ERROR
    assert "unreachable" in (capture.note or "")


def test_nfl_scoreboard_record_and_rows(monkeypatch):
    def fake_fetch(url, timeout):
        return _canned_nfl_payload()

    monkeypatch.setattr(nfl_module, "_request_json", fake_fetch)
    capture = fetch_nfl_scoreboard("2026-09-14", observed_at="2026-09-12T12:00:00Z")
    record = capture.to_record()
    rows = capture.to_rows()
    assert record["nfl_games_listed"] == 1
    assert record["nfl_scoreboard_date"] == "2026-09-14"
    assert rows[0]["home_team_id"] == "21"
    assert rows[0]["scoreboard_date"] == "2026-09-14"
