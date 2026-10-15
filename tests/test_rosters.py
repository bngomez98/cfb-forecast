"""Tests for the roster module (current + historical).

Network and disk fetches are mocked. The historical path uses a locally
written parquet file to stand in for the SportsDataverse download.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cfb_forecast import rosters as rosters_module
from cfb_forecast.rosters import (
    STATUS_CAPTURED,
    STATUS_EMPTY,
    STATUS_NETWORK_ERROR,
    fetch_current_roster,
    fetch_current_rosters,
    fetch_historical_rosters,
    fetch_historical_rosters_span,
    summarize_captures,
)


def _canned_espn_roster_payload() -> dict:
    return {
        "athletes": [
            {
                "position": "offense",
                "items": [
                    {
                        "id": "4567890",
                        "displayName": "John Doe",
                        "jersey": "7",
                        "position": {"abbreviation": "QB", "name": "Quarterback"},
                        "experience": {"displayValue": "Junior"},
                        "height": 75,
                        "weight": 215,
                    },
                    {
                        "id": "4567891",
                        "displayName": "Alex Smith",
                        "jersey": "22",
                        "position": {"abbreviation": "RB"},
                        "experience": {"displayValue": "Senior"},
                        "height": 70,
                        "weight": 205,
                    },
                ],
            },
            {
                "position": "defense",
                "items": [
                    {
                        "id": "4567892",
                        "displayName": "Chris Baker",
                        "jersey": "55",
                        "position": {"abbreviation": "LB"},
                        "experience": {"displayValue": "Sophomore"},
                        "height": 73,
                        "weight": 240,
                    }
                ],
            },
        ]
    }


def test_fetch_current_roster_flattens_position_groups(monkeypatch):
    calls: list[str] = []

    def fake_fetch(url, timeout):
        calls.append(url)
        return _canned_espn_roster_payload()

    monkeypatch.setattr(rosters_module, "_request_json", fake_fetch)
    capture = fetch_current_roster(194, observed_at="2026-10-10T19:00:00Z")
    assert capture.status == STATUS_CAPTURED
    assert capture.count == 3
    assert capture.rows[0]["player_name"] == "John Doe"
    assert capture.rows[0]["position"] == "QB"
    assert capture.rows[0]["height_inches"] == 75.0
    assert capture.rows[2]["position"] == "LB"
    assert "roster" in calls[0]


def test_fetch_current_roster_returns_sentinel_on_empty_payload(monkeypatch):
    def fake_fetch(url, timeout):
        return {"athletes": []}

    monkeypatch.setattr(rosters_module, "_request_json", fake_fetch)
    capture = fetch_current_roster(194)
    assert capture.status == STATUS_EMPTY
    assert capture.count == 0


def test_fetch_current_roster_returns_sentinel_on_network_failure(monkeypatch):
    def always_fails(url, timeout):
        raise OSError("no network")

    monkeypatch.setattr(rosters_module, "_request_json", always_fails)
    capture = fetch_current_roster(194)
    assert capture.status == STATUS_NETWORK_ERROR
    assert "no network" in (capture.note or "")


def test_fetch_current_rosters_deduplicates(monkeypatch):
    seen: list[int] = []

    def fake_fetch(url, timeout):
        seen.append(int(url.rsplit("/", 2)[1]))
        return _canned_espn_roster_payload()

    monkeypatch.setattr(rosters_module, "_request_json", fake_fetch)
    captures = fetch_current_rosters([194, 213, 194])
    assert [capture.team_id for capture in captures] == [194, 213]
    assert seen == [194, 213]


def test_roster_capture_to_frame_round_trip(monkeypatch):
    def fake_fetch(url, timeout):
        return _canned_espn_roster_payload()

    monkeypatch.setattr(rosters_module, "_request_json", fake_fetch)
    capture = fetch_current_roster(194, observed_at="2026-10-10T19:00:00Z")
    frame = capture.to_frame()
    assert len(frame) == 3
    assert list(frame.columns)[:4] == ["season", "team_id", "athlete_id", "player_name"]
    assert frame.iloc[0]["source"] == "espn.site.cfb.roster"


def test_summarize_captures(monkeypatch):
    def fake_fetch(url, timeout):
        return _canned_espn_roster_payload()

    monkeypatch.setattr(rosters_module, "_request_json", fake_fetch)
    captures = fetch_current_rosters([194, 213])
    summary = summarize_captures(captures)
    assert summary["captures"] == 2
    assert summary["status_counts"][STATUS_CAPTURED] == 2
    assert summary["total_players"] == 6


def test_fetch_historical_rosters_reads_cached_parquet(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "season": 2024,
                "team_id": 194,
                "athlete_id": "4567890",
                "full_name": "John Doe",
                "position": "QB",
                "jersey": "7",
                "year_class": "Junior",
                "height": 75,
                "weight": 215,
            },
            {
                "season": 2024,
                "team_id": 194,
                "athlete_id": "4567891",
                "full_name": "Alex Smith",
                "position": "RB",
                "jersey": "22",
                "year_class": "Senior",
                "height": 70,
                "weight": 205,
            },
        ]
    )
    cache = tmp_path / "rosters"
    cache.mkdir()
    frame.to_parquet(cache / "cfb_rosters_2024.parquet")
    capture = fetch_historical_rosters(2024, cache_dir=cache, observed_at="2026-10-10T19:00:00Z")
    assert capture.status == STATUS_CAPTURED
    assert capture.season == 2024
    assert capture.count == 2
    assert capture.rows[0]["player_name"] == "John Doe"
    assert capture.rows[0]["source"] == "sportsdataverse.cfb_rosters"
    assert capture.rows[0]["height_inches"] == 75.0


def test_fetch_historical_rosters_downloads_when_not_cached(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        [
            {
                "season": 2024,
                "team_id": 213,
                "athlete_id": "9",
                "full_name": "Pat Rivera",
                "position": "WR",
                "jersey": "11",
                "year_class": "Freshman",
                "height": 71,
                "weight": 180,
            }
        ]
    )
    cache = tmp_path / "rosters"
    expected_path = cache / "cfb_rosters_2024.parquet"
    cache.mkdir()

    def fake_download(url, target, timeout=60.0):
        assert "cfb_rosters_2024.parquet" in url
        assert Path(target) == expected_path
        frame.to_parquet(target)

    monkeypatch.setattr(rosters_module, "_download_file", fake_download)
    capture = fetch_historical_rosters(2024, cache_dir=cache)
    assert capture.status == STATUS_CAPTURED
    assert capture.count == 1
    assert expected_path.exists()


def test_fetch_historical_rosters_sentinel_on_download_failure(monkeypatch, tmp_path):
    def always_fails(url, target, timeout=60.0):
        raise OSError("github unreachable")

    monkeypatch.setattr(rosters_module, "_download_file", always_fails)
    capture = fetch_historical_rosters(2024, cache_dir=tmp_path / "rosters")
    assert capture.status == STATUS_NETWORK_ERROR
    assert "github unreachable" in (capture.note or "")


def test_fetch_historical_rosters_span_covers_each_season(monkeypatch, tmp_path):
    seasons_requested: list[int] = []

    def fake_download(url, target, timeout=60.0):
        # Extract season from URL (e.g. cfb_rosters_2021.parquet).
        for token in url.split("_"):
            if token.endswith(".parquet"):
                seasons_requested.append(int(token.split(".")[0]))
                break
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            [
                {
                    "season": seasons_requested[-1],
                    "team_id": 194,
                    "athlete_id": f"X-{seasons_requested[-1]}",
                    "full_name": "Rolling Player",
                    "position": "QB",
                    "jersey": "7",
                    "year_class": "Senior",
                    "height": 75,
                    "weight": 215,
                }
            ]
        )
        frame.to_parquet(target)

    monkeypatch.setattr(rosters_module, "_download_file", fake_download)
    captures = fetch_historical_rosters_span(2021, 2024, cache_dir=tmp_path / "rosters")
    assert [capture.season for capture in captures] == [2021, 2022, 2023, 2024]
    assert all(capture.status == STATUS_CAPTURED for capture in captures)
    assert seasons_requested == [2021, 2022, 2023, 2024]


def test_fetch_historical_rosters_span_rejects_inverted_range(tmp_path):
    try:
        fetch_historical_rosters_span(2024, 2021, cache_dir=tmp_path / "rosters")
    except ValueError as exc:
        assert "end_season" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for inverted range")


def test_fetch_historical_rosters_span_continues_past_single_failure(monkeypatch, tmp_path):
    calls: list[int] = []

    def half_fails(url, target, timeout=60.0):
        season = int(url.rsplit("_", 1)[-1].split(".")[0])
        calls.append(season)
        if season == 2022:
            raise OSError("gh down")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "season": season,
                    "team_id": 194,
                    "athlete_id": "A",
                    "full_name": "A Name",
                    "position": "QB",
                    "jersey": "1",
                    "year_class": "Senior",
                    "height": 75,
                    "weight": 215,
                }
            ]
        ).to_parquet(target)

    monkeypatch.setattr(rosters_module, "_download_file", half_fails)
    captures = fetch_historical_rosters_span(2021, 2023, cache_dir=tmp_path / "rosters")
    statuses = {capture.season: capture.status for capture in captures}
    assert statuses[2021] == STATUS_CAPTURED
    assert statuses[2022] == STATUS_NETWORK_ERROR
    assert statuses[2023] == STATUS_CAPTURED
    assert calls == [2021, 2022, 2023]
