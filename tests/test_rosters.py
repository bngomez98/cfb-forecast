"""Tests for the roster module (current + historical).

Network and disk fetches are mocked. The historical path uses a locally
written parquet file to stand in for the SportsDataverse download.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cfb_forecast.ingest import rosters as rosters_module
from cfb_forecast.ingest.rosters import (
    STATUS_CAPTURED,
    STATUS_EMPTY,
    STATUS_NETWORK_ERROR,
    fetch_historical_rosters,
    fetch_historical_rosters_span,
    summarize_captures,
)


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
    captures = fetch_historical_rosters_span(
        2021, 2024, cache_dir=tmp_path / "rosters"
    )
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


def test_fetch_historical_rosters_span_continues_past_single_failure(
    monkeypatch, tmp_path
):
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
    captures = fetch_historical_rosters_span(
        2021, 2023, cache_dir=tmp_path / "rosters"
    )
    statuses = {capture.season: capture.status for capture in captures}
    assert statuses[2021] == STATUS_CAPTURED
    assert statuses[2022] == STATUS_NETWORK_ERROR
    assert statuses[2023] == STATUS_CAPTURED
    assert calls == [2021, 2022, 2023]
