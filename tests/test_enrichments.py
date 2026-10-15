"""Tests for the venues registry and the weather enrichment pipeline.

Network access is never used in the test suite. The weather module's HTTP
entry point is monkeypatched to return canned payloads so the behavior of
the enrichment logic can be verified without a live Open-Meteo call.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from cfb_forecast import weather
from cfb_forecast.venues import load_venues
from cfb_forecast.weather import (
    WeatherSnapshot,
    capture_weather,
    enrich_forecasts_with_weather,
    summarize_captures,
)


def _canned_hourly_payload(kickoff: datetime, temperature_c: float = 11.0) -> dict:
    hours = [
        (kickoff.replace(minute=0, second=0, microsecond=0)).isoformat().replace("+00:00", "Z"),
    ]
    return {
        "hourly": {
            "time": hours,
            "temperature_2m": [temperature_c],
            "wind_speed_10m": [8.3],
            "precipitation": [0.0],
            "cloud_cover": [45.0],
            "weather_code": [3],
        }
    }


def test_capture_weather_returns_indoor_sentinel_without_calling_api(monkeypatch):
    called = []

    def should_not_be_called(url, timeout):
        called.append(url)
        return {}

    monkeypatch.setattr(weather, "_request_json", should_not_be_called)
    snapshot = capture_weather(
        "2026-10-10T19:00:00Z",
        latitude=36.09,
        longitude=-115.18,
        indoor=True,
        observed_at="2026-10-09T19:00:00Z",
    )
    assert snapshot.status == "indoor_venue_no_api_call"
    assert snapshot.temperature_c == 21.0
    assert called == []


def test_capture_weather_marks_missing_coordinates(monkeypatch):
    snapshot = capture_weather("2026-10-10T19:00:00Z", observed_at="2026-10-09T19:00:00Z")
    assert snapshot.status == "venue_not_in_registry"
    assert snapshot.temperature_c is None


def test_capture_weather_selects_forecast_endpoint_inside_horizon(monkeypatch):
    kickoff = datetime(2026, 10, 10, 19, 0, tzinfo=UTC)
    captured_urls = []

    def fake_fetch(url, timeout):
        captured_urls.append(url)
        return _canned_hourly_payload(kickoff)

    monkeypatch.setattr(weather, "_request_json", fake_fetch)
    now = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
    snapshot = capture_weather(
        kickoff,
        latitude=40.0,
        longitude=-83.0,
        observed_at="2026-10-07T12:00:00Z",
        now=now,
    )
    assert snapshot.status == "captured"
    assert captured_urls and "api.open-meteo.com/v1/forecast" in captured_urls[0]
    assert snapshot.temperature_c == 11.0
    assert snapshot.wind_speed_kph == 8.3


def test_capture_weather_switches_to_archive_for_historical_games(monkeypatch):
    kickoff = datetime(2024, 11, 2, 17, 30, tzinfo=UTC)
    captured_urls = []

    def fake_fetch(url, timeout):
        captured_urls.append(url)
        return _canned_hourly_payload(kickoff, temperature_c=6.5)

    monkeypatch.setattr(weather, "_request_json", fake_fetch)
    now = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
    snapshot = capture_weather(
        kickoff,
        latitude=40.0,
        longitude=-83.0,
        observed_at="2026-10-07T12:00:00Z",
        now=now,
    )
    assert snapshot.status == "captured"
    assert captured_urls and "archive-api.open-meteo.com" in captured_urls[0]
    assert snapshot.temperature_c == 6.5


def test_capture_weather_rejects_games_outside_both_windows(monkeypatch):
    kickoff = datetime(2027, 6, 15, 23, 0, tzinfo=UTC)
    now = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
    snapshot = capture_weather(
        kickoff,
        latitude=40.0,
        longitude=-83.0,
        observed_at="2026-10-07T12:00:00Z",
        now=now,
    )
    assert snapshot.status == "outside_forecast_and_archive_window"
    assert snapshot.temperature_c is None


def test_capture_weather_returns_sentinel_on_network_failure(monkeypatch):
    def always_fails(url, timeout):
        raise OSError("simulated network outage")

    monkeypatch.setattr(weather, "_request_json", always_fails)
    kickoff = datetime(2026, 10, 10, 19, 0, tzinfo=UTC)
    now = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
    snapshot = capture_weather(
        kickoff,
        latitude=40.0,
        longitude=-83.0,
        observed_at="2026-10-07T12:00:00Z",
        now=now,
    )
    assert snapshot.status == "network_or_parse_error"
    assert "simulated network outage" in (snapshot.note or "")
    assert snapshot.temperature_c is None


def test_enrich_forecasts_tolerates_missing_venues(monkeypatch):
    forecasts = pd.DataFrame(
        [
            {"game_id": 1, "home_id": 194, "kickoff": "2026-10-10T19:00:00Z"},
            {"game_id": 2, "home_id": 99999, "kickoff": "2026-10-10T19:00:00Z"},
        ]
    )
    venues = pd.DataFrame(
        [
            {
                "home_id": 194,
                "team": "Ohio State",
                "stadium": "Ohio Stadium",
                "latitude": 40.0017,
                "longitude": -83.0197,
                "indoor": False,
            }
        ]
    )

    def fake_fetch(url, timeout):
        return _canned_hourly_payload(datetime(2026, 10, 10, 19, 0, tzinfo=UTC))

    monkeypatch.setattr(weather, "_request_json", fake_fetch)
    now = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
    enriched, captures = enrich_forecasts_with_weather(
        forecasts, venues, observed_at="2026-10-07T12:00:00Z", now=now
    )
    assert len(enriched) == 2
    assert list(enriched.weather_status) == ["captured", "venue_not_in_registry"]
    assert captures[0]["game_id"] == 1
    assert captures[1]["weather_status"] == "venue_not_in_registry"

    summary = summarize_captures(captures)
    assert summary["games"] == 2
    assert summary["statuses"]["captured"] == 1
    assert summary["statuses"]["venue_not_in_registry"] == 1
    assert summary["captured_at_least_once"] is True


def test_summarize_captures_handles_empty_input():
    summary = summarize_captures([])
    assert summary == {"games": 0, "statuses": {}, "captured_at_least_once": False}


def test_weather_snapshot_record_round_trip():
    snapshot = WeatherSnapshot(
        status="captured",
        temperature_c=12.0,
        wind_speed_kph=9.1,
        precipitation_mm=0.4,
        cloud_cover_percent=55.0,
        weather_code=61,
        source="https://api.open-meteo.com/v1/forecast",
        observed_at="2026-10-07T12:00:00+00:00",
    )
    record = snapshot.to_record()
    assert record["weather_status"] == "captured"
    assert record["weather_temperature_c"] == 12.0
    assert record["weather_code"] == 61


def test_load_venues_returns_empty_frame_when_file_missing(tmp_path):
    frame = load_venues(tmp_path / "does-not-exist.csv")
    assert list(frame.columns) == [
        "home_id",
        "team",
        "stadium",
        "latitude",
        "longitude",
        "indoor",
    ]
    assert len(frame) == 0


def test_load_venues_rejects_duplicate_home_id(tmp_path):
    path = tmp_path / "stadiums.csv"
    path.write_text(
        "home_id,team,stadium,latitude,longitude,indoor\n"
        "1,A,A Stadium,10.0,10.0,false\n"
        "1,B,B Stadium,20.0,20.0,false\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_venues(path)


def test_load_venues_rejects_invalid_coordinates(tmp_path):
    path = tmp_path / "stadiums.csv"
    path.write_text("home_id,team,stadium,latitude,longitude,indoor\n1,A,A Stadium,91.0,0.0,false\n")
    with pytest.raises(ValueError, match="latitude"):
        load_venues(path)


def test_bundled_stadiums_csv_loads():
    frame = load_venues("data/stadiums.csv")
    assert len(frame) > 20
    assert set(frame.columns) >= {
        "home_id",
        "team",
        "stadium",
        "latitude",
        "longitude",
        "indoor",
    }
    assert not frame.home_id.duplicated().any()
    assert (frame.latitude.abs() <= 90).all()
    assert (frame.longitude.abs() <= 180).all()
