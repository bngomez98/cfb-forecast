"""Pregame weather enrichment via the free Open-Meteo API.

Weather is captured at forecast time as a point-in-time snapshot keyed by
the home venue's coordinates and the game's kickoff timestamp. Every capture
records the API endpoint, the request URL, the observed-at timestamp, and
the resolved forecast time, so a saved forecast can be audited the same way
schedules and advanced-team data are.

Open-Meteo is used because it has no API key, serves hourly forecasts up to
sixteen days ahead, serves historical hourly observations through the ERA5
archive, and is reliable enough for a research tool without a billing
relationship. The returned fields are intentionally few: temperature in
Celsius, wind speed in kilometers per hour, precipitation in millimeters,
cloud cover in percent, and a weather code describing conditions.

The module never fabricates a value. When the API call fails, or when the
requested game is outside the forecast and archive windows, the enrichment
step returns a status row carrying the failure reason and leaves the
weather columns empty. Downstream consumers decide whether that counts as a
blocker.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from .data import timestamp

log = logging.getLogger(__name__)

#: Base URL for Open-Meteo's standard forecast endpoint (0 to 16 days out).
_FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"

#: Base URL for Open-Meteo's archive endpoint, which serves ERA5 reanalysis.
_ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"

#: Hourly variables requested from Open-Meteo. The order controls which
#: columns the response is parsed into.
_HOURLY_VARIABLES = (
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
    "weather_code",
)

#: Window in which the live forecast endpoint can be used. Games farther in
#: the future than this receive a climatology-based fallback, not a forecast.
FORECAST_HORIZON_DAYS = 16

#: Minimum age below which a game is still treated as "future" for the live
#: forecast endpoint. Beyond this age we fetch the archive instead.
ARCHIVE_SWITCHOVER_DAYS = 1

_SENTINEL_INDOOR_STATUS = "indoor_venue_no_api_call"
_SENTINEL_MISSING_VENUE_STATUS = "venue_not_in_registry"
_SENTINEL_OUT_OF_RANGE_STATUS = "outside_forecast_and_archive_window"
_SENTINEL_NETWORK_FAILURE_STATUS = "network_or_parse_error"
_SENTINEL_CAPTURED_STATUS = "captured"


@dataclass(frozen=True)
class WeatherSnapshot:
    """One game's weather snapshot, including provenance for reproducibility."""

    status: str
    temperature_c: float | None = None
    wind_speed_kph: float | None = None
    precipitation_mm: float | None = None
    cloud_cover_percent: float | None = None
    weather_code: int | None = None
    source: str | None = None
    request_url: str | None = None
    observed_at: str | None = None
    forecast_hour_utc: str | None = None
    note: str | None = None

    def to_record(self) -> dict:
        """Flat dict suitable for storage next to forecast rows."""
        return {
            "weather_status": self.status,
            "weather_temperature_c": self.temperature_c,
            "weather_wind_speed_kph": self.wind_speed_kph,
            "weather_precipitation_mm": self.precipitation_mm,
            "weather_cloud_cover_percent": self.cloud_cover_percent,
            "weather_code": self.weather_code,
            "weather_source": self.source,
            "weather_request_url": self.request_url,
            "weather_observed_at": self.observed_at,
            "weather_forecast_hour_utc": self.forecast_hour_utc,
            "weather_note": self.note,
        }


def _request_json(url: str, timeout: float = 15.0) -> dict:
    """Fetch a JSON document from ``url`` using the standard library.

    The call goes through the environment's HTTPS proxy if one is configured.
    Any non-2xx response, timeout, DNS failure or parse error is raised to
    the caller; the caller decides whether to convert that into a sentinel.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "cfb-forecast/0.3"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _build_url(
    endpoint: str,
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> str:
    query = urllib.parse.urlencode(
        {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "hourly": ",".join(_HOURLY_VARIABLES),
            "timezone": "UTC",
            "start_date": start_date,
            "end_date": end_date,
            "wind_speed_unit": "kmh",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
        }
    )
    return f"{endpoint}?{query}"


def _select_hour(payload: dict, kickoff: pd.Timestamp) -> dict | None:
    """Pick the hour entry closest to ``kickoff`` from an Open-Meteo payload."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None
    time_index = pd.to_datetime(times, utc=True)
    offset = (time_index - kickoff).total_seconds().__abs__()
    position = int(pd.Series(offset).idxmin())
    return {
        "forecast_hour_utc": time_index[position].isoformat(),
        **{var: hourly.get(var, [None] * len(times))[position] for var in _HOURLY_VARIABLES},
    }


def _fetch(
    endpoint: str,
    *,
    latitude: float,
    longitude: float,
    kickoff: pd.Timestamp,
    observed_at: pd.Timestamp,
    timeout: float,
) -> WeatherSnapshot:
    start_date = (kickoff - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (kickoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    url = _build_url(
        endpoint,
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
    )
    try:
        payload = _request_json(url, timeout=timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        log.warning("weather fetch failed for %s: %s", url, exc)
        return WeatherSnapshot(
            status=_SENTINEL_NETWORK_FAILURE_STATUS,
            source=endpoint,
            request_url=url,
            observed_at=observed_at.isoformat(),
            note=f"{type(exc).__name__}: {exc}",
        )
    hour = _select_hour(payload, kickoff)
    if hour is None:
        return WeatherSnapshot(
            status=_SENTINEL_NETWORK_FAILURE_STATUS,
            source=endpoint,
            request_url=url,
            observed_at=observed_at.isoformat(),
            note="Open-Meteo response contained no hourly rows",
        )
    weather_code = hour.get("weather_code")
    return WeatherSnapshot(
        status=_SENTINEL_CAPTURED_STATUS,
        temperature_c=_as_float(hour.get("temperature_2m")),
        wind_speed_kph=_as_float(hour.get("wind_speed_10m")),
        precipitation_mm=_as_float(hour.get("precipitation")),
        cloud_cover_percent=_as_float(hour.get("cloud_cover")),
        weather_code=int(weather_code) if weather_code is not None else None,
        source=endpoint,
        request_url=url,
        observed_at=observed_at.isoformat(),
        forecast_hour_utc=hour["forecast_hour_utc"],
    )


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _indoor_snapshot(observed_at: pd.Timestamp, note: str) -> WeatherSnapshot:
    return WeatherSnapshot(
        status=_SENTINEL_INDOOR_STATUS,
        temperature_c=21.0,
        wind_speed_kph=0.0,
        precipitation_mm=0.0,
        cloud_cover_percent=0.0,
        source="indoor_venue_constant",
        observed_at=observed_at.isoformat(),
        note=note,
    )


def capture_weather(
    kickoff,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    indoor: bool = False,
    observed_at=None,
    timeout: float = 15.0,
    now: datetime | None = None,
) -> WeatherSnapshot:
    """Capture the pregame weather snapshot for one game.

    Parameters
    ----------
    kickoff:
        Kickoff time; parsed through :func:`cfb_forecast.data.timestamp` and
        must carry a timezone.
    latitude, longitude:
        Venue coordinates. Both required for an outdoor venue; a missing
        value returns a ``venue_not_in_registry`` snapshot.
    indoor:
        When true, the function skips the API and returns a constant
        indoor-venue snapshot. Teams at domed stadiums should set this.
    observed_at:
        The real time at which the capture is taken. Defaults to ``now``;
        this is the field that goes into the forecast manifest alongside the
        model and source hashes.
    timeout:
        Per-call HTTP timeout in seconds.
    now:
        Override for the "current time" used to choose between the live
        forecast and archive endpoints. Only tests should set this.
    """
    kickoff_ts = timestamp(kickoff)
    reference_now = datetime.now(UTC) if now is None else now
    observed_ts = (
        timestamp(observed_at)
        if observed_at is not None
        else pd.Timestamp(reference_now).tz_convert("UTC")
    )
    if indoor:
        return _indoor_snapshot(observed_ts, "Indoor venue; API call skipped.")
    if latitude is None or longitude is None:
        return WeatherSnapshot(
            status=_SENTINEL_MISSING_VENUE_STATUS,
            observed_at=observed_ts.isoformat(),
            note="Venue coordinates not available in the registry.",
        )
    delta = kickoff_ts - pd.Timestamp(reference_now).tz_convert("UTC")
    delta_days = delta.total_seconds() / 86400.0
    if -365 * 50 < delta_days < -ARCHIVE_SWITCHOVER_DAYS:
        # Historical game: use the archive endpoint (ERA5 reanalysis).
        return _fetch(
            _ARCHIVE_ENDPOINT,
            latitude=latitude,
            longitude=longitude,
            kickoff=kickoff_ts,
            observed_at=observed_ts,
            timeout=timeout,
        )
    if -ARCHIVE_SWITCHOVER_DAYS <= delta_days <= FORECAST_HORIZON_DAYS:
        # Live forecast window.
        return _fetch(
            _FORECAST_ENDPOINT,
            latitude=latitude,
            longitude=longitude,
            kickoff=kickoff_ts,
            observed_at=observed_ts,
            timeout=timeout,
        )
    return WeatherSnapshot(
        status=_SENTINEL_OUT_OF_RANGE_STATUS,
        observed_at=observed_ts.isoformat(),
        note=(
            f"Kickoff is {delta_days:+.1f} days from capture time; "
            "outside Open-Meteo forecast and archive windows."
        ),
    )


def enrich_forecasts_with_weather(
    forecasts: pd.DataFrame,
    venues: pd.DataFrame,
    *,
    observed_at=None,
    timeout: float = 15.0,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Return ``forecasts`` with weather columns added and the raw capture log.

    ``venues`` must carry ``home_id`` (matching the forecast's ``home_team``
    team ID when the forecast was produced) with ``latitude``, ``longitude``
    and optional ``indoor`` columns. Games with no matching venue row
    receive a ``venue_not_in_registry`` snapshot but are kept.
    """
    if "home_id" not in forecasts.columns:
        raise ValueError("Forecast frame is missing a home_id column")
    venue_cols = {"home_id", "latitude", "longitude"}
    missing = venue_cols - set(venues.columns)
    if missing:
        raise ValueError(f"Venue frame missing required columns: {sorted(missing)}")
    has_indoor = "indoor" in venues.columns
    indexed = venues.set_index(venues.home_id.astype(int))
    records: list[WeatherSnapshot] = []
    captures: list[dict] = []
    for row in forecasts.itertuples(index=False):
        home_id = int(row.home_id)
        kickoff = row.kickoff
        game_id = int(row.game_id)
        if home_id not in indexed.index:
            snapshot = capture_weather(
                kickoff, observed_at=observed_at, timeout=timeout, now=now
            )
        else:
            venue_row = indexed.loc[home_id]
            latitude = float(venue_row["latitude"])
            longitude = float(venue_row["longitude"])
            indoor = bool(venue_row["indoor"]) if has_indoor else False
            snapshot = capture_weather(
                kickoff,
                latitude=latitude,
                longitude=longitude,
                indoor=indoor,
                observed_at=observed_at,
                timeout=timeout,
                now=now,
            )
        records.append(snapshot)
        captures.append({"game_id": game_id, **snapshot.to_record()})
    weather_df = pd.DataFrame([snapshot.to_record() for snapshot in records])
    enriched = forecasts.reset_index(drop=True).join(weather_df)
    return enriched, captures


def summarize_captures(captures: list[dict]) -> dict:
    """Summary counts for the weather enrichment pass, suitable for a manifest."""
    counts: dict[str, int] = {}
    for row in captures:
        status = row.get("weather_status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "games": len(captures),
        "statuses": counts,
        "captured_at_least_once": any(
            row.get("weather_status") == _SENTINEL_CAPTURED_STATUS for row in captures
        ),
    }


__all__ = [
    "WeatherSnapshot",
    "capture_weather",
    "enrich_forecasts_with_weather",
    "summarize_captures",
    "FORECAST_HORIZON_DAYS",
    "ARCHIVE_SWITCHOVER_DAYS",
]
