"""Venue coordinates for home teams, used by the weather enrichment step.

Expected CSV schema (``data/stadiums.csv``):

- ``home_id``: integer ESPN team ID of the home team (matches the schedule's
  ``home_id`` field).
- ``team``: display name of the team (not used for matching, for human
  review only).
- ``stadium``: display name of the stadium.
- ``latitude``, ``longitude``: WGS-84 decimal degrees.
- ``indoor``: ``true`` or ``false`` — whether the stadium is a controlled
  environment (dome, retractable roof usually closed, or arena).

Rows for teams that are not in the registry simply mean the weather
enrichment pass will mark those games as ``venue_not_in_registry`` instead
of calling the API. The registry can be extended by the user without a code
change.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .data import booleans, require_columns

log = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ["home_id", "team", "stadium", "latitude", "longitude", "indoor"]


def load_venues(path) -> pd.DataFrame:
    """Load and validate a stadium registry CSV.

    Returns an empty frame with the expected schema when the file does not
    exist, so callers can treat "missing registry" and "empty registry" the
    same way.
    """
    path = Path(path)
    if not path.exists():
        log.info("venues: no registry at %s; weather enrichment will skip venues", path)
        return pd.DataFrame(columns=_REQUIRED_COLUMNS)
    frame = pd.read_csv(path)
    require_columns(frame, _REQUIRED_COLUMNS, "venues")
    frame = frame.copy()
    frame["home_id"] = pd.to_numeric(frame.home_id, errors="raise").astype("int64")
    frame["latitude"] = pd.to_numeric(frame.latitude, errors="raise").astype(float)
    frame["longitude"] = pd.to_numeric(frame.longitude, errors="raise").astype(float)
    frame["indoor"] = booleans(frame.indoor, "indoor")
    if frame.home_id.duplicated().any():
        raise ValueError("Stadium registry contains duplicate home_id rows")
    if (frame.latitude.abs() > 90).any() or (frame.longitude.abs() > 180).any():
        raise ValueError("Stadium registry contains invalid latitude/longitude values")
    return frame.reset_index(drop=True)


__all__ = ["load_venues"]
