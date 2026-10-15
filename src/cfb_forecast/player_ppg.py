"""Player-level season-to-date points-per-game.

Player PPG for college football is not a native box-score field: the box
score records who scored each touchdown, who kicked the extra points and
the field goals, but it does not sum them to a player total. This module
does that sum properly from a player-game-stats frame and then averages
over games played to date, with the same strict cutoff rule the team PPG
and the dynamic ratings use.

Expected input is a player-game frame — one row per player per game — with
the columns a normalized college-football release carries: a player
identifier, a team identifier, a game identifier, and the per-game count
columns ``passing_td``, ``rushing_td``, ``receiving_td``, ``field_goals_made``,
``extra_points_made`` and ``two_point_conversions``. We compute points as
``6 * (passing_td + rushing_td + receiving_td) + 3 * field_goals_made +
extra_points_made + 2 * two_point_conversions``. Defensive and return
touchdowns usually show up on a separate stat line in these releases; when
the frame includes them under a ``defensive_td`` or ``return_td`` column
they are counted the same way.

The dataset itself has to be loaded upstream, since player-game stats are
not shipped in this repo. The ``cfbd`` free tier and the SportsDataverse
player stats parquet releases both provide compatible frames; the
``fetch_player_game_stats`` helper reads the latter and normalizes the
columns the compute function needs.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .data import timestamp

log = logging.getLogger(__name__)

_SPORTSDATAVERSE_PLAYER_STATS = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "cfb_player_stats/cfb_player_stats_{year}.parquet"
)

STATUS_CAPTURED = "captured"
STATUS_EMPTY = "no_player_rows"
STATUS_NETWORK_ERROR = "network_or_parse_error"

PLAYER_PPG_COLUMNS = [
    "player_id",
    "player_name",
    "team_id",
    "position",
    "games_played_to_date",
    "points_scored_total",
    "ppg_scored",
    "passing_td_total",
    "rushing_td_total",
    "receiving_td_total",
    "field_goals_made_total",
    "extra_points_made_total",
    "two_point_conversions_total",
    "as_of",
]

#: Columns the compute function expects on a player-game frame. Missing
#: columns are treated as zero so a dataset that does not break out, for
#: example, two-point conversions still works.
_SCORING_COMPONENTS = (
    ("passing_td", 6),
    ("rushing_td", 6),
    ("receiving_td", 6),
    ("defensive_td", 6),
    ("return_td", 6),
    ("field_goals_made", 3),
    ("extra_points_made", 1),
    ("two_point_conversions", 2),
)


@dataclass(frozen=True)
class PlayerStatsCapture:
    """Result of a player-game-stats fetch."""

    season: int
    status: str
    observed_at: str
    request_url: str
    rows: int
    note: str | None = None


def _as_int(value) -> int:
    if value is None:
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if number != number:  # NaN
        return 0
    return int(number)


def compute_player_points_column(player_games: pd.DataFrame) -> pd.Series:
    """Compute per-row points scored for each player-game.

    Missing scoring columns are treated as zero. The function never
    mutates the input frame.
    """
    points = pd.Series(0, index=player_games.index, dtype="int64")
    for column, weight in _SCORING_COMPONENTS:
        if column in player_games.columns:
            series = player_games[column].fillna(0).astype(int)
            points = points + series * weight
    return points


def compute_player_ppg(
    player_games: pd.DataFrame,
    *,
    as_of,
    season: int | None = None,
    min_games: int = 1,
) -> pd.DataFrame:
    """Return a per-player season-to-date PPG table.

    Parameters
    ----------
    player_games:
        Player-game frame with at least ``player_id``, ``team_id``,
        ``game_id``, ``game_date`` (or ``start_date``) and the scoring
        component columns described in the module docstring. ``player_name``
        and ``position`` are used when present and preserved in the output.
    as_of:
        Cutoff timestamp. A player-game only contributes once its game
        date is strictly earlier than this time.
    season:
        Optional season filter. When set, only rows whose ``season``
        matches are included.
    min_games:
        Drop players with fewer than this many usable games before the
        cutoff. Defaults to 1 (keep anyone who played at least once).

    Returns
    -------
    pd.DataFrame
        One row per player, columns in :data:`PLAYER_PPG_COLUMNS`.
    """
    if player_games.empty:
        return pd.DataFrame(columns=PLAYER_PPG_COLUMNS)
    cutoff = timestamp(as_of)
    frame = player_games.copy()
    if season is not None and "season" in frame.columns:
        frame = frame[frame.season.eq(int(season))]
    date_column = (
        "game_date"
        if "game_date" in frame.columns
        else ("start_date" if "start_date" in frame.columns else None)
    )
    if date_column is None:
        raise ValueError("player_games needs a 'game_date' or 'start_date' column for the cutoff.")
    frame[date_column] = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
    frame = frame[frame[date_column].notna() & frame[date_column].lt(cutoff)]
    if frame.empty:
        return pd.DataFrame(columns=PLAYER_PPG_COLUMNS)
    frame = frame.copy()
    frame["_points"] = compute_player_points_column(frame)
    group_keys = ["player_id"]
    aggregations: dict[str, tuple[str, str]] = {
        "games_played_to_date": ("game_id", "nunique") if "game_id" in frame.columns else ("_points", "size"),
        "points_scored_total": ("_points", "sum"),
    }
    for column, _weight in _SCORING_COMPONENTS:
        if column in frame.columns and column.endswith("_td"):
            aggregations[f"{column}_total"] = (column, "sum")
    for column, _weight in _SCORING_COMPONENTS:
        if column in frame.columns and not column.endswith("_td"):
            aggregations[f"{column}_total"] = (column, "sum")
    grouped = frame.groupby(group_keys, as_index=False).agg(**aggregations)
    # Attach the most recent team, name and position for each player.
    sort_cols = [date_column]
    latest = frame.sort_values(sort_cols).groupby("player_id").tail(1)
    identity_cols = ["player_id"]
    for optional in ("player_name", "team_id", "position"):
        if optional in latest.columns:
            identity_cols.append(optional)
    identity = latest[identity_cols].drop_duplicates("player_id")
    merged = identity.merge(grouped, on="player_id", how="right")
    merged["ppg_scored"] = merged.points_scored_total / merged.games_played_to_date
    merged["as_of"] = cutoff.isoformat()
    merged = merged[merged.games_played_to_date >= int(min_games)]
    for column in PLAYER_PPG_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
    log.info(
        "player_ppg: players=%d min_games=%d as_of=%s",
        len(merged),
        min_games,
        cutoff.isoformat(),
    )
    return merged[PLAYER_PPG_COLUMNS].reset_index(drop=True)


def summarize_player_ppg(player_ppg: pd.DataFrame, *, top_n: int = 10) -> dict:
    """Return a short rollup: how many players, the top scorers, averages."""
    if player_ppg.empty:
        return {"players": 0, "top_scorers": [], "mean_ppg": None}
    sorted_frame = player_ppg.sort_values("ppg_scored", ascending=False)
    top = sorted_frame.head(top_n)
    top_list = [
        {
            "player_id": row.player_id,
            "player_name": getattr(row, "player_name", None),
            "team_id": getattr(row, "team_id", None),
            "ppg_scored": float(row.ppg_scored),
            "games_played_to_date": int(row.games_played_to_date),
        }
        for row in top.itertuples()
    ]
    return {
        "players": int(len(player_ppg)),
        "top_scorers": top_list,
        "mean_ppg": float(player_ppg.ppg_scored.mean()),
    }


def _download_file(url: str, target: Path, timeout: float = 60.0) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "cfb-forecast/0.4"})
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=timeout) as response, target.open("wb") as handle:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(chunk)


def fetch_player_game_stats(
    season: int,
    *,
    cache_dir=None,
    timeout: float = 60.0,
) -> tuple[pd.DataFrame, PlayerStatsCapture]:
    """Download one season of player-game stats from SportsDataverse.

    Returns the frame (empty on failure) alongside a capture record for
    the audit log. The parquet is cached under ``cache_dir`` (default
    ``data/player_stats``).
    """
    season = int(season)
    url = _SPORTSDATAVERSE_PLAYER_STATS.format(year=season)
    cache_root = Path(cache_dir) if cache_dir is not None else Path("data/player_stats")
    cached_path = cache_root / f"cfb_player_stats_{season}.parquet"
    observed = datetime.now(UTC).isoformat()
    try:
        if not cached_path.exists():
            _download_file(url, cached_path, timeout=timeout)
        frame = pd.read_parquet(cached_path)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        OSError,
        ValueError,
        ImportError,
    ) as exc:
        log.warning("player stats fetch failed for season %s: %s", season, exc)
        return pd.DataFrame(), PlayerStatsCapture(
            season=season,
            status=STATUS_NETWORK_ERROR,
            observed_at=observed,
            request_url=url,
            rows=0,
            note=f"{type(exc).__name__}: {exc}",
        )
    if frame.empty:
        return frame, PlayerStatsCapture(
            season=season,
            status=STATUS_EMPTY,
            observed_at=observed,
            request_url=url,
            rows=0,
            note="SportsDataverse returned an empty player stats frame.",
        )
    return frame, PlayerStatsCapture(
        season=season,
        status=STATUS_CAPTURED,
        observed_at=observed,
        request_url=url,
        rows=int(len(frame)),
    )


__all__ = [
    "PLAYER_PPG_COLUMNS",
    "PlayerStatsCapture",
    "STATUS_CAPTURED",
    "STATUS_EMPTY",
    "STATUS_NETWORK_ERROR",
    "compute_player_points_column",
    "compute_player_ppg",
    "fetch_player_game_stats",
    "summarize_player_ppg",
]
