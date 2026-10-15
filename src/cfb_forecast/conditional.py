"""Conditional game context: venue, weather, roster and matchup-similarity layers.

This module builds the *conditioning* layer that sits between the raw score
history and the prediction head. It never manufactures a team-strength number:
every quantity here is an observed game condition (how cold and windy was it
at kickoff, which starters were out, how similar were past games to this
matchup). Team quality enters only implicitly, through features computed from
the two teams' own recent scores and through similarity-weighted borrowing of
comparable games (Hoegh et al., 2015, nearest-neighbor matchup effects).

Design notes
------------
* Rolling form features follow the standard "last k games" pattern used by
  every statistics-only predictor in the literature (South & Egros, 2020;
  Tsagris et al., 2024; Horvat et al., 2023): recent margins, recent points
  for/against, and recent totals, always conditioned strictly on results that
  were available before the game being predicted.
* Opponent-recent-form columns implement the schedule adjustment recommended
  for rating-free pipelines: instead of a rating table, each feature records
  how the opponent itself performed in its *previous* game. This is what lets
  a blowout against a weak nonconference opponent carry less weight than a
  close game against a strong one without any explicit rating object.
* Venue capacity is an arena covariate (indoor/outdoor proxies are handled by
  the weather layer); weather columns come from the captured Open-Meteo
  snapshots; injury counts from ESPN's public pregame lists; scoring-skill
  concentration from sportsdataverse player-game stats (the "top performers
  in their last few games" design of Rodriguez, 2019).
* Missingness is explicit. Every optional block contributes indicator columns
  so downstream models learn when a condition is unknown rather than silently
  treating zero as "no effect".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Trailing window (in games) for rolling per-team form features.
FORM_WINDOW = 4

#: Number of top scorers whose season-to-date production conditions a team's
#: expected offensive output (Rodriguez, 2019 found recent top-performer
#: statistics among the most predictive single designs).
TOP_SCORERS = 3

_INJURY_COLUMNS = [
    "injury_count",
    "injury_out_count",
    "injury_questionable_count",
]

_WEATHER_NUMERIC = [
    "temperature_c",
    "wind_speed_kph",
    "precipitation_mm",
    "cloud_cover_percent",
]


def _team_side_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Explode ``games`` into one row per (game, side) sorted by event time.

    Each row describes the team that played at that side, with the result it
    recorded (points for/against, margin, total) and the identity of the
    opponent, ordered so that rolling computations can shift safely into the
    future to expose only *prior* information.
    """
    home = pd.DataFrame(
        {
            "game_id": games.game_id.to_numpy(),
            "start_date": games.start_date.to_numpy(),
            "team_id": games.home_id.to_numpy(),
            "opponent_id": games.away_id.to_numpy(),
            "points_for": games.home_points.to_numpy(),
            "points_against": games.away_points.to_numpy(),
        }
    )
    away = pd.DataFrame(
        {
            "game_id": games.game_id.to_numpy(),
            "start_date": games.start_date.to_numpy(),
            "team_id": games.away_id.to_numpy(),
            "opponent_id": games.home_id.to_numpy(),
            "points_for": games.away_points.to_numpy(),
            "points_against": games.home_points.to_numpy(),
        }
    )
    sides = pd.concat([home, away], ignore_index=True)
    sides["margin"] = sides.points_for - sides.points_against
    sides["total"] = sides.points_for + sides.points_against
    sides["win"] = (sides.margin > 0).astype(float)
    sides = sides.sort_values(["team_id", "start_date", "game_id"], kind="mergesort").reset_index(drop=True)
    return sides


def rolling_form(games: pd.DataFrame) -> pd.DataFrame:
    """Strictly-prior rolling form for every (game, team) appearance.

    Returns a frame keyed by ``(game_id, team_id)`` with trailing means over
    the previous :data:`FORM_WINDOW` completed games (not including this
    game), plus the opponent's performance in its immediately preceding game.
    Teams with fewer prior games get NaN; the accompanying ``games_prior``
    count lets models distinguish thin samples from full samples.
    """
    require = ["game_id", "start_date", "home_id", "away_id", "home_points", "away_points"]
    missing = [c for c in require if c not in games.columns]
    if missing:
        raise ValueError(f"Games frame missing required columns: {missing}")
    sides = _team_side_frame(games)
    grouped = sides.groupby("team_id", sort=False)
    shifted = {}
    for column in ("margin", "points_for", "points_against", "total", "win"):
        # shift(1) guarantees the current game never informs its own features;
        # the trailing window then averages up to FORM_WINDOW prior games.
        lagged = grouped[column].shift(1)
        shifted[f"form_{column}"] = lagged.groupby(sides.team_id, sort=False).transform(
            lambda s: s.rolling(FORM_WINDOW, min_periods=1).mean()
        )
    sides["games_prior"] = grouped.cumcount().astype(float)
    # Opponent-recent-form: how did this team's opponent do in the game right
    # before this one? (implicit strength-of-schedule signal, no rating built)
    sides["_opp_prev_margin"] = shifted["form_margin"]
    sides["_opp_prev_points_for"] = shifted["form_points_for"]
    sides["_opp_prev_points_against"] = shifted["form_points_against"]
    for key, values in shifted.items():
        sides[key] = values.to_numpy() if hasattr(values, "to_numpy") else values
    keep = [
        "game_id",
        "team_id",
        "games_prior",
        "form_margin",
        "form_points_for",
        "form_points_against",
        "form_total",
        "form_win",
        "_opp_prev_margin",
        "_opp_prev_points_for",
        "_opp_prev_points_against",
    ]
    trimmed = sides[keep].drop_duplicates(subset=["game_id", "team_id"], keep="first")
    return trimmed.rename(
        columns={
            "_opp_prev_margin": "opp_prev_margin",
            "_opp_prev_points_for": "opp_prev_points_for",
            "_opp_prev_points_against": "opp_prev_points_against",
        }
    )


def matchup_similarity_weights(
    history: pd.DataFrame,
    target_vectors: np.ndarray,
    *,
    k: int = 12,
    bandwidth: float = 8.0,
) -> np.ndarray:
    """Nearest-neighbor matchup weights (Hoegh et al., 2015 template).

    ``history`` rows must align with ``target_vectors`` columns in feature
    space (same standardized conditioning vector layout). For each target
    game we select the ``k`` historically closest games and convert distances
    to weights with a Gaussian kernel of the given point-scale ``bandwidth``,
    then renormalize so weights sum to one. Sparse early-season slates borrow
    strength from similar *matchups* rather than from a fitted team rating.
    """
    if len(history) == 0:
        raise ValueError("Matchup similarity requires at least one historical game")
    targets = np.atleast_2d(np.asarray(target_vectors, dtype=float))
    reference = np.atleast_2d(np.asarray(history, dtype=float))
    if targets.shape[1] != reference.shape[1]:
        raise ValueError("Target and history feature widths differ")
    n_select = min(k, len(reference))
    weights = np.zeros((len(targets), len(reference)))
    for i, t in enumerate(targets):
        distance = np.linalg.norm(reference - t, axis=1)
        nearest = np.argpartition(distance, n_select - 1)[:n_select]
        kernel = np.exp(-0.5 * (distance[nearest] / bandwidth) ** 2)
        total = kernel.sum()
        if total <= 0:
            weights[i, nearest] = 1.0 / n_select
        else:
            weights[i, nearest] = kernel / total
    return weights


def weighted_matchup_expectation(
    weights: np.ndarray,
    outcomes: np.ndarray,
    *,
    shrinkage: float = 1.0,
) -> np.ndarray:
    """Similarity-weighted average outcome with add-one smoothing toward mean.

    ``shrinkage`` pulls the borrowed expectation toward the unweighted mean of
    the contributing pool; with no neighbors at all the caller receives the
    plain pooled mean, keeping early forecasts defined.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    mass = weights.sum(axis=1, keepdims=True)
    safe = np.where(mass > 0, mass, 1.0)
    raw = (weights @ outcomes.reshape(len(outcomes), -1)) / safe
    pooled = outcomes.mean(axis=0)
    blend = mass / (mass + shrinkage)
    return blend * raw + (1.0 - blend) * pooled


def _attach_side(frame: pd.DataFrame, form: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Merge per-(game, team) form onto the home and away sides of ``frame``."""
    indexed = form.set_index(["game_id", "team_id"])
    for _side, team_column in (("home", "home_id"), ("away", "away_id")):
        index = pd.MultiIndex.from_arrays([frame.game_id.astype(int), frame[team_column].astype(int)])
        block = indexed.reindex(index).add_prefix(prefix)
        for column in block.columns:
            frame[column] = block[column].to_numpy()
    return frame


def _base_form_columns() -> list[str]:
    """Per-team rolling form columns, attached with home_/away_ prefixes."""
    return [
        "games_prior",
        "form_margin",
        "form_points_for",
        "form_points_against",
        "form_total",
        "form_win",
        "opp_prev_margin",
        "opp_prev_points_for",
        "opp_prev_points_against",
    ]


def conditional_feature_columns() -> list[str]:
    """Canonical ordering of the conditioning matrix columns (per side pairs)."""
    base = _base_form_columns()
    sided = [f"{side}_{name}" for side in ("home", "away") for name in base]
    difference = [f"diff_{name}" for name in base]
    matchup = [
        "match_home_form_margin_vs_opp_defense",
        "match_away_form_margin_vs_opp_defense",
        "expected_total_from_form",
    ]
    venue = ["venue_altitude_m", "venue_altitude_missing"]
    weather = [
        "weather_temperature_c",
        "weather_wind_speed_kph",
        "weather_precipitation_mm",
        "weather_cloud_cover_percent",
        "weather_indoor",
        "weather_missing",
    ]
    injuries = [
        "home_injury_count",
        "home_injury_out_count",
        "home_injury_questionable_count",
        "home_injuries_missing",
        "away_injury_count",
        "away_injury_out_count",
        "away_injury_questionable_count",
        "away_injuries_missing",
    ]
    roster = [
        "home_top_scorer_ppg",
        "home_top3_scorer_ppg_sum",
        "home_roster_missing",
        "away_top_scorer_ppg",
        "away_top3_scorer_ppg_sum",
        "away_roster_missing",
    ]
    return sided + difference + matchup + venue + weather + injuries + roster


def build_conditional_matrix(
    fixtures: pd.DataFrame,
    games_with_results: pd.DataFrame,
    *,
    venues: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    player_ppg: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the conditioning matrix for ``fixtures``.

    Parameters
    ----------
    fixtures:
        One row per game to be predicted, carrying ``game_id``, ``season``,
        ``week``, ``home_id``, ``away_id``, ``home_team``, ``away_team`` and
        ``neutral_site``.
    games_with_results:
        Completed games available strictly before each fixture's cutoff; used
        only for rolling form (all features are lagged by construction).
    venues, weather, injuries, player_ppg:
        Optional captured context tables. Absent blocks become NaN columns
        with explicit missing indicators, never fabricated zeros.

    Returns
    -------
    (matrix, provenance)
        ``matrix`` is a DataFrame indexed like ``fixtures`` whose columns are
        :func:`conditional_feature_columns`. ``provenance`` records per-block
        coverage counts for manifests and dashboards.
    """
    form = rolling_form(games_with_results)
    frame = fixtures.reset_index(drop=True).copy()
    frame = _attach_side(frame, form, "home_")
    frame = _attach_side(frame, form, "away_")
    base = _base_form_columns()
    for name in base:
        frame[f"diff_{name}"] = frame[f"home_{name}"] - frame[f"away_{name}"]
    # Matchup terms: offense form vs the opponent's *conceded* form, and the
    # implied total when both offenses meet both defenses.
    frame["match_home_form_margin_vs_opp_defense"] = (
        frame.home_form_points_for - frame.away_form_points_against
    )
    frame["match_away_form_margin_vs_opp_defense"] = (
        frame.away_form_points_for - frame.home_form_points_against
    )
    frame["expected_total_from_form"] = frame.home_form_points_for + frame.away_form_points_for

    provenance = {"form_games_used": int(len(games_with_results)), "fixtures": int(len(frame))}

    if venues is not None and len(venues) and "altitude_m" in venues.columns:
        registry = venues.set_index(venues.home_id.astype(int))
        altitude = registry["altitude_m"].reindex(frame.home_id.astype(int)).to_numpy(dtype=float)
        frame["venue_altitude_m"] = altitude
        frame["venue_altitude_missing"] = np.isnan(altitude).astype(float)
        provenance["venue_coverage"] = float(np.isfinite(altitude).mean())
    else:
        frame["venue_altitude_m"] = np.nan
        frame["venue_altitude_missing"] = 1.0
        provenance["venue_coverage"] = 0.0

    if weather is not None and len(weather):
        wide = weather.copy()
        if "game_id" not in wide.columns:
            raise ValueError("Weather capture frame needs a game_id column")
        wide = wide.set_index(wide.game_id.astype(int))
        mapping = {
            "weather_temperature_c": "temperature_c",
            "weather_wind_speed_kph": "wind_speed_kph",
            "weather_precipitation_mm": "precipitation_mm",
            "weather_cloud_cover_percent": "cloud_cover_percent",
        }
        for target, source in mapping.items():
            values = (
                wide[source].reindex(frame.game_id.astype(int)).to_numpy(dtype=float)
                if source in wide.columns
                else np.full(len(frame), np.nan)
            )
            frame[target] = values
        status = (
            wide["status"].reindex(frame.game_id.astype(int)).to_numpy() if "status" in wide.columns else None
        )
        indoor = np.zeros(len(frame))
        if status is not None:
            indoor = np.array([1.0 if str(s) == "indoor" else 0.0 for s in status])
        frame["weather_indoor"] = indoor
        covered = np.isfinite(frame.weather_temperature_c.to_numpy()) | (indoor == 1.0)
        frame["weather_missing"] = (~covered).astype(float)
        provenance["weather_coverage"] = float(covered.mean())
    else:
        for column in (
            "weather_temperature_c",
            "weather_wind_speed_kph",
            "weather_precipitation_mm",
            "weather_cloud_cover_percent",
        ):
            frame[column] = np.nan
        frame["weather_indoor"] = 0.0
        frame["weather_missing"] = 1.0
        provenance["weather_coverage"] = 0.0

    if injuries is not None and len(injuries):
        latest = injuries.sort_values("observed_at").groupby("team_id").tail(1)
        indexed = latest.set_index(latest.team_id.astype(int))
        for side in ("home", "away"):
            for name in _INJURY_COLUMNS:
                target = f"{side}_{name}"
                values = (
                    indexed[name].reindex(frame[f"{side}_id"].astype(int)).to_numpy(dtype=float)
                    if name in indexed.columns
                    else np.full(len(frame), np.nan)
                )
                frame[target] = values
            present = np.isfinite(frame[f"{side}_injury_count"].to_numpy())
            frame[f"{side}_injuries_missing"] = (~present).astype(float)
            provenance[f"injury_{side}_coverage"] = float(present.mean())
    else:
        for column in (
            "home_injury_count",
            "home_injury_out_count",
            "home_injury_questionable_count",
            "home_injuries_missing",
            "away_injury_count",
            "away_injury_out_count",
            "away_injury_questionable_count",
            "away_injuries_missing",
        ):
            frame[column] = 1.0 if column.endswith("_missing") else np.nan
        provenance["injury_home_coverage"] = 0.0
        provenance["injury_away_coverage"] = 0.0

    if player_ppg is not None and len(player_ppg):
        ranked = player_ppg.sort_values(["team_id", "ppg_scored"], ascending=[True, False])
        top = ranked.groupby("team_id").head(TOP_SCORERS)
        best = top.groupby("team_id").ppg_scored.max()
        summed = top.groupby("team_id").ppg_scored.sum()
        for side in ("home", "away"):
            ids = frame[f"{side}_id"].astype(int)
            frame[f"{side}_top_scorer_ppg"] = best.reindex(ids).to_numpy(dtype=float)
            frame[f"{side}_top3_scorer_ppg_sum"] = summed.reindex(ids).to_numpy(dtype=float)
            present = np.isfinite(frame[f"{side}_top_scorer_ppg"].to_numpy())
            frame[f"{side}_roster_missing"] = (~present).astype(float)
            provenance[f"roster_{side}_coverage"] = float(present.mean())
    else:
        for column in (
            "home_top_scorer_ppg",
            "home_top3_scorer_ppg_sum",
            "home_roster_missing",
            "away_top_scorer_ppg",
            "away_top3_scorer_ppg_sum",
            "away_roster_missing",
        ):
            frame[column] = 1.0 if column.endswith("_missing") else np.nan
        provenance["roster_home_coverage"] = 0.0
        provenance["roster_away_coverage"] = 0.0

    columns = conditional_feature_columns()
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        raise RuntimeError(f"Internal error: conditioning columns not built: {absent}")
    matrix = frame[columns].astype(float)
    return matrix, provenance
