"""Freeze one prospective as-of state, then score the games due in a window.

Two entry points share every stage of the pipeline but differ only in how the
rating state is obtained:

- :func:`forecast` rebuilds the state from scratch by replaying all results
  available before the cutoff (see :func:`cfb_forecast.forecast_state.replay_state`).
- :func:`advance_forecast_snapshot` resumes an exact saved state and decays it
  forward, but refuses to run unless model hashes, source hashes, and result
  eligibility all still match the frozen provenance.

Both write the same artifact set into ``out_root``: ``forecasts.json/csv``,
optional distribution tables, ``state_snapshot.npz``, ``data_availability.json``
and a ``forecast_manifest.json`` provenance receipt. The anti-leakage contract
is enforced at load time: the model must be trained strictly before the
forecast season, the state clock never crosses the cutoff, and the advance mode
aborts if any final result became eligible between cutoffs.
"""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .availability import availability_snapshot
from .data import load_data, normalize_schedule, read_frame, sha256, timestamp
from .forecast_sim import DISTRIBUTION_STEMS, forecast_rows, frame_from_rows, pregame_features
from .forecast_state import replay_state, restore_state
from .market import read_market_history
from .model import ScoreHead, load_model

#: Human-readable description of how each mode produced its numbers; stored
#: verbatim in every forecast row so downstream consumers can tell modes apart.
PREDICTION_METHODS = {
    "replay": "dynamic Gaussian state + regularized score head + prior-season residual covariance",
    "advance": (
        "advanced exact saved state + dynamic Gaussian decay + regularized score head"
        " + prior-season residual covariance"
    ),
}

#: Caveat text recorded in each manifest receipt, per mode.
INFORMATION_LIMITS = {
    "replay": (
        "Downloaded record versions may postdate individual historical stat publication times."
        " Prospective current-season source capture is checked separately in data_availability.json."
    ),
    "advance": (
        "Advanced from an exact saved state only because no final result became newly eligible"
        " between cutoffs. Historical source vintages remain the same fixed hashes."
    ),
}


# ---------------------------------------------------------------------------
# Stage 1: validate inputs and load the leak-checked model
# ---------------------------------------------------------------------------

def _validate_window(as_of, through):
    """Return validated ``(cutoff, window end)`` UTC timestamps."""
    at, end = timestamp(as_of), timestamp(through)
    if end <= at:
        raise ValueError("Forecast window must end after as-of time")
    return at, end


def _load_leak_checked_model(model_root, season):
    """Load the model and refuse heads trained on the forecast season."""
    model = load_model(model_root)
    # A fitted head containing the forecast season would be retrospective leakage.
    if model["training_through_season"] >= season:
        raise ValueError("Model training must end before the forecast season")
    return model


# ---------------------------------------------------------------------------
# Stage 2: resolve sources and verify their integrity
# ---------------------------------------------------------------------------

def _load_sources(data_root):
    """Normalized schedules, advanced summaries, and source paths for a data root."""
    return load_data(data_root)


def _verify_saved_source_hashes(data_root, expected_hashes):
    """Confirm every source file behind a saved forecast is byte-identical.

    Returns the resolved paths in manifest order. Raises on any mismatch so an
    advance can never silently score against edited history.
    """
    source_paths = []
    for name, expected in expected_hashes.items():
        p = Path(data_root) / name
        if not p.exists() or sha256(p) != expected:
            raise ValueError(f"Source hash mismatch for {name}")
        source_paths.append(p)
    return source_paths


def _assert_model_matches_saved_state(model_root, old_manifest):
    if sha256(Path(model_root) / "model.json") != old_manifest["model_sha256"]:
        raise ValueError("Model no longer matches saved forecast state")


# ---------------------------------------------------------------------------
# Stage 3: pick target games inside the window
# ---------------------------------------------------------------------------

def _select_targets(schedules, at, end, season, target_game_ids):
    """FBS-v-FBS games of ``season`` kicking off within ``(at, end]``."""
    targets = schedules[
        schedules.fbs_game
        & (schedules.season == season)
        & (schedules.start_date > at)
        & (schedules.start_date <= end)
    ].copy()
    if target_game_ids is not None:
        wanted = {int(x) for x in target_game_ids}
        targets = targets[targets.game_id.astype(int).isin(wanted)].copy()
        missing = wanted - set(targets.game_id.astype(int))
        if missing:
            raise ValueError(
                f"Requested target game IDs are not eligible in the forecast window: {sorted(missing)}"
            )
    if targets.empty:
        raise ValueError("No FBS-v-FBS games in the requested forecast window")
    return targets


def _assert_no_newly_eligible_results(schedules, old_at, at):
    """Abort an advance if any usable final result landed between the cutoffs."""
    newly_eligible = schedules[
        schedules.result_usable & (schedules.result_available_at > old_at) & (schedules.result_available_at <= at)
    ]
    if len(newly_eligible):
        raise ValueError(
            f"{len(newly_eligible)} result(s) became available after the saved state; full replay is required"
        )


# ---------------------------------------------------------------------------
# Stage 4: score the targets
# ---------------------------------------------------------------------------

def _score_targets(targets, state, model, at, market_path, market_provider, draws,
                   prior_fbs_games, prediction_method, save_distribution):
    """Run the score head and Monte Carlo simulation for every target game.

    Returns ``(rows, joint, margins, totals)`` where the last three are the
    per-game probability tables collected when ``save_distribution`` is true.
    """
    F = pregame_features(state, targets)
    mu = ScoreHead.from_dict(model["head"]).predict(F)
    market = read_market_history(market_path) if market_path else None
    joint, margins, totals = [], [], []
    rows = forecast_rows(
        targets, mu, model, state, at, market, market_provider, draws,
        prior_fbs_games=prior_fbs_games,
        prediction_method=prediction_method,
        joint=joint, margins=margins, totals=totals,
        save_distribution=save_distribution,
    )
    return rows, joint, margins, totals, F


# ---------------------------------------------------------------------------
# Stage 5: write the artifact set
# ---------------------------------------------------------------------------

def _write_outputs(out, rows, joint, margins, totals, save_distribution):
    """Write forecasts JSON/CSV plus optional distribution tables.

    Distribution tables prefer parquet and fall back to CSV when no parquet
    engine is installed; the actual filenames are returned for the receipt.
    """
    (out / "forecasts.json").write_text(json.dumps(rows, indent=2))
    frame_from_rows(rows).to_csv(out / "forecasts.csv", index=False)
    distribution_files = []
    if save_distribution:
        for stem, frames in zip(DISTRIBUTION_STEMS, (joint, margins, totals)):
            table = pd.concat(frames, ignore_index=True)
            try:
                table.to_parquet(out / f"{stem}.parquet", index=False)
                distribution_files.append(f"{stem}.parquet")
            except (ImportError, ModuleNotFoundError):
                table.to_csv(out / f"{stem}.csv", index=False)
                distribution_files.append(f"{stem}.csv")
    return distribution_files


def _save_state(out, state, F):
    np.savez_compressed(out / "state_snapshot.npz", team_ids=state.teams, mean=state.mean, cov=state.cov, features=F)


def _copy_market(out, market_path):
    """Archive the market file next to the forecasts; return its stored name."""
    if not market_path:
        return None
    mp = Path(market_path)
    market_file = f"market_history{mp.suffix.lower()}"
    shutil.copy2(mp, out / market_file)
    return market_file


# ---------------------------------------------------------------------------
# Stage 6: provenance receipts
# ---------------------------------------------------------------------------

def _manifest_trace(at, end, model_root, *, source_hashes, market_path, market_provider,
                    used_game_ids, latest_result_available_at, distribution_files,
                    target_game_ids, data_availability_check, information_limit,
                    extra=None):
    """Assemble the provenance receipt written as ``forecast_manifest.json``."""
    trace = {
        "as_of": at.isoformat(),
        "through": end.isoformat(),
        "model_sha256": sha256(Path(model_root) / "model.json"),
        "source_hashes": source_hashes,
        "market_sha256": sha256(market_path) if market_path else None,
        "market_provider": market_provider,
        "used_game_ids": used_game_ids,
        "latest_result_available_at": latest_result_available_at,
        "created_at": datetime.now(UTC).isoformat(),
        "distribution_files": distribution_files,
        "target_game_ids": target_game_ids,
        "data_availability_check": data_availability_check,
        "information_limit": information_limit,
    }
    if extra:
        trace.update(extra)
    return trace


def _write_availability(out, avail):
    (out / "data_availability.json").write_text(json.dumps(avail, indent=2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast(data_root, model_root, out_root, as_of, through, season, market_path=None,
             draws=50000, market_provider=None, save_distribution=True, target_game_ids=None):
    """Full replay: rebuild the rating state from all sources available at ``as_of``."""
    at, end = _validate_window(as_of, through)
    model = _load_leak_checked_model(model_root, season)
    schedules, summaries, paths = _load_sources(data_root)
    population = schedules[
        schedules.fbs_game & schedules.season.between(model["warmup_season"], season)
    ].copy().reset_index(drop=True)
    targets = _select_targets(schedules, at, end, season, target_game_ids)
    past = population[population.result_usable & (population.result_available_at < at)].copy().reset_index(drop=True)
    if not (past.season == model["warmup_season"]).any():
        raise ValueError("Historical warmup season missing")
    state = replay_state(population, past, summaries, model, at)
    rows, joint, margins, totals, F = _score_targets(
        targets, state, model, at, market_path, market_provider, draws,
        prior_fbs_games=len(past),
        prediction_method=PREDICTION_METHODS["replay"],
        save_distribution=save_distribution,
    )
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    market_file = _copy_market(out, market_path)
    distribution_files = _write_outputs(out, rows, joint, margins, totals, save_distribution)
    _save_state(out, state, F)
    avail = availability_snapshot(data_root, at, schedules, summaries, paths)
    _write_availability(out, avail)
    trace = _manifest_trace(
        at, end, model_root,
        source_hashes={p.name: sha256(p) for p in paths},
        market_path=market_path, market_provider=market_provider,
        used_game_ids=past.game_id.astype(int).tolist(),
        latest_result_available_at=state.latest_update.isoformat(),
        distribution_files=distribution_files,
        target_game_ids=targets.game_id.astype(int).tolist(),
        data_availability_check=avail["prospective_current_season_source_check"],
        information_limit=INFORMATION_LIMITS["replay"],
        extra={"market_file": market_file},
    )
    (out / "forecast_manifest.json").write_text(json.dumps(trace, indent=2))
    return frame_from_rows(rows)


def advance_forecast_snapshot(previous_forecast_dir, data_root, model_root, out_root, as_of,
                              through, season, market_path=None, draws=50000,
                              market_provider=None, save_distribution=True, target_game_ids=None):
    """Advance an exact saved forecast state when no new result became eligible.

    This is useful for prospective freezes between game completions. It refuses
    to cross any newly available final result because doing so would require the
    game-level advanced summaries to perform the corresponding state update.
    """
    prev = Path(previous_forecast_dir)
    data_root = Path(data_root)
    old_manifest = json.loads((prev / "forecast_manifest.json").read_text())
    old_at = timestamp(old_manifest["as_of"])
    at, end = _validate_window(as_of, through)
    if at <= old_at:
        raise ValueError("Advanced forecast cutoff must be later than the saved state cutoff")
    model = _load_leak_checked_model(model_root, season)
    _assert_model_matches_saved_state(model_root, old_manifest)
    source_paths = _verify_saved_source_hashes(data_root, old_manifest["source_hashes"])
    schedule_paths = sorted(data_root.glob("schedule_*.csv"))
    schedules = normalize_schedule(pd.concat([read_frame(p) for p in schedule_paths], ignore_index=True))
    _assert_no_newly_eligible_results(schedules, old_at, at)
    targets = _select_targets(schedules, at, end, season, target_game_ids)
    state = restore_state(np.load(prev / "state_snapshot.npz"), old_manifest, model, old_at, at)
    prior_count = int(pd.read_csv(prev / "forecasts.csv").prior_fbs_games.iloc[0])
    rows, joint, margins, totals, F = _score_targets(
        targets, state, model, at, market_path, market_provider, draws,
        prior_fbs_games=prior_count,
        prediction_method=PREDICTION_METHODS["advance"],
        save_distribution=save_distribution,
    )
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    distribution_files = _write_outputs(out, rows, joint, margins, totals, save_distribution)
    _save_state(out, state, F)
    avail = availability_snapshot(data_root, at, schedules, None, source_paths)
    avail["advanced_from_saved_state"] = {
        "previous_forecast": str(prev),
        "previous_as_of": old_at.isoformat(),
        "newly_eligible_results_between_cutoffs": 0,
    }
    _write_availability(out, avail)
    trace = _manifest_trace(
        at, end, model_root,
        source_hashes=old_manifest["source_hashes"],
        market_path=market_path, market_provider=market_provider,
        used_game_ids=old_manifest.get("used_game_ids", []),
        latest_result_available_at=state.latest_update.isoformat() if state.latest_update is not None else None,
        distribution_files=distribution_files,
        target_game_ids=targets.game_id.astype(int).tolist(),
        data_availability_check=avail["prospective_current_season_source_check"],
        information_limit=INFORMATION_LIMITS["advance"],
        extra={
            "state_transition": {
                "from_as_of": old_at.isoformat(),
                "to_as_of": at.isoformat(),
                "newly_eligible_results": 0,
            },
        },
    )
    (out / "forecast_manifest.json").write_text(json.dumps(trace, indent=2))
    return frame_from_rows(rows)
