"""Replay only available prior results, then freeze one as-of state snapshot."""

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import numpy as np
import pandas as pd
from .config import StateConfig
from .data import load_data, timestamp, sha256
from .features import observations
from .ratings import DynamicRatings, build_features
from .model import load_model, ScoreHead
from .market import market_at_cutoff, market_history_summary, read_market_history
from .availability import availability_snapshot
from .analytic import score_summary
from .decision import moneyline_diagnostics
from .strength import strength_probability


class SnapshotReplayRequired(ValueError):
    """The cached state cannot be reused; replay the current source snapshot."""


def archive_market(path, out):
    if not path or not Path(path).exists():
        return {"market_file": None, "market_sha256": None}
    source = Path(path)
    name = f"market_history{source.suffix.lower()}"
    target = out / name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return {"market_file": name, "market_sha256": sha256(target)}


def forecast(
    data_root,
    model_root,
    out_root,
    as_of,
    through,
    season,
    market_path=None,
    draws=50000,
    market_provider=None,
    save_distribution=True,
    target_game_ids=None,
    analytic=True,
):
    analytic = True
    save_distribution = False
    at = timestamp(as_of)
    end = timestamp(through)
    if end <= at:
        raise ValueError("Forecast window must end after as-of time")
    model = load_model(model_root)
    if model.get("adaptive_training_cutoff") and timestamp(model["adaptive_training_cutoff"]) > at:
        raise ValueError("Adaptive model was trained after the requested forecast cutoff")
    if model.get("adaptive_latest_result") and timestamp(model["adaptive_latest_result"]) >= at:
        raise ValueError("Adaptive model includes results at or after forecast cutoff")
    # A fitted head containing the forecast season would be retrospective leakage.
    if model["training_through_season"] >= season:
        raise ValueError("Model training must end before the forecast season")
    schedules, summaries, paths = load_data(data_root)
    population = (
        schedules[
            schedules.fbs_game
            & schedules.season.between(model["warmup_season"], season)
        ]
        .copy()
        .reset_index(drop=True)
    )
    targets = population[
        (population.season == season)
        & (population.start_date > at)
        & (population.start_date <= end)
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
    past = (
        population[population.result_usable & (population.result_available_at < at)]
        .copy()
        .reset_index(drop=True)
    )
    if not (past.season == model["warmup_season"]).any():
        raise ValueError("Historical warmup season missing")
    teams = sorted(set(population.home_id) | set(population.away_id))
    state = DynamicRatings(
        teams, model["center"], model["scale"], StateConfig(**model["state_config"])
    )
    Y = observations(past, summaries)
    events = []
    for year, g in population.groupby("season"):
        reset = g.start_date.min() - pd.Timedelta(days=21)
        if reset < at:
            events.append((reset.value, 0, int(year)))
    records = list(past.itertuples(index=False))
    for k, r in enumerate(records):
        events.append((r.result_available_at.value, 1, k))
    for tm, typ, k in sorted(events):
        event_at = pd.Timestamp(tm, tz="UTC")
        if typ == 0:
            state.offseason(event_at)
        else:
            r = records[k]
            state.update(r.home_id, r.away_id, r.neutral_site, Y[k], event_at)
    state.advance(at)
    F = np.stack(
        [
            state.features(r.home_id, r.away_id, r.neutral_site, int(r.week) <= 4)
            for r in targets.itertuples(index=False)
        ]
    )
    historical_F, _, _ = build_features(
        past, Y, model["center"], model["scale"],
        StateConfig(**model["state_config"]), warmup=model["warmup_season"]
    )
    fit_mask = np.isfinite(historical_F).all(axis=(1, 2)) & np.isfinite(Y).all(axis=(1, 2))
    if int(fit_mask.sum()) < 50:
        raise ValueError("At least 50 completed historical games are required for a live score-head fit")
    age_days = (at - pd.to_datetime(past.result_available_at, utc=True)).dt.total_seconds().to_numpy() / 86400.0
    decay = float(model.get("reinforcement_decay", 0.002))
    live_head = ScoreHead(model["variant"], model["head"]["alpha"]).fit(
        historical_F, Y, fit_mask, np.exp(-decay * np.maximum(age_days, 0.0))
    )
    mu = live_head.predict(F)
    market = read_market_history(market_path) if market_path else None
    strength_p = None
    rows = []
    joint = []
    margins = []
    totals = []
    for position, (r, means) in enumerate(zip(targets.itertuples(index=False), mu)):
        m = (
            market_at_cutoff(
                market, r.game_id, at, r.start_date, provider=market_provider
            )
            if market is not None and not market.empty
            else {}
        )
        hist = (
            market_history_summary(
                market, r.game_id, at, r.start_date, provider=market_provider
            )
            if market is not None and not market.empty
            else {"market_snapshot_count": 0}
        )
        sim = score_summary(means, model["score_covariance"], **m)
        sim["score_head_home_win_probability"] = sim["home_win_probability"]
        sim["raw_home_win_probability"] = (
            float(strength_p[position])
            if strength_p is not None
            else sim["home_win_probability"]
        )
        sim["probability_source"] = (
            "dynamic_strength"
            if strength_p is not None
            else model["variant"] + "_score_head"
        )
        sim["home_win_probability"] = float(sim["raw_home_win_probability"])
        sim.update(
            moneyline_diagnostics(
                sim["home_win_probability"],
                m.get("home_ml_decimal"),
                m.get("away_ml_decimal"),
            )
        )
        rows.append(
            {
                "game_id": int(r.game_id),
                "season": int(r.season),
                "home_team": r.home_team,
                "away_team": r.away_team,
                "kickoff": r.start_date.isoformat(),
                "as_of": at.isoformat(),
                "latest_result_available_at": state.latest_update.isoformat()
                if state.latest_update is not None
                else None,
                "prior_fbs_games": len(past),
                "model_variant": model["variant"],
                "head_training_through_season": model["training_through_season"],
                "prediction_method": "live dynamic Bayesian hierarchical ratings + recency-weighted score head + posterior score covariance",
                **sim,
                **m,
                **hist,
            }
        )
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    market_archive = archive_market(market_path, out)
    market_file = market_archive["market_file"]
    (out / "forecasts.json").write_text(json.dumps(rows, indent=2))
    pd.DataFrame(rows).to_csv(out / "forecasts.csv", index=False)
    distribution_files = []
    if save_distribution:
        for stem, frames in [
            ("score_distribution", joint),
            ("margin_distribution", margins),
            ("total_distribution", totals),
        ]:
            table = pd.concat(frames, ignore_index=True)
            try:
                table.to_parquet(out / f"{stem}.parquet", index=False)
                distribution_files.append(f"{stem}.parquet")
            except (ImportError, ModuleNotFoundError):
                table.to_csv(out / f"{stem}.csv", index=False)
                distribution_files.append(f"{stem}.csv")
    np.savez_compressed(
        out / "state_snapshot.npz",
        team_ids=state.teams,
        mean=state.mean,
        cov=state.cov,
        features=F,
    )
    avail = availability_snapshot(data_root, at, schedules, summaries, paths)
    (out / "data_availability.json").write_text(json.dumps(avail, indent=2))
    trace = {
        "as_of": at.isoformat(),
        "through": end.isoformat(),
        "model_sha256": sha256(Path(model_root) / "model.json"),
        "source_hashes": {p.name: sha256(p) for p in paths},
        "market_sha256": market_archive["market_sha256"],
        "market_file": market_file,
        "market_provider": market_provider,
        "used_game_ids": past.game_id.astype(int).tolist(),
        "latest_result_available_at": state.latest_update.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "distribution_files": distribution_files,
        "target_game_ids": targets.game_id.astype(int).tolist(),
        "data_availability_check": avail["prospective_current_season_source_check"],
        "information_limit": "Downloaded record versions may postdate individual historical stat publication times. Prospective current-season source capture is checked separately in data_availability.json.",
    }
    (out / "forecast_manifest.json").write_text(json.dumps(trace, indent=2))
    return pd.DataFrame(rows)


def advance_forecast_snapshot(
    previous_forecast_dir,
    data_root,
    model_root,
    out_root,
    as_of,
    through,
    season,
    market_path=None,
    draws=50000,
    market_provider=None,
    save_distribution=True,
    target_game_ids=None,
    analytic=True,
):
    """Advance an exact saved forecast state when no new result became eligible.

    This is useful for prospective freezes between game completions. It refuses to
    cross any newly available final result because doing so would require the
    game-level advanced summaries to perform the corresponding state update.
    """
    analytic = True
    save_distribution = False
    from .data import normalize_schedule

    prev = Path(previous_forecast_dir)
    data_root = Path(data_root)
    out = Path(out_root)
    old_manifest = json.loads((prev / "forecast_manifest.json").read_text())
    old_at = timestamp(old_manifest["as_of"])
    at = timestamp(as_of)
    end = timestamp(through)
    if at <= old_at:
        raise ValueError(
            "Advanced forecast cutoff must be later than the saved state cutoff"
        )
    if end <= at:
        raise ValueError("Forecast window must end after as-of time")
    model = load_model(model_root)
    if model["training_through_season"] >= season:
        raise ValueError("Model training must end before the forecast season")
    if sha256(Path(model_root) / "model.json") != old_manifest["model_sha256"]:
        raise ValueError("Model no longer matches saved forecast state")
    source_paths = []
    for name, expected in old_manifest["source_hashes"].items():
        p = data_root / name
        if not p.exists() or sha256(p) != expected:
            raise SnapshotReplayRequired(
                f"Source hash mismatch for {name}; full replay is required"
            )
        source_paths.append(p)
    schedule_paths = sorted(data_root.glob("schedule_*.csv"))
    schedules = normalize_schedule(
        pd.concat([pd.read_csv(p) for p in schedule_paths], ignore_index=True)
    )
    newly_eligible = schedules[
        schedules.result_usable
        & (schedules.result_available_at >= old_at)
        & (schedules.result_available_at < at)
    ]
    if len(newly_eligible):
        raise SnapshotReplayRequired(
            f"{len(newly_eligible)} result(s) became available after the saved state; full replay is required"
        )
    targets = schedules[
        (schedules.fbs_game)
        & (schedules.season == season)
        & (schedules.start_date > at)
        & (schedules.start_date <= end)
    ].copy()
    if target_game_ids is not None:
        ids = {int(x) for x in target_game_ids}
        targets = targets[targets.game_id.astype(int).isin(ids)].copy()
    if targets.empty:
        raise ValueError("No FBS-v-FBS games in the requested forecast window")
    snap = np.load(prev / "state_snapshot.npz")
    state = DynamicRatings(
        snap["team_ids"].astype(int).tolist(),
        model["center"],
        model["scale"],
        StateConfig(**model["state_config"]),
    )
    state.mean = snap["mean"].copy()
    state.cov = snap["cov"].copy()
    state.clock = old_at
    latest = old_manifest.get("latest_result_available_at")
    state.latest_update = timestamp(latest) if latest else None
    state.advance(at)
    F = np.stack(
        [
            state.features(r.home_id, r.away_id, r.neutral_site, int(r.week) <= 4)
            for r in targets.itertuples(index=False)
        ]
    )
    mu = ScoreHead.from_dict(model["head"]).predict(F)
    market = read_market_history(market_path) if market_path else None
    prior_count = int(pd.read_csv(prev / "forecasts.csv").prior_fbs_games.iloc[0])
    strength_p = (
        strength_probability(schedules, targets, at, model["win_strength_model"])
        if model.get("win_strength_model")
        else None
    )
    rows = []
    joint = []
    margins = []
    totals = []
    for position, (r, means) in enumerate(zip(targets.itertuples(index=False), mu)):
        m = (
            market_at_cutoff(
                market, r.game_id, at, r.start_date, provider=market_provider
            )
            if market is not None and not market.empty
            else {}
        )
        hist = (
            market_history_summary(
                market, r.game_id, at, r.start_date, provider=market_provider
            )
            if market is not None and not market.empty
            else {"market_snapshot_count": 0}
        )
        sim = score_summary(means, model["score_covariance"], **m)
        sim["score_head_home_win_probability"] = sim["home_win_probability"]
        sim["raw_home_win_probability"] = (
            float(strength_p[position])
            if strength_p is not None
            else sim["home_win_probability"]
        )
        sim["probability_source"] = (
            "dynamic_strength"
            if strength_p is not None
            else model["variant"] + "_score_head"
        )
        sim["home_win_probability"] = float(sim["raw_home_win_probability"])
        sim.update(
            moneyline_diagnostics(
                sim["home_win_probability"],
                m.get("home_ml_decimal"),
                m.get("away_ml_decimal"),
            )
        )
        rows.append(
            {
                "game_id": int(r.game_id),
                "season": int(r.season),
                "home_team": r.home_team,
                "away_team": r.away_team,
                "kickoff": r.start_date.isoformat(),
                "as_of": at.isoformat(),
                "latest_result_available_at": state.latest_update.isoformat()
                if state.latest_update is not None
                else None,
                "prior_fbs_games": prior_count,
                "model_variant": model["variant"],
                "head_training_through_season": model["training_through_season"],
                "prediction_method": "advanced exact saved state + dynamic Gaussian decay + regularized score head + prior-season residual covariance",
                **sim,
                **m,
                **hist,
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    market_archive = archive_market(market_path, out)
    (out / "forecasts.json").write_text(json.dumps(rows, indent=2))
    pd.DataFrame(rows).to_csv(out / "forecasts.csv", index=False)
    distribution_files = []
    if save_distribution:
        for stem, frames in [
            ("score_distribution", joint),
            ("margin_distribution", margins),
            ("total_distribution", totals),
        ]:
            table = pd.concat(frames, ignore_index=True)
            try:
                table.to_parquet(out / f"{stem}.parquet", index=False)
                distribution_files.append(f"{stem}.parquet")
            except (ImportError, ModuleNotFoundError):
                table.to_csv(out / f"{stem}.csv", index=False)
                distribution_files.append(f"{stem}.csv")
    np.savez_compressed(
        out / "state_snapshot.npz",
        team_ids=state.teams,
        mean=state.mean,
        cov=state.cov,
        features=F,
    )
    avail = availability_snapshot(data_root, at, schedules, None, source_paths)
    avail["advanced_from_saved_state"] = {
        "previous_forecast": str(prev),
        "previous_as_of": old_at.isoformat(),
        "newly_eligible_results_between_cutoffs": 0,
    }
    (out / "data_availability.json").write_text(json.dumps(avail, indent=2))
    trace = {
        "as_of": at.isoformat(),
        "through": end.isoformat(),
        "model_sha256": sha256(Path(model_root) / "model.json"),
        "source_hashes": old_manifest["source_hashes"],
        "market_sha256": market_archive["market_sha256"],
        "market_provider": market_provider,
        "used_game_ids": old_manifest.get("used_game_ids", []),
        "latest_result_available_at": state.latest_update.isoformat()
        if state.latest_update is not None
        else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "distribution_files": distribution_files,
        "target_game_ids": targets.game_id.astype(int).tolist(),
        "data_availability_check": avail["prospective_current_season_source_check"],
        "state_transition": {
            "from_as_of": old_at.isoformat(),
            "to_as_of": at.isoformat(),
            "newly_eligible_results": 0,
        },
        "information_limit": "Advanced from an exact saved state only because no final result became newly eligible between cutoffs. Historical source vintages remain the same fixed hashes.",
    }
    trace.update(market_archive)
    (out / "forecast_manifest.json").write_text(json.dumps(trace, indent=2))
    return pd.DataFrame(rows)
