"""Replay only available prior results, then freeze one as-of state snapshot."""

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import numpy as np
import pandas as pd
from ..config import StateConfig
from ..ingest.data import load_data, timestamp, sha256
from ..ingest.features import observations, pregame_history, load_history_facts
from .ratings import DynamicRatings, build_features, early_season
from .model import load_model, ScoreHead
from ..odds.market import closing_lines, market_at_cutoff, market_history_summary, read_market_history
from ..ingest.availability import availability_snapshot
from ..odds.decision import moneyline_diagnostics
from .forecast_model import METHOD, ForecastModel, known_future_matrix, rest_days
from ..paths import database_path


LOCATION_HEAD = "location_head.json"


class SnapshotReplayRequired(ValueError):
    """The cached state cannot be reused; replay the current source snapshot."""


def _known_frame(games, schedules, market_rows, extra):
    frame = games.reset_index(drop=True).copy()
    rested = rest_days(frame, schedules)
    frame = frame.merge(rested, on="game_id", how="left")
    for column in ("home_spread", "total_line", "home_ml_decimal", "away_ml_decimal"):
        frame[column] = [row.get(column) for row in market_rows]
    if extra is not None and len(extra) and "game_id" in extra.columns:
        keep = [column for column in extra.columns if column == "game_id" or column in frame.columns or column.startswith(("weather_", "home_", "away_"))]
        extra_keep = extra[keep].drop_duplicates("game_id")
        overlap = [column for column in extra_keep.columns if column != "game_id" and column in frame.columns]
        frame = frame.drop(columns=overlap).merge(extra_keep, on="game_id", how="left")
    return frame


def _project_rows(targets, means, fitted, covariance, schedules, market_rows, history_rows, extra, metadata, history=None):
    known = known_future_matrix(_known_frame(targets, schedules, market_rows, extra))
    projected = fitted.apply(
        means,
        known,
        covariance,
        market_rows,
        targets.game_id.astype(int).tolist(),
        history=history,
    )
    rows = []
    for game, sim, market, history in zip(targets.itertuples(index=False), projected, market_rows, history_rows):
        sim["score_head_home_win_probability"] = float(sim["gaussian_home_win_probability"])
        sim["raw_home_win_probability"] = float(sim["home_win_probability"])
        sim["probability_source"] = "forecast_model"
        sim.update(
            moneyline_diagnostics(
                sim["home_win_probability"],
                market.get("home_ml_decimal"),
                market.get("away_ml_decimal"),
            )
        )
        rows.append(
            {
                "game_id": int(game.game_id),
                "season": int(game.season),
                "week": int(game.week),
                "home_team": game.home_team,
                "away_team": game.away_team,
                "kickoff": game.start_date.isoformat(),
                "prediction_method": METHOD,
                **metadata,
                **sim,
                **market,
                **history,
            }
        )
    return rows


def _out_of_fold_location(model, features, observed, mask, weights, seasons):
    """Score locations for completed games, each from a head fitted without that game's season.

    The line, rest and team-rate adjustments and the win-chance curve are fitted
    on how far real scores landed from these locations. A head that had already
    seen a game sits closer to its score than it would for a new game, which
    makes the adjustments too small and the curve too sure of itself. Holding
    out one season at a time keeps every location a genuine out-of-sample one.
    Returns None when a season cannot be held out with enough games left.
    """
    rows = np.flatnonzero(mask)
    held = seasons[rows]
    location = np.empty((len(rows), 2))
    for season in np.unique(held):
        hold = np.zeros(len(mask), dtype=bool)
        hold[rows[held == season]] = True
        train = mask & ~hold
        if int(train.sum()) < 50:
            return None
        head = ScoreHead(model["variant"], model["head"]["alpha"]).fit(features, observed, train, weights)
        location[held == season] = head.predict(features[hold])
    return location


def _store_history_facts(data_root, out):
    try:
        teams, players = load_history_facts(data_root, database=database_path())
        if teams is None or len(teams) == 0:
            return
        from contextlib import closing
        from ..storage.database import connect, record_history_facts
        with closing(connect(database_path())) as conn:
            record_history_facts(conn, teams, players)
    except Exception as exc:
        (Path(out) / "history_database_error.txt").write_text(str(exc))


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
    known_future=None,
    closing_path=None,
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
            state.features(r.home_id, r.away_id, r.neutral_site, early_season(r))
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
    past_location = _out_of_fold_location(
        model, historical_F, Y, fit_mask, np.exp(-decay * np.maximum(age_days, 0.0)), past.season.to_numpy()
    )
    location_fit = "out_of_fold_by_season"
    if past_location is None:
        past_location = live_head.predict(historical_F[fit_mask])
        location_fit = "in_sample"
    past_games = past.loc[fit_mask].reset_index(drop=True)
    # Completed games learn from the last line before their kickoff. Captured
    # quotes come first; the backfilled closing table fills games never captured.
    closing = closing_lines(closing_path if closing_path is not None else Path(data_root) / "market_closing_history.csv")
    past_market = [
        (
            market_at_cutoff(market, game.game_id, game.start_date, game.start_date, provider=market_provider)
            if market is not None and not market.empty
            else {}
        )
        or closing.get(int(game.game_id), {})
        for game in past_games.itertuples(index=False)
    ]
    history_past = pregame_history(past_games, data_root, past, database=database_path())
    fitted = ForecastModel.fit(
        past_location,
        Y[fit_mask][:, :, 0],
        # Rest days count every earlier kickoff, including games against non-FBS teams.
        known_future_matrix(_known_frame(past_games, schedules, past_market, None)),
        (past_games.home_points.to_numpy() > past_games.away_points.to_numpy()).astype(int),
        weeks=past_games.season.astype(str) + "-" + past_games.week.astype(str),
        history=history_past,
    )
    market_rows = []
    history_rows = []
    for game in targets.itertuples(index=False):
        market_rows.append(
            market_at_cutoff(market, game.game_id, at, game.start_date, provider=market_provider)
            if market is not None and not market.empty
            else {}
        )
        history_rows.append(
            market_history_summary(market, game.game_id, at, game.start_date, provider=market_provider)
            if market is not None and not market.empty
            else {"market_snapshot_count": 0}
        )
    history_future = pregame_history(targets, data_root, past, database=database_path())
    rows = _project_rows(
        targets,
        mu,
        fitted,
        model["score_covariance"],
        schedules,
        market_rows,
        history_rows,
        known_future,
        {
            "as_of": at.isoformat(),
            "latest_result_available_at": state.latest_update.isoformat() if state.latest_update is not None else None,
            "prior_fbs_games": len(past),
            "model_variant": model["variant"],
            "head_training_through_season": model["training_through_season"],
            "second_stage_location": location_fit,
        },
        history=history_future,
    )
    joint = []
    margins = []
    totals = []
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    _store_history_facts(data_root, out)
    market_archive = archive_market(market_path, out)
    market_file = market_archive["market_file"]
    (out / "forecasts.json").write_text(json.dumps(rows, indent=2))
    (out / "forecast_model.json").write_text(json.dumps(fitted.to_dict(), indent=2))
    (out / LOCATION_HEAD).write_text(json.dumps(live_head.to_dict(), indent=2))
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
    from ..ingest.data import normalize_schedule

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
            state.features(r.home_id, r.away_id, r.neutral_site, early_season(r))
            for r in targets.itertuples(index=False)
        ]
    )
    # The forecast that saved this state refit its score head on every result
    # before its cutoff; reuse that head so no-new-result advances match a replay.
    saved_head = prev / LOCATION_HEAD
    head = ScoreHead.from_dict(json.loads(saved_head.read_text())) if saved_head.exists() else ScoreHead.from_dict(model["head"])
    mu = head.predict(F)
    market = read_market_history(market_path) if market_path else None
    prior_count = int(pd.read_csv(prev / "forecasts.csv").prior_fbs_games.iloc[0])
    saved_model = prev / "forecast_model.json"
    fitted = ForecastModel.from_dict(json.loads(saved_model.read_text())) if saved_model.exists() else ForecastModel.identity()
    market_rows = []
    history_rows = []
    for game in targets.itertuples(index=False):
        market_rows.append(
            market_at_cutoff(market, game.game_id, at, game.start_date, provider=market_provider)
            if market is not None and not market.empty
            else {}
        )
        history_rows.append(
            market_history_summary(market, game.game_id, at, game.start_date, provider=market_provider)
            if market is not None and not market.empty
            else {"market_snapshot_count": 0}
        )
    completed = schedules[
        schedules.fbs_game
        & schedules.season.between(model["warmup_season"], season)
        & schedules.result_usable
        & (schedules.result_available_at < at)
    ]
    history_future = pregame_history(targets, data_root, completed, database=database_path())
    rows = _project_rows(
        targets,
        mu,
        fitted,
        model["score_covariance"],
        schedules,
        market_rows,
        history_rows,
        None,
        {
            "as_of": at.isoformat(),
            "latest_result_available_at": state.latest_update.isoformat() if state.latest_update is not None else None,
            "prior_fbs_games": prior_count,
            "model_variant": model["variant"],
            "head_training_through_season": model["training_through_season"],
        },
        history=history_future,
    )
    joint = []
    margins = []
    totals = []
    out.mkdir(parents=True, exist_ok=True)
    market_archive = archive_market(market_path, out)
    (out / "forecasts.json").write_text(json.dumps(rows, indent=2))
    (out / "forecast_model.json").write_text(json.dumps(fitted.to_dict(), indent=2))
    (out / LOCATION_HEAD).write_text(json.dumps(head.to_dict(), indent=2))
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
