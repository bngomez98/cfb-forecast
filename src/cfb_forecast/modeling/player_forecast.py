"""Database-backed production adapter for a single team/player posterior."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest.data import sha256, timestamp
from ..ingest.features import load_history_facts
from ..ingest.availability import availability_snapshot
from ..ingest.player_matchups import player_evidence
from ..odds.decision import moneyline_diagnostics
from ..odds.market import read_market_history, market_at_cutoff
from ..paths import database_path
from .player_hierarchy import PlayerHierarchy, observed_lineup
from .forecast_model import KNOWN_COLUMNS, rest_days

TEAM_METRICS = ("points_for", "points_against", "yards_per_play", "success_rate", "explosive_rate", "turnover_rate", "epa")


def _json_values(value):
    if isinstance(value, dict):
        return {str(k): _json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    return value


def _team_context(past, targets, facts, at):
    """Historical rates gated by final availability, not kickoff/week order."""
    rows = []
    for game in past.itertuples(index=False):
        for team, points, allowed in ((game.home_id, game.home_points, game.away_points),
                                       (game.away_id, game.away_points, game.home_points)):
            rows.append({"game_id": int(game.game_id), "team_id": int(team), "season": int(game.season),
                         "available_at": game.result_available_at, "points_for": points, "points_against": allowed})
    history = pd.DataFrame(rows)
    if facts is not None and len(facts):
        columns = [c for c in TEAM_METRICS if c not in ("points_for", "points_against") and c in facts]
        history = history.merge(facts[["game_id", "team_id", *columns]], on=["game_id", "team_id"], how="left", validate="one_to_one")
    for column in TEAM_METRICS:
        if column not in history:
            history[column] = np.nan
    by_team = {int(team): frame for team, frame in history.groupby("team_id")}
    result = np.full((len(targets), 2, len(TEAM_METRICS) * 2), np.nan)
    for slot, game in enumerate(targets.itertuples(index=False)):
        cutoff = min(timestamp(game.start_date), at)
        team_values = []
        for team in (game.home_id, game.away_id):
            part = by_team.get(int(team))
            if part is None:
                team_values.append(np.full(len(TEAM_METRICS), np.nan)); continue
            part = part.loc[(part.available_at < cutoff) & part.season.between(int(game.season) - 1, int(game.season))]
            team_values.append(part[list(TEAM_METRICS)].mean().to_numpy(float))
        result[slot, 0] = np.r_[team_values[0], team_values[1]]
        result[slot, 1] = np.r_[team_values[1], team_values[0]]
    return result


def _known_context(query, schedules, snapshots, at):
    rest = rest_days(query, schedules)
    result = np.full((len(query), 2, len(KNOWN_COLUMNS)), np.nan)
    if len(snapshots):
        snapshots = snapshots.copy()
        snapshots["observed_at"] = pd.to_datetime(snapshots.observed_at, utc=True)
        snapshots = snapshots.sort_values(["observed_at", "provider"], kind="stable")
    for slot, game in enumerate(query.itertuples(index=False)):
        cutoff = min(timestamp(game.start_date), at)
        values = {"home_rest_days": rest.iloc[slot].home_rest_days, "away_rest_days": rest.iloc[slot].away_rest_days}
        if len(snapshots):
            eligible = snapshots.loc[(snapshots.game_id == int(game.game_id)) & (snapshots.observed_at < cutoff)]
            # Merge eligible fields in observation order; sparse reports do not
            # erase older observed fields. No post-cutoff report can enter.
            for record in eligible.itertuples(index=False):
                values.update(json.loads(record.features_json))
        h = np.array([np.nan if values.get(k) is None else values.get(k, np.nan) for k in KNOWN_COLUMNS], float)
        a = h.copy()
        a[0] = -h[0]
        a[2] = 1 - h[2]
        a[3], a[4] = h[4], h[3]
        a[7], a[8] = h[8], h[7]
        result[slot] = (h, a)
    return result


def _lineup(players, lineups, game, team, at):
    cutoff = min(timestamp(game.start_date), at)
    if len(lineups):
        part = lineups.loc[(lineups.game_id == int(game.game_id)) & (lineups.team_id == int(team))].copy()
        if len(part):
            # Only a lineup recorded before kickoff and the forecast cutoff may
            # enter, and the latest such record wins regardless of row order.
            part["observed_at"] = pd.to_datetime(part["observed_at"], utc=True)
            part = part.loc[part["observed_at"] < cutoff].sort_values("observed_at", kind="stable")
            if len(part):
                row = part.iloc[-1]
                return json.loads(row.players_json), "timestamped_lineup", row.observed_at.isoformat()
    if players.empty:
        return [], "missing_player_history", None
    part = players.loc[(players.team_id == int(team)) & (players.start_date < cutoff)]
    if part.empty:
        return [], "missing_player_history", None
    # Latest observed participation is evidence of usage, not a confirmed future
    # roster. Do not mix all previous-season players into a new-season lineup.
    latest = part.start_date.max()
    latest_rows = part.loc[part.start_date == latest]
    return observed_lineup(latest_rows, int(team)), "last_observed_usage_unconfirmed", latest.isoformat()


def forecast_players(data_root, model_root, out_root, at, end, season, schedules, paths,
                     market_path=None, market_provider=None, target_game_ids=None, database=None):
    population = schedules.loc[schedules.fbs_game & schedules.season.le(season)].copy()
    past = population.loc[population.result_usable & (population.result_available_at < at)].copy().reset_index(drop=True)
    targets = population.loc[(population.season == season) & (population.start_date > at) & (population.start_date <= end)].copy()
    if target_game_ids is not None:
        wanted = {int(x) for x in target_game_ids}
        targets = targets.loc[targets.game_id.isin(wanted)]
        if wanted - set(targets.game_id.astype(int)):
            raise ValueError("Requested games are outside the eligible joint forecast window")
    if targets.empty:
        raise ValueError("No FBS-v-FBS games in the requested forecast window")
    database = database or database_path()
    paths = sorted(set(Path(p) for p in paths) | set(Path(data_root).glob("pbp_*.parquet")))
    players, lineups, evidence_hash = player_evidence(data_root, database, past, at)
    if players.empty:
        raise ValueError("Joint player forecast requires player observations in the database or historical parquet; import-player-evidence accepts identified source records. Use --legacy-score only to reproduce the previous team-only calculation.")
    facts, _ = load_history_facts(data_root, database=database)
    from ..storage.database import connect
    with closing(connect(database)) as conn:
        context_rows = pd.read_sql_query("SELECT * FROM game_context_snapshots", conn)
    if len(context_rows):
        context_rows = context_rows.loc[pd.to_datetime(context_rows.observed_at, utc=True) < at]
    train_context = np.concatenate([_team_context(past, past, facts, at),
                                    _known_context(past, schedules, context_rows, at)], axis=2)
    future_context = np.concatenate([_team_context(past, targets, facts, at),
                                     _known_context(targets, schedules, context_rows, at)], axis=2)
    specs = []
    for slot, game in enumerate(targets.itertuples(index=False)):
        pair = []
        for side, (team, opponent) in enumerate(((game.home_id, game.away_id), (game.away_id, game.home_id))):
            lineup, status, observed = _lineup(players, lineups, game, team, at)
            if not lineup:
                raise ValueError(f"No player lineup/history for team {team}; joint forecast cannot silently become team-only")
            pair.append({"team": int(team), "opponent": int(opponent), "season": int(game.season),
                         "venue": 0 if game.neutral_site else (1 if side == 0 else -1),
                         "lineup": lineup, "context": future_context[slot, side],
                         "lineup_status": status, "lineup_observed_at": observed})
        specs.append(pair)
    model = PlayerHierarchy.fit(past, players, train_context, [s for pair in specs for s in pair], at)
    market = read_market_history(market_path) if market_path else None
    records = []
    for game, pair in zip(targets.itertuples(index=False), specs):
        quote = market_at_cutoff(market, game.game_id, at, game.start_date, provider=market_provider) if market is not None and not market.empty else {}
        summary = model.predict(*pair, market=quote)
        summary["raw_home_win_probability"] = summary["home_win_probability"]
        summary.update(moneyline_diagnostics(summary["home_win_probability"], quote.get("home_ml_decimal"), quote.get("away_ml_decimal")))
        records.append({"game_id": int(game.game_id), "season": int(game.season), "week": int(game.week),
                        "home_team": game.home_team, "away_team": game.away_team, "kickoff": game.start_date.isoformat(),
                        "as_of": at.isoformat(), "latest_result_available_at": model.latest_result,
                        "prediction_method": "joint conditional Gaussian team/player/opponent hierarchy",
                        "model_variant": "joint_player", "prior_fbs_games": len(past),
                        "training_player_observations": model.training_player_observations,
                        "home_lineup_status": pair[0]["lineup_status"], "away_lineup_status": pair[1]["lineup_status"],
                        "input_evidence_sha256": evidence_hash, "player_availability_status": "unknown_unless_explicitly_supplied",
                        **summary, **quote})
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out)
    availability = availability_snapshot(data_root, at, schedules, None, paths)
    (out / "data_availability.json").write_text(json.dumps(availability, indent=2))
    # Store the exact fitted observations, query lineups and context; replay
    # never needs to trust subsequently edited SQL rows.
    inputs = {"cutoff": at.isoformat(), "players": players.to_dict("records"),
              "lineups": lineups.to_dict("records"), "targets": targets.to_dict("records"),
              "games": past.to_dict("records"), "team_facts": None if facts is None else facts.to_dict("records"),
              "contexts": context_rows.to_dict("records"),
              "query_designs": [[{**s, "context": s["context"].tolist()} for s in pair] for pair in specs]}
    (out / "joint_player_inputs.json").write_text(json.dumps(_json_values(inputs), indent=2, allow_nan=False))
    (out / "forecasts.json").write_text(json.dumps(records, indent=2, allow_nan=False))
    pd.DataFrame(records).to_csv(out / "forecasts.csv", index=False)
    from .forecast import archive_market
    market_archive = archive_market(market_path, out)
    artifact_files = ("joint_player_model.json", "joint_player_precision.npz", "joint_player_inputs.json")
    manifest = {"as_of": at.isoformat(), "through": end.isoformat(), "family": "joint_player_gaussian",
                "created_at": datetime.now(timezone.utc).isoformat(), "source_hashes": {p.name: sha256(p) for p in paths},
                "model_sha256": sha256(Path(model_root) / "model.json"),
                "joint_artifact_hashes": {name: sha256(out / name) for name in artifact_files},
                "player_evidence_sha256": evidence_hash, "latest_result_available_at": model.latest_result,
                "used_game_ids": past.game_id.astype(int).tolist(), "target_game_ids": targets.game_id.astype(int).tolist(),
                "training_games": model.training_games, "training_player_observations": model.training_player_observations,
                "probability_family": "availability_mixture_of_joint_gaussians", **market_archive,
                "data_availability_check": availability["prospective_current_season_source_check"],
                "limitations": ["Gaussian working likelihood, conditional fixed prior variances and normalization",
                                "Season-indexed ability evolution; within-season form enters pregame history covariates",
                                "Player-game likelihoods conditionally independent; shared-play measurement dependence is not fitted",
                                "No fitted drive/scoring-event model or event-consistent player stat totals",
                                "Historical name identities are team/season scoped; no inferred transfers",
                                "Unconfirmed usage is not a verified future roster; unknown availability has no invented probability",
                                "Predictive intervals use moment-matched Gaussian quantiles",
                                "New family requires chronological validation; previous accuracy does not apply"]}
    (out / "forecast_manifest.json").write_text(json.dumps(manifest, indent=2))
    return pd.DataFrame(records)
