"""Chronological holdout check for the joint team/player forecast.

Each holdout season is replayed slate by slate. For every slate the joint
posterior is refitted from results available before the forecast time
(first kickoff of the slate minus ``lead_hours``) and the slate's games are
predicted with lineups and team summaries known at that time. Nothing from
the predicted games, or anything later, enters their own forecasts.

The same replays supply the score-noise calibration: the noise used for a
holdout season is measured only on earlier replayed seasons, so it never sees
the season it is graded on. Tuning of the fixed prior settings uses earlier
seasons and a separate confirmation season; the final season is graded once.
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest.data import load_data
from ..ingest.features import load_history_facts
from ..ingest.player_matchups import player_evidence
from ..modeling.analytic import win_probability
from ..modeling.player_forecast import CONTEXT_HALF_LIFE_DAYS, joint_designs
from ..modeling.player_hierarchy import (NOISE_CALIBRATION_FILE, HierarchyConfig, PlayerHierarchy,
                                         holdout_noise_covariance)

FOLDS = ((2022, 2023), (2023, 2024), (2024, 2025))
# Predicted only to measure score noise for the first fold, never graded.
CALIBRATION_SEASON = 2022
REQUIRED_SEASONS = range(2021, 2026)
TUNED_SETTINGS = ("team_variance", "player_variance", "defense_variance", "matchup_variance",
                  "context_variance", "season_innovation_variance", "offseason_carry")
LIMITATIONS = [
    "Final-result times are kickoff plus eight hours; the schedule has no recorded final-whistle time",
    "Historical team and player statistics were downloaded after the fact and may include later revisions; "
    "their original publication times are unknown, so they are gated by result time only",
    "Historical lineups are last observed participation before the cutoff, not confirmed pregame lineups",
    "Games where either team has no player history before the cutoff are skipped and counted",
    "Closing spreads are the last public lines before kickoff, retrieved now; they carry no moneyline, "
    "so the market has no win-probability (Brier) comparison",
]


def load_holdout_inputs(data_root, database, seasons=REQUIRED_SEASONS, min_games=500):
    """Schedules, player evidence, team facts and context reports for the replay.

    Refuses to run when any required season is missing or thin, so accuracy
    is never reported from incomplete history.
    """
    schedules, _, _ = load_data(data_root)
    fbs = schedules.loc[schedules.fbs_game].copy()
    far = fbs.result_available_at.max() + pd.Timedelta(days=1)
    players, lineups, evidence_hash = player_evidence(data_root, database, fbs, far)
    facts, _ = load_history_facts(data_root, database=database)
    from ..storage.database import connect
    with closing(connect(database)) as conn:
        contexts = pd.read_sql_query("SELECT * FROM game_context_snapshots", conn)
    coverage = {}
    for season in seasons:
        games = int((fbs.season.eq(season) & fbs.result_usable).sum())
        with_players = int(players.loc[players.season.eq(season), "game_id"].nunique()) if len(players) else 0
        coverage[int(season)] = {"usable_fbs_games": games, "games_with_player_stats": with_players}
        if games < min_games or with_players < min_games:
            raise ValueError(f"Season {season} is incomplete: {coverage[int(season)]}")
    return {"schedules": schedules, "players": players, "lineups": lineups, "facts": facts,
            "contexts": contexts, "evidence_sha256": evidence_hash, "coverage": coverage}


def season_slates(schedules, season, lead_hours=48.):
    """Completed FBS games grouped by week, each with its forecast time."""
    games = schedules.loc[schedules.fbs_game & schedules.season.eq(season) & schedules.result_usable].copy()
    games["season_type"] = games.season_type.fillna("regular").astype(str)
    slates = []
    for (kind, week), part in games.dropna(subset=["week"]).groupby(["season_type", "week"]):
        label = f"postseason-{int(week)}" if kind.lower() == "postseason" else str(int(week))
        slates.append((label, part.sort_values("start_date"), part.start_date.min() - pd.Timedelta(hours=lead_hours)))
    return sorted(slates, key=lambda item: item[2])


def replay_season(inputs, season, config=None, context_mode="opponent_adjusted",
                  half_life_days=CONTEXT_HALF_LIFE_DAYS, refit_every=1, lead_hours=48.):
    """Predict every completed game of ``season`` from strictly earlier information.

    ``refit_every`` > 1 fits once per group of slates at the group's first
    forecast time (a cheaper, staler replay used for tuning).
    """
    config = config or HierarchyConfig()
    schedules, players, lineups = inputs["schedules"], inputs["players"], inputs["lineups"]
    facts, contexts = inputs["facts"], inputs["contexts"]
    population = schedules.loc[schedules.fbs_game & schedules.season.le(season)]
    if len(contexts):
        observed = pd.to_datetime(contexts.observed_at, utc=True)
    slates = season_slates(schedules, season, lead_hours)
    rows = []
    for start in range(0, len(slates), refit_every):
        group = slates[start:start + refit_every]
        at = group[0][2]
        past = population.loc[population.result_usable & (population.result_available_at < at)].reset_index(drop=True)
        targets = pd.concat([part for _, part, _ in group])
        eligible = players.loc[players.available_at < at] if len(players) else players
        context_rows = contexts.loc[observed < at] if len(contexts) else contexts
        train_context, specs, skipped = joint_designs(past, targets, schedules, eligible, lineups, facts, context_rows,
                                                      at, context_mode, half_life_days, skip_missing=True)
        model = PlayerHierarchy.fit(past, eligible, train_context, [s for _, pair in specs for s in pair], at, config)
        by_id = targets.set_index("game_id")
        labels = {int(g): label for label, part, _ in group for g in part.game_id}
        for game_id, pair in specs:
            game = by_id.loc[game_id]
            summary = model.predict(*pair, players=False)
            parameter = np.asarray(summary["score_covariance"]) - model.noise_covariance
            rows.append({"game_id": game_id, "season": int(season), "week": int(game.week), "slate": labels[game_id],
                         "start_date": game.start_date, "result_available_at": game.result_available_at,
                         "forecast_at": at, "latest_result_used": model.latest_result,
                         "training_games": model.training_games, "home_team": game.home_team,
                         "away_team": game.away_team, "home_points": float(game.home_points),
                         "away_points": float(game.away_points),
                         "expected_home_score": summary["expected_home_score"],
                         "expected_away_score": summary["expected_away_score"],
                         "parameter_cov_home": parameter[0, 0], "parameter_cov_cross": parameter[0, 1],
                         "parameter_cov_away": parameter[1, 1],
                         "home_lineup_status": pair[0]["lineup_status"], "away_lineup_status": pair[1]["lineup_status"],
                         "skipped": False})
        for game_id in skipped:
            game = by_id.loc[game_id]
            rows.append({"game_id": game_id, "season": int(season), "week": int(game.week), "slate": labels[game_id],
                         "start_date": game.start_date, "forecast_at": at, "skipped": True})
    frame = pd.DataFrame(rows)
    leaks = frame.loc[~frame.skipped & ~(pd.to_datetime(frame.latest_result_used, utc=True) < frame.forecast_at)]
    if len(leaks) or (frame.forecast_at >= frame.start_date).any():
        raise RuntimeError("Replay used a result at or after its forecast time")
    return frame


def grade(predictions, noise_covariance=None):
    """Winner accuracy, score errors and probability accuracy for replayed games."""
    graded = predictions.loc[~predictions.skipped].copy()
    actual_margin = graded.home_points - graded.away_points
    margin = graded.expected_home_score - graded.expected_away_score
    result = {"games": int(len(graded)), "skipped_games": int(predictions.skipped.sum()),
              "winner_accuracy": float(((margin > 0) == (actual_margin > 0)).mean()),
              "margin_mae": float((margin - actual_margin).abs().mean()),
              "score_mae": float(((graded.expected_home_score - graded.home_points).abs()
                                  + (graded.expected_away_score - graded.away_points).abs()).mean() / 2),
              "score_mse": float((((graded.expected_home_score - graded.home_points) ** 2
                                   + (graded.expected_away_score - graded.away_points) ** 2) / 2).mean())}
    if noise_covariance is not None:
        noise = np.asarray(noise_covariance, float)
        p = np.array([float(win_probability([h, a], noise + np.array([[ch, cx], [cx, ca]])))
                      for h, a, ch, cx, ca in graded[["expected_home_score", "expected_away_score", "parameter_cov_home",
                                                      "parameter_cov_cross", "parameter_cov_away"]].to_numpy(float)])
        y = (actual_margin > 0).to_numpy(float)
        result.update(brier=float(np.mean((p - y) ** 2)),
                      log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                      mean_win_probability_of_pick=float(np.mean(np.maximum(p, 1 - p))))
    return result


def assumed_noise(predictions_or_inputs, season):
    """The fit's default noise for a season: historical score SD squared, independent."""
    schedules = predictions_or_inputs["schedules"]
    games = schedules.loc[schedules.fbs_game & schedules.result_usable & schedules.season.lt(season)]
    scale = max(float(games[["home_points", "away_points"]].to_numpy(float).std()), 1.)
    return np.eye(2) * scale ** 2


def market_lines(paths):
    """Closing home spreads from SportsDataverse betting files (benchmark only)."""
    frames = []
    for path in paths:
        raw = pd.read_csv(path)
        if "game_spread_available" in raw:
            raw = raw.loc[raw.game_spread_available.astype(str).str.lower().isin(["true", "1"])]
        frames.append(pd.DataFrame({"game_id": pd.to_numeric(raw.game_id, errors="coerce"),
                                    "market_home_margin": -pd.to_numeric(raw.home_team_spread, errors="coerce")}))
    lines = pd.concat(frames).dropna().drop_duplicates("game_id", keep="last")
    lines["game_id"] = lines.game_id.astype("int64")
    return lines


def _margin_grade(frame, column):
    actual = frame.home_points - frame.away_points
    picked = frame[column] != 0
    return {"games": int(len(frame)),
            "winner_accuracy": float(((frame[column] > 0) == (actual > 0))[picked].mean()),
            "pick_em_games": int((~picked).sum()),
            "margin_mae": float((frame[column] - actual).abs().mean())}


def _replay_job(args):
    inputs, season, kwargs = args
    return season, kwargs, replay_season(inputs, season, **kwargs)


def _run_jobs(inputs, jobs, workers):
    if workers <= 1:
        return [_replay_job((inputs, season, kwargs)) for season, kwargs in jobs]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_replay_job, [(inputs, season, kwargs) for season, kwargs in jobs]))


def candidate_configs(base=None, half_life_days=CONTEXT_HALF_LIFE_DAYS):
    """A small one-setting-at-a-time grid around the current values."""
    base = base or HierarchyConfig()
    candidates = [("current", base, half_life_days)]
    for name in TUNED_SETTINGS:
        value = getattr(base, name)
        options = (0.6, 0.95) if name == "offseason_carry" else (value / 2, value * 2)
        for option in options:
            if option != value:
                candidates.append((f"{name}={option:g}", replace(base, **{name: float(option)}), half_life_days))
    for option in (half_life_days / 2, half_life_days * 2):
        candidates.append((f"half_life_days={option:g}", base, option))
    # All prior variances together, in case every one is too wide or narrow.
    variances = [name for name in TUNED_SETTINGS if name != "offseason_carry"]
    for factor in (0.25, 0.5, 2.):
        candidates.append((f"all_variances*{factor:g}",
                           replace(base, **{name: getattr(base, name) * factor for name in variances}), half_life_days))
    return candidates


def tune_settings(inputs, tuning_seasons=(2022, 2023), confirm_season=2024, refit_every=4, workers=1,
                  min_noise_games=100):
    """Choose prior settings on earlier seasons, then check them on a later one.

    Each candidate's score noise is measured on its own tuning-season
    replays, as production does with earlier holdout seasons, and candidates
    are ranked by win-probability log loss pooled over the tuning seasons
    (this judges both the predicted scores and the stated uncertainty). The
    confirmation season is graded with noise from the tuning seasons only.
    The best is kept only if it also beats the current settings there;
    otherwise the current settings are retained. The final holdout season is
    never used here.
    """
    candidates = candidate_configs()
    jobs = [(season, {"config": config, "half_life_days": half_life, "refit_every": refit_every, "_name": name})
            for name, config, half_life in candidates for season in tuning_seasons]
    results = _run_jobs(inputs, [(s, {k: v for k, v in kw.items() if k != "_name"}) for s, kw in jobs], workers)
    frames = {}
    for (season, kwargs), (_, _, frame) in zip(jobs, results):
        frames.setdefault(kwargs["_name"], []).append(frame)
    trials, ranking, noises = [], [], {}
    for name, parts in frames.items():
        pooled = pd.concat(parts)
        noises[name], _ = holdout_noise_covariance(pooled.loc[~pooled.skipped], min_games=min_noise_games)
        for frame in parts:
            trials.append({"candidate": name, "season": int(frame.season.iloc[0]), **grade(frame, noises[name])})
        ranking.append({"candidate": name, **grade(pooled, noises[name]), "noise_covariance": noises[name].tolist()})
    trials = pd.DataFrame(trials)
    ranking = pd.DataFrame(ranking).sort_values("log_loss").reset_index(drop=True)
    best = ranking.candidate.iloc[0]
    chosen = {name: (config, half_life) for name, config, half_life in candidates}
    compared = list(dict.fromkeys(["current", best]))
    confirm_jobs = [(confirm_season, {"config": chosen[name][0], "half_life_days": chosen[name][1],
                                      "refit_every": refit_every}) for name in compared]
    confirmed = {name: grade(frame, noises[name])
                 for name, (_, _, frame) in zip(compared, _run_jobs(inputs, confirm_jobs, workers))}
    adopted = best if best != "current" and confirmed[best]["log_loss"] < confirmed["current"]["log_loss"] else "current"
    return {"tuning_seasons": list(tuning_seasons), "confirm_season": confirm_season, "refit_every_slates": refit_every,
            "criterion": "win-probability log loss pooled over tuning seasons, noise measured on those seasons",
            "best_on_tuning_seasons": best, "confirmation": confirmed, "adopted": adopted,
            "adopted_config": asdict(chosen[adopted][0]), "adopted_half_life_days": chosen[adopted][1],
            "adopted_noise_covariance": noises[adopted].tolist(),
            "ranking": ranking.to_dict("records"),
            "trials": trials.to_dict("records")}


def run_chronological_holdout(data_root, database, out_root, betting_paths=(), legacy_predictions=None,
                              workers=1, lead_hours=48., tune=True, tuning_refit_every=4):
    """Replay the folds, calibrate noise on earlier seasons only, and report.

    ``legacy_predictions`` maps a test season to the previous team-only
    model's held-out predictions (run_backtest predictions.csv) so both are
    graded on identical games.
    """
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    inputs = load_holdout_inputs(data_root, database)
    seasons = sorted({CALIBRATION_SEASON, *(test for _, test in FOLDS)})
    modes = ("opponent_adjusted", "season_mean")
    jobs = [(season, {"context_mode": mode, "lead_hours": lead_hours}) for mode in modes for season in seasons]
    replays = {}
    for season, kwargs, frame in _run_jobs(inputs, jobs, workers):
        replays[(kwargs["context_mode"], season)] = frame
        frame.to_csv(out / f"joint_{kwargs['context_mode']}_{season}.csv", index=False)
    lines = market_lines(betting_paths) if betting_paths else None
    summary, noise_report = [], {}
    for train_through, test in FOLDS:
        for mode in modes:
            earlier = pd.concat([replays[(mode, s)] for s in seasons if s < test])
            noise, source = holdout_noise_covariance(earlier.loc[~earlier.skipped])
            noise_report[f"{mode}_{test}"] = {"noise_covariance": noise.tolist(), **source}
            frame = replays[(mode, test)]
            for label, cov in (("assumed_noise", assumed_noise(inputs, test)), ("holdout_noise", noise)):
                summary.append({"season": test, "trained_through": train_through, "model": f"joint_{mode}",
                                "noise": label, "games_scope": "all_replayed", **grade(frame, cov)})
        # Identical games across every available model for this season.
        common = replays[("opponent_adjusted", test)].loc[lambda f: ~f.skipped, ["game_id"]]
        common = common.merge(replays[("season_mean", test)].loc[lambda f: ~f.skipped, ["game_id"]], on="game_id")
        legacy = None
        if legacy_predictions and test in legacy_predictions:
            legacy = legacy_predictions[test]
            common = common.merge(legacy[["game_id"]], on="game_id")
        if lines is not None:
            common = common.merge(lines[["game_id"]], on="game_id")
        for mode in modes:
            earlier = pd.concat([replays[(mode, s)] for s in seasons if s < test])
            noise, _ = holdout_noise_covariance(earlier.loc[~earlier.skipped])
            frame = replays[(mode, test)].merge(common, on="game_id")
            summary.append({"season": test, "trained_through": train_through, "model": f"joint_{mode}",
                            "noise": "holdout_noise", "games_scope": "identical_games", **grade(frame, noise)})
        if legacy is not None:
            frame = legacy.merge(common, on="game_id")
            y = (frame.home_points > frame.away_points).astype(float)
            p = frame.legacy_home_win_probability.clip(1e-9, 1 - 1e-9)
            summary.append({"season": test, "trained_through": train_through, "model": "previous_team_only",
                            "noise": "fitted_validation_covariance", "games_scope": "identical_games",
                            **_margin_grade(frame, "legacy_home_margin"),
                            "score_mae": float(((frame.legacy_home_score - frame.home_points).abs()
                                                + (frame.legacy_away_score - frame.away_points).abs()).mean() / 2),
                            "brier": float(((p - y) ** 2).mean()),
                            "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())})
        if lines is not None:
            frame = replays[("opponent_adjusted", test)].merge(common, on="game_id").merge(lines, on="game_id")
            summary.append({"season": test, "trained_through": train_through, "model": "closing_spread",
                            "noise": None, "games_scope": "identical_games", **_margin_grade(frame, "market_home_margin")})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    # Production calibration: every replayed season, all earlier than the next forecast season.
    everything = pd.concat([replays[("opponent_adjusted", s)] for s in seasons])
    noise, source = holdout_noise_covariance(everything.loc[~everything.skipped])
    (out / NOISE_CALIBRATION_FILE).write_text(json.dumps({"noise_covariance": noise.tolist(), "source": source}, indent=2))
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "folds": [list(f) for f in FOLDS],
              "calibration_season": CALIBRATION_SEASON, "lead_hours": lead_hours,
              "data_coverage": inputs["coverage"], "player_evidence_sha256": inputs["evidence_sha256"],
              "noise_calibration_by_fold": noise_report, "production_noise_calibration": source,
              "limitations": LIMITATIONS}
    if tune:
        tuning = tune_settings(inputs, refit_every=tuning_refit_every, workers=workers)
        final_season = FOLDS[-1][1]
        if tuning["adopted"] != "current":
            config = HierarchyConfig(**tuning["adopted_config"])
            frame = replay_season(inputs, final_season, config=config, lead_hours=lead_hours,
                                  half_life_days=tuning["adopted_half_life_days"])
            frame.to_csv(out / f"joint_tuned_{final_season}.csv", index=False)
            earlier = pd.concat([replays[("opponent_adjusted", s)] for s in seasons if s < final_season])
            noise, _ = holdout_noise_covariance(earlier.loc[~earlier.skipped])
            # Each set of settings is graded with noise measured on its own
            # earlier-season replays; neither sees the final season.
            tuning["final_season"] = {"season": final_season,
                                      "tuned": grade(frame, tuning["adopted_noise_covariance"]),
                                      "current": grade(replays[("opponent_adjusted", final_season)], noise)}
        report["tuning"] = tuning
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return summary, report


def legacy_holdout_predictions(data_root, out_root, test):
    """The previous team-only model's held-out predictions for one season.

    Trains with run_backtest using only seasons before ``test`` (warmup three
    seasons earlier, selection on the season before) and returns its chosen
    variant's predictions for ``test``.
    """
    from .backtest import run_backtest
    out = Path(out_root) / f"previous_team_only_{test}"
    if not (out / "predictions.csv").exists():
        run_backtest(data_root, out, warmup=test - 3, train_start=test - 2, validation=test - 1, test=test)
    spec = json.loads((out / "run_manifest.json").read_text())
    variant = spec["selected_variant"]
    raw = pd.read_csv(out / "predictions.csv")
    return pd.DataFrame({"game_id": raw.game_id.astype("int64"), "home_points": raw.home_points,
                         "away_points": raw.away_points, "legacy_home_score": raw[f"{variant}_home_score"],
                         "legacy_away_score": raw[f"{variant}_away_score"],
                         "legacy_home_margin": raw[f"{variant}_home_margin"],
                         "legacy_home_win_probability": raw[f"{variant}_p"]})
