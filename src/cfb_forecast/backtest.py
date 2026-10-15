"""Chronological backtest: warmup, fit, validation and held-out evaluation.

The default chronology is warmup 2021, fit 2022-2023, select and calibrate on
2024, evaluate on 2025. The test year is never used for hyperparameter or
variant selection. Production refitting through the test season only runs
after the held-out predictions have been written to disk.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import evaluate_elo
from .config import METRICS, StateConfig
from .data import audit, load_data, sha256, timestamp
from .evaluate import calibration, metrics, paired_bootstrap
from .features import normalization, observations
from .model import ScoreHead, residual_covariance, write_model
from .ratings import build_features
from .simulate import win_probability

log = logging.getLogger(__name__)

#: Hyperparameter grid for the state-model process variance (standardized
#: units per week). Values span one order of magnitude so slow- and
#: fast-moving teams are both covered.
_PROCESS_VARIANCE_GRID = (0.005, 0.02, 0.06)

#: Grid for the offseason-carry shrinkage factor.
_OFFSEASON_CARRY_GRID = (0.5, 0.8)

#: Grid for the ridge regularization strength. Expanded from the original
#: three-value grid to a six-value approximately log-uniform grid so the
#: selection cannot land on an endpoint of the search space.
_RIDGE_ALPHA_GRID = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)

_REQUIRED_ARCHIVE_FIELDS = {
    "game_id",
    "snapshot_utc",
    "snapshot_commit",
    "snapshot_file",
    "phwin",
    "lineavg",
    "lineca",
}


def verified_archive(path, games: pd.DataFrame):
    """Load and verify an external archived-forecast file.

    Independently retained commit metadata is cross-checked against each
    snapshot's recorded commit and timestamp, and every archived probability
    and line is compared bit-exactly to its original CSV source. Returns the
    cleaned archive frame and a dict mapping each game ID to its archived
    snapshot timestamp (used as an explicit forecast cutoff downstream).
    """
    if path is None:
        return None, {}
    archive = pd.read_csv(path)
    missing = sorted(_REQUIRED_ARCHIVE_FIELDS - set(archive.columns))
    if missing:
        raise ValueError(f"Archive missing fields {missing}")
    if archive.game_id.duplicated().any():
        raise ValueError("Archive must have one snapshot per game")
    meta_path = Path(path).parent / "snapshot_commits.json"
    if not meta_path.exists():
        raise ValueError("Archive needs snapshot_commits.json for independently retained commit metadata")
    meta = {record["file"]: record for record in json.loads(meta_path.read_text())}
    for name, group in archive.groupby("snapshot_file"):
        if Path(name).name != name or name not in meta:
            raise ValueError("Unrecognized archive snapshot file")
        commit = meta[name]
        if (
            not group.snapshot_commit.eq(commit["sha"]).all()
            or not pd.to_datetime(group.snapshot_utc, utc=True).eq(timestamp(commit["date"])).all()
        ):
            raise ValueError(f"Snapshot commit/timestamp mismatch: {name}")
        raw_path = Path(path).parent / name
        if not raw_path.exists():
            raise ValueError(f"Missing original archive snapshot: {name}")
        raw = pd.read_csv(raw_path)
        if raw.duplicated(["home", "road"]).any():
            raise ValueError(f"Ambiguous raw archive match: {name}")
        check = group.merge(
            raw[["home", "road", "phwin", "lineavg", "lineca"]],
            on=["home", "road"],
            how="left",
            suffixes=("", "_raw"),
            validate="many_to_one",
        )
        for column in ("phwin", "lineavg", "lineca"):
            if not np.allclose(check[column], check[column + "_raw"], equal_nan=True, rtol=1e-10, atol=1e-10):
                raise ValueError(
                    f"Archive probabilities/lines differ from original snapshot: {name} {column}"
                )
    archive = archive.merge(games[["game_id", "start_date"]], on="game_id", validate="one_to_one")
    archive["snapshot_utc"] = pd.to_datetime(archive.snapshot_utc, utc=True)
    valid = (
        (archive.snapshot_utc < archive.start_date)
        & archive.snapshot_commit.astype(str).str.fullmatch(r"[a-fA-F0-9]{40}")
        & archive.phwin.between(0, 1)
    )
    archive = archive[valid].drop(columns="start_date")
    return archive, dict(zip(archive.game_id.astype(int), archive.snapshot_utc, strict=True))


def _validate_season_boundaries(warmup: int, train_start: int, validation: int, test: int) -> None:
    if not warmup < train_start < validation < test:
        raise ValueError("Need warmup < training start < validation < test year")


def _ensure_seasons_present(games: pd.DataFrame, warmup: int, test: int) -> None:
    for year in range(warmup, test + 1):
        if not (games.season == year).any():
            raise ValueError(f"No completed FBS-v-FBS games for required season {year}")


def _run_validation_grid(
    games: pd.DataFrame,
    Y: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    explicit: dict,
    warmup: int,
) -> tuple[dict, list[dict]]:
    """Score every (process_variance, offseason_carry, variant, alpha) combination.

    Returns a cache of built features keyed by (process_variance, carry) and
    the list of per-combination metric records.
    """
    cache: dict = {}
    trials: list[dict] = []
    for process_variance in _PROCESS_VARIANCE_GRID:
        for offseason_carry in _OFFSEASON_CARRY_GRID:
            cfg = StateConfig(
                process_variance=process_variance,
                offseason_carry=offseason_carry,
            )
            F, cutoffs, latest = build_features(games, Y, center, scale, cfg, explicit, warmup)
            cache[(process_variance, offseason_carry)] = (F, cutoffs, latest)
            for variant in ("score", "pbp"):
                for alpha in _RIDGE_ALPHA_GRID:
                    head = ScoreHead(variant, alpha).fit(F, Y, train)
                    train_mu = head.predict(F[train])
                    val_mu = head.predict(F[val])
                    # In-sample covariance is only used for ranking the
                    # validation grid; the released calibration covariance
                    # comes from out-of-fit validation residuals below.
                    sigma = residual_covariance(Y[train, :, 0], train_mu)
                    p = win_probability(val_mu, sigma)
                    scores = metrics((Y[val, 0, 0] > Y[val, 1, 0]).astype(int), p)
                    trials.append(
                        {
                            "variant": variant,
                            "q": process_variance,
                            "carry": offseason_carry,
                            "alpha": alpha,
                            **scores,
                        }
                    )
            log.info(
                "backtest.validation: completed grid for q=%.3f carry=%.1f",
                process_variance,
                offseason_carry,
            )
    return cache, trials


def _fit_variant(
    variant: str,
    best: dict,
    cache: dict,
    games: pd.DataFrame,
    Y: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    final_train: np.ndarray,
    te: np.ndarray,
    train_start: int,
    test: int,
    warmup: int,
    validation: int,
    paths: list,
    out: Path,
    result: pd.DataFrame,
    summaries_out: list[dict],
) -> tuple[dict, dict]:
    """Fit held-out and production heads for one variant and write artifacts."""
    F, cutoffs, latest = cache[(best["q"], best["carry"])]
    cal_head = ScoreHead(variant, best["alpha"]).fit(F, Y, train)
    cov = residual_covariance(Y[val, :, 0], cal_head.predict(F[val]))
    head = ScoreHead(variant, best["alpha"]).fit(F, Y, final_train)
    mu = head.predict(F[te])
    p = win_probability(mu, cov)
    result[f"{variant}_home_score"] = mu[:, 0]
    result[f"{variant}_away_score"] = mu[:, 1]
    result[f"{variant}_home_margin"] = mu[:, 0] - mu[:, 1]
    result[f"{variant}_total"] = mu.sum(axis=1)
    result[f"{variant}_p"] = p
    result["forecast_cutoff"] = cutoffs[te].astype(str).to_numpy()
    result["latest_result_available_at"] = [
        str(x) if x is not None else "" for x, is_test in zip(latest, te, strict=True) if is_test
    ]
    base = {
        "schema_version": 1,
        "variant": variant,
        "state_config": StateConfig(process_variance=best["q"], offseason_carry=best["carry"]).to_dict(),
        "metric_names": list(METRICS),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "head": head.to_dict(),
        "score_covariance": cov.tolist(),
        "warmup_season": warmup,
        "train_start_season": train_start,
        "training_through_season": validation,
        "calibration_season": validation,
        "selection_season": validation,
        "source_hashes": {p.name: sha256(p) for p in paths},
        "input_note": ("Retrospectively revised historical records; not immutable as-known statistics"),
    }
    write_model(out / "models" / variant / "heldout", base)
    scores = metrics(
        result.home_win,
        p,
        mu[:, 0] - mu[:, 1],
        result.actual_home_margin,
        mu.sum(axis=1),
        result.actual_total,
        mu,
        Y[te, :, 0],
    )
    summaries_out.append({"model": variant, "season": test, **scores})
    calibration(result.home_win, p).to_csv(out / f"calibration_{variant}.csv", index=False)

    # Refit for later forecasting; never overwrite held-out predictions.
    production = dict(base)
    production["training_through_season"] = test
    production["head"] = (
        ScoreHead(variant, best["alpha"])
        .fit(F, Y, games.season.between(train_start, test).to_numpy())
        .to_dict()
    )
    write_model(out / "models" / variant / "production", production)
    log.info("backtest.fit: wrote %s heldout and production models", variant)
    return base, production


def _compare_against_archive(
    archive: pd.DataFrame | None,
    result: pd.DataFrame,
    out: Path,
    summaries_out: list[dict],
    comparisons: list[dict],
    test: int,
) -> None:
    if archive is None or not len(archive):
        return
    matched = result.merge(archive, on="game_id", validate="one_to_one")
    matched = matched[np.isfinite(matched.phwin) & np.isfinite(matched.lineavg)].copy()
    matched["archive_p"] = matched.phwin
    # Tracker lineavg: positive means home advantage, verified in original archive.
    matched["archive_home_margin"] = matched.lineavg
    matched.to_csv(out / "archive_comparison.csv", index=False)
    if not len(matched):
        return
    for name, col, margin_col in (
        ("archive", "archive_p", "archive_home_margin"),
        ("score_matched", "score_p", "score_home_margin"),
        ("pbp_matched", "pbp_p", "pbp_home_margin"),
    ):
        summaries_out.append(
            {
                "model": name,
                "season": test,
                **metrics(
                    matched.home_win,
                    matched[col],
                    matched[margin_col],
                    matched.actual_home_margin,
                ),
            }
        )
    for variant in ("score", "pbp"):
        comparisons.append(paired_bootstrap(matched, f"{variant}_p", "archive_p"))

    # Recorded historical lineca is descriptive archive evidence. Odds and
    # original sportsbook timestamp are not established; no ROI reported.
    spread_rows: list[dict] = []
    for variant in ("score", "pbp"):
        with_line = matched[matched.lineca.notna()].copy()
        edge = with_line[f"{variant}_home_margin"] - with_line.lineca
        actual = with_line.actual_home_margin - with_line.lineca
        chosen = np.where(edge > 0, actual, -actual)
        for threshold in (0.0, 2.0, 4.0):
            take = edge.abs() > threshold
            values = chosen[take]
            spread_rows.append(
                {
                    "variant": variant,
                    "minimum_margin_edge": threshold,
                    "games": int(take.sum()),
                    "wins": int((values > 0).sum()),
                    "losses": int((values < 0).sum()),
                    "pushes": int((values == 0).sum()),
                    "win_rate_excluding_pushes": (
                        float((values > 0).sum() / (values != 0).sum()) if (values != 0).sum() else None
                    ),
                }
            )
    pd.DataFrame(spread_rows).to_csv(out / "archive_spread_diagnostics.csv", index=False)


def run_backtest(
    data_root,
    out_root,
    archive_path=None,
    warmup: int = 2021,
    train_start: int = 2022,
    validation: int = 2024,
    test: int = 2025,
):
    """Run the full chronological backtest and write every artifact to ``out_root``."""
    _validate_season_boundaries(warmup, train_start, validation, test)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    schedules, summaries, paths = load_data(data_root)
    audit(schedules, summaries).to_csv(out / "data_audit.csv", index=False)
    games = (
        schedules[schedules.fbs_game & schedules.result_usable & schedules.season.between(warmup, test)]
        .copy()
        .reset_index(drop=True)
    )
    _ensure_seasons_present(games, warmup, test)
    Y = observations(games, summaries)
    center, scale = normalization(Y, games.season.eq(warmup).to_numpy())
    train = games.season.between(train_start, validation - 1).to_numpy()
    val = games.season.eq(validation).to_numpy()
    te = games.season.eq(test).to_numpy()
    final_train = games.season.between(train_start, validation).to_numpy()
    archive, explicit = verified_archive(archive_path, games)

    cache, trials = _run_validation_grid(games, Y, center, scale, train, val, explicit, warmup)
    pd.DataFrame(trials).to_csv(out / "validation_trials.csv", index=False)

    selected = {
        variant: min(
            (record for record in trials if record["variant"] == variant),
            key=lambda record: record["log_loss"],
        )
        for variant in ("score", "pbp")
    }
    chosen_variant = min(selected, key=lambda variant: selected[variant]["log_loss"])
    log.info(
        "backtest.selection: variant=%s alpha=%s q=%.3f carry=%.1f",
        chosen_variant,
        selected[chosen_variant]["alpha"],
        selected[chosen_variant]["q"],
        selected[chosen_variant]["carry"],
    )

    result = games.loc[
        te,
        [
            "game_id",
            "season",
            "week",
            "start_date",
            "home_id",
            "away_id",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
            "neutral_site",
        ],
    ].copy()
    result["home_win"] = (result.home_points > result.away_points).astype(int)
    result["actual_home_margin"] = result.home_points - result.away_points
    result["actual_total"] = result.home_points + result.away_points

    production: dict = {}
    summaries_out: list[dict] = []
    for variant, best in selected.items():
        _, production_model = _fit_variant(
            variant,
            best,
            cache,
            games,
            Y,
            center,
            scale,
            train,
            val,
            final_train,
            te,
            train_start,
            test,
            warmup,
            validation,
            paths,
            out,
            result,
            summaries_out,
        )
        production[variant] = production_model

    elo_p, elo_trials, elo_chosen = evaluate_elo(games, train, val, final_train, te, explicit)
    result["elo_p"] = elo_p
    pd.DataFrame(elo_trials).to_csv(out / "elo_validation.csv", index=False)
    summaries_out.append({"model": "elo", "season": test, **metrics(result.home_win, elo_p)})
    calibration(result.home_win, elo_p).to_csv(out / "calibration_elo.csv", index=False)
    result.to_csv(out / "predictions.csv", index=False)
    summaries_out.append(
        {"model": "always_home", "season": test, **metrics(result.home_win, np.full(len(result), 0.5))}
    )

    comparisons = [
        paired_bootstrap(result, "pbp_p", "score_p"),
        paired_bootstrap(result, "score_p", "elo_p"),
        paired_bootstrap(result, "pbp_p", "elo_p"),
    ]
    _compare_against_archive(archive, result, out, summaries_out, comparisons, test)

    pd.DataFrame(summaries_out).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    write_model(out / "model", production[chosen_variant])

    spec = {
        "created_at": datetime.now(UTC).isoformat(),
        "warmup": warmup,
        "initial_train_seasons": list(range(train_start, validation)),
        "validation_season": validation,
        "test_season": test,
        "selected_variant": chosen_variant,
        "selected_settings": selected,
        "elo_selected_C": elo_chosen["C"],
        "evaluation_games": len(result),
        "all_cutoffs_before_kickoff": bool(
            (pd.to_datetime(result.forecast_cutoff, utc=True) < result.start_date).all()
        ),
        "all_latest_results_before_cutoff": bool(
            (
                pd.to_datetime(result.latest_result_available_at, utc=True)
                < pd.to_datetime(result.forecast_cutoff, utc=True)
            ).all()
        ),
        "missing_pbp_does_not_remove_games": True,
        "source_hashes": {p.name: sha256(p) for p in paths},
        "archive_sha256": sha256(archive_path) if archive_path else None,
        "limitations": [
            "Historical source statistics may have later revisions",
            "Unknown final-whistle time uses kickoff plus eight hours",
            "FBS-v-FBS rating and evaluation population",
            "Empirical validation covariance includes state-estimation uncertainty; it is not added a second time",
            "No learned player/injury availability, fitted EPA, or drive-level rules engine",
            "Archive comparison is a public forecast tracker, not a bookmaker moneyline baseline",
            "Test season has been studied in earlier work; this is reproducible historical evaluation, not a newly blind experiment",
        ],
    }
    (out / "run_manifest.json").write_text(json.dumps(spec, indent=2))
    log.info("backtest.run_manifest: wrote run_manifest.json")
    return pd.DataFrame(summaries_out), spec
