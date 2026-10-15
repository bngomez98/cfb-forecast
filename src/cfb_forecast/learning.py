"""Continuous prospective learning loop: freeze, settle, and adapt the blend.

The application's core loop lives here. It is *not* a backtest harness — it is
the mechanism by which the system learns from every real game as it happens:

1. ``freeze_upcoming`` writes one immutable prediction record per upcoming
   fixture **before kickoff**, capturing every expert's home-win probability:
   the production model head, the nearest-neighbor matchup expert, the market
   (when a pre-kickoff quote was captured) and the league base rate. Each
   record also stores the blend weights in force at freeze time, so later
   analysis can prove no forecast ever saw information about its own outcome.
2. ``settle_learning`` replays finished games whose results are now available,
   compares them against the frozen predictions, updates each expert's Beta
   posterior, and persists the new blend state. Settlement is idempotent:
   re-running it never double-counts a game.
3. ``blend_forecast_frame`` attaches blended probabilities to live forecast
   output using the current learned weights.

Records are stored as JSONL under ``artifacts/learning/predictions.jsonl``;
the blend posterior lives in ``artifacts/learning/blend_state.json``. Both are
append-or-replace atomically (write temp file, then rename) so an interrupted
nightly worker cannot corrupt the ledger.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .adaptive import AdaptiveBlend, base_rate_expert
from .conditional import build_conditional_matrix, matchup_similarity_weights, weighted_matchup_expectation
from .data import require_columns, timestamp
from .heads import ConditionalHeads
from .market import market_at_cutoff

log = logging.getLogger(__name__)


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    """Normal-approximation 95% confidence interval for a mean of bounded values."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return (max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se))


#: Columns a fixture frame must carry for the learning loop to run.
FIXTURE_REQUIREMENTS = [
    "game_id",
    "season",
    "week",
    "home_id",
    "away_id",
    "home_team",
    "away_team",
    "neutral_site",
]

#: Column subset of the conditioning matrix used for neighbor matching. These
#: are the pure matchup terms (form differentials); weather/injury columns are
#: excluded so neighbors share *who is playing how*, not merely the forecast
#: environment, mirroring Hoegh et al. (2015).
MATCHUP_DISTANCE_COLUMNS = [
    "diff_form_margin",
    "diff_form_points_for",
    "diff_form_points_against",
    "match_home_form_margin_vs_opp_defense",
]

PREDICTIONS_FILE = "predictions.jsonl"
BLEND_STATE_FILE = "blend_state.json"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _home_win_probability_from_history(games: pd.DataFrame, cutoff) -> float:
    """Historical home-win share among decided games before ``cutoff``."""
    prior = games[games.result_available_at < cutoff] if "result_available_at" in games.columns else games
    decided = prior[prior.home_points.notna() & prior.away_points.notna()]
    decided = decided[decided.home_points != decided.away_points]
    if decided.empty:
        return 0.5
    return float((decided.home_points > decided.away_points).mean())


def _matchup_expert_probability(
    history_vectors: np.ndarray,
    history_outcomes: np.ndarray,
    target_vector: np.ndarray,
) -> float | None:
    """Nearest-neighbor borrowed expectation that the home team wins.

    Neighbors are found in matchup-differential space; their realized home-win
    indicators are averaged with similarity weights and shrunk toward the pool
    mean when the neighbor mass is thin.
    """
    if len(history_vectors) < 8:
        return None
    weights = matchup_similarity_weights(history_vectors, target_vector.reshape(1, -1), k=12)
    pooled = weighted_matchup_expectation(weights, history_outcomes.reshape(-1, 1), shrinkage=2.0)
    value = float(pooled[0, 0])
    if not np.isfinite(value):
        return None
    return float(np.clip(value, 1e-6, 1 - 1e-6))


def train_heads(
    data_root,
    heads_dir=None,
    *,
    as_of=None,
    venues: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    player_ppg: pd.DataFrame | None = None,
) -> dict:
    """Refit the conditional score heads on every settled game before ``as_of``.

    This is the production training step of the daily cycle: it never uses a
    team rating; features come exclusively from the conditioning matrix
    (rolling recent scores, opponent-recent-performance, venue, weather,
    injuries, roster production). The fitted experts are persisted so the
    forecast/freeze steps share one brain.
    """
    from .data import load_data

    at = timestamp(as_of) if as_of is not None else None
    schedules, _, _ = load_data(data_root)
    usable = schedules[schedules.result_usable].copy()
    if at is not None and "result_available_at" in usable.columns:
        usable = usable[usable.result_available_at < at]
    usable = usable.sort_values("start_date").reset_index(drop=True)
    # Self-conditioning: each historical game is scored against the form that
    # existed strictly before it (build_conditional_matrix lags by design).
    matrix, provenance = build_conditional_matrix(
        usable[FIXTURE_REQUIREMENTS + ["start_date"]].copy(),
        usable,
        venues=venues,
        weather=weather,
        injuries=injuries,
        player_ppg=player_ppg,
    )
    heads = ConditionalHeads().fit(matrix, usable[["home_points", "away_points"]])
    destination = Path(heads_dir) if heads_dir is not None else Path("artifacts/model-conditional")
    heads.save(destination)
    summary = {
        "trained_at_games": heads.trained_at_games,
        "features": len(heads.feature_names_),
        "probability_scale": heads.probability_scale_,
        "heads_dir": str(destination),
        "provenance": provenance,
    }
    log.info(
        "learning.train_heads: fit %d experts on %d games (%d features)",
        len(ConditionalHeads.EXPERTS),
        heads.trained_at_games,
        len(heads.feature_names_),
    )
    return summary


def freeze_upcoming(
    data_root,
    forecasts: pd.DataFrame,
    *,
    as_of,
    market_path=None,
    venues: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    player_ppg: pd.DataFrame | None = None,
    blend_path=None,
    ledger_dir=None,
    heads_dir=None,
    decay: float = 1.0,
) -> pd.DataFrame:
    """Freeze one pre-kickoff prediction record per fixture row.

    ``forecasts`` must carry ``game_id``, ``home_id``, ``away_id`` and
    ``kickoff`` (or ``start_date``) identity columns, and may carry a legacy
    ``p_home``/``home_win_probability`` column. When trained conditional heads
    exist under ``heads_dir`` they supply the primary model probabilities
    (ridge / logit / gbm experts); otherwise the frame's own model column is
    used so older pipelines keep working. Returns the frame with added
    ``blend_p_home``, ``blend_effective_weights`` and per-expert probability
    columns.
    """
    from .data import load_data

    at = timestamp(as_of)
    schedules, _, _ = load_data(data_root)
    completed = schedules[schedules.result_usable & (schedules.result_available_at < at)].copy()
    completed = completed.sort_values("start_date").reset_index(drop=True)

    fixtures = forecasts.copy()
    kickoff_column = "kickoff" if "kickoff" in fixtures.columns else "start_date"
    require_columns(fixtures, [*FIXTURE_REQUIREMENTS, kickoff_column], "learning fixtures")
    fixtures["game_id"] = fixtures.game_id.astype(int)

    matrix, provenance = build_conditional_matrix(
        fixtures,
        completed,
        venues=venues,
        weather=weather,
        injuries=injuries,
        player_ppg=player_ppg,
    )
    head_predictions = None
    heads_path = Path(heads_dir) if heads_dir is not None else Path("artifacts/model-conditional")
    if (heads_path / "heads.pkl").exists():
        heads = ConditionalHeads.load(heads_path)
        head_predictions = heads.predict(matrix)
        provenance["heads_used"] = True
        provenance["heads_trained_at_games"] = heads.trained_at_games
    # Neighbor pool lives in matchup-only space: the form-differential
    # columns, so similarity means "same kind of matchup", not "same weather".
    history_pool = completed[FIXTURE_REQUIREMENTS + ["start_date"]].copy()
    history_matrix, _ = build_conditional_matrix(history_pool, completed)
    target_vectors = matrix[MATCHUP_DISTANCE_COLUMNS].fillna(0.0).to_numpy()
    history_vectors = history_matrix[MATCHUP_DISTANCE_COLUMNS].fillna(0.0).to_numpy()
    home_wins = (completed.home_points > completed.away_points).to_numpy(dtype=float)

    blend = AdaptiveBlend(decay=decay)
    if blend_path is not None and Path(blend_path).exists():
        blend = AdaptiveBlend.load(blend_path)

    market = None
    if market_path is not None and Path(market_path).exists():
        from .market import read_market_history

        market = read_market_history(market_path)

    base_p = base_rate_expert(_home_win_probability_from_history(schedules, at))
    weights = blend.weights()
    rows = []
    ledger_records = []
    for position, (_, fixture) in enumerate(fixtures.iterrows()):
        game_id = int(fixture.game_id)
        predictions = {
            "ridge": None,
            "logit": None,
            "gbm": None,
            "matchup": None,
            "market": None,
            "base": base_p,
        }
        if head_predictions is not None:
            for name, column in (
                ("ridge", "ridge_p_home"),
                ("logit", "logit_p_home"),
                ("gbm", "gbm_p_home"),
            ):
                value = head_predictions.iloc[position][column]
                if pd.notna(value):
                    predictions[name] = float(np.clip(value, 1e-6, 1 - 1e-6))
        else:
            # Legacy path: the caller's own model probability becomes "model".
            for column in ("p_home", "home_win_probability"):
                if column in fixture and pd.notna(fixture[column]):
                    predictions["ridge"] = float(fixture[column])
                    break
        neighbor = _matchup_expert_probability(history_vectors, home_wins, target_vectors[position])
        predictions["matchup"] = neighbor
        if market is not None and not market.empty:
            quote = market_at_cutoff(market, game_id, at, pd.Timestamp(fixture[kickoff_column], tz="UTC"))
            p_market = quote.get("market_home_win_probability_no_vig")
            if p_market is not None:
                predictions["market"] = float(p_market)
        blended, effective = blend.blend_probability(predictions)
        row = {
            "game_id": game_id,
            "as_of": at.isoformat(),
            "kickoff": pd.Timestamp(fixture[kickoff_column]).isoformat(),
            "expert_ridge_p": predictions["ridge"],
            "expert_logit_p": predictions["logit"],
            "expert_gbm_p": predictions["gbm"],
            "expert_matchup_p": predictions["matchup"],
            "expert_market_p": predictions["market"],
            "expert_base_p": predictions["base"],
            "blend_p_home": blended,
            "blend_effective_weights": effective,
            "blend_weights_at_freeze": weights,
            "conditional_provenance": provenance,
        }
        if head_predictions is not None:
            row["expected_margin"] = float(head_predictions.iloc[position]["ridge_margin"])
            row["expected_total"] = float(head_predictions.iloc[position]["ridge_expected_total"])
        rows.append(row)
        ledger_records.append({"type": "prediction", **row})

    ledger = Path(ledger_dir) if ledger_dir is not None else Path("artifacts/learning")
    path = ledger / PREDICTIONS_FILE
    existing = _read_jsonl(path)
    frozen_ids = {
        r["game_id"] for r in existing if r.get("type") == "prediction" and r.get("settled") is not True
    }
    fresh = [record for record in ledger_records if record["game_id"] not in frozen_ids]
    kept = [
        r
        for r in existing
        if not (r.get("type") == "prediction" and r["game_id"] in {rec["game_id"] for rec in fresh})
    ]
    combined = kept + fresh
    _atomic_write(path, "\n".join(json.dumps(record) for record in combined) + ("\n" if combined else ""))
    log.info("learning.freeze: %d new prediction records (of %d fixtures)", len(fresh), len(fixtures))

    enriched = fixtures.copy()
    enriched["blend_p_home"] = [row["blend_p_home"] for row in rows]
    for name in ("ridge", "logit", "gbm", "matchup", "market", "base"):
        enriched[f"expert_{name}_p"] = [row[f"expert_{name}_p"] for row in rows]
    if head_predictions is not None:
        enriched["expected_margin"] = [row.get("expected_margin") for row in rows]
        enriched["expected_total"] = [row.get("expected_total") for row in rows]
    enriched["blend_effective_weights"] = [json.dumps(row["blend_effective_weights"]) for row in rows]
    return enriched


def settle_learning(data_root, ledger_dir=None, blend_out=None, *, as_of=None, decay: float = 1.0) -> dict:
    """Settle every frozen prediction whose result is now available.

    Reads the ledger, matches each prediction to the schedule's final score
    (only results available strictly after the freeze and at or before
    ``as_of`` count), updates the Beta posteriors, marks records settled, and
    persists both files. Idempotent: already-settled records are skipped.
    """
    from .data import load_data

    ledger = Path(ledger_dir) if ledger_dir is not None else Path("artifacts/learning")
    path = ledger / PREDICTIONS_FILE
    records = _read_jsonl(path)
    if not records:
        return {"settled": 0, "pending": 0, "weights": AdaptiveBlend(decay=decay).weights()}

    schedules, _, _ = load_data(data_root)
    results = schedules.set_index(schedules.game_id.astype(int))
    blend_path = Path(blend_out) if blend_out is not None else ledger / BLEND_STATE_FILE
    blend = AdaptiveBlend.load(blend_path) if blend_path.exists() else AdaptiveBlend(decay=decay)
    cutoff = timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")

    settled_count = 0
    for record in records:
        if record.get("type") != "prediction" or record.get("settled"):
            continue
        game_id = int(record["game_id"])
        if game_id not in results.index:
            continue
        row = results.loc[game_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        if not bool(row.result_usable):
            continue
        available = row.result_available_at
        frozen_at = pd.Timestamp(record["as_of"], tz="UTC")
        if available <= frozen_at or available > cutoff:
            continue
        home_win = bool(row.home_points > row.away_points)
        if row.home_points == row.away_points:
            record["settled"] = "tie_excluded"
            continue
        predictions = {
            "ridge": record.get("expert_ridge_p", record.get("expert_model_p")),
            "matchup": record.get("expert_matchup_p"),
            "market": record.get("expert_market_p"),
            "base": record.get("expert_base_p"),
        }
        update = blend.observe(game_id, predictions, home_win)
        record["settled"] = True
        record["settlement"] = {
            "home_points": float(row.home_points),
            "away_points": float(row.away_points),
            "home_win": home_win,
            # Preserve the legacy "model" key for pre-split ledgers so audit
            # and analytics can read either naming.
            "correctness": {**update["experts"], "model": update["experts"]["ridge"]},
        }
        settled_count += 1

    pending = sum(1 for r in records if r.get("type") == "prediction" and not r.get("settled"))
    _atomic_write(path, "\n".join(json.dumps(record) for record in records) + "\n")
    blend.save(blend_path)
    log.info("learning.settle: settled %d games, %d pending", settled_count, pending)
    return {
        "settled": settled_count,
        "pending": pending,
        "weights": blend.weights(),
        "experts": {name: state.to_dict() for name, state in blend.experts.items()},
        "blend_state_path": str(blend_path),
    }


def blend_forecast_frame(forecasts: pd.DataFrame, blend_path=None, *, decay: float = 1.0) -> pd.DataFrame:
    """Attach blended probabilities to a live forecast frame using stored state."""
    if blend_path is not None and Path(blend_path).exists():
        blend = AdaptiveBlend.load(blend_path)
    else:
        blend = AdaptiveBlend(decay=decay)
    rows = []
    for _, fixture in forecasts.iterrows():
        predictions: dict[str, float | None] = {"base": None}
        for name in ("ridge", "logit", "gbm", "matchup"):
            column = f"expert_{name}_p"
            if column in fixture and pd.notna(fixture[column]):
                predictions[name] = float(fixture[column])
            elif name == "ridge" and "p_home" in fixture and pd.notna(fixture.p_home):
                predictions[name] = float(fixture.p_home)  # legacy model column
            else:
                predictions[name] = None
        if "market_home_win_probability_no_vig" in fixture and pd.notna(
            fixture.market_home_win_probability_no_vig
        ):
            predictions["market"] = float(fixture.market_home_win_probability_no_vig)
        else:
            predictions["market"] = None
        usable = {k: v for k, v in predictions.items() if v is not None}
        if not usable:
            rows.append(np.nan)
            continue
        blended, _ = blend.blend_probability(usable)
        rows.append(blended)
    enriched = forecasts.copy()
    enriched["blend_p_home"] = rows
    enriched["blend_weights"] = json.dumps(blend.weights())
    return enriched


# --------------------------------------------------------------------- audit
#: Expert keys as stored inside each settled ledger record's settlement block.
_EXPERT_KEYS = ("ridge", "logit", "gbm", "matchup", "market", "base")

#: Legacy audit keys: ledgers written before the ridge/logit/gbm split stored
#: the production model under "model"; report it as an alias of ridge.
_LEGACY_EXPERT_KEYS = {"model": "ridge"}


def _correctness_entry(record: dict, key: str) -> dict:
    """Fetch a settled record's per-expert correctness block with aliasing."""
    correctness = record["settlement"]["correctness"]
    entry = correctness.get(key)
    if entry is None and key in _LEGACY_EXPERT_KEYS:
        entry = correctness.get(_LEGACY_EXPERT_KEYS[key])
    return entry if isinstance(entry, dict) else {}


def _record_expert_keys(records: list[dict]) -> tuple[str, ...]:
    """Expert keys to audit, honoring legacy ledgers that predate head split.

    Canonical ledgers carry ridge/logit/gbm columns.  Older ledgers (and test
    fixtures) only ever wrote the single production expert under ``model``;
    for those we audit the legacy name directly instead of fabricating empty
    canonical entries.
    """
    sample = records[0].get("settlement", {}).get("correctness", {}) if records else {}
    if "ridge" not in sample and "model" in sample:
        return ("model", "matchup", "market", "base")
    return _EXPERT_KEYS


def learning_audit(ledger_dir=None, blend_path=None) -> dict:
    """Statistical post-mortem of everything the running application has frozen and settled.

    This is how the system *learns from its own history* rather than merely
    accumulating it. For every expert tracked by the adaptive blend we compute,
    on the settled prospective (pre-kickoff) predictions only:

    * hit rate with an exact Clopper-Pearson binomial interval
      (:func:`scipy.stats.beta`) — no normal approximation at small samples;
    * Brier score and mean log loss (:mod:`sklearn.metrics`), the two proper
      scoring rules the research write-up requires for winner forecasting;
    * calibration slope and intercept from an L2-penalized logistic refit of
      the outcome on the expert's predicted probability
      (:class:`sklearn.linear_model.LogisticRegression`). A slope near 1 with
      an intercept near 0 means the probabilities are honest; slope < 1 means
      overconfidence, which is exactly the failure mode early-season blowout
      data induces;
    * a paired bootstrap comparing the blend against each single expert:
      resample settled games with replacement and test whether the blend's
      log-loss advantage exceeds zero (block-free i.i.d. draws are fine here
      because each draw is one independently settled game).

    Returns a plain dict so both the CLI and the dashboard can render it.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss

    ledger = Path(ledger_dir) if ledger_dir is not None else Path("artifacts/learning")
    records = [r for r in _read_jsonl(ledger / PREDICTIONS_FILE) if r.get("settled") is True]
    result = {
        "settled_games": len(records),
        "experts": {},
        "paired_bootstrap": {},
        "notes": [],
    }
    if len(records) < 5:
        result["notes"].append("Fewer than five settled prospective games; audit withheld.")
        return result

    outcomes = np.array([1.0 if r["settlement"]["home_win"] else 0.0 for r in records])

    # Per-expert arrays (NaN where the expert abstained).
    audit_keys = _record_expert_keys(records)
    per_expert: dict[str, np.ndarray] = {}
    for key in audit_keys:
        values = []
        for r in records:
            entry = r["settlement"]["correctness"].get(key, {})
            p = entry.get("p") if isinstance(entry, dict) else None
            values.append(float(p) if p is not None else np.nan)
        per_expert[key] = np.array(values, dtype=float)

    blend_p = np.array([float(r["blend_p_home"]) for r in records], dtype=float)

    def _score_one(probs: np.ndarray) -> dict:
        mask = np.isfinite(probs)
        used = int(mask.sum())
        if used < 5:
            return {"games": used, "status": "insufficient"}
        y = outcomes[mask]
        p = np.clip(probs[mask], 1e-6, 1 - 1e-6)
        hits = int(((p >= 0.5) == (y == 1.0)).sum())
        # Exact Clopper-Pearson interval via the Beta-binomial identity.
        lo = float(sps.beta.ppf(0.025, hits, used - hits + 1)) if hits > 0 else 0.0
        hi = float(sps.beta.ppf(0.975, hits + 1, used - hits)) if hits < used else 1.0
        design = np.log(p / (1.0 - p)).reshape(-1, 1)
        try:
            # C=np.inf => unpenalized maximum-likelihood refit (the standard
            # logistic calibration-reciprocal-of-slope form); sklearn>=1.8
            # deprecates the explicit ``penalty`` keyword in favor of this.
            clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000)
            clf.fit(design, y.astype(int))
            slope = float(clf.coef_[0, 0])
            intercept = float(clf.intercept_[0])
        except ValueError:  # single-class outcome slice: calibration undefined
            slope = intercept = float("nan")
        return {
            "games": used,
            "hit_rate": hits / used,
            "hit_rate_ci95": [lo, hi],
            "brier": float(brier_score_loss(y.astype(int), p)),
            "log_loss": float(log_loss(y.astype(int), p, labels=[0, 1])),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "mean_p": float(p.mean()),
        }

    for name in audit_keys:
        result["experts"][name] = _score_one(per_expert[name])
    result["experts"]["blend"] = _score_one(blend_p)

    # Paired bootstrap: blend vs each single expert on shared games.
    rng = np.random.default_rng(20261011)
    ll_blend = np.where(
        np.isfinite(blend_p),
        -(
            outcomes * np.log(np.clip(blend_p, 1e-6, 1))
            + (1 - outcomes) * np.log(np.clip(1 - blend_p, 1e-6, 1))
        ),
        np.nan,
    )
    for name in audit_keys:
        shared = np.isfinite(ll_blend) & np.isfinite(per_expert[name])
        count = int(shared.sum())
        if count < 10:
            result["paired_bootstrap"][name] = {"games": count, "status": "insufficient"}
            continue
        lb = ll_blend[shared]
        le = -(
            outcomes[shared] * np.log(np.clip(per_expert[name][shared], 1e-6, 1))
            + (1 - outcomes[shared]) * np.log(np.clip(1 - per_expert[name][shared], 1e-6, 1))
        )
        diff = le - lb  # positive => blend better
        draws = rng.integers(0, count, size=(2000, count))
        stats = diff[draws].mean(axis=1)
        p_value = float((stats <= 0).mean())
        result["paired_bootstrap"][name] = {
            "games": count,
            "mean_logloss_gain": float(diff.mean()),
            "ci95": [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))],
            "p_blend_better": p_value,
        }

    weights_state: dict | None = None
    if blend_path is not None and Path(blend_path).exists():
        weights_state = AdaptiveBlend.load(blend_path).weights()
    result["current_weights"] = weights_state
    return result
