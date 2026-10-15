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

Persistence contract (docs/GLOSSARY.md §2 "Single database"): PostgreSQL is
the only store. When a DSN is configured (``CFB_POSTGRES_DSN`` or explicit
``dsn=``), every record flows through ``db.py`` into the canonical schema —
sealed ``predictions`` rows, append-only ``prediction_outcomes``, snapshotted
``blend_weights``. There is deliberately no file-based artifact fallback: if
the database is unreachable the cycle fails loudly rather than silently
writing local artifacts that could diverge from the single source of truth.
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

from .adaptive import AdaptiveBlend, ExpertState, base_rate_expert
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


#: Expert names as stored in the ledger's settlement block, canonical order.
EXPERT_KEYS = ("ridge", "logit", "gbm", "matchup", "market", "base")


def _dsn(dsn: str | None) -> str:
    """Resolve the single-database DSN; fail loudly (no artifact fallback)."""
    from . import db as _db

    return _db.resolve_dsn(dsn)


def _db_records(conn, *, settled_only: bool) -> list[dict]:
    """Reconstruct ledger records from the canonical database.

    ``predictions`` joined to ``prediction_outcomes`` yields exactly the same
    record shape the blend and audit consume, so all downstream logic is
    identical whether a game was frozen yesterday or months ago. Column
    naming follows schema/schema.sql; glossary §4 defines the expert registry.
    """
    where = "WHERE o.prediction_id IS NOT NULL" if settled_only else ""
    rows = conn.execute(
        f"""
        SELECT p.espn_game_id, p.expert, p.cutoff_at, p.model_version,
               p.win_prob, p.margin_points, p.total_points,
               o.home_win, o.log_loss, o.abs_margin_error, o.abs_total_error,
               o.settled_at
        FROM predictions p
        LEFT JOIN prediction_outcomes o USING (prediction_id)
        {where}
        ORDER BY p.cutoff_at, p.espn_game_id, p.expert
        """
    ).fetchall()
    grouped: dict[tuple[str, object], dict] = {}
    for row in rows:
        key = (row["espn_game_id"], row["cutoff_at"])
        record = grouped.setdefault(
            key,
            {
                "game_id": int(row["espn_game_id"]),
                "as_of": row["cutoff_at"].isoformat(),
                "settled": False,
                "experts": {},
                "blend_p_home": None,
                "expected_margin": None,
                "expected_total": None,
            },
        )
        name = row["expert"]
        record["experts"][name] = float(row["win_prob"])
        if row["margin_points"] is not None and record["expected_margin"] is None:
            record["expected_margin"] = float(row["margin_points"])
            record["expected_total"] = float(row["total_points"])
        if row["home_win"] is not None:
            record["settled"] = True
            record["settlement"] = {"home_win": bool(row["home_win"]), "correctness": {}}
    # Second pass: per-expert correctness blocks + blended probability.
    weights = _latest_blend_weights_dict(conn)
    outcome_rows = [r for r in rows if r["home_win"] is not None]
    outcome_by_key: dict[tuple[str, object], list] = {}
    for row in outcome_rows:
        outcome_by_key.setdefault((row["espn_game_id"], row["cutoff_at"]), []).append(row)
    for key, group in outcome_by_key.items():
        record = grouped[key]
        correctness = {}
        for row in group:
            entry = {
                "p": float(row["win_prob"]),
                "correct": (float(row["win_prob"]) >= 0.5) == bool(row["home_win"]),
                "log_loss": float(row["log_loss"]) if row["log_loss"] is not None else None,
            }
            correctness[row["expert"]] = entry
            if row["expert"] == "ridge":
                correctness["model"] = entry  # legacy alias key
        record["settlement"]["correctness"] = correctness
        if weights:
            blended, _ = _blend_from_weights(weights).blend_probability(record["experts"])
            record["blend_p_home"] = blended
    return sorted(grouped.values(), key=lambda r: (r["as_of"], r["game_id"]))


def _blend_from_weights(weights: dict[str, float], decay: float = 1.0) -> AdaptiveBlend:
    """Approximate a blend whose softmax weights match a snapshot.

    ``AdaptiveBlend.weights()`` is a temperature-scaled softmax over posterior
    credibility; inverting it exactly needs the posteriors themselves, which
    live in ``prediction_outcomes`` and are rebuilt precisely by
    :func:`_load_blend`. This helper exists only for *displaying* historical
    blend probabilities on reconstructed records; the learning path always
    uses the exact replay. We store each weight as a pseudo-count credibility
    (Beta(w·K, (1−w)·K), K = prior total) so the softmax reproduces the
    snapshot to within the floor/renormalization the blend applies.
    """
    from .adaptive import PRIOR_ALPHA, PRIOR_BETA

    k = PRIOR_ALPHA + PRIOR_BETA
    experts = {
        name: ExpertState(name, alpha=k * w, beta=k * (1 - w))
        for name, w in weights.items()
        if 0.0 < w < 1.0
    }
    return AdaptiveBlend(experts=experts, decay=decay)


def _latest_blend_weights_dict(conn) -> dict[str, float]:
    from . import db as _db

    return _db.latest_blend_weights(conn)


def _load_blend(dsn: str, decay: float = 1.0) -> AdaptiveBlend:
    """Rebuild the learned blend exactly from settled history in Postgres.

    Every row of ``prediction_outcomes`` (the append-only learning ledger,
    glossary §4) is one Bernoulli observation for one expert. Replaying them
    in settlement order reproduces each Beta posterior exactly — with
    ``decay < 1`` applying geometric recency discounting — so a restarted
    application resumes with precisely the knowledge it accumulated. The
    database *is* the model state; nothing else needs to survive a deploy.
    """
    from . import db as _db

    with _db.connect(dsn) as conn:
        rows = conn.execute(
            """
            SELECT expert, win_prob, home_win
            FROM prediction_outcomes
            ORDER BY settled_at, prediction_id
            """
        ).fetchall()
    if not rows:
        return AdaptiveBlend(decay=decay)
    experts: dict[str, ExpertState] = {}
    total = len(rows)
    for index, row in enumerate(rows):
        name = row["expert"]
        state = experts.get(name)
        if state is None:
            state = ExpertState(name)
            experts[name] = state
        correct = (float(row["win_prob"]) >= 0.5) == bool(row["home_win"])
        factor = decay ** (total - 1 - index) if decay < 1 else 1.0
        state.update(correct, decay=factor)
    return AdaptiveBlend(experts=experts, decay=decay)


def _save_blend(dsn: str, blend: AdaptiveBlend) -> None:
    from . import db as _db

    diagnostics = {
        name: {
            "hit_rate": state.hits / state.n if state.n else None,
            "n_settled": state.n,
        }
        for name, state in blend.experts.items()
    }
    with _db.connect(dsn) as conn:
        _db.record_blend_weights(conn, blend.weights(), diagnostics)


def _seal_record(conn, record: dict) -> int:
    """Insert one frozen multi-expert prediction into the sealed ledger.

    The DB trigger enforces cutoff < kickoff_at (schema integrity, glossary
    §1): a row whose game is missing or already past kickoff is rejected by
    PostgreSQL itself, not by application code. Idempotency comes from the
    ON CONFLICT DO NOTHING keys in db.insert_prediction. Returns the number
    of new rows sealed.
    """
    from . import db as _db

    cutoff = pd.Timestamp(record["as_of"]).to_pydatetime()
    sealed = 0
    for name in EXPERT_KEYS:
        p = record.get(f"expert_{name}_p")
        if p is None:
            continue
        prediction_id = _db.insert_prediction(
            conn,
            espn_game_id=str(record["game_id"]),
            expert=name,
            model_version=record.get("model_version", "conditional-heads"),
            cutoff_at=cutoff,
            win_prob=float(p),
            input_hash=record.get("input_hash", ""),
            margin_points=record.get("expected_margin"),
            total_points=record.get("expected_total"),
        )
        sealed += 1 if prediction_id is not None else 0
    return sealed


def _settle_record(conn, record: dict, row) -> dict | None:
    """Join truth to every sealed prediction of one freeze event.

    Returns the settlement block (home_win + per-expert correctness) built
    from what the database accepted, or None when nothing was pending.
    """
    from . import db as _db

    home_win = bool(row.home_points > row.away_points)
    margin = int(row.home_points - row.away_points)
    total = int(row.home_points + row.away_points)
    pending = conn.execute(
        """
        SELECT prediction_id, expert, win_prob FROM predictions
        WHERE espn_game_id = %s AND cutoff_at = %s
          AND prediction_id NOT IN (SELECT prediction_id FROM prediction_outcomes)
        """,
        (str(record["game_id"]), pd.Timestamp(record["as_of"]).to_pydatetime()),
    ).fetchall()
    if not pending:
        return None
    correctness: dict[str, dict] = {}
    for pred in pending:
        err = None
        terr = None
        if pred["expert"] == "ridge" and record.get("expected_margin") is not None:
            err = abs(float(record["expected_margin"]) - margin)
            terr = abs(float(record["expected_total"]) - total)
        _db.settle_prediction(
            conn,
            prediction_id=int(pred["prediction_id"]),
            home_win=home_win,
            margin=margin,
            total=total,
            abs_margin_error=err,
            abs_total_error=terr,
        )
        p = float(pred["win_prob"])
        entry = {
            "p": p,
            "correct": (p >= 0.5) == home_win,
            "abs_margin_error": err,
            "abs_total_error": terr,
        }
        correctness[pred["expert"]] = entry
        if pred["expert"] == "ridge":
            correctness["model"] = entry  # legacy alias for audit readers
    return {"home_win": home_win, "correctness": correctness}


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
    heads_dir=None,
    decay: float = 1.0,
    dsn: str | None = None,
) -> pd.DataFrame:
    """Freeze one pre-kickoff prediction record per fixture into Postgres.

    ``forecasts`` must carry ``game_id``, ``home_id``, ``away_id`` and
    ``kickoff`` (or ``start_date``) identity columns, and may carry a legacy
    ``p_home``/``home_win_probability`` column. When trained conditional heads
    exist under ``heads_dir`` they supply the primary model probabilities
    (ridge / logit / gbm experts); otherwise the frame's own model column is
    used so older pipelines keep working. Returns the frame with added
    ``blend_p_home``, ``blend_effective_weights`` and per-expert probability
    columns.

    Persistence goes exclusively through the canonical database: every sealed
    row passes the ``predictions_cutoff_before_kickoff`` trigger, so the
    single store — not application code — guarantees no forecast was made
    after kickoff (glossary §1). Requires ``CFB_POSTGRES_DSN``; there is no
    file fallback by design.
    """
    from . import db as _db
    from .data import load_data

    resolved_dsn = _dsn(dsn)
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
    heads_path = Path(heads_dir) if heads_dir is not None else None
    if heads_path is not None and (heads_path / "heads.pkl").exists():
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

    blend = _load_blend(resolved_dsn, decay)

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

    sealed = 0
    with _db.connect(resolved_dsn) as conn:
        for record in ledger_records:
            sealed += _seal_record(conn, record)
    log.info(
        "learning.freeze: sealed %d expert rows for %d fixtures into the canonical ledger",
        sealed,
        len(fixtures),
    )

    enriched = fixtures.copy()
    enriched["blend_p_home"] = [row["blend_p_home"] for row in rows]
    for name in ("ridge", "logit", "gbm", "matchup", "market", "base"):
        enriched[f"expert_{name}_p"] = [row[f"expert_{name}_p"] for row in rows]
    if head_predictions is not None:
        enriched["expected_margin"] = [row.get("expected_margin") for row in rows]
        enriched["expected_total"] = [row.get("expected_total") for row in rows]
    enriched["blend_effective_weights"] = [json.dumps(row["blend_effective_weights"]) for row in rows]
    return enriched


def settle_learning(data_root, *, as_of=None, decay: float = 1.0, dsn: str | None = None) -> dict:
    """Settle every frozen prediction whose result is now available.

    Reads pending freezes from the canonical ``predictions`` table, joins each
    to the schedule's final score (only results available strictly after the
    freeze and at or before ``as_of`` count), writes append-only
    ``prediction_outcomes`` rows, replays the full settled history to rebuild
    each expert's Beta posterior exactly, and snapshots the new blend weights.
    Idempotent: outcomes already present are never re-inserted or re-counted.
    """
    from . import db as _db
    from .data import load_data

    resolved_dsn = _dsn(dsn)
    schedules, _, _ = load_data(data_root)
    results = schedules.set_index(schedules.game_id.astype(int))
    cutoff = timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")

    settled_count = 0
    with _db.connect(resolved_dsn) as conn:
        pending_rows = conn.execute(
            """
            SELECT DISTINCT p.espn_game_id, p.cutoff_at
            FROM predictions p
            WHERE NOT EXISTS (SELECT 1 FROM prediction_outcomes o WHERE o.prediction_id = p.prediction_id)
            ORDER BY p.cutoff_at, p.espn_game_id
            """
        ).fetchall()
        for key in pending_rows:
            game_id = int(key["espn_game_id"])
            if game_id not in results.index:
                continue
            row = results.loc[game_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            if not bool(row.result_usable):
                continue
            available = row.result_available_at
            frozen_at = pd.Timestamp(key["cutoff_at"])
            if available <= frozen_at or available > cutoff:
                continue
            if row.home_points == row.away_points:
                continue  # ties excluded from binary learning
            record = {"game_id": game_id, "as_of": frozen_at.isoformat()}
            if _settle_record(conn, record, row) is not None:
                settled_count += 1

    # Rebuild posteriors exactly from the full settled ledger, then snapshot.
    blend = _load_blend(resolved_dsn, decay)
    _save_blend(resolved_dsn, blend)
    pending = sum(
        1
        for r in _read_ledger_counts(resolved_dsn)
        if r["settled"] is False
    )
    log.info("learning.settle: settled %d games, %d freeze events pending", settled_count, pending)
    return {
        "settled": settled_count,
        "pending": pending,
        "weights": blend.weights(),
        "experts": {name: state.to_dict() for name, state in blend.experts.items()},
    }


def _read_ledger_counts(dsn: str) -> list[dict]:
    from . import db as _db

    with _db.connect(dsn) as conn:
        return [
            {"game_id": r["game_id"], "settled": r["settled"]}
            for r in _db_records(conn, settled_only=False)
        ]


def blend_forecast_frame(forecasts: pd.DataFrame, *, decay: float = 1.0, dsn: str | None = None) -> pd.DataFrame:
    """Attach blended probabilities to a live forecast frame using learned state."""
    blend = _load_blend(_dsn(dsn), decay)
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


def learning_audit(*, dsn: str | None = None) -> dict:
    """Statistical post-mortem of everything the application has frozen and settled.

    This is how the system *learns from its own history* rather than merely
    accumulating it. For every expert tracked by the adaptive blend we compute,
    on the settled prospective (pre-kickoff) predictions only — read directly
    from the canonical ``prediction_outcomes`` ledger:

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

    from . import db as _db

    resolved_dsn = _dsn(dsn)
    with _db.connect(resolved_dsn) as conn:
        records = _db_records(conn, settled_only=True)
        current_weights = _db.latest_blend_weights(conn)
    records = [r for r in records if r.get("settled") is True]
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

    blend_values = [r["blend_p_home"] for r in records]
    blend_p = np.array(
        [float(v) if v is not None else np.nan for v in blend_values], dtype=float
    )

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
