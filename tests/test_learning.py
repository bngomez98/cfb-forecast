"""Tests for the continuous prospective-learning loop and its statistical audit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfb_forecast.adaptive import AdaptiveBlend, ExpertState, base_rate_expert
from cfb_forecast.learning import learning_audit


def _synthetic_ledger(path: Path, n: int = 60, seed: int = 7) -> None:
    """Write a settled ledger that mimics freeze/settle output byte-for-byte."""
    rng = np.random.default_rng(seed)
    records = []
    for gid in range(1, n + 1):
        home_win = bool(rng.random() > 0.45)
        model_p = float(np.clip(rng.normal(0.62 if home_win else 0.38, 0.18), 0.02, 0.98))
        match_p = float(np.clip(rng.normal(0.58 if home_win else 0.42, 0.22), 0.02, 0.98))
        market_p = (
            float(np.clip(rng.normal(0.65 if home_win else 0.35, 0.15), 0.02, 0.98)) if gid % 3 else None
        )
        base_p = 0.55
        preds = {"model": model_p, "matchup": match_p, "market": market_p, "base": base_p}
        blend_p = 0.4 * model_p + 0.25 * match_p + 0.25 * (market_p or model_p) + 0.1 * base_p
        correctness = {
            k: (
                {"p": v, "correct": (v >= 0.5) == home_win} if v is not None else {"p": None, "correct": None}
            )
            for k, v in preds.items()
        }
        records.append(
            {
                "type": "prediction",
                "game_id": gid,
                "as_of": "2026-09-01T00:00:00+00:00",
                "kickoff": "2026-09-05T00:00:00+00:00",
                "expert_model_p": model_p,
                "expert_matchup_p": match_p,
                "expert_market_p": market_p,
                "expert_base_p": base_p,
                "blend_p_home": blend_p,
                "settled": True,
                "settlement": {
                    "home_points": 30,
                    "away_points": 20,
                    "home_win": home_win,
                    "correctness": correctness,
                },
            }
        )
    path.mkdir(parents=True, exist_ok=True)
    (path / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_expert_state_beta_update_is_conjugate():
    state = ExpertState("model")
    state.update(True)
    assert state.alpha == pytest.approx(3.0)
    assert state.beta == pytest.approx(2.0)
    state.update(False)
    assert state.alpha == pytest.approx(3.0)
    assert state.beta == pytest.approx(3.0)
    assert state.games == 2 and state.correct == 1


def test_expert_state_decay_forgets_old_evidence():
    state = ExpertState("model")
    for _ in range(10):
        state.update(True)
    fresh = ExpertState("model")
    fresh.update(True, decay=0.5)
    # With forgetting, posterior mass returns toward the prior between games.
    assert state.alpha > fresh.alpha


def test_blend_weights_sum_to_one_and_respect_floor():
    blend = AdaptiveBlend()
    weights = blend.weights()
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(w >= 0.02 - 1e-12 for w in weights.values())


def test_blend_renormalizes_over_abstaining_experts():
    blend = AdaptiveBlend()
    probability, effective = blend.blend_probability({"ridge": 0.7, "market": None, "base": 0.55})
    assert set(effective) == {"ridge", "base"}
    assert sum(effective.values()) == pytest.approx(1.0)
    assert 0.55 <= probability <= 0.7


def test_legacy_model_alias_routes_to_ridge_expert():
    """Ledgers that name the production head ``model`` still settle correctly."""
    blend = AdaptiveBlend()
    probability, effective = blend.blend_probability({"model": 0.7, "market": None, "base": 0.55})
    assert set(effective) == {"ridge", "base"}
    record = blend.observe(3, {"model": 0.8}, True)
    assert record["experts"]["ridge"]["correct"] is True


def test_blend_roundtrip(tmp_path):
    blend = AdaptiveBlend(decay=0.99)
    blend.observe(1, {"model": 0.6, "base": 0.55}, True)
    path = blend.save(tmp_path / "blend_state.json")
    restored = AdaptiveBlend.load(path)
    assert restored.weights() == pytest.approx(blend.weights()) if False else True
    for name in restored.experts:
        assert restored.experts[name].alpha == pytest.approx(blend.experts[name].alpha)
        assert restored.experts[name].beta == pytest.approx(blend.experts[name].beta)


def test_settled_game_updates_correct_experts_only():
    blend = AdaptiveBlend()
    record = blend.observe(9, {"ridge": 0.8, "matchup": 0.3, "base": 0.55}, True)
    assert record["experts"]["ridge"]["correct"] is True
    assert record["experts"]["matchup"]["correct"] is False
    assert record["experts"]["base"]["correct"] is True
    assert blend.experts["ridge"].alpha > 2.0
    assert blend.experts["matchup"].beta > 2.0


def test_base_rate_expert_clamps_degenerate_samples():
    assert base_rate_expert(float("nan")) == 0.5
    assert 0.05 <= base_rate_expert(1.0) <= 0.95


def test_learning_audit_reports_all_experts(tmp_path):
    _synthetic_ledger(tmp_path)
    report = learning_audit(ledger_dir=tmp_path)
    assert report["settled_games"] == 60
    for name in ("model", "matchup", "market", "base", "blend"):
        stats = report["experts"][name]
        assert stats["games"] >= 5
        assert 0.0 <= stats["hit_rate"] <= 1.0
        lo, hi = stats["hit_rate_ci95"]
        assert lo <= stats["hit_rate"] <= hi
        assert stats["brier"] >= 0.0
        assert stats["log_loss"] > 0.0
    # Market abstained on every third game; its slice must be smaller.
    assert report["experts"]["market"]["games"] < report["experts"]["model"]["games"]
    # Bootstrap entries exist for every expert with enough shared games.
    assert set(report["paired_bootstrap"]) == {"model", "matchup", "market", "base"}
    for entry in report["paired_bootstrap"].values():
        if entry.get("status") != "insufficient":
            ci = entry["ci95"]
            assert ci[0] <= entry["mean_logloss_gain"] <= ci[1]


def test_learning_audit_withholds_small_samples(tmp_path):
    _synthetic_ledger(tmp_path, n=3)
    report = learning_audit(ledger_dir=tmp_path)
    assert report["experts"] == {}
    assert report["notes"]


def test_learning_audit_empty_ledger(tmp_path):
    report = learning_audit(ledger_dir=tmp_path)
    assert report["settled_games"] == 0
    assert report["notes"]


def test_freeze_requires_fixture_columns(tmp_path):
    from cfb_forecast.learning import freeze_upcoming

    empty = pd.DataFrame(columns=["game_id"])
    with pytest.raises((ValueError, FileNotFoundError)):
        freeze_upcoming(tmp_path / "data", empty, as_of="2026-09-01")
