"""Settle frozen forecasts using later outcomes without changing predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..ingest.data import timestamp
from ..storage.evidence import canonical_json, verify_evidence


def gaussian_crps(mean, sd, outcome):
    """CRPS in points for a Normal predictive distribution."""
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("CRPS requires a positive predictive standard deviation")
    z = (outcome - mean) / sd
    return float(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))


def settle_evidence(conn, run_key, schedules, evaluated_at, out=None, evidence_dir=None):
    at = timestamp(evaluated_at)
    verify_evidence(conn, run_key, evidence_dir)
    if schedules.game_id.duplicated().any():
        raise ValueError("Duplicate game results")
    outcomes = schedules[schedules.result_usable & (schedules.result_available_at <= at)].set_index("game_id")
    settled = []
    pending = 0
    evidence = conn.execute("SELECT * FROM forecast_evidence WHERE run_key=? ORDER BY game_id", (str(run_key),)).fetchall()
    for frozen in evidence:
        pred = json.loads(frozen["payload_json"])
        gid = frozen["game_id"]
        if gid not in outcomes.index:
            pending += 1
            continue
        actual = outcomes.loc[gid]
        # Legacy forecasts without a declared cutoff cannot be labelled prospective.
        cutoff = timestamp(pred["as_of"]) if pred.get("as_of") else None
        kickoff = timestamp(pred["kickoff"])
        if cutoff is not None and cutoff >= kickoff:
            raise ValueError(f"Forecast {gid} was not issued before kickoff")
        if actual.home_team != pred["home_team"] or actual.away_team != pred["away_team"]:
            raise ValueError(f"Result team identity mismatch for {gid}")
        home, away = float(actual.home_points), float(actual.away_points)
        if home == away:
            raise ValueError(f"Tied final score for {gid}")
        probability = float(pred["home_win_probability"])
        if not 0 <= probability <= 1:
            raise ValueError(f"Invalid probability for {gid}")
        ph, pa = float(pred["expected_home_score"]), float(pred["expected_away_score"])
        margin, total = home - away, home + away
        winner = int(home > away)
        clipped = np.clip(probability, 1e-15, 1 - 1e-15)
        row = {
            "run_id": str(run_key), "game_id": gid,
            "home_team": pred["home_team"], "away_team": pred["away_team"],
            "forecast_as_of": pred.get("as_of"), "kickoff": pred["kickoff"],
            "forecast_mode": pred.get("forecast_mode", "legacy_unspecified"),
            "prospective_cutoff_recorded": cutoff is not None,
            "forecast_recorded_at": frozen["recorded_at"],
            "prospective_record": cutoff is not None and timestamp(frozen["recorded_at"]) < kickoff,
            "expected_home_score": ph, "expected_away_score": pa,
            "actual_home_score": home, "actual_away_score": away,
            "predicted_winner": pred["home_team"] if probability >= .5 else pred["away_team"],
            "actual_winner": pred["home_team"] if winner else pred["away_team"],
            "winner_correct": bool((probability >= .5) == winner),
            "home_win_probability": probability,
            "projected_home_margin": ph - pa, "actual_home_margin": margin,
            "projected_total": ph + pa, "actual_total": total,
            "home_score_error": ph - home, "away_score_error": pa - away,
            "margin_error": ph - pa - margin, "total_error": ph + pa - total,
            "brier": float((probability - winner) ** 2),
            "log_loss": float(-winner * np.log(clipped) - (1 - winner) * np.log(1 - clipped)),
        }
        for name, mean, value in (("margin", ph - pa, margin), ("total", ph + pa, total)):
            sd = pred.get(f"{name}_standard_deviation")
            if sd is not None:
                row[f"{name}_crps"] = gaussian_crps(mean, float(sd), value)
                radius = norm.ppf(.9) * float(sd)
                row[f"{name}_coverage_80"] = bool(mean - radius <= value <= mean + radius)
        outcome = {"game_id": gid, "home": home, "away": away, "available_at": actual.result_available_at.isoformat()}
        digest = hashlib.sha256(canonical_json(outcome).encode()).hexdigest()
        with conn:
            conn.execute("INSERT OR IGNORE INTO forecast_evaluations VALUES(?,?,?,?,?)", (str(run_key), gid, digest, at.isoformat(), canonical_json(row)))
        settled.append(row)
    frame = pd.DataFrame(settled)
    report = {"run_id": str(run_key), "evaluated_at": at.isoformat(), "settled_games": len(frame), "pending_games": pending, "error_sign": "prediction minus actual"}
    if len(frame):
        report.update(accuracy=float(frame.winner_correct.mean()), margin_mae=float(frame.margin_error.abs().mean()), total_mae=float(frame.total_error.abs().mean()), team_score_mae=float((frame.home_score_error.abs().mean() + frame.away_score_error.abs().mean()) / 2), brier=float(frame.brier.mean()), log_loss=float(frame.log_loss.mean()))
        for column in ("margin_crps", "total_crps", "margin_coverage_80", "total_coverage_80"):
            if column in frame:
                report[column] = float(frame[column].mean())
    if out:
        root = Path(out)
        root.mkdir(parents=True, exist_ok=True)
        frame.to_csv(root / "evaluation.csv", index=False)
        (root / "evaluation.json").write_text(json.dumps(report, indent=2))
    return frame, report
