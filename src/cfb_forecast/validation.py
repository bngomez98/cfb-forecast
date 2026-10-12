"""Validate saved pregame predictions against real completed FBS outcomes."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import sha256, timestamp
from .evaluate import metrics, calibration


def validate_runs(run_root, schedules, as_of):
    root = Path(run_root)
    at = timestamp(as_of)
    reports, rows = [], []
    outcomes = schedules[
        schedules.fbs_game
        & schedules.result_usable
        & (schedules.result_available_at <= at)
    ]
    for path in sorted(root.glob("*/forecasts.csv")):
        manifest_path = path.parent / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if sha256(path) != manifest["prediction_sha256"]:
            raise ValueError(f"Saved predictions changed: {path}")
        for name, expected in manifest.get("input_hashes", {}).items():
            evidence = path.parent / "inputs" / name
            if not evidence.exists() or sha256(evidence) != expected:
                raise ValueError(f"Saved pregame inputs changed: {evidence}")
        pred = pd.read_csv(path)
        if (
            pd.to_datetime(pred.as_of, utc=True)
            >= pd.to_datetime(pred.kickoff, utc=True)
        ).any():
            raise ValueError(f"A saved prediction was issued after kickoff: {path}")
        matched = pred.merge(
            outcomes[["game_id", "home_points", "away_points"]],
            on="game_id",
            validate="one_to_one",
        )
        matched["home_win"] = (matched.home_points > matched.away_points).astype(int)
        matched["actual_margin"] = matched.home_points - matched.away_points
        matched["actual_total"] = matched.home_points + matched.away_points
        matched["run_id"] = path.parent.name
        report = {
            "run_id": path.parent.name,
            "forecast_games": len(pred),
            "settled_games": len(matched),
            "pending_games": len(pred) - len(matched),
            "validated_at": at.isoformat(),
        }
        if len(matched):
            report.update(
                metrics(
                    matched.home_win,
                    matched.home_win_probability,
                    matched.projected_home_margin,
                    matched.actual_margin,
                    matched.projected_total,
                    matched.actual_total,
                )
            )
            if "market_home_win_probability_no_vig" in matched:
                priced = matched.dropna(subset=["market_home_win_probability_no_vig"])
                if len(priced):
                    report["market_probability_metrics"] = metrics(
                        priced.home_win, priced.market_home_win_probability_no_vig
                    )
            matched["hypothetical_profit_units"] = np.nan
            for side, outcome in [
                ("home", matched.home_win),
                ("away", 1 - matched.home_win),
            ]:
                column = f"{side}_ml_decimal"
                if column in matched:
                    mask = (matched.moneyline_candidate == side) & matched[
                        column
                    ].notna()
                    matched.loc[mask, "hypothetical_profit_units"] = (
                        outcome[mask] * matched.loc[mask, column] - 1
                    )
            bets = matched.hypothetical_profit_units.dropna()
            report["price_available_candidates"] = len(bets)
            report["hypothetical_profit_units"] = float(bets.sum())
            report["hypothetical_roi"] = float(bets.mean()) if len(bets) else None
            report["betting_definition"] = (
                "One unit per eligible saved candidate at its recorded price. No wager execution is inferred."
            )
            calibration(matched.home_win, matched.home_win_probability).to_csv(
                path.parent / "validation_calibration.csv", index=False
            )
            matched.to_csv(path.parent / "validation.csv", index=False)
            rows.append(matched)
        (path.parent / "validation.json").write_text(json.dumps(report, indent=2))
        reports.append(report)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary = {
        "validated_at": at.isoformat(),
        "runs": reports,
        "settled_games": len(combined),
    }
    if len(combined):
        summary["metrics"] = metrics(combined.home_win, combined.home_win_probability)
    return combined, summary
