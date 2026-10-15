"""Validate saved pregame predictions against real completed FBS outcomes."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from ..ingest.data import timestamp
from .evaluate import metrics, calibration, calibration_test, spread_regression


def validate_runs(run_root, schedules, as_of):
    root = Path(run_root)
    at = timestamp(as_of)
    reports, rows = [], []
    outcomes = schedules[
        schedules.fbs_game
        & schedules.result_usable
        & (schedules.result_available_at <= at)
    ]
    paths = list(root.glob("*/forecasts.csv"))
    if root.name == "runs":
        paths += list((root.parent / "weekly").glob("*/forecasts.csv"))
    for path in sorted(paths):
        manifest_path = path.parent / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        from ..pipeline.integrity import verify_run_integrity
        verify_run_integrity(path.parent, manifest)
        pred = pd.read_csv(path)
        if pred.empty:
            continue
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
            if "projected_home_margin" in matched and matched.projected_home_margin.notna().any():
                report["spread_mae"] = float(np.mean(np.abs(matched.actual_margin - matched.projected_home_margin)))
            if "home_spread" in matched:
                priced_spread = matched.dropna(subset=["home_spread"])
                if len(priced_spread) >= 8:
                    report["market_spread_regression"] = spread_regression(
                        priced_spread.actual_margin, priced_spread.home_spread
                    )
            if "market_home_win_probability_no_vig" in matched:
                priced = matched.dropna(subset=["market_home_win_probability_no_vig"])
                if len(priced):
                    report["market_probability_metrics"] = metrics(
                        priced.home_win, priced.market_home_win_probability_no_vig
                    )
            for series in ("football_only", "market_conditioned"):
                column = f"{series}_home_win_probability"
                if column in matched:
                    subset = matched.dropna(subset=[column])
                    if len(subset):
                        report[f"{series}_metrics"] = metrics(subset.home_win, subset[column],
                            subset[f"{series}_projected_home_margin"], subset.actual_margin,
                            subset[f"{series}_projected_total"], subset.actual_total)
                        report[f"{series}_metrics"]["games"] = int(len(subset))
            series_columns = ["football_only_home_win_probability", "market_conditioned_home_win_probability"]
            if all(column in matched for column in series_columns):
                paired = matched.dropna(subset=series_columns)
                if len(paired):
                    report["paired_series_comparison"] = {
                        "games": int(len(paired)),
                        "football_only": metrics(paired.home_win, paired[series_columns[0]]),
                        "market_conditioned": metrics(paired.home_win, paired[series_columns[1]]),
                    }
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
            report["calibration"] = calibration_test(matched.home_win, matched.home_win_probability)
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
        summary["metrics"] = metrics(combined.home_win, combined.home_win_probability, combined.projected_home_margin, combined.actual_margin, combined.projected_total, combined.actual_total)
        summary["calibration"] = calibration_test(combined.home_win, combined.home_win_probability)
    return combined, summary
