"""Command-line interface for live FBS forecasts and measured model research."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="FBS forecasts, Sunday validation, and model research"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    live = sub.add_parser(
        "sunday-run",
        help="Collect game inputs, validate prior predictions, then forecast the coming slate",
    )
    live.add_argument("--data", default="data")
    live.add_argument("--artifacts", default="artifacts")
    live.add_argument("--season", type=int, default=2026)
    live.add_argument("--model", default=None)
    live.add_argument("--timezone", default="America/Chicago")
    live.add_argument(
        "--scheduled",
        action="store_true",
        help="Run only at Sunday 10 AM in the selected timezone",
    )
    live.add_argument(
        "--offline",
        action="store_true",
        help="Use existing data; live game context still requires public sources",
    )
    validation = sub.add_parser(
        "validate", help="Validate saved real predictions against current final results"
    )
    validation.add_argument("--data", default="data")
    validation.add_argument("--runs", default="artifacts/runs")
    validation.add_argument("--out", default="artifacts/validation")
    fetch = sub.add_parser(
        "fetch", help="Download schedules and advanced summaries from SportsDataverse"
    )
    fetch.add_argument("--data", default="data")
    fetch.add_argument("--seasons", nargs="+", type=int, required=True)
    fetch.add_argument("--refresh", action="store_true")
    fetch.add_argument("--raw-pbp", action="store_true")
    train = sub.add_parser(
        "backtest",
        help="Train score/PBP models and measure their historical performance",
    )
    train.add_argument("--data", default="data")
    train.add_argument("--out", default="artifacts/training")
    train.add_argument("--archive", default=None)
    train.add_argument("--warmup", type=int, default=2021)
    train.add_argument("--train-start", type=int, default=2022)
    train.add_argument("--validation", type=int, default=2024)
    train.add_argument("--test", type=int, default=2025)
    strength = sub.add_parser(
        "train-strength",
        help="Evaluate opponent-adjusted dynamic strength with earlier-season calibration",
    )
    strength.add_argument("--data", default="data")
    strength.add_argument("--out", default="artifacts/training/strength")
    audit = sub.add_parser(
        "probability-audit",
        help="Evaluate prior-season calibration, constrained fitting, and robustness",
    )
    audit.add_argument("--input", required=True)
    audit.add_argument("--out", required=True)
    audit.add_argument("--probability-column", default="score_p")
    audit.add_argument("--outcome-column", default="home_win")
    audit.add_argument("--min-games", type=int, default=100)
    audit.add_argument("--regularization", type=float, default=0.01)
    audit.add_argument("--max-change", type=float, default=None)
    audit.add_argument("--stress-radius", type=float, default=0.1)
    audit.add_argument(
        "--model", default=None, help="Optionally write a separate candidate model"
    )
    forecast = sub.add_parser(
        "forecast", help="Calculate predictions for a pregame window"
    )
    forecast.add_argument("--data", default="data")
    forecast.add_argument("--model", default="artifacts/training/model")
    forecast.add_argument("--out", default="artifacts/manual-forecast")
    forecast.add_argument("--as-of", required=True)
    forecast.add_argument("--through", required=True)
    forecast.add_argument("--season", type=int, required=True)
    forecast.add_argument("--market", default=None)
    forecast.add_argument("--market-provider", default=None)
    args = parser.parse_args()
    if args.command == "sunday-run":
        from .sunday import sunday_run

        print(
            json.dumps(
                sunday_run(
                    args.data,
                    args.artifacts,
                    args.season,
                    args.model,
                    args.timezone,
                    args.scheduled,
                    not args.offline,
                ),
                indent=2,
            )
        )
    elif args.command == "validate":
        from .data import load_data
        from .sources import utc_now
        from .validation import validate_runs

        schedules, _, _ = load_data(args.data)
        games, summary = validate_runs(args.runs, schedules, utc_now())
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        games.to_csv(out / "games.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    elif args.command == "fetch":
        from .data import fetch

        fetch(args.data, args.seasons, args.refresh, args.raw_pbp)
    elif args.command == "backtest":
        from .backtest import run_backtest

        result, _ = run_backtest(
            args.data,
            args.out,
            args.archive,
            args.warmup,
            args.train_start,
            args.validation,
            args.test,
        )
        print(result.to_string(index=False))
    elif args.command == "train-strength":
        from .strength import train_strength

        print(json.dumps(train_strength(args.data, args.out), indent=2))
    elif args.command == "probability-audit":
        from .probability import audit_probabilities

        print(
            json.dumps(
                audit_probabilities(
                    args.input,
                    args.out,
                    args.probability_column,
                    args.outcome_column,
                    args.min_games,
                    args.regularization,
                    args.model,
                    args.max_change,
                    args.stress_radius,
                ),
                indent=2,
            )
        )
    elif args.command == "forecast":
        from .forecast import forecast

        result = forecast(
            args.data,
            args.model,
            args.out,
            args.as_of,
            args.through,
            args.season,
            market_path=args.market,
            market_provider=args.market_provider,
            analytic=True,
        )
        print(
            result[
                [
                    "away_team",
                    "home_team",
                    "home_win_probability",
                    "expected_away_score",
                    "expected_home_score",
                ]
            ].to_string(index=False)
        )
