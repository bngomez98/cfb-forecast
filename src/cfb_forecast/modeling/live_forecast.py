"""Live 2026 joint forecast orchestration from stored real matchup inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .joint_forecast import JointSimulationConfig, simulate_joint_game


REQUIRED_INPUTS = (
    "home_team",
    "away_team",
    "home_players",
    "away_players",
    "home_units",
    "away_units",
    "home_availability",
    "away_availability",
    "conditions",
)


def load_matchup_input(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    missing = [key for key in REQUIRED_INPUTS if key not in payload]
    if missing:
        raise ValueError(f"Live matchup input missing required fields: {missing}")
    if payload.get("season") != 2026:
        raise ValueError("The live joint forecast adapter is restricted to season 2026")
    if not payload.get("information_cutoff"):
        raise ValueError("Live matchup input requires information_cutoff")
    if payload.get("synthetic"):
        raise ValueError("Synthetic matchup inputs are forbidden for live forecasts")
    return payload


def forecast_live_2026(path: str | Path, draws: int = 50000, seed: int | None = 0) -> dict:
    payload = load_matchup_input(path)
    home = dict(payload["home_team"])
    away = dict(payload["away_team"])
    home["availability_probabilities"] = payload["home_availability"]
    away["availability_probabilities"] = payload["away_availability"]
    result = simulate_joint_game(
        home,
        away,
        conditions=payload["conditions"],
        covariance=payload.get("joint_covariance"),
        market=payload.get("market", {}),
        config=JointSimulationConfig(draws=draws, seed=seed),
    )
    return {
        "season": 2026,
        "game_id": payload["game_id"],
        "information_cutoff": payload["information_cutoff"],
        "source_manifest": payload.get("source_manifest", []),
        "football_forecast": result["football_forecast"],
        "market_conditioned_forecast": result["market_conditioned_forecast"],
        "simulated_data": False,
        "draws_are_uncertainty_propagation": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forecast one real 2026 matchup from stored live inputs")
    parser.add_argument("input", help="JSON matchup input with source receipts and cutoff")
    parser.add_argument("--draws", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(forecast_live_2026(args.input, args.draws, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
