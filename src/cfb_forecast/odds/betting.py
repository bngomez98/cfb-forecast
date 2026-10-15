"""Moneyline pricing and bankroll decisions for forecast probabilities."""
from __future__ import annotations

import math
from typing import Any


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if not math.isfinite(odds) or odds == 0:
        raise ValueError("American odds must be finite and non-zero")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def decimal_to_implied(decimal_odds: float) -> float:
    odds = float(decimal_odds)
    if not math.isfinite(odds) or odds <= 1:
        raise ValueError("Decimal odds must exceed 1")
    return 1.0 / odds


def american_to_implied(american_odds: float) -> float:
    return decimal_to_implied(american_to_decimal(american_odds))


def no_vig_two_way(home_decimal: float, away_decimal: float) -> tuple[float, float, float]:
    home = decimal_to_implied(home_decimal)
    away = decimal_to_implied(away_decimal)
    hold = home + away - 1.0
    total = home + away
    return home / total, away / total, hold


def kelly_fraction(probability: float, decimal_odds: float, fraction: float = 0.25, cap: float = 0.05) -> float:
    p = float(probability)
    d = float(decimal_odds)
    fraction = float(fraction)
    cap = float(cap)
    if not 0 <= p <= 1 or d <= 1 or fraction < 0 or cap < 0:
        raise ValueError("Invalid probability, odds, fraction, or cap")
    b = d - 1.0
    full = (p * d - 1.0) / b
    return max(0.0, min(cap, full * fraction))


def moneyline_decision(
    model_home_probability: float,
    home_decimal: float | None = None,
    away_decimal: float | None = None,
    home_american: float | None = None,
    away_american: float | None = None,
    min_edge: float = 0.02,
    kelly_fraction_size: float = 0.25,
    kelly_cap: float = 0.05,
) -> dict[str, Any]:
    """Compare model probability against a two-sided no-vig moneyline."""
    p_home = float(model_home_probability)
    if not 0 < p_home < 1:
        raise ValueError("Model probability must be between 0 and 1")
    if home_decimal is None and home_american is not None:
        home_decimal = american_to_decimal(home_american)
    if away_decimal is None and away_american is not None:
        away_decimal = american_to_decimal(away_american)
    if home_decimal is None or away_decimal is None:
        return {"bet_status": "no_market", "bet_side": None, "model_home_probability": p_home}
    fair_home, fair_away, hold = no_vig_two_way(home_decimal, away_decimal)
    candidates = {
        "home": (p_home, fair_home, home_decimal),
        "away": (1.0 - p_home, fair_away, away_decimal),
    }
    side, (p, fair, odds) = max(candidates.items(), key=lambda item: item[1][0] - item[1][1])
    edge = p - fair
    ev = p * (odds - 1.0) - (1.0 - p)
    bet = edge >= float(min_edge) and ev > 0
    return {
        "market_home_probability_no_vig": fair_home,
        "market_away_probability_no_vig": fair_away,
        "market_hold": hold,
        "home_decimal": float(home_decimal),
        "away_decimal": float(away_decimal),
        "model_home_probability": p_home,
        "best_bet_side": side,
        "best_bet_probability": p,
        "best_bet_fair_probability": fair,
        "best_bet_edge": edge,
        "best_bet_ev_per_unit": ev,
        "best_bet_kelly_fraction_full": max(0.0, ev / (odds - 1.0)),
        "recommended_stake_fraction": kelly_fraction(p, odds, kelly_fraction_size, kelly_cap) if bet else 0.0,
        "minimum_edge": float(min_edge),
        "bet_status": "bet" if bet else "pass",
    }
