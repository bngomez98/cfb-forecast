"""Moneyline price comparisons, separate from probability estimation."""

import math


def american_to_decimal(odds):
    odds = float(odds)
    if not math.isfinite(odds) or abs(odds) < 100:
        raise ValueError("American odds must be <= -100 or >= 100")
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def moneyline_diagnostics(
    probability,
    home_decimal=None,
    away_decimal=None,
    min_edge=0.02,
    kelly_fraction=0.25,
    stake_cap=0.01,
):
    """Return price diagnostics and a capped hypothetical bankroll fraction."""
    p = float(probability)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Win probability must be in [0, 1]")
    if not all(
        math.isfinite(x) and 0 <= x <= 1 for x in [min_edge, kelly_fraction, stake_cap]
    ):
        raise ValueError("Decision settings must be finite fractions in [0, 1]")
    result = {
        "moneyline_status": "missing_two_sided_prices",
        "moneyline_candidate": "none",
        "hypothetical_bankroll_fraction": 0.0,
    }
    if home_decimal is None or away_decimal is None:
        return result
    h, a = float(home_decimal), float(away_decimal)
    if not all(math.isfinite(x) and x > 1 for x in [h, a]):
        raise ValueError("Moneyline decimal odds must exceed 1")
    overround = 1 / h + 1 / a
    fair = (1 / h) / overround
    home_ev, away_ev = p * h - 1, (1 - p) * a - 1
    home_kelly, away_kelly = max(0.0, home_ev / (h - 1)), max(0.0, away_ev / (a - 1))
    home_edge, away_edge = p - fair, fair - p
    candidates = []
    for side, edge, ev, kelly in [
        ("home", home_edge, home_ev, home_kelly),
        ("away", away_edge, away_ev, away_kelly),
    ]:
        if edge >= min_edge and ev > 0:
            candidates.append((ev, side, kelly))
    result.update(
        moneyline_status="two_sided_prices_available",
        market_home_win_probability_no_vig=fair,
        market_overround=overround - 1,
        home_moneyline_edge=home_edge,
        away_moneyline_edge=away_edge,
        home_moneyline_expected_net_per_unit=home_ev,
        away_moneyline_expected_net_per_unit=away_ev,
        home_full_kelly_fraction=home_kelly,
        away_full_kelly_fraction=away_kelly,
        minimum_moneyline_edge=min_edge,
        kelly_multiplier=kelly_fraction,
        per_game_stake_cap=stake_cap,
    )
    if candidates:
        _, side, kelly = max(candidates)
        result.update(
            moneyline_candidate=side,
            hypothetical_bankroll_fraction=min(stake_cap, kelly_fraction * kelly),
        )
    return result
