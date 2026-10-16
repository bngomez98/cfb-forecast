"""Shared hypothetical exposure rules for Sunday and weekly publications."""
import pandas as pd
from ..odds.decision import moneyline_diagnostics


def apply_decisions(predictions, complete=None):
    frame = predictions.copy()
    frame["inputs_complete"] = False if complete is None else complete
    decisions = []
    for row in frame.to_dict("records"):
        prices = [row.get(f"{side}_ml_decimal") for side in ("home", "away")]
        prices = [None if price is None or pd.isna(price) else price for price in prices]
        decision = moneyline_diagnostics(row["home_win_probability"], *prices)
        if not row["inputs_complete"]:
            decision.update(moneyline_candidate="none", hypothetical_bankroll_fraction=0.0,
                            decision_reason="incomplete_weather_injury_lineup_or_stats_inputs")
        else:
            decision["decision_reason"] = "price_and_probability_thresholds"
        decisions.append(decision)
    for column in pd.DataFrame(decisions):
        frame[column] = pd.DataFrame(decisions)[column].to_numpy()
    exposure = float(frame.hypothetical_bankroll_fraction.sum())
    if exposure > 0.05:
        frame.hypothetical_bankroll_fraction *= 0.05 / exposure
    frame["slate_exposure_cap"] = 0.05
    return frame
