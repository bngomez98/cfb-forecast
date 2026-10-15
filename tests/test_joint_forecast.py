from cfb_forecast.modeling.joint_forecast import JointSimulationConfig, simulate_joint_game


def test_joint_forecast_returns_football_and_market_distributions():
    result = simulate_joint_game(
        {"offense": 1.0, "defense": 0.5, "tempo": 1.0, "drive_sustainability": .5, "special_teams": .2, "availability_probabilities": [.9, .8]},
        {"offense": 0.2, "defense": 0.0, "tempo": .8, "drive_sustainability": .2, "special_teams": .1, "availability_probabilities": [1.0]},
        {"wind_speed": 5, "precipitation": 0},
        market={"home_decimal": 1.8, "away_decimal": 2.1},
        config=JointSimulationConfig(draws=2000, seed=7),
    )
    football = result["football_forecast"]
    conditioned = result["market_conditioned_forecast"]
    assert 0 <= football["home_win_probability"] <= 1
    assert football["expected_home_score"] >= 0
    assert football["expected_away_score"] >= 0
    assert "market_weight" in conditioned
