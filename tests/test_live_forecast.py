import json
from pathlib import Path

import pytest

from cfb_forecast.modeling.live_forecast import forecast_live_2026, load_matchup_input


def live_payload():
    return {
        "season": 2026,
        "game_id": 1,
        "information_cutoff": "2026-09-04T12:00:00Z",
        "home_team": {"offense": 1, "defense": .5, "tempo": 1, "drive_sustainability": .5, "special_teams": .2},
        "away_team": {"offense": .2, "defense": 0, "tempo": .8, "drive_sustainability": .2, "special_teams": .1},
        "home_players": [],
        "away_players": [],
        "home_units": {},
        "away_units": {},
        "home_availability": [1.0],
        "away_availability": [1.0],
        "conditions": {"wind_speed": 5, "precipitation": 0},
        "market": {"home_decimal": 1.8, "away_decimal": 2.1},
        "source_manifest": [{"provider": "test", "retrieved_at": "2026-09-04T11:00:00Z", "sha256": "test"}],
        "synthetic": False,
    }


def test_live_2026_forecast_requires_real_complete_inputs(tmp_path: Path):
    path = tmp_path / "matchup.json"
    path.write_text(json.dumps(live_payload()))
    result = forecast_live_2026(path, draws=1000, seed=2)
    assert result["season"] == 2026
    assert result["simulated_data"] is False
    assert "football_forecast" in result
    assert "market_conditioned_forecast" in result


def test_live_forecast_rejects_synthetic_and_wrong_season(tmp_path: Path):
    payload = live_payload()
    payload["synthetic"] = True
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Synthetic"):
        load_matchup_input(path)
    payload["synthetic"] = False
    payload["season"] = 2025
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="season 2026"):
        load_matchup_input(path)
