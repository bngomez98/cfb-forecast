"""Hierarchical player ratings: recovery, pooling, point-in-time safety, feature layout."""
import numpy as np
import pandas as pd

from cfb_forecast.features import attach_player_features, feature_columns
from cfb_forecast.players import (
    PlayerConfig,
    build_player_layer,
    fit_player_ratings,
    player_feature_matrix,
    player_features_by_week,
    player_plays,
)


def _raw_play(game_id, team, defn, role, yards):
    is_pass = role == "pass"
    return {"game_id": game_id, "pos_team_id": team, "def_pos_team_id": defn,
            "rush": not is_pass, "pass": is_pass, "sack": False, "statYardage": yards,
            "passer_player_id": 1000 if is_pass else np.nan,
            "rusher_player_id": np.nan if is_pass else 2000}


def test_player_plays_credits_the_ball_carrier():
    raw = pd.DataFrame([
        _raw_play(1, 10, 20, "pass", 12), _raw_play(1, 10, 20, "rush", 5),
        {"game_id": 1, "pos_team_id": 10, "def_pos_team_id": 20, "rush": False, "pass": False,
         "sack": False, "statYardage": 0, "passer_player_id": np.nan, "rusher_player_id": np.nan},  # not a run/pass
    ])
    pp = player_plays(raw)
    assert set(pp.role) == {"pass", "rush"}
    assert pp[pp.role == "pass"].player_id.iloc[0] == 1000
    assert len(pp) == 2  # the non-carrier play is dropped


def _synthetic_games():
    """Two offenses with a known gap facing a shared defense; truth: team A >> team B."""
    rng = np.random.default_rng(0)
    rows = []
    gid = 1
    start = pd.Timestamp("2026-08-30T17:00:00Z")
    for week in range(1, 9):
        for team, base in ((10, 2.0), (11, -2.0)):
            opp = 11 if team == 10 else 10
            for role, carrier in (("pass", 100 + team), ("rush", 200 + team)):
                yards = rng.normal(6 + base if role == "pass" else 4 + base, 3, size=25)
                for y in yards:
                    rows.append({"game_id": gid, "role": role, "player_id": carrier, "team_id": team,
                                 "def_id": opp, "plays": 1, "yards": float(y), "yards_per_play": float(y),
                                 "season": 2026, "week": week, "start_date": start + pd.Timedelta(weeks=week)})
            gid += 1
    agg = pd.DataFrame(rows).groupby(["game_id", "role", "player_id", "team_id", "def_id", "season", "week", "start_date"],
                                     as_index=False).agg(plays=("yards", "size"), yards=("yards", "sum"), yards_per_play=("yards", "mean"))
    return agg


def test_ratings_recover_known_offense_gap_and_are_pooled():
    agg = _synthetic_games()
    ratings, team = build_player_layer(agg, "2026-11-01T00:00:00Z", PlayerConfig(half_life_weeks=50))
    off = team.set_index("team_id").player_offense
    assert off[10] > off[11]  # the stronger offense is rated higher
    # shrinkage keeps standardized ratings bounded well inside the raw ~1.3 SD gap
    assert off.abs().max() < 2.0


def test_point_in_time_excludes_games_after_cutoff():
    agg = _synthetic_games()
    early = fit_player_ratings(agg, "2026-09-10T00:00:00Z")
    late = fit_player_ratings(agg, "2026-11-01T00:00:00Z")
    assert early["league"]["pass"]["games"] < late["league"]["pass"]["games"]
    # a cutoff before any game yields an empty fit, never a peek at the future
    empty = fit_player_ratings(agg, "2026-08-01T00:00:00Z")
    assert empty["players"] == {}


def test_feature_matrix_layout_matches_columns():
    pf = pd.DataFrame(
        {"home_player_offense": [1.0], "away_player_offense": [-1.0],
         "home_player_defense": [0.5], "away_player_defense": [-0.5], "player_net_signal": [0.0]},
        index=pd.Index([99], name="game_id"))
    block = player_feature_matrix([99], pf)
    assert block.shape == (1, 2, 2)
    assert block[0, 0, 0] == 1.0 and block[0, 0, 1] == -0.5   # home offense, away defense
    assert block[0, 1, 0] == -1.0 and block[0, 1, 1] == 0.5   # away offense, home defense
    F = np.zeros((1, 2, 26))
    F28 = attach_player_features(F, [99], pf)
    assert F28.shape == (1, 2, 28)
    assert feature_columns("pbp_player") == list(range(28))
    assert F28[0, 0, 26] == 1.0 and F28[0, 0, 27] == -0.5
    # a game missing from the table defaults to league-average zero, never NaN
    assert not np.isnan(player_feature_matrix([12345], pf)).any()


def test_player_features_by_week_is_one_fit_per_week():
    agg = _synthetic_games()
    games = agg[["game_id", "season", "week", "team_id"]].drop_duplicates("game_id").copy()
    games = games.rename(columns={"team_id": "home_id"})
    games["away_id"] = np.where(games.home_id == 10, 11, 10)
    games["start_date"] = agg.groupby("game_id").start_date.first().reindex(games.game_id).to_numpy()
    pf = player_features_by_week(agg, games, lead_hours=48.0)
    assert set(["home_player_offense", "away_player_offense", "home_player_defense", "away_player_defense"]).issubset(pf.columns)
    assert pf.index.name == "game_id" and len(pf) == len(games)
    assert np.isfinite(pf.to_numpy()).all()


def test_cached_player_features_match_a_full_pass_and_fit_only_new_weeks(tmp_path, monkeypatch):
    import cfb_forecast.players as players

    agg = _synthetic_games()
    games = agg[["game_id", "season", "week", "team_id"]].drop_duplicates("game_id").copy()
    games = games.rename(columns={"team_id": "home_id"})
    games["away_id"] = np.where(games.home_id == 10, 11, 10)
    games["start_date"] = agg.groupby("game_id").start_date.first().reindex(games.game_id).to_numpy()
    full = player_features_by_week(agg, games, lead_hours=48.0)
    cache = tmp_path / "player_features_cache.parquet"
    weeks = sorted(games.week.unique())
    early = games[games.week < weeks[-1]]
    players.cached_player_features(agg, early, cache)
    fitted = []
    original = players.player_features_by_week
    monkeypatch.setattr(players, "player_features_by_week", lambda a, g, *args, **kw: fitted.append(sorted(g.week.unique())) or original(a, g, *args, **kw))
    out = players.cached_player_features(agg, games, cache)
    assert fitted == [[weeks[-1]]]  # only the new week is fit
    pd.testing.assert_frame_equal(out.loc[full.index].sort_index(), full.sort_index(), check_exact=False)
    # a different configuration discards the cache
    fitted.clear()
    players.cached_player_features(agg, games, cache, lead_hours=24.0)
    assert len(fitted) == 1
