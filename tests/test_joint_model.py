"""End-to-end tests of the joint model on a small synthetic league with known strengths."""
import numpy as np
import pandas as pd
import pytest

from cfb_forecast import football_model as fm
from cfb_forecast import forecasting as fc
from cfb_forecast import model_spec as sp
from cfb_forecast.database import connect
from cfb_forecast.observations import derive_observations, load_model_data
from synthetic_league import build

SEASONS = [2023, 2024, 2025]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    folder = tmp_path_factory.mktemp("league")
    conn = connect(folder / "db.sqlite3")
    truth = build(conn)
    derive_observations(conn, SEASONS)
    data = load_model_data(conn, SEASONS)
    return conn, truth, data, folder


def test_observations_cover_the_designed_streams(world):
    conn, _, data, _ = world
    names = {r[0] for r in conn.execute("SELECT DISTINCT metric FROM derived_team_observations")}
    assert {"pass_epa", "rush_epa", "success_rate", "sack_avoid", "scoring_opp_created"} <= names
    assert data.T == 8 and data.Y == 3
    assert data.drive_gd.size > 0 and set(data.drive_class) <= set(range(len(sp.OUTCOMES)))
    for g in sp.PLAYER_GROUPS:
        assert len(data.nodes[g]["player_id"]) > 0
        assert (data.nodes[g]["parent"] >= 0).any()


def test_joint_fit_recovers_team_strength_and_forecasts_end_to_end(world):
    conn, truth, data, folder = world
    hidden = pd.read_sql_query("SELECT game_id, week, season FROM games WHERE result_usable=0", conn)
    keep = sorted({2 * sp.WEEKS + w - 1 for w in hidden.week})
    f = fm.fit(data, steps=450, lr=0.03, log_every=0, log=None, keep_periods=keep)
    assert f.losses[-1] < f.losses[0] * 0.6                      # one joint objective improved
    bundle = fc.extract_bundle(f, n_draws=24, batch=12)
    state = bundle.draws["S_keep"].mean(0)[:, 0, :]              # (teams, components) at the first forecast week
    col = sp.col("OFF", "pass_efficiency")
    corr = np.corrcoef(state[:, col], truth["offense"] + truth["qb"])[0, 1]
    assert corr > 0.5
    # the whole thing lands in the database
    fc.store_estimates(conn, bundle)
    assert conn.execute("SELECT COUNT(*) FROM team_state_estimates").fetchone()[0] == data.T * (20 + 18)
    assert conn.execute("SELECT COUNT(*) FROM unit_state_estimates").fetchone()[0] == data.T * (3 + 4 + 6)
    assert conn.execute("SELECT COUNT(*) FROM player_state_estimates").fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM statistical_parameter_values").fetchone()[0] > 100
    # forecast the hidden weeks
    out = fc.forecast_games(conn, bundle, hidden.game_id.tolist(), folder)
    assert len(out) == len(hidden)
    assert out.home_win.between(0.01, 0.99).all()
    assert (out.expected_total > 10).all() and (out.expected_total < 120).all()
    assert fc.store_availability(conn, bundle, fc.Forecaster(bundle, conn), hidden.game_id.tolist()) > 0
    # stronger offense-minus-defense side should be favored more often than not
    strength = truth["offense"] + truth["qb"] - truth["defense"]
    g = pd.read_sql_query(f"SELECT game_id, home_id, away_id FROM games WHERE game_id IN ({','.join(map(str, hidden.game_id))})", conn).set_index("game_id")
    edge = np.array([strength[g.loc[i].home_id - 1] - strength[g.loc[i].away_id - 1] for i in out.game_id])
    assert np.corrcoef(edge, out.expected_margin)[0, 1] > 0.2


def test_score_distribution_is_exact_and_symmetric():
    probs = np.tile([0.5, 0.2, 0.1, 0.05, 0.08, 0.05, 0.015, 0.005], (4, 1))
    td = np.tile([0.05, 0.9, 0.05], (4, 1))
    ro = np.tile([0.2, 0.3, 0.2, 0.2, 0.1], (4, 1))
    J = fc.game_distribution(probs, probs, td, ro, ro, np.full(4, 11.0), np.full(4, 80.0))
    s = fc.summarize(J)
    assert abs(J.sum() - 1) < 1e-9
    assert abs(s["home_win"] - 0.5) < 1e-6 and abs(s["home_points"] - s["away_points"]) < 1e-6
    # 5.5 possessions each; per possession 0.2*7.0 + 0.1*3 + 0.005*4.0 own points, plus 0.005*4.0 credit
    assert abs(s["home_points"] - 5.5 * (1.72 + 0.02)) < 0.3
    assert abs(fc.cover_probability(s["margin"], 0.0) - 0.5) < 0.02 + s["tie_mass"]
