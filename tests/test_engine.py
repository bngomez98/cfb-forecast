import numpy as np
import pandas as pd
import pytest
from cfb_forecast.config import StateConfig
from cfb_forecast.data import normalize_schedule,timestamp
from cfb_forecast.features import clean_plays
from cfb_forecast.ratings import DynamicRatings,build_features
from cfb_forecast.model import ScoreHead,residual_covariance
from cfb_forecast.gaussian import checked_covariance,win_probability
from cfb_forecast.analytic import score_summary
from cfb_forecast.market import market_at_cutoff
from cfb_forecast.evaluate import metrics

def fixture_games():
    return normalize_schedule(pd.DataFrame([
        dict(game_id=1,season=2021,week=1,start_date="2021-09-04T16:00Z",completed=True,neutral_site=False,home_id=1,away_id=2,home_team="A",away_team="B",home_points=28,away_points=14,fbs_game=True),
        dict(game_id=2,season=2022,week=1,start_date="2022-09-03T16:00Z",completed=True,neutral_site=False,home_id=2,away_id=1,home_team="B",away_team="A",home_points=17,away_points=24,fbs_game=True),
        dict(game_id=3,season=2022,week=2,start_date="2022-09-10T16:00Z",completed=True,neutral_site=True,home_id=1,away_id=2,home_team="A",away_team="B",home_points=31,away_points=21,fbs_game=True)]))

def test_future_outcomes_cannot_change_earlier_forecast():
    g=fixture_games(); Y=np.ones((3,2,5)); Y[:,:,0]=g[["home_points","away_points"]].to_numpy()
    kwargs=dict(center=np.zeros(5),scale=np.ones(5),config=StateConfig(),warmup=2021)
    F,cuts,latest=build_features(g,Y,**kwargs)
    changed=Y.copy(); changed[1:]*=100
    alt,_,_=build_features(g,changed,**kwargs)
    np.testing.assert_array_equal(F[1],alt[1])
    assert not np.allclose(F[2],alt[2])
    assert all(pd.Timestamp(last)<cut for last,cut in zip(latest[1:],cuts.iloc[1:]))

def test_result_at_cutoff_is_excluded():
    g=fixture_games(); g.loc[0,"result_available_at"]=pd.Timestamp("2022-08-29T00:00Z")
    Y=np.ones((3,2,5)); F,cuts,last=build_features(g,Y,np.zeros(5),np.ones(5),StateConfig())
    assert last[1] is None

def test_state_updates_have_positive_covariance_and_opponent_effect():
    state=DynamicRatings([1,2],np.zeros(5),np.ones(5),StateConfig())
    for day in range(10):
        state.update(1,2,True,np.array([[2.]*5,[-2.]*5]),pd.Timestamp("2024-09-01T00:00Z")+pd.Timedelta(days=day))
    for matrix in state.cov:
        assert np.linalg.eigvalsh(matrix).min()>0
        np.testing.assert_allclose(matrix,matrix.T)
    assert state.mean[0,0]>0  # good home offense
    assert state.mean[0,3]<0  # weak opposing defense

def test_state_cannot_move_backward():
    s=DynamicRatings([1,2],np.zeros(5),np.ones(5),StateConfig())
    s.advance("2024-09-02T00:00Z")
    with pytest.raises(ValueError): s.advance("2024-09-01T00:00Z")

def test_neutral_swapped_matchup_symmetry():
    s=DynamicRatings([1,2],np.zeros(5),np.ones(5),StateConfig())
    s.update(1,2,True,np.array([[2.]*5,[-2.]*5]),"2024-09-01T00:00Z")
    np.testing.assert_allclose(s.features(1,2,True),s.features(2,1,True)[::-1])

def test_home_venue_is_signed():
    s=DynamicRatings([1,2],np.zeros(5),np.ones(5),StateConfig())
    np.testing.assert_array_equal(s.features(1,2,False)[:,22],[1,-1])
    np.testing.assert_array_equal(s.features(1,2,True)[:,22],[0,0])

def test_head_json_roundtrip():
    rng=np.random.default_rng(2); F=rng.normal(size=(100,2,26)); Y=rng.normal(25,5,size=(100,2,5))
    h=ScoreHead().fit(F,Y,np.ones(100,dtype=bool))
    np.testing.assert_allclose(h.predict(F),ScoreHead.from_dict(h.to_dict()).predict(F))

def test_analytic_summary_is_deterministic_and_probabilities_are_complete():
    cov=np.array([[100,15],[15,100]])
    a=score_summary([25,25],cov,home_spread=-3,total_line=50,home_decimal=1.91,away_decimal=1.91)
    b=score_summary([25,25],cov,home_spread=-3,total_line=50,home_decimal=1.91,away_decimal=1.91)
    assert a==b
    assert a["home_win_probability"]==pytest.approx(.5)
    assert abs(a["home_cover_probability"]+a["away_cover_probability"]+a["spread_push_probability"]-1)<1e-10
    assert abs(a["over_probability"]+a["under_probability"]+a["total_push_probability"]-1)<1e-10
    assert a["home_spread_expected_net_per_unit"]==pytest.approx(a["home_cover_probability"]*.91-a["away_cover_probability"])

def test_win_probabilities_respect_team_swap():
    cov=[[100,10],[10,90]]
    assert win_probability([30,20],cov)+win_probability([20,30],cov)==pytest.approx(1)

@pytest.mark.parametrize("cov", [[[1,2],[2,1]],[[1,2],[0,1]],[[1,0],[0,0]]])
def test_invalid_score_covariance_rejected(cov):
    with pytest.raises(ValueError): checked_covariance(cov)

def test_market_cannot_use_future_snapshot_and_home_spread_sign_is_preserved():
    f=pd.DataFrame({"game_id":[1,1],"observed_at":["2024-09-01T12:00Z","2024-09-01T14:00Z"],"home_spread":[-3,-7],"home_ml_decimal":[2.,1.5],"away_ml_decimal":[2.,2.5]})
    r=market_at_cutoff(f,1,"2024-09-01T13:00Z","2024-09-01T15:00Z")
    assert r["home_spread"]==-3
    assert r["market_home_win_probability_no_vig"]==.5

def test_duplicate_schedule_rejected():
    g=fixture_games()
    with pytest.raises(ValueError,match="Duplicate"):
        normalize_schedule(pd.concat([g,g]))

def test_missing_final_score_is_quarantined_without_inventing_an_outcome():
    g=fixture_games(); g.loc[0,"home_points"]=np.nan
    clean=normalize_schedule(g)
    assert clean.loc[0,"invalid_completed_outcome"]
    assert not clean.loc[0,"result_usable"]
    assert pd.isna(clean.loc[0,"home_points"])

def test_elo_is_independent_of_the_game_being_predicted():
    from cfb_forecast.baseline import elo_features
    g=fixture_games(); F=elo_features(g)
    changed=g.copy(); changed.loc[1,["home_points","away_points"]]=[99,0]
    alt=elo_features(changed)
    np.testing.assert_array_equal(F[1],alt[1])
    assert not np.allclose(F[2],alt[2])

def test_timezone_required():
    with pytest.raises(ValueError): timestamp("2024-09-01")

def test_play_filters_and_success_thresholds():
    row=dict(game_id=1,pos_team_id=1,def_pos_team_id=2,down=1,distance=10,statYardage=5,rush=True,**{"pass":False},sack=False,kneel_down=False,penalty_no_play=False,period=1,**{"start.pos_score_diff":0})
    p=pd.DataFrame([
        dict(row,game_play_number=1),
        dict(row,game_play_number=1),
        dict(row,game_play_number=2,down=2,statYardage=6),
        dict(row,game_play_number=3,kneel_down=True),
        dict(row,game_play_number=4,period=5),
        dict(row,game_play_number=5,**{"start.pos_score_diff":30}),
    ])
    clean=clean_plays(p)
    assert len(clean)==4
    np.testing.assert_array_equal(clean.success,[1,0,1,1])

def test_single_class_evaluation_and_score_errors():
    r=metrics([1,1],[.8,.9],margin=[10,20],actual_margin=[7,21])
    assert r["accuracy"]==1
    assert r["margin_mae"]==2
    assert np.isfinite(r["log_loss"])

def test_residual_covariance_calibration_rejects_tiny_sample():
    with pytest.raises(ValueError): residual_covariance(np.ones((2,2)),np.ones((2,2)))

from cfb_forecast.distribution import normal_crps
from cfb_forecast.market import normalize_market_history, market_history_summary


def test_normal_crps_prefers_truth_and_matches_the_closed_form():
    assert float(normal_crps(4.0,2.0,4.0))==pytest.approx(2.0*(2*0.3989422804014327-1/np.sqrt(np.pi)))
    assert float(normal_crps(14.0,2.0,4.0))>float(normal_crps(4.0,2.0,4.0))
    with pytest.raises(ValueError): normal_crps(0.0,0.0,1.0)


def test_market_history_supports_multiple_providers_and_consensus_cutoff():
    f=pd.DataFrame({
        "game_id":[1,1,1,1],
        "provider":["a","b","a","b"],
        "observed_at":["2026-10-01T12:00Z","2026-10-01T12:00Z","2026-10-02T12:00Z","2026-10-02T12:00Z"],
        "home_spread":[-3,-4,-5,-6],
        "total_line":[50,51,52,53],
    })
    n=normalize_market_history(f)
    r=market_at_cutoff(n,1,"2026-10-02T13:00Z","2026-10-03T16:00Z")
    assert r["market_provider_count"]==2
    assert r["home_spread"]==-5.5
    s=market_history_summary(n,1,"2026-10-02T13:00Z","2026-10-03T16:00Z")
    assert s["market_open_spread"]==-3.5
    assert s["market_current_spread"]==-5.5
    assert s["market_spread_move"]==-2.0

from cfb_forecast.chronology import select_due_games


def test_chronological_due_games_respect_each_games_own_cutoff():
    g=fixture_games()
    due=select_due_games(g,2022,"2022-09-02T17:00:00Z",24)
    assert due.game_id.astype(int).tolist()==[2]
    assert due.iloc[0].scheduled_cutoff==pd.Timestamp("2022-09-02T16:00:00Z")
    none=select_due_games(g,2022,"2022-09-02T17:00:00Z",24,already_forecasted=[2])
    assert none.empty


def test_chronological_due_games_never_include_started_games():
    g=fixture_games()
    due=select_due_games(g,2022,"2022-09-09T17:00:00Z",48)
    assert 2 not in due.game_id.astype(int).tolist()
    assert 3 in due.game_id.astype(int).tolist()


def test_prospective_collection_records_market_failure_and_still_runs(monkeypatch, tmp_path):
    from pathlib import Path
    import cfb_forecast.collector as collector
    dummy_ledger=pd.DataFrame({"game_id":[1]})
    dummy_added=pd.DataFrame(columns=["game_id"])
    def fail_market(*args, **kwargs):
        raise RuntimeError("source unavailable")
    def fake_chronology(**kwargs):
        assert str(kwargs["now"]) == "2026-10-02 19:26:22+00:00"
        return dummy_ledger, dummy_added, {"status":"no_due_games","games_added":0}
    monkeypatch.setattr(collector,"capture_sportsdataverse_snapshot",fail_market)
    monkeypatch.setattr(collector,"run_chronological_update",fake_chronology)
    _,_,receipt=collector.run_prospective_collection(
        data_root=tmp_path, model_root=tmp_path, ledger_root=tmp_path/"ledger",
        season=2026, now="2026-10-02T19:26:22Z", market_store=tmp_path/"market.csv"
    )
    assert receipt["market_capture"]["status"]=="failed"
    assert receipt["chronology"]["status"]=="no_due_games"
    assert Path(receipt["receipt_path"]).exists()

def test_state_update_matches_textbook_kalman_filter():
    rng=np.random.default_rng(1); cfg=StateConfig()
    center=np.array([28,5.5,7,4.5,.3]); scale=np.array([14,1.2,2,1.5,.08])
    s=DynamicRatings([1,2,3,4],center,scale,cfg)
    for j in range(5):
        A=rng.normal(size=(8,8)); s.cov[j]=A@A.T/8+np.eye(8)*.3; s.mean[j]=rng.normal(size=8)*.5
    s.clock=timestamp("2025-09-01T00:00Z"); before_mean,before_cov=s.mean.copy(),s.cov.copy()
    values=np.array([[35,6.1,8.0,5.0,.35],[17,4.9,6.0,3.9,.27]])
    s.update(1,2,False,values,"2025-09-01T00:00Z")
    H=np.zeros((2,8)); H[0,0]=1; H[0,5]=-1; H[1,1]=1; H[1,4]=-1
    R=np.array([[1,cfg.observation_correlation],[cfg.observation_correlation,1]])
    for j in range(5):
        home=np.array([1,-1])*cfg.home_margin/2/scale[0] if j==0 else np.zeros(2)
        P=before_cov[j]; K=P@H.T@np.linalg.inv(H@P@H.T+R)
        z=(values[:,j]-center[j])/scale[j]-home
        np.testing.assert_allclose(s.mean[j],before_mean[j]+K@(z-H@before_mean[j]),atol=1e-12)
        np.testing.assert_allclose(s.cov[j],(np.eye(8)-K@H)@P,atol=1e-12)
