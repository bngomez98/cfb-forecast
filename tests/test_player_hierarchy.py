"""Joint evidence, leakage, covariance, identity and entry-point regression tests."""
import json

import numpy as np
import pandas as pd
import pytest

from cfb_forecast.ingest.data import normalize_schedule
from cfb_forecast.ingest.player_matchups import import_player_evidence, player_evidence
from cfb_forecast.modeling.player_hierarchy import PlayerHierarchy, availability_scenarios
from cfb_forecast.modeling.player_forecast import TEAM_METRICS, _team_context, _known_context
from cfb_forecast.storage.database import connect


def history():
    games, players = [], []
    for i in range(12):
        date = pd.Timestamp('2025-09-01T18:00Z') + pd.Timedelta(days=7*i)
        games.append(dict(game_id=i+1, season=2025, week=i+1, start_date=date,
                          result_available_at=date+pd.Timedelta(hours=4), result_usable=True,
                          home_id=1, away_id=2+(i%2), home_points=28.+i%4, away_points=17.+i%3,
                          neutral_site=False))
        for team, opponent in ((1, 2+i%2), (2+i%2, 1)):
            for k, role in enumerate(('passing','rushing','receiving')):
                players.append(dict(game_id=i+1, team_id=team, opponent_id=opponent,
                                    player_id=f'{team}-{role}', role=role, opportunities=20.+k,
                                    yards=100.+i*2, epa=(.3 if team==1 else -.1)+i*.01+k*.02,
                                    available_at=date+pd.Timedelta(hours=4), season=2025, start_date=date))
    return pd.DataFrame(games), pd.DataFrame(players)


def spec(team=1, opponent=2, season=2026):
    return dict(team=team, opponent=opponent, season=season, venue=0,
                lineup=[dict(player_id=f'{team}-{role}', role=role, opportunity_share=1.)
                        for role in ('passing','rushing','receiving')], context=np.zeros(2))


def fit(players=None, future=None):
    games, original = history()
    return PlayerHierarchy.fit(games, original if players is None else players,
                               np.zeros((len(games),2,2)), future or [spec(),spec(2,1)], '2026-09-01T00:00Z')


def test_player_observations_change_both_player_and_game_posterior():
    _, players = history()
    baseline = fit(players)
    changed = players.copy()
    changed.loc[changed.player_id=='1-passing','epa'] -= 1.
    altered = fit(changed)
    a, b = baseline.predict(spec(),spec(2,1)), altered.predict(spec(),spec(2,1))
    assert a['expected_home_score'] != pytest.approx(b['expected_home_score'])
    assert a['player_forecasts']['home'][0]['conditional_rate_mean'] != pytest.approx(b['player_forecasts']['home'][0]['conditional_rate_mean'])
    assert a['home_win_probability'] != pytest.approx(b['home_win_probability'])


def test_score_observations_update_player_effect_in_same_posterior():
    games, players = history()
    baseline = fit(players)
    games.loc[games.index%2==0, 'home_points'] += 18
    changed = PlayerHierarchy.fit(games, players,np.zeros((12,2,2)),[spec(),spec(2,1)],'2026-09-01T00:00Z')
    key = baseline._key('player','1-passing','passing',2025)
    assert baseline.posterior_mean[baseline.index[key]] != pytest.approx(changed.posterior_mean[changed.index[key]])


def test_future_evidence_and_future_results_are_rejected():
    games, players = history()
    players.loc[0,'available_at'] = pd.Timestamp('2026-09-01T00:00Z')
    with pytest.raises(ValueError,match='cutoff'):
        fit(players)
    with pytest.raises(ValueError,match='eligible'):
        PlayerHierarchy.fit(games,players,np.zeros((12,2,2)),[spec()], '2025-09-01T00:00Z')


def test_unknown_opponent_effect_keeps_prior_uncertainty():
    unseen = spec(1,99)
    model = fit(future=[spec(),spec(2,1),unseen,spec(99,1)])
    row = model.predict(unseen,spec(99,1))
    assert np.linalg.eigvalsh(row['score_covariance']).min()>0
    assert row['score_covariance'][0][0] > model.noise_covariance[0,0]
    key = model._key('matchup','1-passing',99,'passing')
    assert model.posterior_mean[model.index[key]] == pytest.approx(0.)
    X=model._matrix([{key:1.}]).toarray()
    assert (X@model.factor.solve(X.T)).item() == pytest.approx(model.config.matchup_variance)


def test_availability_redistributes_within_role_and_integrates_probabilities():
    home=spec()
    home['lineup']=[dict(player_id='1-passing',role='passing',opportunity_share=.8,availability_probability=.4),
                    dict(player_id='backup',role='passing',opportunity_share=.2)]
    components=availability_scenarios(home['lineup'])
    assert sum(p for p,_ in components)==pytest.approx(1.)
    assert components[0][1][0]['player_id']=='backup'
    assert components[0][1][0]['opportunity_share']==pytest.approx(1.)
    model=fit(future=[home,spec(2,1)])
    row=model.predict(home,spec(2,1),{'home_spread':0.,'total_line':45.})
    assert row['availability_components']==2
    assert row['home_win_probability']==pytest.approx(row['home_cover_probability'])
    assert row['over_probability']+row['under_probability']+row['total_push_probability']==pytest.approx(1.)
    scenarios=[]
    for probability,lineup in components:
        fixed={**home,'lineup':lineup}
        scenarios.append((probability,model.predict(fixed,spec(2,1))))
    assert row['home_win_probability']==pytest.approx(sum(p*r['home_win_probability'] for p,r in scenarios))


def test_new_season_query_does_not_feed_prior_evidence_backward():
    base=fit()
    extra=fit(future=[spec(),spec(2,1),spec(season=2027),spec(2,1,season=2027)])
    assert base.predict(spec(),spec(2,1))['home_win_probability']==pytest.approx(extra.predict(spec(),spec(2,1))['home_win_probability'],abs=1e-10)


def test_artifact_roundtrip_and_neutral_team_swap(tmp_path):
    home,away=spec(),spec(2,1)
    model=fit()
    expected=model.predict(home,away)
    model.save(tmp_path)
    restored=PlayerHierarchy.load(tmp_path)
    assert restored.predict(home,away)==expected
    swapped=restored.predict(away,home)
    assert expected['home_win_probability']+swapped['home_win_probability']==pytest.approx(1.)
    assert expected['expected_home_score']==pytest.approx(swapped['expected_away_score'])


def payload():
    return dict(players=[dict(player_id='qb',display_name='Quarterback',source_ids=[dict(provider='test',source_player_id='1')])],
                observations=[dict(game_id=1,team_id=1,opponent_id=2,player_id='qb',role='passing',opportunities=20.,yards=200.,epa=.2,
                                   available_at='2025-09-02T00:00Z',provider='test',source_url='https://example.test/game/1',source_sha256='a'*64)],
                lineups=[dict(game_id=99,team_id=1,observed_at='2026-09-01T00:00Z',provider='test',source_url='https://example.test/lineup',source_sha256='b'*64,
                              players=[dict(player_id='qb',role='passing',opportunity_share=1.,availability_probability=.4)])])


def test_database_migration_is_nondestructive_and_import_atomic(tmp_path):
    db=tmp_path/'db.sqlite'
    conn=connect(db)
    conn.execute("INSERT INTO teams VALUES(1,'Home')");conn.commit();conn.close()
    data=payload();import_player_evidence(db,data)
    broken=payload();broken['players'][0]['source_ids'][0]['source_player_id']='new'
    broken['lineups'][0]['players'][0]['opportunity_share']=.5
    with pytest.raises(ValueError,match='sum to one'):
        import_player_evidence(db,broken)
    conn=connect(db)
    assert conn.execute('SELECT name FROM teams').fetchone()[0]=='Home'
    assert conn.execute('SELECT COUNT(*) FROM player_source_ids').fetchone()[0]==1
    assert conn.execute('SELECT COUNT(*) FROM player_observations').fetchone()[0]==1
    conn.close()


def test_source_alias_collision_is_rejected(tmp_path):
    db=tmp_path/'db.sqlite';data=payload();import_player_evidence(db,data)
    data['players'][0]['player_id']='someone-else'
    with pytest.raises(ValueError,match='different player'):
        import_player_evidence(db,data)


def test_observations_and_lineups_exclude_equality_and_future_revisions(tmp_path):
    db=tmp_path/'db.sqlite';data=payload();import_player_evidence(db,data)
    games,_=history()
    first,_,_=player_evidence(tmp_path,db,games,'2025-09-02T00:00Z')
    assert first.empty
    later,lineups,_=player_evidence(tmp_path,db,games,'2026-09-01T00:00Z')
    assert len(later)==1
    assert lineups.empty
    data['observations'][0]['available_at']='2026-10-01T00:00Z'
    data['observations'][0]['epa']=1000
    import_player_evidence(db,data)
    same,_,_=player_evidence(tmp_path,db,games,'2026-09-01T00:00Z')
    assert same.epa.iloc[0]==pytest.approx(.2)


def test_team_context_excludes_result_not_yet_available_at_kickoff():
    games,_=history()
    games.loc[0,'result_available_at']=games.loc[1,'start_date']
    context=_team_context(games,games,pd.DataFrame(),pd.Timestamp('2026-09-01T00:00Z'))
    assert np.isnan(context[1,0,0])


def test_context_uses_only_timestamped_pregame_reports():
    games,_=history()
    reports=pd.DataFrame([dict(game_id=1,observed_at='2025-09-01T17:00Z',provider='test',features_json=json.dumps({'weather_temperature_f':70.})),
                          dict(game_id=1,observed_at='2025-09-01T18:00Z',provider='test',features_json=json.dumps({'weather_temperature_f':100.}))])
    result=_known_context(games,games,reports,pd.Timestamp('2026-09-01T00:00Z'))
    assert result[0,0,5]==70.


def test_production_entrypoint_uses_database_and_freezes_joint_artifacts(tmp_path,monkeypatch):
    from test_model_artifact_validation import model_value
    from cfb_forecast.modeling.forecast import forecast, advance_forecast_snapshot
    from cfb_forecast.evaluation.holdout import freeze_holdout, verify_holdout
    from cfb_forecast.pipeline.publisher import board_row
    from cfb_forecast.ingest.data import sha256
    data,model,out=[tmp_path/name for name in ('data','model','out')]
    for path in (data,model):path.mkdir()
    db=data/'facts.sqlite'
    monkeypatch.setenv('CFB_DATABASE',str(db))
    games,players=history()
    games['home_team']='Home';games['away_team']='Away';games['completed']=True;games['fbs_game']=True
    future=dict(game_id=99,season=2026,week=1,start_date=pd.Timestamp('2026-09-05T18:00Z'),completed=False,
                home_id=1,away_id=2,home_points=None,away_points=None,home_team='Home',away_team='Away',neutral_site=False,fbs_game=True)
    scheduled=pd.concat([games,pd.DataFrame([future])],ignore_index=True)
    for year,group in scheduled.groupby('season'):
        group.to_csv(data/f'schedule_{year}.csv',index=False)
    (data/'source_manifest.json').write_text(json.dumps({'files':{'schedule_2026.csv':{'downloaded_at':'2026-08-31T00:00Z'}}}))
    metadata={**model_value(),'training_through_season':2025}
    (model/'model.json').write_text(json.dumps(metadata))
    identities=[dict(player_id=x,display_name=x) for x in players.player_id.unique()]
    observations=[]
    for row in players.to_dict('records'):
        observations.append({**row,'available_at':row['available_at'].isoformat(),'provider':'fixture',
                             'source_url':'https://example.test/fixture','source_sha256':'a'*64})
    import_player_evidence(db,{'players':identities,'observations':observations})
    result=forecast(data,model,out,'2026-09-01T00:00Z','2026-09-06T00:00Z',2026)
    assert result.model_variant.tolist()==['joint_player']
    assert result.iloc[0].player_forecasts['home']
    row=board_row(pd.read_csv(out/'forecasts.csv').iloc[0].to_dict())
    assert row['player_forecasts']['home']
    frozen,_=freeze_holdout(out,data,model,tmp_path/'holdouts',freeze_at='2026-09-01T01:00Z')
    verify_holdout(frozen)
    assert (frozen/'forecast/joint_player_precision.npz').exists()
    again=advance_forecast_snapshot(out,data,model,tmp_path/'advanced','2026-09-02T00:00Z','2026-09-06T00:00Z',2026)
    assert again.iloc[0].home_win_probability==pytest.approx(result.iloc[0].home_win_probability)
    (out/'joint_player_precision.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact hash'):
        freeze_holdout(out,data,model,tmp_path/'tampered',freeze_at='2026-09-01T01:00Z')


def test_raw_player_ids_survive_database_roundtrip_and_follow_source_identity(tmp_path):
    from cfb_forecast.storage.database import record_history_facts
    db=tmp_path/'facts.sqlite';games,_=history()
    conn=connect(db)
    record_history_facts(conn,pd.DataFrame(),pd.DataFrame([dict(game_id=1,team_id=1,player_name='Name',position_group='passing',
                                                             season=2025,plays=20,yards=200,epa=.2,source_player_id='123')]))
    conn.close()
    rows,_,_=player_evidence(tmp_path,db,games,'2026-09-01T00:00Z')
    assert rows.player_id.tolist()==['source:sportsdataverse:123']


def test_installed_fallback_schema_matches_checkout_player_tables():
    import sqlite3
    from pathlib import Path
    from cfb_forecast.storage.database import SCHEMA
    a,b=sqlite3.connect(':memory:'),sqlite3.connect(':memory:')
    a.executescript(SCHEMA)
    b.executescript((Path(__file__).parents[1]/'sql/schema.sql').read_text())
    for table in ('player_game_facts','player_observations','player_identities','player_source_ids','player_lineups','game_context_snapshots'):
        assert a.execute(f'PRAGMA table_info({table})').fetchall()==b.execute(f'PRAGMA table_info({table})').fetchall()
    a.close();b.close()


def test_lineup_uses_latest_record_before_cutoff_regardless_of_row_order():
    from cfb_forecast.modeling.player_forecast import _lineup
    game = pd.DataFrame([dict(game_id=99, start_date=pd.Timestamp('2026-09-05T18:00Z'))]).iloc[0]
    def entry(observed, player):
        return dict(game_id=99, team_id=1, observed_at=observed,
                    players_json=json.dumps([dict(player_id=player, role='passing', opportunity_share=1.)]))
    # Rows deliberately out of time order; the last row is after the cutoff.
    lineups = pd.DataFrame([entry('2026-09-04T12:00Z', 'newest'), entry('2026-09-02T12:00Z', 'oldest'),
                            entry('2026-09-03T12:00Z', 'middle'), entry('2026-09-05T12:00Z', 'after_cutoff')])
    lineup, status, observed = _lineup(pd.DataFrame(), lineups, game, 1, pd.Timestamp('2026-09-05T00:00Z'))
    assert status == 'timestamped_lineup'
    assert lineup[0]['player_id'] == 'newest'
    assert pd.Timestamp(observed) == pd.Timestamp('2026-09-04T12:00Z')
    lineup, status, _ = _lineup(pd.DataFrame(), lineups.iloc[[3]], game, 1, pd.Timestamp('2026-09-05T00:00Z'))
    assert status == 'missing_player_history' and lineup == []


def test_forecast_reads_the_database_it_is_given(tmp_path, monkeypatch):
    from test_model_artifact_validation import model_value
    from cfb_forecast.modeling.forecast import forecast
    data, model, out = [tmp_path/name for name in ('data', 'model', 'out')]
    for path in (data, model):
        path.mkdir()
    db = tmp_path/'chosen.sqlite'
    # The default database stays empty; only the explicit argument has evidence.
    monkeypatch.setenv('CFB_DATABASE', str(tmp_path/'default.sqlite'))
    games, players = history()
    games['home_team'] = 'Home'; games['away_team'] = 'Away'; games['completed'] = True; games['fbs_game'] = True
    future = dict(game_id=99, season=2026, week=1, start_date=pd.Timestamp('2026-09-05T18:00Z'), completed=False,
                  home_id=1, away_id=2, home_points=None, away_points=None, home_team='Home', away_team='Away',
                  neutral_site=False, fbs_game=True)
    for year, group in pd.concat([games, pd.DataFrame([future])], ignore_index=True).groupby('season'):
        group.to_csv(data/f'schedule_{year}.csv', index=False)
    (model/'model.json').write_text(json.dumps({**model_value(), 'training_through_season': 2025}))
    observations = [{**row, 'available_at': row['available_at'].isoformat(), 'provider': 'fixture',
                     'source_url': 'https://example.test/fixture', 'source_sha256': 'a'*64}
                    for row in players.to_dict('records')]
    import_player_evidence(db, {'players': [dict(player_id=x, display_name=x) for x in players.player_id.unique()],
                                'observations': observations})
    with pytest.raises(ValueError, match='requires player observations'):
        forecast(data, model, out, '2026-09-01T00:00Z', '2026-09-06T00:00Z', 2026)
    result = forecast(data, model, out, '2026-09-01T00:00Z', '2026-09-06T00:00Z', 2026, database=db)
    assert result.model_variant.tolist() == ['joint_player']
    assert result.training_player_observations.iloc[0] > 0


def _context_games(rows):
    games = []
    for i, (home, away, hp, ap, day) in enumerate(rows):
        date = pd.Timestamp('2025-09-01T18:00Z') + pd.Timedelta(days=day)
        games.append(dict(game_id=i + 1, season=2025, start_date=date, result_available_at=date + pd.Timedelta(hours=4),
                          home_id=home, away_id=away, home_points=float(hp), away_points=float(ap)))
    return pd.DataFrame(games)


def test_team_context_weights_recent_games_more_than_old_ones():
    # Team 1 scored 50 long ago and 10 last week, against opponents with no
    # other games, so no opponent adjustment applies.
    games = _context_games([(1, 2, 50, 20, 0), (1, 3, 10, 20, 140)])
    target = _context_games([(1, 4, 0, 0, 150)])
    at = pd.Timestamp('2026-09-01T00:00Z')
    plain = _team_context(games, target, pd.DataFrame(), at, mode='season_mean')
    weighted = _team_context(games, target, pd.DataFrame(), at)
    assert plain[0, 0, 0] == pytest.approx(30.)
    assert 10. < weighted[0, 0, 0] < 20.


def test_team_context_credits_points_against_strong_defenses():
    # Teams 1 and 3 both scored 30 once. Team 2's defense allowed 10 to its
    # other opponent, team 4's allowed 50, and the league averages 30, so 30
    # against team 2 is the stronger showing.
    games = _context_games([(5, 2, 10, 30, 0), (6, 4, 50, 30, 0), (1, 2, 30, 30, 7), (3, 4, 30, 30, 7)])
    target = _context_games([(1, 3, 0, 0, 14)])
    context = _team_context(games, target, pd.DataFrame(), pd.Timestamp('2026-09-01T00:00Z'))
    home_points_for, away_points_for = context[0, 0, 0], context[0, 0, len(TEAM_METRICS)]
    assert home_points_for > 30. > away_points_for
    plain = _team_context(games, target, pd.DataFrame(), pd.Timestamp('2026-09-01T00:00Z'), mode='season_mean')
    assert plain[0, 0, 0] == pytest.approx(plain[0, 0, len(TEAM_METRICS)])


def test_team_context_never_uses_results_after_the_cutoff_for_opponent_strength():
    games = _context_games([(1, 2, 30, 20, 0), (5, 2, 70, 20, 20)])
    target = _context_games([(1, 3, 0, 0, 10)])
    before = _team_context(games.iloc[:1], target, pd.DataFrame(), pd.Timestamp('2026-09-01T00:00Z'))
    after = _team_context(games, target, pd.DataFrame(), pd.Timestamp('2026-09-01T00:00Z'))
    np.testing.assert_array_equal(np.isnan(before), np.isnan(after))
    np.testing.assert_allclose(before[np.isfinite(before)], after[np.isfinite(after)])
