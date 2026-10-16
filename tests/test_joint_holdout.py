"""Chronological replay of the joint player model: no leakage, grading, tuning grid."""
import numpy as np
import pandas as pd
import pytest

from cfb_forecast.evaluation.joint_holdout import (candidate_configs, grade, market_lines, replay_season,
                                                   season_slates, TUNED_SETTINGS)
from cfb_forecast.modeling.player_hierarchy import HierarchyConfig, holdout_noise_covariance


def inputs(seasons=(2024, 2025), teams=6, weeks=4):
    rng = np.random.default_rng(7)
    games, players = [], []
    gid = 0
    for season in seasons:
        for week in range(1, weeks + 1):
            date = pd.Timestamp(f'{season}-09-06T18:00Z') + pd.Timedelta(days=7 * (week - 1))
            order = rng.permutation(teams) + 1
            for home, away in zip(order[::2], order[1::2]):
                gid += 1
                hp, ap = float(rng.integers(10, 45)), float(rng.integers(7, 40))
                if hp == ap:
                    hp += 3
                games.append(dict(game_id=gid, season=season, week=week, season_type='regular', start_date=date,
                                  result_available_at=date + pd.Timedelta(hours=8), result_usable=True, fbs_game=True,
                                  completed=True, home_id=int(home), away_id=int(away), home_points=hp, away_points=ap,
                                  neutral_site=False, home_team=f'T{home}', away_team=f'T{away}'))
                for team, opponent in ((home, away), (away, home)):
                    for k, role in enumerate(('passing', 'rushing', 'receiving')):
                        players.append(dict(game_id=gid, team_id=int(team), opponent_id=int(opponent),
                                            player_id=f'{team}-{role}', role=role, opportunities=20. + k,
                                            yards=float(rng.normal(120, 30)), epa=float(rng.normal(.05 * team, .2)),
                                            available_at=date + pd.Timedelta(hours=8), season=season, start_date=date))
    return {'schedules': pd.DataFrame(games), 'players': pd.DataFrame(players), 'lineups': pd.DataFrame(),
            'facts': pd.DataFrame(), 'contexts': pd.DataFrame()}


def test_replay_predicts_each_game_only_from_earlier_results():
    data = inputs()
    frame = replay_season(data, 2025)
    graded = frame.loc[~frame.skipped]
    assert len(graded) == (data['schedules'].season == 2025).sum()
    assert (pd.to_datetime(graded.latest_result_used, utc=True) < graded.forecast_at).all()
    assert (graded.forecast_at < graded.start_date).all()
    # Each week is refitted, so later weeks learn from more games.
    assert graded.groupby('week').training_games.first().is_monotonic_increasing
    assert graded.groupby('week').training_games.first().iloc[-1] > graded.training_games.min()


def test_replay_of_a_week_ignores_that_weeks_and_later_outcomes():
    data = inputs()
    base = replay_season(data, 2025).set_index('game_id')
    changed = inputs()
    late = changed['schedules'].season.eq(2025) & changed['schedules'].week.ge(3)
    changed['schedules'].loc[late, 'home_points'] += 40
    after = replay_season(changed, 2025).set_index('game_id')
    early = base.week <= 3
    np.testing.assert_allclose(after.loc[early[early].index, 'expected_home_score'],
                               base.loc[early, 'expected_home_score'])
    assert not np.allclose(after.loc[~early, 'expected_home_score'], base.loc[~early, 'expected_home_score'])


def test_grade_and_noise_calibration_use_replayed_games():
    data = inputs(seasons=(2023, 2024, 2025))
    earlier = replay_season(data, 2024)
    later = replay_season(data, 2025)
    noise, source = holdout_noise_covariance(earlier.loc[~earlier.skipped], min_games=5)
    result = grade(later, noise)
    assert result['games'] == len(later)
    assert 0 <= result['winner_accuracy'] <= 1 and 0 <= result['brier'] <= 1
    assert source['games'] == len(earlier)


def test_slates_put_bowl_weeks_after_the_regular_season():
    data = inputs(seasons=(2025,))
    bowl = data['schedules'].iloc[[0]].assign(game_id=999, week=1, season_type='postseason',
                                              start_date=pd.Timestamp('2025-12-20T18:00Z'))
    schedules = pd.concat([data['schedules'], bowl], ignore_index=True)
    labels = [label for label, _, _ in season_slates(schedules, 2025)]
    assert labels[-1] == 'postseason-1' and labels[0] == '1'


def test_tuning_grid_moves_one_setting_at_a_time_and_includes_current():
    candidates = candidate_configs()
    assert candidates[0][0] == 'current' and candidates[0][1] == HierarchyConfig()
    single = [c for c in candidates[1:] if not c[0].startswith('all_variances')]
    assert {name.split('=')[0] for name, _, _ in single} == set(TUNED_SETTINGS) | {'half_life_days'}
    for name, config, _ in single:
        assert config == HierarchyConfig() or sum(
            getattr(config, s) != getattr(HierarchyConfig(), s) for s in TUNED_SETTINGS) == 1
    joint = {name: config for name, config, _ in candidates if name.startswith('all_variances')}
    assert joint['all_variances*0.5'].team_variance == pytest.approx(HierarchyConfig().team_variance / 2)
    assert joint['all_variances*0.5'].offseason_carry == HierarchyConfig().offseason_carry


def test_market_lines_flip_home_spread_into_home_margin(tmp_path):
    path = tmp_path / 'betting_2025.csv'
    pd.DataFrame([dict(game_id=1, home_team_spread=-7.5, game_spread_available=True),
                  dict(game_id=2, home_team_spread=3., game_spread_available=False)]).to_csv(path, index=False)
    lines = market_lines([path])
    assert lines.game_id.tolist() == [1]
    assert lines.market_home_margin.iloc[0] == pytest.approx(7.5)


def test_tuning_ranks_on_earlier_seasons_and_confirms_on_a_later_one():
    from cfb_forecast.evaluation.joint_holdout import tune_settings
    data = inputs(seasons=(2021, 2022, 2023, 2024), weeks=3)
    result = tune_settings(data, tuning_seasons=(2022, 2023), confirm_season=2024, refit_every=3, min_noise_games=5)
    assert {t['season'] for t in result['trials']} == {2022, 2023}
    losses = [r['log_loss'] for r in result['ranking']]
    assert losses == sorted(losses)
    assert result['best_on_tuning_seasons'] == result['ranking'][0]['candidate']
    assert result['adopted'] in ('current', result['best_on_tuning_seasons'])
    assert 'current' in result['confirmation']
