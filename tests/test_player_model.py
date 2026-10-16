"""Player facts must change the fitted score model through causal feature states."""
import numpy as np
import pandas as pd
import pytest

from cfb_forecast.ingest.features import pregame_history, player_game_rates
from cfb_forecast.modeling.forecast_model import ForecastModel, HISTORY_COLUMNS, KNOWN_COLUMNS, AdjustmentConfig
from cfb_forecast.storage.database import connect, record_history_facts


def games():
    return pd.DataFrame([
        dict(game_id=1, season=2025, week=1, home_id=10, away_id=20, home_points=28., away_points=21., start_date=pd.Timestamp('2025-09-01T18:00Z'), result_available_at=pd.Timestamp('2025-09-02T02:00Z')),
        dict(game_id=2, season=2025, week=2, home_id=10, away_id=30, home_points=35., away_points=14., start_date=pd.Timestamp('2025-09-08T18:00Z'), result_available_at=pd.Timestamp('2025-09-09T02:00Z')),
        dict(game_id=3, season=2026, week=1, home_id=10, away_id=40, home_points=24., away_points=20., start_date=pd.Timestamp('2026-09-01T18:00Z'), result_available_at=pd.Timestamp('2026-09-02T02:00Z')),
    ])


def test_role_rates_use_recorded_opportunities_and_weight_epa():
    players = pd.DataFrame([
        dict(game_id=1, team_id=10, position_group='passing', plays=30., yards=240., epa=.2),
        dict(game_id=1, team_id=10, position_group='passing', plays=10., yards=60., epa=.6),
    ])
    rates = player_game_rates(players).iloc[0]
    assert rates.player_passing_yards_per_play == pytest.approx(7.5)
    assert rates.player_passing_epa == pytest.approx(.3)
    assert rates.player_passing_usage_concentration == pytest.approx(.625)
    assert pd.isna(rates.player_rushing_yards_per_play)


def test_player_facts_enter_pregame_model_at_result_availability(tmp_path):
    database = tmp_path / 'facts.sqlite'
    conn = connect(database)
    players = pd.DataFrame([
        dict(game_id=1, team_id=10, player_name='QB', position_group='passing', season=2025, plays=30., yards=240., epa=.2),
        dict(game_id=2, team_id=10, player_name='QB', position_group='passing', season=2025, plays=20., yards=120., epa=.1),
    ])
    record_history_facts(conn, pd.DataFrame(), players)
    conn.close()
    completed = games()
    query = completed.iloc[[2]].copy()
    values = pregame_history(query, tmp_path, completed.iloc[:2], database)
    column = HISTORY_COLUMNS.index('home_player_passing_yards_per_play')
    assert values[0, column] == pytest.approx(7.)
    early = completed.iloc[[1]].assign(cutoff_utc=pd.Timestamp('2025-09-02T02:00Z'))
    values = pregame_history(early, tmp_path, completed.iloc[:2], database)
    assert values is None  # A final at exactly cutoff is still excluded.
    after = early.assign(cutoff_utc=pd.Timestamp('2025-09-02T02:00:01Z'))
    values = pregame_history(after, tmp_path, completed.iloc[:2], database)
    assert values[0, column] == pytest.approx(8.)


def test_player_efficiency_changes_scores_and_every_probability_coherently():
    rng = np.random.default_rng(22)
    n = 200
    history = np.full((n, len(HISTORY_COLUMNS)), np.nan)
    home_column = HISTORY_COLUMNS.index('home_player_passing_yards_per_play')
    away_column = HISTORY_COLUMNS.index('away_player_passing_yards_per_play')
    history[:, home_column] = rng.uniform(4., 10., n)
    history[:, away_column] = rng.uniform(4., 10., n)
    location = np.full((n, 2), 24.)
    actual = location + 2 * (history[:, [home_column, away_column]] - 7.)
    known = np.full((n, len(KNOWN_COLUMNS)), np.nan)
    model = ForecastModel.fit(location, actual, known, (actual[:, 0] > actual[:, 1]).astype(int), history=history)
    query = history[:2].copy()
    query[:, away_column] = 7.
    query[:, home_column] = [5., 9.]
    rows = model.apply(location[:2], known[:2], [[64., 8.], [8., 64.]], [{}, {}], [1, 2], history=query)
    assert rows[1]['expected_home_score'] > rows[0]['expected_home_score'] + 7.
    assert rows[1]['home_win_probability'] > rows[0]['home_win_probability']
    for row in rows:
        assert row['historical_player_feature_count'] == 2
        assert row['home_poisson_rate'] == row['expected_home_score']
        total = row['home_base_score'] + sum(row[f'home_{group}_score_adjustment'] for group in ('team', 'player', 'market', 'rest', 'weather', 'availability')) + row['home_score_bound_adjustment']
        assert row['expected_home_score'] == pytest.approx(total)
    restored = ForecastModel.from_dict(model.to_dict())
    assert restored.apply(location[:2], known[:2], [[64., 8.], [8., 64.]], [{}, {}], [1, 2], history=query) == rows


def test_no_missing_season_is_used_as_a_previous_season(tmp_path):
    completed = games().iloc[:1].copy()
    query = games().iloc[[2]].copy()
    query['season'] = 2027
    query['start_date'] = pd.Timestamp('2027-09-01T18:00Z')
    assert pregame_history(query, tmp_path, completed) is None


@pytest.mark.parametrize('settings', [{'ridge': 0}, {'min_games': 1}, {'min_feature_coverage': 0}])
def test_invalid_model_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        AdjustmentConfig(**settings)


def test_production_forecast_fits_player_features_and_exports_their_score_effects(tmp_path):
    """Exercise the real replay, out-of-fold adjustment fit and output writer."""
    import json
    from cfb_forecast.config import StateConfig
    from cfb_forecast.ingest.features import feature_columns
    from cfb_forecast.modeling.forecast import forecast
    from cfb_forecast.modeling.model import write_model
    data = tmp_path / 'data'
    data.mkdir()
    schedule, advanced, players = [], [], []
    for year in (2021, 2022, 2023):
        for number in range(80):
            gid = year * 1000 + number
            schedule.append(dict(game_id=gid, season=year, week=number // 5 + 1,
                start_date=(pd.Timestamp(f'{year}-08-01', tz='UTC') + pd.Timedelta(days=number)).isoformat(),
                completed=True, neutral_site=False, home_id=1, away_id=2,
                home_team='A', away_team='B', home_points=35 + number % 7,
                away_points=14 + number % 5, fbs_game=True))
            for team in (1, 2):
                advanced.append(dict(game_id=gid, pos_team_id=team, yards_per_play=5 + team,
                    yards_per_pass=7 + team, yards_per_rush=4 + team, first_downs_created_rate=.2 + team / 20))
                players.append(dict(game_id=gid, team_id=team, player_name=f'Player {team}',
                    position_group='passing', season=year, plays=20., yards=100. + number + 20 * team, epa=.01 * number))
    schedule.append(dict(game_id=999, season=2024, week=1, start_date='2024-09-01T18:00Z',
        completed=False, neutral_site=False, home_id=1, away_id=2, home_team='A', away_team='B',
        home_points=None, away_points=None, fbs_game=True))
    pd.DataFrame(schedule).to_csv(data / 'schedule_history.csv', index=False)
    pd.DataFrame(advanced).to_parquet(data / 'adv_team_history.parquet', index=False)
    database = tmp_path / 'facts.sqlite'
    conn = connect(database)
    record_history_facts(conn, pd.DataFrame(), pd.DataFrame(players))
    conn.close()
    cols = feature_columns('score')
    model = dict(schema_version=1, metric_names=["points", "yards_per_play", "yards_per_pass", "yards_per_rush", "first_downs_created_rate"], variant='score', warmup_season=2021,
        train_start_season=2022, training_through_season=2022,
        center=[24, 6, 8, 5, .3], scale=[10, 1, 1, 1, .1],
        state_config=StateConfig().to_dict(), score_covariance=[[144, 0], [0, 144]],
        head=dict(variant='score', alpha=10, cols=cols, mean=[0] * len(cols),
                  scale=[1] * len(cols), coef=[0] * len(cols), intercept=24))
    base = tmp_path / 'model'
    write_model(base, model)
    output = tmp_path / 'forecast'
    result = forecast(data, base, output, '2024-08-25T15:00Z', '2024-09-02T00:00Z',
                      2024, database=database, adjustment_config={'ridge': .5})
    row = result.iloc[0]
    assert row.historical_player_feature_count == 6
    assert row.pregame_home_player_passing_yards_per_play > 0
    assert np.isfinite(row.home_player_score_adjustment)
    fitted = json.loads((output / 'forecast_model.json').read_text())
    assert fitted['ready']
    assert 'home_player_passing_yards_per_play' in fitted['input_columns']
    assert fitted['adjustment_config']['ridge'] == .5
    assert row.home_win_probability == row.gaussian_home_win_probability
    assert row.home_poisson_rate == row.expected_home_score


def test_insufficient_player_training_keeps_the_base_score_location():
    n = 40
    known = np.full((n, len(KNOWN_COLUMNS)), np.nan)
    history = np.full((n, len(HISTORY_COLUMNS)), np.nan)
    history[:, HISTORY_COLUMNS.index('home_player_passing_yards_per_play')] = np.linspace(4., 10., n)
    actual = np.column_stack([np.linspace(21., 35., n), np.full(n, 24.)])
    model = ForecastModel.fit(np.full((n, 2), 24.), actual, known,
                             (actual[:, 0] > actual[:, 1]).astype(int), history=history,
                             config={'min_games': 60})
    assert not model.ready
    row = model.apply(np.array([[40., 20.]]), known[:1], [[64., 8.], [8., 64.]], [{}], [1], history=history[:1])[0]
    assert row['expected_home_score'] == 40.
    assert row['home_player_score_adjustment'] == 0.


def test_full_forecast_uses_player_history_and_persists_fitted_variables(tmp_path):
    import json
    from cfb_forecast.config import StateConfig
    from cfb_forecast.ingest.features import feature_columns
    from cfb_forecast.modeling.forecast import forecast
    from cfb_forecast.modeling.model import write_model

    data, base, out = [tmp_path / name for name in ('data', 'model', 'forecast')]
    data.mkdir()
    player_rows = []
    for year in (2021, 2022, 2023):
        schedules, advanced = [], []
        for number in range(60):
            game_id = year * 1000 + number
            kickoff = pd.Timestamp(f'{year}-08-01T18:00Z') + pd.Timedelta(days=number)
            schedules.append(dict(game_id=game_id, season=year, week=number // 5 + 1,
                start_date=kickoff.isoformat(), completed=True, neutral_site=False,
                home_id=1, away_id=2, home_team='Home', away_team='Away',
                home_points=28 + number % 8, away_points=21 + number % 6, fbs_game=True))
            for team in (1, 2):
                advanced.append(dict(game_id=game_id, pos_team_id=team, yards_per_play=5. + team,
                    yards_per_pass=7. + team, yards_per_rush=4. + team, first_downs_created_rate=.2 + team / 20))
                player_rows.append(dict(game_id=game_id, team_id=team, season=year,
                    player_name=f'QB {team}', position_group='passing', plays=30.,
                    yards=180. + (number % 8) * 10 + team * 5, epa=.1 * team))
        pd.DataFrame(schedules).to_csv(data / f'schedule_{year}.csv', index=False)
        pd.DataFrame(advanced).to_parquet(data / f'adv_team_{year}.parquet', index=False)
    pd.DataFrame([dict(game_id=999, season=2024, week=1, start_date='2024-09-07T18:00Z',
        completed=False, neutral_site=False, home_id=1, away_id=2, home_team='Home',
        away_team='Away', home_points=None, away_points=None, fbs_game=True)]).to_csv(data / 'schedule_2024.csv', index=False)
    columns = feature_columns('score')
    write_model(base, dict(schema_version=1, metric_names=["points", "yards_per_play", "yards_per_pass", "yards_per_rush", "first_downs_created_rate"], variant='score', warmup_season=2021,
        train_start_season=2022, training_through_season=2023,
        center=[24, 6, 8, 5, .3], scale=[10, 1, 1, 1, .1], state_config=StateConfig().to_dict(),
        score_covariance=[[144, 0], [0, 144]], head=dict(variant='score', alpha=10,
        cols=columns, mean=[0] * len(columns), scale=[1] * len(columns),
        coef=[0] * len(columns), intercept=24)))
    database = tmp_path / 'facts.sqlite'
    conn = connect(database)
    record_history_facts(conn, pd.DataFrame(), pd.DataFrame(player_rows))
    conn.close()
    result = forecast(data, base, out, '2024-09-01T15:00Z', '2024-09-08T00:00Z', 2024,
                      database=database, adjustment_config={'ridge': 2.})
    row = result.iloc[0]
    assert row.historical_player_feature_count == 6
    assert np.isfinite(row.pregame_home_player_passing_yards_per_play)
    assert np.isfinite(row.home_player_score_adjustment)
    saved = json.loads((out / 'forecast_model.json').read_text())
    assert saved['ready']
    assert saved['adjustment_config']['ridge'] == 2.
    assert 'home_player_passing_yards_per_play' in saved['input_columns']
    assert row.home_poisson_rate == row.expected_home_score
    assert row.home_win_probability == row.gaussian_home_win_probability
    assert row.expected_home_score == pytest.approx(row.home_base_score + row.home_team_score_adjustment + row.home_player_score_adjustment + row.home_known_future_score_adjustment + row.home_score_bound_adjustment)


def test_insufficient_history_keeps_base_location_instead_of_crashing():
    history = np.full((10, len(HISTORY_COLUMNS)), 1.)
    known = np.full((10, len(KNOWN_COLUMNS)), np.nan)
    location = np.full((10, 2), 24.)
    model = ForecastModel.fit(location, location + 6., known, np.arange(10) % 2, history=history)
    rows = model.apply(location[:1], known[:1], [[64., 0.], [0., 64.]], [{}], [1], history=history[:1])
    assert not model.ready
    assert rows[0]['expected_home_score'] == 24.
    assert rows[0]['home_player_score_adjustment'] == 0.
