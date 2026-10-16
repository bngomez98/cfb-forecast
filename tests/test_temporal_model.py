"""Football adaptations of lag, rolling-window and expanding forecasting features."""
import numpy as np
import pandas as pd
import pytest

from cfb_forecast.ingest.features import pregame_history
from cfb_forecast.modeling.forecast_model import HISTORY_COLUMNS, KNOWN_COLUMNS, ForecastModel


def history():
    rows = []
    for index, points in enumerate((10., 20., 30., 40., 50.)):
        start = pd.Timestamp('2025-09-01T18:00Z') + pd.Timedelta(days=7 * index)
        rows.append(dict(game_id=index + 1, season=2025, week=index + 1,
            home_id=10, away_id=20, home_points=points, away_points=7., start_date=start,
            result_available_at=start + pd.Timedelta(hours=8)))
    return pd.DataFrame(rows)


def test_lags_windows_and_expanding_use_only_the_teams_available_finals(tmp_path):
    past = history()
    query = past.iloc[[-1]].copy()
    query['game_id'] = 99
    query['start_date'] = pd.Timestamp('2025-10-10T18:00Z')
    query['cutoff_utc'] = pd.Timestamp('2025-10-09T18:00Z')
    values = pregame_history(query, tmp_path, past)[0]
    def value(name):
        return values[HISTORY_COLUMNS.index(name)]
    assert value('home_points_for_lag_1') == 50.
    assert value('home_points_for_lag_2') == 40.
    assert value('home_points_for_rolling_3_mean') == 40.
    assert value('home_points_for_rolling_5_mean') == 30.
    assert value('home_points_for_rolling_3_std') == pytest.approx(np.std([30., 40., 50.]))
    assert value('home_points_for_expanding_mean') == 30.
    assert value('away_points_for_lag_1') == 7.
    # A final received at the cutoff must not update lag 1.
    query['cutoff_utc'] = past.result_available_at.iloc[-1]
    values = pregame_history(query, tmp_path, past)[0]
    assert values[HISTORY_COLUMNS.index('home_points_for_lag_1')] == 40.
    assert np.isnan(values[HISTORY_COLUMNS.index('home_points_for_rolling_5_mean')])


def test_future_scores_cannot_change_earlier_temporal_features(tmp_path):
    past = history()
    query = past.iloc[[2]].assign(cutoff_utc=pd.Timestamp('2025-09-10T18:00Z'))
    before = pregame_history(query, tmp_path, past)
    changed = past.copy()
    changed.loc[2:, 'home_points'] = 500.
    after = pregame_history(query, tmp_path, changed)
    np.testing.assert_allclose(before, after, equal_nan=True)


def test_temporal_feature_changes_the_fitted_score_and_supports_ablation():
    rng = np.random.default_rng(51)
    n = 100
    values = np.full((n, len(HISTORY_COLUMNS)), np.nan)
    column = HISTORY_COLUMNS.index('home_points_for_rolling_3_mean')
    values[:, column] = rng.normal(30., 5., n)
    base = np.full((n, 2), 24.)
    actual = base.copy()
    actual[:, 0] += .5 * (values[:, column] - 30.)
    known = np.full((n, len(KNOWN_COLUMNS)), np.nan)
    wins = (actual[:, 0] > actual[:, 1]).astype(int)
    model = ForecastModel.fit(base, actual, known, wins, history=values)
    disabled = ForecastModel.fit(base, actual, known, wins, history=values, config={'use_temporal_features': False})
    assert HISTORY_COLUMNS[column] in model.input_columns
    assert not disabled.input_columns
    target = values[:2].copy()
    target[:, column] = [20., 40.]
    rows = model.apply(base[:2], known[:2], [[64., 0.], [0., 64.]], [{}, {}], [1, 2], history=target)
    assert rows[1]['expected_home_score'] > rows[0]['expected_home_score']
    assert rows[1]['home_win_probability'] > rows[0]['home_win_probability']
    assert rows[0]['historical_temporal_feature_count'] == 1


def test_football_only_series_does_not_change_with_a_market_quote():
    rng = np.random.default_rng(8)
    n = 150
    base = np.full((n, 2), 24.)
    known = np.full((n, len(KNOWN_COLUMNS)), np.nan)
    known[:, 0] = rng.normal(0., 6., n)
    actual = base + np.column_stack([-known[:, 0], known[:, 0]]) * .5
    model = ForecastModel.fit(base, actual, known, (actual[:, 0] > actual[:, 1]).astype(int))
    quotes = known[:2].copy()
    quotes[:, 0] = [-10., 10.]
    rows = model.apply(base[:2], quotes, [[64., 0.], [0., 64.]], [{}, {}], [1, 2])
    assert rows[0]['football_only_home_win_probability'] == rows[1]['football_only_home_win_probability']
    assert rows[0]['market_conditioned_home_win_probability'] > rows[1]['market_conditioned_home_win_probability']
    assert rows[0]['market_conditioned_available']


def test_missing_game_slots_are_not_skipped_by_lags_or_rolling_windows(tmp_path):
    past = history()
    pd.DataFrame([dict(game_id=1, pos_team_id=10, yards_per_play=5.),
                  dict(game_id=2, pos_team_id=10, yards_per_play=6.),
                  dict(game_id=3, pos_team_id=10, yards_per_play=np.nan)]).to_parquet(tmp_path / 'adv_team_2025.parquet', index=False)
    query = past.iloc[[3]].assign(cutoff_utc=pd.Timestamp('2025-09-20T18:00Z'))
    row = pregame_history(query, tmp_path, past)[0]
    assert np.isnan(row[HISTORY_COLUMNS.index('home_yards_per_play_lag_1')])
    assert row[HISTORY_COLUMNS.index('home_yards_per_play_lag_2')] == 6.
    assert np.isnan(row[HISTORY_COLUMNS.index('home_yards_per_play_rolling_3_mean')])
    assert row[HISTORY_COLUMNS.index('home_yards_per_play_expanding_mean')] == 5.5


def test_series_validation_compares_only_saved_common_games(tmp_path):
    import json
    from cfb_forecast.evaluation.validation import validate_runs
    from cfb_forecast.ingest.data import normalize_schedule, sha256
    run = tmp_path / 'runs' / 'saved'
    run.mkdir(parents=True)
    predictions = []
    schedules = []
    for game_id in (1, 2):
        row = dict(game_id=game_id, as_of='2026-10-11T15:00Z', kickoff='2026-10-17T18:00Z',
                   home_win_probability=.6, projected_home_margin=4., projected_total=50., moneyline_candidate='none')
        for series in ('football_only', 'market_conditioned'):
            row[f'{series}_home_win_probability'] = .6 if series == 'football_only' or game_id == 1 else np.nan
            row[f'{series}_projected_home_margin'] = 4.
            row[f'{series}_projected_total'] = 50.
        predictions.append(row)
        schedules.append(dict(game_id=game_id, season=2026, week=7, home_id=10, away_id=20,
            home_team='Home', away_team='Away', start_date='2026-10-17T18:00Z', completed=True,
            neutral_site=False, home_points=28, away_points=24, fbs_game=True))
    pd.DataFrame(predictions).to_csv(run / 'forecasts.csv', index=False)
    (run / 'run_manifest.json').write_text(json.dumps({'prediction_sha256': sha256(run / 'forecasts.csv')}))
    _, summary = validate_runs(tmp_path / 'runs', normalize_schedule(pd.DataFrame(schedules)), '2026-10-19T15:00Z')
    report = summary['runs'][0]
    assert report['football_only_metrics']['games'] == 2
    assert report['market_conditioned_metrics']['games'] == 1
    assert report['paired_series_comparison']['games'] == 1
