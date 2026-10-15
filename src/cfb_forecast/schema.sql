-- CFB forecast database: the chronological record of the football information
-- available to the forecasting process. Applied on every connect; existing
-- databases are migrated (legacy tables renamed, missing columns added).
--
-- Provenance columns used throughout:
--   source_id     -> sources.source_id
--   retrieval_id  -> source_retrievals.id (exact response the value came from)
--   available_at  -> when the value could first be known (chronology guard)
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- 1. sources
CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY,
  name TEXT,
  url TEXT NOT NULL,
  role TEXT,
  schema_version TEXT
);

CREATE TABLE IF NOT EXISTS source_retrievals (
  id INTEGER PRIMARY KEY,
  source_id TEXT REFERENCES sources(source_id),
  season INTEGER,
  name TEXT,
  url TEXT,
  status TEXT,
  retrieved_at TEXT,
  schema_version TEXT,
  sha256 TEXT,
  bytes INTEGER,
  receipt_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(source_id, season, retrieved_at)
);

CREATE TABLE IF NOT EXISTS source_files (
  id INTEGER PRIMARY KEY,
  retrieval_id INTEGER REFERENCES source_retrievals(id),
  path TEXT NOT NULL,
  format TEXT,
  sha256 TEXT NOT NULL,
  bytes INTEGER,
  stored_at TEXT,
  UNIQUE(path, sha256)
);

CREATE TABLE IF NOT EXISTS source_rows (
  id INTEGER PRIMARY KEY,
  season INTEGER NOT NULL,
  source TEXT NOT NULL,
  game_id INTEGER,
  row_json TEXT NOT NULL,
  collected_at TEXT NOT NULL
);

-- ----------------------------------------------------------------- entities
CREATE TABLE IF NOT EXISTS teams (
  team_id INTEGER PRIMARY KEY,
  name TEXT,
  abbreviation TEXT,
  conference TEXT,
  division TEXT,
  latitude REAL,
  longitude REAL,
  timezone TEXT
);

CREATE TABLE IF NOT EXISTS players (
  player_id TEXT PRIMARY KEY,
  name TEXT,
  position TEXT,
  height REAL,
  weight REAL,
  birth_date TEXT
);

CREATE TABLE IF NOT EXISTS coaches (
  coach_id TEXT PRIMARY KEY,
  name TEXT,
  team_id INTEGER,
  role TEXT,
  start_season INTEGER,
  end_season INTEGER
);

CREATE TABLE IF NOT EXISTS venues (
  venue_id TEXT PRIMARY KEY,
  name TEXT,
  city TEXT,
  state TEXT,
  latitude REAL,
  longitude REAL,
  elevation REAL,
  surface TEXT,
  indoor INTEGER,
  timezone TEXT
);

-- -------------------------------------------------------------------- games
CREATE TABLE IF NOT EXISTS games (
  game_id INTEGER PRIMARY KEY,
  season INTEGER NOT NULL,
  week INTEGER,
  start_date TEXT NOT NULL,
  completed INTEGER NOT NULL,
  neutral_site INTEGER NOT NULL,
  fbs_game INTEGER NOT NULL,
  result_usable INTEGER NOT NULL,
  result_available_at TEXT,
  home_id INTEGER NOT NULL,
  away_id INTEGER NOT NULL,
  home_team TEXT,
  away_team TEXT,
  home_points REAL,
  away_points REAL,
  venue_id TEXT,
  conference TEXT,
  game_status TEXT
);

CREATE TABLE IF NOT EXISTS game_teams (
  game_id INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
  team_id INTEGER NOT NULL,
  team_name TEXT,
  is_home INTEGER NOT NULL,
  points REAL,
  PRIMARY KEY(game_id, team_id)
);

CREATE TABLE IF NOT EXISTS plays (
  play_id TEXT PRIMARY KEY,
  game_id INTEGER NOT NULL,
  drive_id TEXT,
  offense_team_id INTEGER,
  defense_team_id INTEGER,
  period INTEGER,
  clock TEXT,
  down INTEGER,
  distance REAL,
  yard_line REAL,
  yards_to_goal REAL,
  score_difference REAL,
  play_type TEXT,
  yards_gained REAL,
  first_down INTEGER,
  touchdown INTEGER,
  completion INTEGER,
  incompletion INTEGER,
  interception INTEGER,
  fumble INTEGER,
  fumble_lost INTEGER,
  sack INTEGER,
  tackle_for_loss INTEGER,
  penalty INTEGER,
  penalty_yards REAL,
  field_goal_attempt INTEGER,
  field_goal_result TEXT,
  punt INTEGER,
  kickoff INTEGER,
  return_yards REAL,
  scrimmage_play INTEGER,
  is_pass INTEGER,
  is_rush INTEGER,
  qb_hurry INTEGER,
  early_down INTEGER,
  late_down INTEGER,
  red_zone INTEGER,
  goal_to_go INTEGER,
  stuffed_run INTEGER,
  short_yardage_attempt INTEGER,
  short_yardage_success INTEGER,
  scoring_opportunity INTEGER,
  points_scored REAL,
  passer_id TEXT,
  rusher_id TEXT,
  receiver_id TEXT,
  air_yards REAL,
  yards_after_catch REAL,
  penalty_team_id INTEGER,
  punt_yards REAL,
  kickoff_yards REAL,
  touchback INTEGER,
  fg_distance REAL,
  source_id TEXT,
  retrieval_id INTEGER,
  available_at TEXT
);

CREATE TABLE IF NOT EXISTS drives (
  drive_id TEXT PRIMARY KEY,
  game_id INTEGER NOT NULL,
  team_id INTEGER,
  start_field_position REAL,
  end_field_position REAL,
  plays INTEGER,
  yards REAL,
  duration_seconds REAL,
  first_downs INTEGER,
  result TEXT,
  outcome_class TEXT,
  points REAL,
  opponent_points REAL,
  start_period INTEGER,
  source_id TEXT,
  retrieval_id INTEGER,
  available_at TEXT
);

-- ------------------------------------------------------ observed statistics
CREATE TABLE IF NOT EXISTS team_game_statistics (
  game_id INTEGER NOT NULL,
  team_id INTEGER NOT NULL,
  source_id TEXT NOT NULL DEFAULT 'sportsdataverse',
  season INTEGER,
  pass_attempts REAL,
  rush_attempts REAL,
  passing_yards REAL,
  rushing_yards REAL,
  yards_per_play REAL,
  yards_per_pass REAL,
  yards_per_rush REAL,
  first_downs REAL,
  first_downs_created_rate REAL,
  third_down_attempts REAL,
  third_down_conversions REAL,
  fourth_down_attempts REAL,
  fourth_down_conversions REAL,
  red_zone_entries REAL,
  red_zone_touchdowns REAL,
  turnovers REAL,
  sacks_allowed REAL,
  sacks_created REAL,
  tackles_for_loss REAL,
  penalties REAL,
  possession_time REAL,
  retrieval_id INTEGER,
  available_at TEXT,
  PRIMARY KEY(game_id, team_id, source_id)
);

CREATE TABLE IF NOT EXISTS player_game_statistics (
  game_id INTEGER NOT NULL,
  player_id TEXT NOT NULL,
  source_id TEXT NOT NULL,
  team_id INTEGER,
  position TEXT,
  games_played REAL,
  starts REAL,
  pass_attempts REAL,
  completions REAL,
  passing_yards REAL,
  interceptions REAL,
  carries REAL,
  rushing_yards REAL,
  targets REAL,
  receptions REAL,
  receiving_yards REAL,
  touchdowns REAL,
  sacks REAL,
  tackles REAL,
  interceptions_made REAL,
  field_goal_attempts REAL,
  field_goal_makes REAL,
  punt_attempts REAL,
  punt_yards REAL,
  return_attempts REAL,
  return_yards REAL,
  retrieval_id INTEGER,
  available_at TEXT,
  PRIMARY KEY(game_id, player_id, source_id)
);

CREATE TABLE IF NOT EXISTS player_usage (
  id INTEGER PRIMARY KEY,
  player_id TEXT NOT NULL,
  team_id INTEGER,
  season INTEGER NOT NULL,
  game_id INTEGER,
  source_id TEXT,
  usage_overall REAL,
  usage_pass REAL,
  usage_rush REAL,
  usage_first_down REAL,
  usage_second_down REAL,
  usage_third_down REAL,
  usage_standard_downs REAL,
  usage_passing_downs REAL,
  retrieval_id INTEGER,
  available_at TEXT
);

-- ----------------------------------------------------------------- personnel
CREATE TABLE IF NOT EXISTS rosters (
  id INTEGER PRIMARY KEY,
  season INTEGER NOT NULL,
  team_id INTEGER NOT NULL,
  player_id TEXT NOT NULL,
  position TEXT,
  class_year TEXT,
  jersey TEXT,
  height REAL,
  weight REAL,
  depth_chart_position TEXT,
  source_id TEXT,
  retrieval_id INTEGER,
  available_at TEXT,
  UNIQUE(season, team_id, player_id, source_id)
);

CREATE TABLE IF NOT EXISTS player_team_membership (
  id INTEGER PRIMARY KEY,
  player_id TEXT NOT NULL,
  team_id INTEGER NOT NULL,
  start_date TEXT,
  end_date TEXT,
  status TEXT,
  source_id TEXT,
  available_at TEXT
);

CREATE TABLE IF NOT EXISTS transfers (
  id INTEGER PRIMARY KEY,
  player_id TEXT,
  season INTEGER,
  previous_team_id INTEGER,
  new_team_id INTEGER,
  transfer_date TEXT,
  rating REAL,
  eligibility TEXT,
  source_id TEXT,
  available_at TEXT
);

CREATE TABLE IF NOT EXISTS recruiting (
  id INTEGER PRIMARY KEY,
  player_id TEXT,
  season INTEGER,
  team_id INTEGER,
  position TEXT,
  rating REAL,
  stars INTEGER,
  national_rank INTEGER,
  source_id TEXT,
  available_at TEXT
);

CREATE TABLE IF NOT EXISTS returning_production (
  id INTEGER PRIMARY KEY,
  season INTEGER NOT NULL,
  team_id INTEGER NOT NULL,
  total_ppa REAL,
  total_passing_ppa REAL,
  total_receiving_ppa REAL,
  total_rushing_ppa REAL,
  percent_ppa REAL,
  usage REAL,
  source_id TEXT,
  available_at TEXT,
  UNIQUE(season, team_id, source_id)
);

CREATE TABLE IF NOT EXISTS player_availability (
  id INTEGER PRIMARY KEY,
  game_id INTEGER,
  player_id TEXT NOT NULL,
  team_id INTEGER,
  reported_status TEXT,
  confirmed_inactive INTEGER,
  depth_chart_position TEXT,
  reported_at TEXT,
  source_id TEXT,
  retrieval_id INTEGER,
  available_at TEXT
);

-- -------------------------------------------------------------- environment
CREATE TABLE IF NOT EXISTS weather (
  id INTEGER PRIMARY KEY,
  game_id INTEGER NOT NULL,
  temperature REAL,
  wind_speed REAL,
  precipitation REAL,
  humidity REAL,
  conditions TEXT,
  observed_at TEXT,
  source_id TEXT,
  retrieval_id INTEGER,
  available_at TEXT
);

CREATE TABLE IF NOT EXISTS schedule_conditions (
  game_id INTEGER PRIMARY KEY,
  home_days_rest REAL,
  away_days_rest REAL,
  home_short_week INTEGER,
  away_short_week INTEGER,
  home_bye_week INTEGER,
  away_bye_week INTEGER,
  home_travel_distance REAL,
  away_travel_distance REAL,
  home_timezone_change REAL,
  away_timezone_change REAL,
  computed_at TEXT
);

-- ------------------------------------------------- ratings and market quotes
CREATE TABLE IF NOT EXISTS external_ratings (
  id INTEGER PRIMARY KEY,
  season INTEGER NOT NULL,
  week INTEGER,
  team_id INTEGER NOT NULL,
  rating_source TEXT NOT NULL,
  rating REAL,
  rank INTEGER,
  source_id TEXT,
  retrieval_id INTEGER,
  available_at TEXT,
  UNIQUE(season, week, team_id, rating_source, available_at)
);

CREATE TABLE IF NOT EXISTS market_quotes (
  id INTEGER PRIMARY KEY,
  game_id INTEGER NOT NULL,
  sportsbook TEXT NOT NULL,
  market_type TEXT NOT NULL,
  side TEXT,
  line REAL,
  price REAL,
  retrieved_at TEXT NOT NULL,
  source_id TEXT,
  retrieval_id INTEGER
);

-- ------------------------------------------------ derived football observations
CREATE TABLE IF NOT EXISTS derived_play_observations (
  play_id TEXT NOT NULL,
  derivation_model TEXT NOT NULL,
  derivation_model_version TEXT NOT NULL,
  expected_points_before REAL,
  expected_points_after REAL,
  epa REAL,
  expected_completion_probability REAL,
  completion_over_expected REAL,
  success_indicator INTEGER,
  explosive_play_indicator INTEGER,
  negative_play_indicator INTEGER,
  input_definition TEXT,
  calculation_timestamp TEXT NOT NULL,
  PRIMARY KEY(play_id, derivation_model, derivation_model_version)
);

CREATE TABLE IF NOT EXISTS derived_drive_observations (
  drive_id TEXT NOT NULL,
  derivation_model TEXT NOT NULL,
  derivation_model_version TEXT NOT NULL,
  metric TEXT NOT NULL,
  value REAL,
  input_definition TEXT,
  calculation_timestamp TEXT NOT NULL,
  PRIMARY KEY(drive_id, derivation_model, derivation_model_version, metric)
);

CREATE TABLE IF NOT EXISTS derived_team_observations (
  game_id INTEGER NOT NULL,
  team_id INTEGER NOT NULL,
  derivation_model TEXT NOT NULL,
  derivation_model_version TEXT NOT NULL,
  metric TEXT NOT NULL,
  value REAL,
  input_definition TEXT,
  calculation_timestamp TEXT NOT NULL,
  PRIMARY KEY(game_id, team_id, derivation_model, derivation_model_version, metric)
);

CREATE TABLE IF NOT EXISTS derived_player_observations (
  game_id INTEGER NOT NULL,
  player_id TEXT NOT NULL,
  derivation_model TEXT NOT NULL,
  derivation_model_version TEXT NOT NULL,
  metric TEXT NOT NULL,
  value REAL,
  input_definition TEXT,
  calculation_timestamp TEXT NOT NULL,
  PRIMARY KEY(game_id, player_id, derivation_model, derivation_model_version, metric)
);

-- ----------------------------- statistical parameters and football-state estimates
CREATE TABLE IF NOT EXISTS statistical_parameter_sets (
  parameter_set_id TEXT PRIMARY KEY,
  calculation_version TEXT NOT NULL,
  data_cutoff TEXT NOT NULL,
  source_manifest_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS statistical_parameter_values (
  parameter_set_id TEXT NOT NULL REFERENCES statistical_parameter_sets(parameter_set_id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  estimated_value REAL,
  uncertainty REAL,
  PRIMARY KEY(parameter_set_id, name)
);

-- component rows: one per (entity, side/unit, component, time index)
CREATE TABLE IF NOT EXISTS team_state_estimates (
  parameter_set_id TEXT NOT NULL REFERENCES statistical_parameter_sets(parameter_set_id) ON DELETE CASCADE,
  team_id INTEGER NOT NULL,
  side TEXT NOT NULL,
  component TEXT NOT NULL,
  as_of TEXT NOT NULL,
  mean REAL,
  sd REAL,
  PRIMARY KEY(parameter_set_id, team_id, side, component, as_of)
);

CREATE TABLE IF NOT EXISTS player_state_estimates (
  parameter_set_id TEXT NOT NULL REFERENCES statistical_parameter_sets(parameter_set_id) ON DELETE CASCADE,
  player_id TEXT NOT NULL,
  position_group TEXT NOT NULL,
  component TEXT NOT NULL,
  as_of TEXT NOT NULL,
  mean REAL,
  sd REAL,
  PRIMARY KEY(parameter_set_id, player_id, position_group, component, as_of)
);

CREATE TABLE IF NOT EXISTS unit_state_estimates (
  parameter_set_id TEXT NOT NULL REFERENCES statistical_parameter_sets(parameter_set_id) ON DELETE CASCADE,
  team_id INTEGER NOT NULL,
  unit TEXT NOT NULL,
  component TEXT NOT NULL,
  as_of TEXT NOT NULL,
  mean REAL,
  sd REAL,
  PRIMARY KEY(parameter_set_id, team_id, unit, component, as_of)
);

CREATE TABLE IF NOT EXISTS availability_estimates (
  parameter_set_id TEXT NOT NULL REFERENCES statistical_parameter_sets(parameter_set_id) ON DELETE CASCADE,
  game_id INTEGER NOT NULL,
  player_id TEXT NOT NULL,
  participation_probability REAL,
  expected_usage_share REAL,
  as_of TEXT NOT NULL,
  PRIMARY KEY(parameter_set_id, game_id, player_id, as_of)
);

-- ------------------------------------------------ forecasts and evaluation
CREATE TABLE IF NOT EXISTS forecast_runs (
  run_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  information_cutoff TEXT NOT NULL,
  parameter_set_id TEXT REFERENCES statistical_parameter_sets(parameter_set_id),
  status TEXT NOT NULL DEFAULT 'completed',
  manifest_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS forecasts (
  forecast_id INTEGER PRIMARY KEY,
  run_id TEXT REFERENCES forecast_runs(run_id),
  game_id INTEGER NOT NULL,
  creation_timestamp TEXT NOT NULL,
  information_cutoff TEXT NOT NULL,
  parameter_set_id TEXT REFERENCES statistical_parameter_sets(parameter_set_id),
  home_expected_points REAL NOT NULL,
  away_expected_points REAL NOT NULL,
  home_win_probability REAL NOT NULL,
  away_win_probability REAL NOT NULL,
  expected_margin REAL NOT NULL,
  margin_variance REAL,
  expected_total REAL NOT NULL,
  total_variance REAL,
  forecast_distribution_artifact TEXT,
  team_estimate_reference TEXT,
  player_estimate_reference TEXT,
  availability_reference TEXT,
  weather_reference TEXT,
  market_quote_reference TEXT,
  UNIQUE(game_id, creation_timestamp, parameter_set_id)
);

CREATE TABLE IF NOT EXISTS forecast_inputs (
  id INTEGER PRIMARY KEY,
  forecast_id INTEGER NOT NULL REFERENCES forecasts(forecast_id) ON DELETE CASCADE,
  input_type TEXT NOT NULL,
  reference_table TEXT,
  reference_key TEXT,
  payload_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS game_results (
  game_id INTEGER PRIMARY KEY,
  actual_home_points REAL NOT NULL,
  actual_away_points REAL NOT NULL,
  actual_margin REAL NOT NULL,
  actual_total REAL NOT NULL,
  actual_winner TEXT,
  result_retrieved_at TEXT NOT NULL,
  source_id TEXT
);

CREATE TABLE IF NOT EXISTS forecast_evaluations (
  forecast_id INTEGER PRIMARY KEY REFERENCES forecasts(forecast_id) ON DELETE CASCADE,
  evaluated_at TEXT NOT NULL,
  margin_absolute_error REAL,
  total_absolute_error REAL,
  home_absolute_error REAL,
  away_absolute_error REAL,
  brier REAL,
  log_loss REAL,
  winner_correct INTEGER,
  crps REAL,
  interval_coverage_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_games_season_start ON games(season, start_date);
CREATE INDEX IF NOT EXISTS idx_plays_game ON plays(game_id);
CREATE INDEX IF NOT EXISTS idx_drives_game ON drives(game_id);
CREATE INDEX IF NOT EXISTS idx_team_game_statistics_team ON team_game_statistics(team_id, season);
CREATE INDEX IF NOT EXISTS idx_player_game_statistics_player ON player_game_statistics(player_id);
CREATE INDEX IF NOT EXISTS idx_rosters_team ON rosters(season, team_id);
CREATE INDEX IF NOT EXISTS idx_weather_game ON weather(game_id);
CREATE INDEX IF NOT EXISTS idx_market_quotes_game ON market_quotes(game_id, retrieved_at);
CREATE INDEX IF NOT EXISTS idx_source_rows_source ON source_rows(season, source);
CREATE INDEX IF NOT EXISTS idx_forecasts_game ON forecasts(game_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_run ON forecasts(run_id);

-- ------------------------------------------- operational tables (run bookkeeping)
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  run_key TEXT NOT NULL UNIQUE,
  season INTEGER NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  status TEXT NOT NULL DEFAULT 'completed',
  manifest_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  path TEXT NOT NULL,
  training_through_season INTEGER NOT NULL,
  variant TEXT NOT NULL,
  created_at TEXT NOT NULL,
  manifest_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_sources (
  id INTEGER PRIMARY KEY,
  run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
  provider TEXT NOT NULL,
  url TEXT,
  retrieved_at TEXT,
  sha256 TEXT,
  status TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS run_forecasts (
  id INTEGER PRIMARY KEY,
  run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
  model_id INTEGER NOT NULL REFERENCES models(id),
  game_id INTEGER NOT NULL,
  season INTEGER NOT NULL,
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  kickoff TEXT NOT NULL,
  as_of TEXT NOT NULL,
  expected_home_score REAL NOT NULL,
  expected_away_score REAL NOT NULL,
  projected_home_margin REAL NOT NULL,
  projected_total REAL NOT NULL,
  home_win_probability REAL NOT NULL,
  market_home_probability_no_vig REAL,
  best_bet_side TEXT,
  best_bet_edge REAL,
  best_bet_ev_per_unit REAL,
  recommended_stake_fraction REAL,
  bet_status TEXT,
  payload_json TEXT NOT NULL,
  UNIQUE(model_id, game_id, as_of)
);

CREATE TABLE IF NOT EXISTS results (
  id INTEGER PRIMARY KEY,
  game_id INTEGER NOT NULL,
  season INTEGER NOT NULL,
  home_points REAL,
  away_points REAL,
  completed INTEGER NOT NULL DEFAULT 0,
  result_available_at TEXT,
  source TEXT,
  payload_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(game_id, season)
);

CREATE TABLE IF NOT EXISTS data_files (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  size_bytes INTEGER NOT NULL,
  modified_ns INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  status TEXT NOT NULL,
  checked_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS watchdog_events (
  id INTEGER PRIMARY KEY,
  event_type TEXT NOT NULL,
  path TEXT,
  status TEXT NOT NULL,
  message TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  details_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_runs_season ON runs(season);
CREATE INDEX IF NOT EXISTS idx_run_sources_run ON run_sources(run_id);
CREATE INDEX IF NOT EXISTS idx_run_forecasts_season_kickoff ON run_forecasts(season, kickoff);
CREATE INDEX IF NOT EXISTS idx_run_forecasts_game ON run_forecasts(game_id);
CREATE INDEX IF NOT EXISTS idx_results_season ON results(season);
CREATE INDEX IF NOT EXISTS idx_data_files_status ON data_files(status);
CREATE INDEX IF NOT EXISTS idx_watchdog_events_time ON watchdog_events(observed_at);

CREATE TABLE IF NOT EXISTS forecast_log (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL,
  game_id INTEGER NOT NULL,
  season INTEGER,
  kickoff TEXT,
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  expected_home_score REAL,
  expected_away_score REAL,
  home_win_probability REAL,
  predicted_winner TEXT,
  confidence REAL,
  published_at TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(run_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_forecast_log_run ON forecast_log(run_id);

