-- Canonical SQLite schema for CFB Forecast.
-- Applied by Python's sqlite3 layer and usable directly by RSQLite.
PRAGMA foreign_keys = ON;

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

CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY,
  run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
  provider TEXT NOT NULL,
  url TEXT,
  retrieved_at TEXT,
  sha256 TEXT,
  status TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS forecasts (
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

CREATE TABLE IF NOT EXISTS source_registry (
  source_id INTEGER PRIMARY KEY,
  source_key TEXT NOT NULL UNIQUE,
  provider TEXT NOT NULL,
  source_url TEXT,
  schema_version TEXT
);

CREATE TABLE IF NOT EXISTS source_retrievals (
  retrieval_id INTEGER PRIMARY KEY,
  source_id INTEGER NOT NULL REFERENCES source_registry(source_id),
  retrieved_at TEXT NOT NULL,
  cutoff_at TEXT,
  http_status INTEGER,
  content_sha256 TEXT,
  byte_count INTEGER,
  status TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS source_files (
  file_id INTEGER PRIMARY KEY,
  retrieval_id INTEGER NOT NULL REFERENCES source_retrievals(retrieval_id),
  path TEXT NOT NULL UNIQUE,
  content_sha256 TEXT,
  byte_count INTEGER,
  format TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS teams (
  team_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
  game_id INTEGER PRIMARY KEY,
  season INTEGER NOT NULL,
  week INTEGER NOT NULL,
  kickoff_datetime TEXT NOT NULL,
  venue_id INTEGER,
  neutral_site INTEGER NOT NULL,
  status TEXT NOT NULL,
  source_id INTEGER REFERENCES source_registry(source_id),
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS game_teams (
  game_id INTEGER NOT NULL REFERENCES games(game_id),
  team_id INTEGER NOT NULL REFERENCES teams(team_id),
  side TEXT NOT NULL,
  points REAL,
  PRIMARY KEY (game_id, side)
);

CREATE TABLE IF NOT EXISTS statistical_parameter_sets (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  data_cutoff TEXT NOT NULL,
  source_manifest_json TEXT NOT NULL,
  parameters_json TEXT NOT NULL,
  uncertainty_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS forecast_runs (
  id INTEGER PRIMARY KEY,
  parameter_set_id INTEGER NOT NULL REFERENCES statistical_parameter_sets(id),
  season INTEGER NOT NULL,
  information_cutoff TEXT NOT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL,
  source_manifest_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_versions (
  version TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  cutoff TEXT NOT NULL,
  status TEXT NOT NULL,
  model_path TEXT NOT NULL,
  report_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_examples (
  version TEXT NOT NULL REFERENCES learning_versions(version),
  game_id INTEGER NOT NULL,
  season INTEGER NOT NULL,
  forecast_cutoff TEXT NOT NULL,
  result_available_at TEXT NOT NULL,
  latest_feature_result TEXT,
  features_json TEXT NOT NULL,
  observations_json TEXT NOT NULL,
  PRIMARY KEY(version, game_id)
);

CREATE INDEX IF NOT EXISTS training_examples_season ON training_examples(season);
CREATE INDEX IF NOT EXISTS idx_runs_season ON runs(season);
CREATE INDEX IF NOT EXISTS idx_sources_run ON sources(run_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_season_kickoff ON forecasts(season, kickoff);
CREATE INDEX IF NOT EXISTS idx_forecasts_game ON forecasts(game_id);
CREATE INDEX IF NOT EXISTS idx_results_season ON results(season);
CREATE INDEX IF NOT EXISTS idx_data_files_status ON data_files(status);
CREATE INDEX IF NOT EXISTS idx_watchdog_events_time ON watchdog_events(observed_at);
CREATE TABLE IF NOT EXISTS forecast_log (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL,
  season INTEGER NOT NULL,
  game_id INTEGER NOT NULL,
  kickoff TEXT NOT NULL,
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  expected_home_score REAL NOT NULL,
  expected_away_score REAL NOT NULL,
  winner TEXT NOT NULL,
  confidence REAL NOT NULL,
  home_win_probability REAL NOT NULL,
  logged_at TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  projected_home_margin REAL,
  model_spread REAL,
  market_spread REAL,
  home_cover_probability REAL,
  home_moneyline REAL,
  away_moneyline REAL,
  UNIQUE(run_id, game_id)
);

CREATE TABLE IF NOT EXISTS weekly_validation (
  id INTEGER PRIMARY KEY,
  season INTEGER NOT NULL,
  week INTEGER,
  run_id TEXT NOT NULL,
  validated_at TEXT NOT NULL,
  games INTEGER NOT NULL,
  status TEXT NOT NULL,
  log_loss REAL,
  brier REAL,
  margin_mae REAL,
  spread_mae REAL,
  report_json TEXT NOT NULL,
  UNIQUE(season, week, run_id)
);

CREATE TABLE IF NOT EXISTS team_game_facts (
  game_id INTEGER NOT NULL,
  team_id INTEGER NOT NULL,
  season INTEGER,
  week INTEGER,
  plays INTEGER,
  points_for REAL,
  points_against REAL,
  yards_per_play REAL,
  success_rate REAL,
  explosive_rate REAL,
  turnover_rate REAL,
  pass_yards REAL,
  rush_yards REAL,
  receiving_yards REAL,
  epa REAL,
  PRIMARY KEY (game_id, team_id)
);

CREATE TABLE IF NOT EXISTS player_game_facts (
  game_id INTEGER NOT NULL,
  team_id INTEGER NOT NULL,
  player_name TEXT NOT NULL,
  position_group TEXT NOT NULL,
  season INTEGER,
  plays INTEGER,
  yards REAL,
  epa REAL,
  PRIMARY KEY (game_id, team_id, player_name, position_group)
);

CREATE INDEX IF NOT EXISTS idx_team_game_facts_team ON team_game_facts(team_id, season);
CREATE INDEX IF NOT EXISTS idx_player_game_facts_team ON player_game_facts(team_id, season);
CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week);
CREATE INDEX IF NOT EXISTS idx_forecast_runs_season ON forecast_runs(season);
CREATE INDEX IF NOT EXISTS idx_forecast_log_season_kickoff ON forecast_log(season, kickoff);

-- Database loop: ingested outcomes, per-forecast validation, and the independent recomputation from stored rows.
CREATE TABLE IF NOT EXISTS external_results (
  id INTEGER PRIMARY KEY,
  season INTEGER NOT NULL,
  game_id INTEGER NOT NULL,
  observed_at TEXT NOT NULL,
  sources_agree INTEGER,
  schedule_agrees INTEGER,
  source_scores TEXT,
  schedule_score TEXT,
  UNIQUE(season, game_id, observed_at)
);

CREATE TABLE IF NOT EXISTS forecast_validation (
  run_id TEXT NOT NULL,
  game_id INTEGER NOT NULL,
  season INTEGER NOT NULL,
  week INTEGER,
  logged_at TEXT NOT NULL,
  kickoff TEXT NOT NULL,
  home_win_probability REAL NOT NULL,
  projected_home_margin REAL,
  home_win INTEGER NOT NULL,
  actual_margin REAL NOT NULL,
  result_source TEXT,
  validated_at TEXT NOT NULL,
  PRIMARY KEY(run_id, game_id)
);

CREATE TABLE IF NOT EXISTS db_validation (
  run_id TEXT NOT NULL,
  computed_at TEXT NOT NULL,
  games INTEGER NOT NULL,
  log_loss REAL NOT NULL,
  brier REAL NOT NULL,
  margin_mae REAL,
  PRIMARY KEY(run_id)
);
