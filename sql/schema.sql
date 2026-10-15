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
