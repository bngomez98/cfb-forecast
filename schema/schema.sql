-- CFB Forecast — canonical PostgreSQL schema (single database: cfb_forecast)
-- Vocabulary contract: docs/GLOSSARY.md. Column names, CHECK values, and
-- timestamp suffixes (_at) follow glossary §7. Forward-only migrations live in
-- schema/migrations/. This file is the full declarative state at version 1.

BEGIN;

CREATE SCHEMA IF NOT EXISTS public;

-- ---------------------------------------------------------------------------
-- Reference entities
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS teams (
    team              text        NOT NULL,           -- canonical display name
    season            smallint    NOT NULL,           -- four-digit calendar year
    espn_team_id      integer,
    conference        text,
    division          text,                           -- e.g. 'East', 'West' (ETG level)
    abbreviation      text,
    PRIMARY KEY (team, season)
);

CREATE TABLE IF NOT EXISTS venues (
    venue             text        NOT NULL,
    latitude          double precision,
    longitude         double precision,
    dome              boolean     NOT NULL DEFAULT false,
    PRIMARY KEY (venue)
);

-- ---------------------------------------------------------------------------
-- Core game data (raw scores stored untouched — no smoothing, glossary §3)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS games (
    espn_game_id      text        NOT NULL,
    season            smallint    NOT NULL,
    week              smallint    NOT NULL CHECK (week >= 1),   -- ESPN weeks start at 1
    kickoff_at        timestamptz,
    home_team         text        NOT NULL,
    away_team         text        NOT NULL,
    venue             text REFERENCES venues(venue),
    site_type         text        NOT NULL DEFAULT 'home'
                    CHECK (site_type IN ('home', 'neutral')),
    home_points       smallint CHECK (home_points IS NULL OR home_points >= 0),
    away_points       smallint CHECK (away_points IS NULL OR away_points >= 0),
    win_score         smallint,                     -- W_g = winning score
    lose_score        smallint,                     -- L_g = losing score
    margin            smallint,                     -- M_g = W_g - L_g (positive for winner side)
    total             smallint,                     -- T_g = W_g + L_g
    winner_team       text,                         -- NULL while scheduled/incomplete
    status            text        NOT NULL DEFAULT 'scheduled'
                    CHECK (status IN ('scheduled', 'in_progress', 'final')),
    source            text        NOT NULL DEFAULT 'espn'
                    CHECK (source IN ('espn', 'cfb_data', 'manual')),
    captured_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (espn_game_id)
);

CREATE INDEX IF NOT EXISTS games_season_week_idx ON games (season, week);

CREATE TABLE IF NOT EXISTS team_games (
    season            smallint    NOT NULL,
    team              text        NOT NULL,
    week              smallint    NOT NULL CHECK (week >= 1),
    espn_game_id      text        NOT NULL REFERENCES games(espn_game_id),
    side              text        NOT NULL CHECK (side IN ('home', 'away')),
    opponent          text        NOT NULL,
    points_for        smallint    NOT NULL CHECK (points_for >= 0),
    points_against    smallint    NOT NULL CHECK (points_against >= 0),
    margin            smallint    NOT NULL,          -- signed from this team's perspective
    total             smallint    NOT NULL,          -- identical to games.total
    won               boolean     NOT NULL,
    PRIMARY KEY (season, team, week, espn_game_id, side)
);

CREATE INDEX IF NOT EXISTS team_games_team_form_idx ON team_games (team, season, week);

-- ---------------------------------------------------------------------------
-- Conditional feature layers (glossary §3)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS weather_observations (
    espn_game_id      text        NOT NULL REFERENCES games(espn_game_id),
    temperature_f     real,
    wind_mph          real,
    precipitation_in  real,
    conditions        text,
    source            text        NOT NULL DEFAULT 'open_meteo'
                    CHECK (source IN ('open_meteo', 'noaa')),
    captured_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (espn_game_id)
);

CREATE TABLE IF NOT EXISTS injury_reports (
    report_id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    season            smallint    NOT NULL,
    week              smallint    NOT NULL CHECK (week >= 1),
    team              text        NOT NULL,
    player            text        NOT NULL,
    position          text,
    status            text        NOT NULL
                    CHECK (status IN ('out', 'questionable', 'doubtful', 'limited', 'day-to-day')),
    injury            text,
    captured_at       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (season, week, team, player)
);

CREATE TABLE IF NOT EXISTS player_ppg (
    season            smallint    NOT NULL,
    team              text        NOT NULL,
    player            text        NOT NULL,
    position          text,
    points_per_game   real,
    snap_fraction     real,       -- usage share, drives matchup features
    PRIMARY KEY (season, team, player)
);

CREATE TABLE IF NOT EXISTS market_lines (
    espn_game_id      text        NOT NULL REFERENCES games(espn_game_id),
    spread_home       real,                          -- market spread (glossary: not "margin")
    total_line        real,                          -- market over/under line
    provider          text        NOT NULL,
    line_at           timestamptz NOT NULL,
    PRIMARY KEY (espn_game_id, provider, line_at)
);

-- ---------------------------------------------------------------------------
-- Prospective prediction ledger (immutable; glossary §1 cutoff rules)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    espn_game_id      text        NOT NULL REFERENCES games(espn_game_id),
    expert            text        NOT NULL,          -- registry name, glossary §4
    model_version     text        NOT NULL,
    cutoff_at         timestamptz NOT NULL,          -- must precede kickoff_at (enforced below)
    created_at        timestamptz NOT NULL DEFAULT now(),
    win_prob          real        NOT NULL CHECK (win_prob >= 0 AND win_prob <= 1),
    margin_points     real,                          -- predicted margin for home side
    total_points      real,                          -- predicted total
    input_hash        text        NOT NULL,          -- hash-sealed frozen inputs
    UNIQUE (espn_game_id, expert, cutoff_at)
);

CREATE TABLE IF NOT EXISTS settled_truth (
    espn_game_id      text        NOT NULL REFERENCES games(espn_game_id),
    settled_at        timestamptz NOT NULL DEFAULT now(),
    home_win          boolean     NOT NULL,
    margin            smallint    NOT NULL,
    total             smallint    NOT NULL,
    PRIMARY KEY (espn_game_id)
);

-- Append-only learning ledger joining predictions to outcomes (glossary §4).
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    outcome_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prediction_id     bigint      NOT NULL REFERENCES predictions(prediction_id),
    espn_game_id      text        NOT NULL,
    expert            text        NOT NULL,
    win_prob          real        NOT NULL,
    home_win          boolean     NOT NULL,
    abs_margin_error  real,
    abs_total_error   real,
    log_loss          real        NOT NULL,
    settled_at        timestamptz NOT NULL,
    UNIQUE (prediction_id)
);

-- Learned blend weights snapshot after each settle run (glossary §4).
CREATE TABLE IF NOT EXISTS blend_weights (
    weight_id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    updated_at        timestamptz NOT NULL DEFAULT now(),
    expert            text        NOT NULL,
    weight            real        NOT NULL CHECK (weight >= 0 AND weight <= 1),
    hit_rate          real CHECK (hit_rate IS NULL OR (hit_rate >= 0 AND hit_rate <= 1)),
    brier_score       real CHECK (brier_score IS NULL OR (brier_score >= 0 AND brier_score <= 1)),
    log_loss          real,
    calibration_slope real,
    n_settled         integer     NOT NULL DEFAULT 0 CHECK (n_settled >= 0),
    UNIQUE (updated_at, expert)
);

-- ---------------------------------------------------------------------------
-- CRAFT module parameters (values aligned with Zhang et al., IJCAI-25;
-- football mapping per glossary §5). One row per fit; latest wins.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS craft_parameters (
    fit_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    fitted_at         timestamptz NOT NULL DEFAULT now(),
    horizon_weeks     smallint    NOT NULL DEFAULT 7 CHECK (horizon_weeks IN (7, 14, 30)),
    lookback_games    smallint    NOT NULL DEFAULT 6 CHECK (lookback_games > 0),
    embed_dim         integer     NOT NULL DEFAULT 32 CHECK (embed_dim > 0),
    koopman_lambda    real        NOT NULL DEFAULT 0.1 CHECK (koopman_lambda > 0),  -- paper §5 Implementation
    hier_children     integer     NOT NULL DEFAULT 15 CHECK (hier_children > 0),    -- paper m = 15
    demand_beta       real        NOT NULL DEFAULT 0.5 CHECK (demand_beta >= 0),    -- hinge weight
    alpha1            real        NOT NULL DEFAULT 1.0 CHECK (alpha1 >= 0),         -- recovery loss
    alpha2            real        NOT NULL DEFAULT 1.0 CHECK (alpha2 >= 0),         -- bias loss
    alpha3            real        NOT NULL DEFAULT 1.0 CHECK (alpha3 >= 0),         -- reconciliation loss
    batch_size        integer     NOT NULL DEFAULT 256 CHECK (batch_size > 0),      -- paper setting
    learning_rate     real        NOT NULL DEFAULT 0.001 CHECK (learning_rate > 0), -- paper Adam lr
    train_epochs      integer     NOT NULL DEFAULT 2 CHECK (train_epochs > 0),      -- paper setting
    wmape             real CHECK (wmape IS NULL OR wmape >= 0),
    notes             text
);

-- ---------------------------------------------------------------------------
-- Data-cleaning audit trail (quarantine is never silently dropped, glossary §1)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS cleaning_log (
    log_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_at            timestamptz NOT NULL DEFAULT now(),
    source            text        NOT NULL,
    rows_in           integer     NOT NULL CHECK (rows_in >= 0),
    rows_clean        integer     NOT NULL CHECK (rows_clean >= 0),
    rows_quarantined  integer     NOT NULL CHECK (rows_quarantined >= 0),
    reason_breakdown  jsonb       NOT NULL DEFAULT '{}'::jsonb
);

-- Daily-cycle execution record (the app runs itself; glossary §6).
CREATE TABLE IF NOT EXISTS cycle_runs (
    run_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz,
    kind              text        NOT NULL CHECK (kind IN ('daily', 'weekly', 'manual')),
    games_refreshed   integer     NOT NULL DEFAULT 0,
    games_settled     integer     NOT NULL DEFAULT 0,
    predictions_made  integer     NOT NULL DEFAULT 0,
    errors            jsonb       NOT NULL DEFAULT '[]'::jsonb,
    ok                boolean     NOT NULL DEFAULT true
);

-- Cutoff integrity trigger: prospective predictions must be sealed pre-kickoff.
CREATE OR REPLACE FUNCTION enforce_prediction_cutoff() RETURNS trigger AS $$
BEGIN
    IF (SELECT kickoff_at FROM games WHERE espn_game_id = NEW.espn_game_id) IS NOT NULL
       AND NEW.cutoff_at >= (SELECT kickoff_at FROM games WHERE espn_game_id = NEW.espn_game_id)
    THEN
        RAISE EXCEPTION 'cutoff_at (%) must precede kickoff_at for game %',
            NEW.cutoff_at, NEW.espn_game_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS predictions_cutoff_check ON predictions;
CREATE TRIGGER predictions_cutoff_check
    BEFORE INSERT OR UPDATE ON predictions
    FOR EACH ROW EXECUTE FUNCTION enforce_prediction_cutoff();

COMMIT;
