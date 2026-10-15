# CFB Forecast — Project Glossary

**Version:** 1.0 · **Canonical spelling enforced repo-wide.** Every term below is the
single approved name for the concept; code identifiers, database columns, CLI flags,
and documentation must use these spellings. Synonyms in the "Deprecated / avoid"
column must not be introduced in new code or schema.

This glossary is the continuity contract between: Python modules
(`src/cfb_forecast/`), the PostgreSQL schema (`schema/schema.sql`, `db.py`),
the Streamlit dashboard (`dashboard/app.py`), and the docs (`README.md`,
`RESULTS.md`, `CHRONOLOGY.md`).

---

## 1. Domain & data model

| Term | Definition | Where implemented | Deprecated / avoid |
|---|---|---|---|
| **Game** | One completed or scheduled FBS (or FCS-for-record) contest. Identified by `espn_game_id`. Row in `games`. | `clean.py`, `db.py::upsert_games` | "match", "fixture" |
| **Team game** | One team's appearance in one game (2 rows per game). Carries side, points, margin contribution. Row in `team_games`. | `clean.py`, `db.py::upsert_team_games` | "team-game line" |
| **Week** | ESPN competition week. Note: ESPN weeks start at **1**; the research write-up's "weeks 0–6" window maps to `week <= 7` under ESPN numbering. Column `week` everywhere. | `games.week`, `team_games.week` | "round", "gameweek" |
| **Season** | Four-digit calendar year of the season (`season` column). | all tables | "year" as a column name |
| **Margin (M)** | `M = W − L` = winning score minus losing score. Stored signed per team in `team_games.margin` (positive = winner). | `clean.py` | "differential", "spread" (spread = market term, §5) |
| **Total (T)** | `T = W + L` combined points. Stored per team identically in `team_games.total`. | `clean.py` | "over/under" (market term) |
| **Winning score (W) / Losing score (L)** | Final scores of winner/loser. Columns `win_score`, `lose_score` on `games`; mirrored into `team_games.points_for/points_against`. | schema | "home score" as outcome variable |
| **Side** | `home` \| `away` \| `neutral`. Column `side`, constrained by CHECK. | `clean.py`, schema | "position" |
| **Cutoff** | The instant after which no information may enter a prediction. All prospective features are frozen at cutoff. Enforced by assertions in `chronology/` and `holdout.py`. | `chronology/`, `holdout.py` | "as-of date" (use only in prose) |
| **Prospective holdout** | Immutable ledger of predictions made strictly before kickoff, hash-sealed. Tables `predictions` + `settled_truth`. | `holdout.py`, `db.py` | "blind test set" |
| **Settlement** | Joining a prediction to its final result and grading it. Command `settle`; writes `settled_truth`. | `daily.py`, `learning.py` | "scoring" (ambiguous with points) |
| **Quarantine** | Rows failing validation (duplicate espn ids, non-numeric scores, negative margins on losers, etc.). Written to `data/clean/quarantine.csv` and reported in `cleaning_report.json`. Never silently dropped. | `clean.py` | "rejects", "bad rows" |

## 2. Sources (single DB, PostgreSQL)

| Term | Definition | Aligned parameters |
|---|---|---|
| **CFB-Data / sportsdataverse** | Bulk release files (parquet/CSV) for historical games, box summaries, spreads. Used for **training history** (reproducible snapshots). | `config.py` release URLs |
| **ESPN public scoreboard API** | Live schedules/results/injuries/news JSON. Used for **freshness** (daily cycle). No key required. | endpoint `site.api.espn.com/apis/site/v2/sports/football/ncaaf/scoreboard` |
| **Open-Meteo** | Free weather forecast + archive APIs keyed by venue lat/lon. No key required. | `weather.py`, table `weather_observations` |
| **Market (spreads)** | Closing-line consensus from sportsdataverse betting releases. Table `market_lines`. | `market.py` |
| **Single database** | Exactly one Postgres instance/database (`cfb_forecast`) holds all canonical state; SQLite artifacts are legacy caches only, never a second source of truth. DSN via `CFB_POSTGRES_DSN`. | `db.py`, `schema/schema.sql` |

## 3. Conditional feature layers (rating-free)

| Term | Definition | Source justification |
|---|---|---|
| **Rolling form** | Per-team trailing windows over *completed* games strictly before cutoff: rolling margin mean, rolling total mean, rolling win rate. No smoothing of raw data — raw scores are stored untouched; aggregation is explicit feature engineering. | South & Egros 2020 [3]; Wang 2025 [5] |
| **Opponent-recent performance** | Feature arm encoding opponent quality *implicitly* (rolling differential of the opponent's last games) instead of an explicit rating object. | Lampis et al. 2023 [7] caveat handling |
| **Matchup similarity (kNN)** | Nearest-neighbor borrowing from historically similar matchups; the March Madness template. | Hoegh et al. 2015 [6] |
| **Venue / home advantage** | `venue`, dome flag, site type. Home advantage H enters margin regressions directly. | Han et al. 2022 [9]; West & Lamsal 2014 |
| **Weather condition** | Temperature, wind speed, precipitation at kickoff from Open-Meteo; affects totals modeling. | table `weather_observations` |
| **Availability (injuries)** | Missing/limited players per team from ESPN injuries endpoint; player-by-player matchup context. | table `injury_reports` |
| **Roster PPG** | Player-level production (`player_ppg`) used for matchup features, not ratings. | `rosters.py`, `player_ppg.py` |

## 4. Experts & blend (learning loop)

| Term | Definition | Notes |
|---|---|---|
| **Expert** | Any single predictive head emitting a win probability. Registered names (canonical, lowercase): `base_rate`, `margin_elo`, `winner_elo`, `ridge`, `logit`, `gbm`, `ts_arima`, `craft`. | Registry in `heads.py`/`timeheads.py`; persisted in `blend_weights.expert` |
| **Blend** | Convex combination of expert probabilities with learned weights. | `adaptive.py`; table `blend_weights` |
| **Adaptive weights** | Multiplicative-weights (Hedge-style) update on log-loss of settled predictions; updated every `settle`. | `adaptive.py::update_weights` |
| **Calibration slope / intercept** | Logistic refit of each expert's settled predictions; slope < 1 ⇒ overconfident. Computed by `learning_audit` with sklearn. | evaluation section of research doc |
| **Clopper-Pearson interval** | Exact binomial hit-rate CI (scipy.stats.beta); used because early-season samples are small. | `learning.py::learning_audit` |
| **Brier score / Log loss** | Proper scoring rules for probability quality (sklearn.metrics). | `learning.py` |
| **Paired bootstrap** | Block resampling of settled games comparing blend vs each expert; reports Δaccuracy CI and p-value. | `evaluate.py`, `learning.py` |
| **Learning ledger** | Append-only record of every prediction→outcome pair feeding weight updates. Table `prediction_outcomes`. | `learning.py`, `db.py` |

## 5. CRAFT terminology (Zhang et al., IJCAI-25)

Mapped from the hotel-booking paper to college football forecasting. The mapping is
part of this project's vocabulary contract — use these terms consistently.

| CRAFT term | Paper meaning | Football instantiation | Identifier |
|---|---|---|---|
| **Label series Y** | Target time series (hotel demand) | Team's per-game point series (score for / margin) | `Y` |
| **Look-back window L** | Past observations (30/90/180 days) | Last `L` games of the team (default 6 ≈ weeks 0–6 window) | `lookback` |
| **Prediction window P** | Future horizon (7/14/30 days) | Next `P` scheduled games (known future schedule!) | `horizon` |
| **Cross-Future Behavior (CFB)** | Actions now that take effect later (bookings) | **Known future schedule**: committed upcoming opponents, sites, dates — observable before they happen | `C` |
| **Partial CFB (C_P)** | Future bookings observed only up to now | Upcoming-games matrix where far-future cells are unobserved (schedule announced but outcomes unknown) | `C_partial` |
| **Koopman Predictor Module (KPM)** | Linear operator K_C mapping CFB trend → label trend | Ridge-solved shared-MLP embedding operator transferring schedule-trend to scoring-trend | `craft_model.py::KPM` |
| **Internal Trend Mining (ITM)** | Completes unknown region of CFB matrix | Complete-network patches missing future-schedule cells; adaptation network shifts CFB distribution → label distribution | `craft_model.py::ITM` |
| **External Trend Guide (ETG)** | Hierarchical reconciliation (hotel→district→city) | Conference → division → FBS aggregate reconciliation of predicted margins/totals (sum-consistency loss) | `craft_model.py::ETG` |
| **Demand-constrained loss** | Penalizes predictions outside lower/upper demand bounds | Bounds: lower = expected floor from rolling form; upper = cap from opponent-adjusted scoring ceiling; hinge penalty β outside band | `craft_model.py::demand_loss` |
| **Recovery loss (L_be_k)** | Decoder reconstructs known CFB cells | Same, masked to observed schedule cells only | `alpha1` |
| **Bias loss (L_be_y)** | Error on completed C_T_P matrix | Same on imputed future-schedule strength cells | `alpha2` |
| **Reconciliation loss (L_recon)** | Parent = Σ children | Conference-total consistency of child team predictions | `alpha3` |
| **Ridge λ (Koopman solve)** | 0.1 in paper | Kept at **0.1** — parameter parity with source | `koopman_lambda` |
| **Hierarchical sample m** | 15 children per virtual parent | 15 teams per sampled conference group (small conferences merged) | `hier_children` |
| **β (demand hinge)** | paper hyperparameter | default **0.5**, grid-searchable | `demand_beta` |
| **wMAPE** | Weighted MAPE metric | Reported for margin/total regression heads | `evaluate.py` |

## 6. Pipeline operations

| Term | Definition |
|---|---|
| **Daily cycle** | `cfb-forecast daily-cycle`: refresh live results/weather/injuries → settle finished games → update learning ledger + adaptive weights → forecast upcoming slate → persist to Postgres. Idempotent; network failures non-fatal. |
| **Freeze** | Snapshot of exact model inputs at cutoff, hash-sealed (`holdout-freeze`). |
| **Verify** | Recompute hashes of immutable ledgers (`holdout-verify`, `chronology-verify`). |
| **Warmup** | Initial weeks excluded from training so rolling features are populated. |
| **Clean build** | `pytest -q` green + `ruff check` clean + `ruff format --check` clean + git index free of artifacts (`__pycache__`, `*.egg-info`). |

## 7. Naming conventions (continuity rules)

1. **Columns:** snake_case, singular entity names (`game_id`, `team`, `week`, `season`). Probability columns end in `_prob`; point columns end in `_points`; no abbreviations except `margin`, `total`, `elo`, `ppg`.
2. **Expert registry names** are the only legal values for `blend_weights.expert` and `predictions.expert`; validated against the Python registry at write time.
3. **Timestamps:** all UTC, `timestamptz`, suffix `_at` (`kickoff_at`, `cutoff_at`, `settled_at`, `captured_at`).
4. **IDs:** natural keys preferred (`espn_game_id`, `(season, team, week)`); surrogate `bigint GENERATED ALWAYS AS IDENTITY` PKs only where append-only (ledger tables).
5. **One database, one schema:** `public` in database `cfb_forecast`. Migrations are forward-only numbered SQL files under `schema/migrations/`.
6. New terms entering the codebase must be added to this glossary in the same commit — the glossary is versioned with the schema.
