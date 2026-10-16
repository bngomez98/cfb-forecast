# CLI and UI guide

How to operate cfb-forecast from the command line, the Streamlit dashboard and
the HTTP API. See [BUILD.md](BUILD.md) for installation, configuration and the
data layout. Everything here was checked against the code: the dashboard was
driven page by page with Streamlit's `AppTest` (empty and populated artifacts,
no exceptions) and the API with FastAPI's `TestClient`.

## 1. Command line

Three installed entry points (after `pip install -e .`):

| Command | Purpose |
| --- | --- |
| `cfb-forecast` | The application CLI (24 subcommands below). Same as `python -m cfb_forecast`. |
| `cfb-pipeline` | Alias of `cfb-forecast` (identical subcommands). |
| `cfb-data-watchdog` | Checks the data directory for missing or changed files. Options: `--data`, `--database`, `--watch`, `--interval` (seconds). Results are served at `GET /api/watchdog`. |

Every subcommand accepts `--help`. Relative `data/...` and `artifacts/...`
option values resolve against `CFB_DATA_DIR` / `CFB_ARTIFACTS_DIR`. Options
marked * are required. Exit status is non-zero on any failure.

### 1.1 Everyday workflow

```bash
cfb-forecast preflight --season 2026                 # check data/artifact paths
cfb-forecast sunday-run --season 2026                # fetch, train if needed, forecast, validate, publish
streamlit run dashboard/app.py                       # look at the result
```

`sunday-run` is the one command that does the whole weekly job. It is resumable
(each stage is journaled), fetches seasons 2000..season-1 plus the forecast
season, trains a model if none exists (warmup 2021, fit 2022-23, calibrate 2024,
compare 2025), writes the weekly forecast and publishes it. `--scheduled` makes
it exit quietly outside the Sunday window; `--no-refresh` skips downloads;
`--timezone` defaults to America/Chicago.

### 1.2 Commands by task

**Data**

| Command | Options | Notes |
| --- | --- | --- |
| `fetch` | `--seasons*` `--data` `--refresh` `--raw-pbp` | Downloads schedules and play summaries. `--refresh` re-downloads existing files; `--raw-pbp` also stores raw play-by-play. |
| `results-refresh` | `--season*` `--week` `--start` `--end` `--data` `--out` | Pulls live scores from ESPN's public scoreboard into `artifacts/results`. |
| `preflight` | `--data` `--artifacts` `--season` | Validates paths and files; run it first when anything looks wrong. |

**Model and forecasts**

| Command | Options | Notes |
| --- | --- | --- |
| `backtest` | `--data` `--out` `--archive` `--warmup` `--train-start` `--validation` `--test` | Chronological fit and held-out evaluation; writes `artifacts/training`. |
| `weekly-forecast` | `--season*` `--week` `--as-of` `--data` `--model` `--out` `--market` | Forecasts the next upcoming FBS slate. |
| `forecast` | `--as-of*` `--through*` `--season*` `--data` `--model` `--out` `--market` `--market-provider` | Forecasts games in an explicit window from a saved model. |
| `forecast-advance` | `--previous*` `--out*` `--as-of*` `--through*` `--season*` ... | Advances a saved forecast state when no new result became eligible. |
| `season-backtest` | `--season*` `--lead-hours` (48) `--data` `--model` `--out` | Reconstructs and grades weekly forecasts for a finished season. |
| `distribution-backtest` | `--predictions` `--models` `--out` | Scores the Gaussian score distributions on frozen predictions. |
| `probability-audit` | `--input*` `--out*` `--probability-column` `--outcome-column` `--min-games` `--regularization` `--max-change` `--stress-radius` `--model` | Calibration audit of exported predictions. |

**Markets (optional, used for comparison only)**

| Command | Options |
| --- | --- |
| `market-init` | `--store` |
| `market-import` | `--store` `--input*` `--provider` `--source` |
| `market-snapshot-sdv` | `--season*` `--store` `--observed-at` |
| `market-backfill-sdv` | `--seasons*` `--out` |

**Frozen holdouts and the chronological ledger**

| Command | Options |
| --- | --- |
| `holdout-freeze` | `--forecast*` `--data` `--model` `--out` `--batch-id` `--freeze-at` |
| `holdout-settle` / `holdout-verify` | `--holdout*` (settle also `--data`) |
| `chronology-update` | `--season*` `--ledger` `--now` `--lead-hours` `--market` `--market-provider` `--primary-lateness-minutes` `--data` `--model` |
| `chronology-verify` | `--ledger` |
| `chronology-settle` | `--ledger` `--data` `--settled-at` |
| `chronology-bootstrap` | `--frozen-batch*` `--ledger` `--lead-hours` `--primary-lateness-minutes` |
| `prospective-run` | as `chronology-update` plus `--skip-market-capture` |

**Publishing**

| Command | Options | Notes |
| --- | --- | --- |
| `publish-update` | `--run*` `--artifacts` `--database` `--run-id` `--fetch-sources` | Writes a finished forecast directory to SQLite (`forecast_log`), `artifacts/latest/forecast_board.json`, `artifacts/logs/forecast_updates.jsonl` and the email outbox. |

### 1.3 Examples

```bash
# Explicit window forecast
cfb-forecast forecast --season 2026 --as-of 2026-10-15T12:00:00Z \
    --through 2026-10-19T12:00:00Z --out artifacts/forecast

# Publish it
cfb-forecast publish-update --run artifacts/forecast

# Retrain from scratch
cfb-forecast backtest --warmup 2021 --train-start 2022 --validation 2024 --test 2025

# Freeze, then verify and settle after the games
cfb-forecast holdout-freeze --forecast artifacts/forecast
cfb-forecast holdout-verify --holdout artifacts/holdouts/<batch>
cfb-forecast holdout-settle --holdout artifacts/holdouts/<batch>

# Watch the data directory every 10 minutes
cfb-data-watchdog --watch --interval 600
```

## 2. Streamlit dashboard

```bash
pip install -r dashboard/requirements.txt     # streamlit, pandas, pyarrow
streamlit run dashboard/app.py
```

Opens on http://localhost:8501. Configuration is in `.streamlit/config.toml`
(dark theme, primary `#7dd8c0`, background `#0c111b`, usage statistics off);
extra styling is `dashboard/theme.css`.

The dashboard is read-only: it never trains, fetches or forecasts. It shows what
the CLI has already written. The sidebar lists the runtime directories in use
and a **Refresh saved data** button, which clears the page cache and re-reads
the files; it does not start a model run. Point it elsewhere with
`CFB_ARTIFACTS_DIR`, `CFB_DATA_DIR`, `CFB_MODEL_DIR`, `CFB_DATABASE`.

### 2.1 Pages

| Page | What it shows | Reads |
| --- | --- | --- |
| Overview | Run status, game count, as-of time, research-only notice | `latest/run_manifest.json`, `latest/forecast_board.json` |
| Forecast desk | Upcoming games: expected scores, home win probability, winner, confidence; team multiselect; **Download predictions** (CSV) | `latest/forecasts.csv` |
| Lines | Model numbers beside any imported market lines | `latest/forecasts.csv`, market file |
| Statistics | Aggregate tables over the forecasts | `latest/forecasts.csv` |
| Methodology | Plain-language description of the model | static |
| Forecast log | Every published forecast | `forecast_log` table / `logs/forecast_updates.jsonl` |
| Calculation pipeline | Stage status: sources, context, validation, training, probability, decisions, publication ("completed" means the stage ran, not that every provider returned data) | `latest/pipeline.json` |
| Historical training | Verified training inputs and metrics | `training/training_inputs.json`, `training/metrics.csv`, `training/winner_model_selection.json` |
| Weekly validation | Previous week's predictions graded against results, plus calibration | `latest/previous_week_validation.json`, `runs/<id>/validation.csv`, `validation_calibration.csv` |
| Game inputs | Per-game inputs and provider status | `latest/inputs/{game_inputs.json,features.csv,provider_status.json}` |
| Model research | Model selection and context-model comparison | `latest/context_model_selection.json`, training metrics |
| Sources | Historical download manifest | `data/source_manifest.json`, `team_game_facts` / `player_game_facts` |

All paths are under `artifacts/` unless prefixed `data/`.

### 2.2 Empty states

With nothing published, every page renders an explanatory message instead of
failing, for example "No forecast log has been published yet", "No verified
training-input report has been saved yet", "There are no prior saved runs to
validate". Running `sunday-run` (or `weekly-forecast` then `publish-update`)
populates them; press **Refresh saved data**.

## 3. HTTP API

```bash
pip install -e '.[api]'
python -m cfb_forecast.api            # PORT env var, default 8000; CORS open
```

| Endpoint | Behaviour |
| --- | --- |
| `GET /health` | Always 200; reports whether a model is available and the database path. |
| `GET /ready` | 200 when model and database exist, else 503. |
| `GET /awareness` | 200 only when all data checks pass, else 503 with per-check detail. |
| `GET /api/model` | Model manifest; 404 if no model has been trained. |
| `POST /api/forecasts` | Body `{"season": int>=2000, "as_of": ISO time, "through": ISO time}`. Window must be >0 and at most 14 days and the model must predate the season. `model_path` / `market_path` are rejected. Output goes to `artifacts/api-forecasts/<uuid>`. 404 without a model. |
| `GET /api/forecasts?season=` | Published forecasts for a season (empty list if none). |
| `GET /api/learning` | Continual-learning version history. |
| `GET /api/pipeline` | Latest pipeline run; 404 if none. |
| `GET /api/watchdog` | Latest watchdog check; 404 if none. |

Interactive docs are at `/docs` while the server runs.

## 4. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Dashboard pages all say "no ... published" | Run `cfb-forecast sunday-run --season <year>`, then **Refresh saved data**. |
| Dashboard shows an old run | Check `CFB_ARTIFACTS_DIR` is the directory the CLI wrote to (shown in the sidebar). |
| `/ready` returns 503 | No model: run `cfb-forecast backtest` (or `sunday-run`), and confirm `CFB_DATABASE` exists. |
| `/awareness` returns 503 | Run `cfb-forecast fetch --seasons ...`, then inspect the response's `checks`. |
| CLI cannot find files | `cfb-forecast preflight --season <year>` names the missing path. |
