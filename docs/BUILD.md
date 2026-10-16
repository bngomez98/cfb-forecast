# Build documentation

How to build, configure, run, test and operate `cfb-forecast`. The repository is Python, databases
and git; nothing else is required.

Contents: [1 Requirements](#1-requirements) · [2 Build](#2-build) · [3 Repository layout](#3-repository-layout) ·
[4 Configuration](#4-configuration) · [5 Data](#5-data) · [6 Database](#6-database) · [7 Running](#7-running) ·
[8 Command reference](#8-command-reference) · [9 Outputs](#9-outputs) · [10 Scheduling](#10-scheduling) ·
[11 Tests](#11-tests) · [12 Troubleshooting](#12-troubleshooting) · [13 Maintaining](#13-maintaining)

The statistical method is described in [ARCHITECTURE.md](../ARCHITECTURE.md) and the long-form
specification in the [README](../README.md). This document covers the build and operation.

---

## 1. Requirements

| Item | Requirement |
| --- | --- |
| Python | 3.11 or newer (verified on 3.13) |
| Runtime packages | `numpy>=1.26`, `pandas>=2.2,<3`, `scipy>=1.12`, `scikit-learn>=1.4`, `pyarrow>=15,<26` |
| Tested pins | `requirements-tested.txt`: numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, scikit-learn 1.8.0, pyarrow 25.0.1, pytest 9.1.1 |
| Optional: API | `fastapi>=0.115`, `uvicorn>=0.30` (extra `api`) |
| Optional: dashboard | `streamlit>=1.40,<2` (`dashboard/requirements.txt`) |
| Test extras | `pytest>=8`, `httpx>=0.27`, `fastapi>=0.115` (extra `test`) |
| Database | SQLite (Python's built-in `sqlite3`); no server |
| Network | HTTPS access to the public sources in section 5 when fetching data |
| Disk | Several GB for the 2000 to present schedules, team summaries and play-by-play (the first `sunday-run` downloads them all) |

`pandas` 3 is not supported (`pyproject.toml` pins `<3`).

## 2. Build

```bash
git clone <repository> cfb-forecast
cd cfb-forecast
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements-tested.txt
python -m pip install -e '.[test]'   # add ',api' for the API
```

Verify the build:

```bash
python -m pytest -q                  # 88 passed
python -m cfb_forecast --help        # lists the commands in section 8
```

This procedure was verified in a clean virtual environment: install, `88 passed`, CLI loads.

`pip install -e .` registers three console commands (`[project.scripts]` in `pyproject.toml`):

| Command | Entry point | Purpose |
| --- | --- | --- |
| `cfb-forecast` | `cfb_forecast.cli:main` | Every command in section 8 |
| `cfb-pipeline` | `cfb_forecast.pipeline_cli:main` | Alias of `cfb-forecast` |
| `cfb-data-watchdog` | `cfb_forecast.data_watchdog:main` | Checksum and integrity watchdog for `data/` |

`python -m cfb_forecast ...` is equivalent to `cfb-forecast ...`.

There is no compile step and no generated code. A "build" is the editable install plus a green test run.
`python -m build` produces a wheel and sdist from `pyproject.toml` (setuptools backend) if one is needed.

## 3. Repository layout

```text
src/cfb_forecast/     the Python package
tests/                pytest suite (88 tests)
sql/schema.sql        the SQLite schema, applied on every connect
dashboard/            optional Streamlit forecast desk (app.py, desk.py, theme.css)
.streamlit/           dashboard configuration
examples/             market_template.csv, the market-history file format
pyproject.toml        package metadata, entry points, pytest settings
requirements-tested.txt   exact versions the suite was verified against
README.md             long-form specification and method
ARCHITECTURE.md       design notes
RESULTS.md            recorded results
docs/BUILD.md         this document
docs/CLI_AND_UI.md     CLI, dashboard and API guide
data/                 runtime: downloaded sources (created on first run, not in git)
artifacts/            runtime: runs, models, logs and the database (not in git)
```

### Package modules

| Layer | Modules | Role |
| --- | --- | --- |
| Entry points | `cli.py`, `__main__.py`, `pipeline_cli.py`, `external_cli.py`, `api.py` | Command line and the optional FastAPI service |
| Paths and config | `paths.py`, `config.py`, `preflight.py` | Runtime locations, state configuration, mounted-path checks |
| Data intake | `data.py`, `sources.py`, `external_sources.py`, `results.py`, `market.py`, `availability.py` | Schedules and team summaries, public-source collection with receipts, live ESPN scores, timestamped market history, per-cutoff data-availability receipts |
| Features | `features.py`, `ratings.py` | Past-game observations, optional raw-play cleaning, opponent-adjusted team states replayed from prior results |
| Model | `model.py`, `forecast_model.py`, `layered_model.py`, `strength.py`, `baseline.py`, `context_model.py`, `learning.py`, `probability.py`, `gaussian.py` | Score heads, the forecasting model, comparison models, context learning, continual reinforcement, probability calibration, Gaussian helpers |
| Forecast | `forecast.py`, `weekly.py`, `sunday.py`, `analytic.py`, `historical.py`, `chronology.py`, `collector.py` | One-game-window forecasts, the weekly slate, the Sunday service, closed-form spread/total/win probabilities, retraining, the append-only prospective ledger |
| Validation | `validation.py`, `evaluate.py`, `distribution.py`, `holdout.py`, `backtest.py`, `season_backtest.py`, `training_audit.py` | Settle saved predictions, metrics, Gaussian CRPS, frozen holdouts, backtests |
| Decisions | `betting.py`, `decision.py` | Moneyline pricing and hypothetical exposure, separate from probabilities |
| Storage and publishing | `database.py`, `football_database.py`, `watchdog.py`, `data_watchdog.py`, `publisher.py`, `awareness.py`, `pipeline_status.py` | SQLite persistence, integrity watchdogs, publishing the board, status journal |

## 4. Configuration

Configuration is by environment variable; there are no config files. Relative paths resolve against the
project root, not the working directory.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CFB_PROJECT_ROOT` | the checkout (two levels above `paths.py`) | Base for relative paths |
| `CFB_DATA_DIR` | `data` | Downloaded sources and market history |
| `CFB_ARTIFACTS_DIR` | `artifacts` | Runs, models, logs, database |
| `CFB_MODEL_DIR` | `artifacts/training/model` | The production model directory |
| `CFB_DATABASE` | `artifacts/cfb_forecast.sqlite` | SQLite database |
| `CFB_MARKET_FILE` | `data/market_history.csv` | Market snapshot store |
| `CFBD_API_KEY` | unset | Enables CollegeFootballData collection; never commit it |
| `CFB_SMTP_HOST`, `CFB_SMTP_PORT` (587), `CFB_SMTP_USER`, `CFB_SMTP_PASSWORD`, `CFB_SMTP_FROM` | unset | Forecast email; without host and password the message is saved to `artifacts/logs/email_outbox/` |
| `CFB_FORECAST_EMAIL_TO` | `bngomez@ieee.org` | Email recipient |
| `PORT` | `8000` | API port (`python -m cfb_forecast.api`) |
| `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` | system | Set to `1` for reproducible numerics |

`--data`, `--out`, `--artifacts` and `--model` command options override the corresponding directories for one run.

## 5. Data

Everything is collected from public sources; nothing is generated. Every retrieval keeps its URL,
retrieval time, SHA-256 and bytes; failures are recorded rather than replaced.

| Source | Used for | Access |
| --- | --- | --- |
| SportsDataverse releases (`sportsdataverse-data`) | Schedules, per-game advanced team summaries, optional play-by-play | Public download |
| ESPN public scoreboard and game summaries | Live scores, weather, injuries, depth charts, odds as displayed | Public API |
| CollegeFootballData | Lines, weather, team and player statistics, ratings | `CFBD_API_KEY` |
| FootballDB, CFBTrack | Independent team and player statistic tables | Public pages |
| Open-Meteo | Venue weather when ESPN has none | Public API |
| cfbtxt.com, mrtimo | Weekly schedule validation | Public pages |

Files written under `data/` by `fetch`:

| File | Content |
| --- | --- |
| `schedule_<season>.csv` | Games, teams, scores, kickoff, division flags |
| `adv_team_<season>.parquet` | Per-game team rates (yards per play, success, explosiveness) |
| `pbp_<season>.parquet` | Play-by-play, only with `--raw-pbp` |
| `source_manifest.json` | URL, SHA-256, download time and size of every file |
| `market_history.csv` | Timestamped spread/total/moneyline snapshots (format: `examples/market_template.csv`) |
| `market_closing_history.csv` | Benchmark-only closing lines (`market-backfill-sdv`) |

Rules enforced on load (`data.normalize_schedule`): unique game ids, distinct home and away teams,
valid booleans, usable results only when both scores are present, and result availability set to the
recorded completion time or kickoff plus eight hours. A source that marks a future game completed stops the
run. Games with an NA score are quarantined, not repaired.

## 6. Database

SQLite, WAL mode, created on first connect at `CFB_DATABASE`. `database.connect` applies
`sql/schema.sql` (falling back to the copy embedded in `database.py` if the file is not beside the
package) and adds columns that older databases lack. Learning tables are created by `learning.py`.

| Table | Contents |
| --- | --- |
| `runs`, `models`, `forecasts`, `results` | Run ledger, fitted models, per-game forecast rows with betting fields, final results |
| `forecast_log` | The published board: one row per game and run (teams, expected scores, winner, confidence, spread and moneyline fields) |
| `forecast_runs`, `statistical_parameter_sets` | Run provenance and parameter sets with cutoff, source manifest and uncertainty |
| `games`, `game_teams`, `teams` | Normalized schedule |
| `source_registry`, `source_retrievals`, `source_files`, `sources` | Source catalog and every retrieval with hash, status and cutoff |
| `team_game_facts`, `player_game_facts` | Per-game team and player rates read back by the forecast |
| `weekly_validation` | Weekly scoring of saved forecasts (log loss, Brier, margin and spread error) |
| `learning_versions`, `training_examples` | Continual-reinforcement versions and the exact examples used |
| `data_files`, `watchdog_events` | Checksums of `data/` and integrity events |

Inspect it with any SQLite client, for example `sqlite3 artifacts/cfb_forecast.sqlite ".tables"`.

## 7. Running

### First run

```bash
python -m cfb_forecast preflight --season 2026
python -m cfb_forecast sunday-run --season 2026
```

`sunday-run` downloads every missing season from 2000 up to the season before the forecast season
(seasons already on disk are kept), always refreshes the forecast season and its play-by-play, trains
the production model from earlier seasons if none exists (warmup 2021, fit 2022 to 2023, calibrate 2024,
compare 2025), validates previously saved predictions against finals, and writes the coming slate. The first
run is therefore the longest and the largest download; later runs refresh only the current season.

To download or refresh specific seasons yourself, for example before running offline:

```bash
python -m cfb_forecast fetch --seasons 2021 2022 2023 2024 2025 2026 --raw-pbp
```

### Each week

```bash
python -m cfb_forecast sunday-run --season 2026 --scheduled
```

`--scheduled` makes the run exit unless it is 10:00 America/Chicago (see section 10). A completed
run is never overwritten; a failed run is left in `artifacts/work/` and can be retried.

### Other workflows

| Goal | Command |
| --- | --- |
| Forecast the next week without the Sunday service | `weekly-forecast --season 2026` (add `--week N`, `--as-of ISO`) |
| Refresh live scores | `results-refresh --season 2026` |
| Republish an existing run | `publish-update --run artifacts/weekly/2026-week-7 --run-id 2026-week-7` |
| Retrain and compare models | `backtest` |
| Grade a whole season week by week | `season-backtest --season 2025` |
| Freeze and later settle a batch | `holdout-freeze`, `holdout-verify`, `holdout-settle` |
| Append-only prospective ledger | `chronology-update`, `chronology-verify`, `chronology-settle`, `prospective-run` |
| Check calibration of exported predictions | `probability-audit --input FILE --out DIR` |
| Watch `data/` for changes | `cfb-data-watchdog --watch` |
| Serve the API | `python -m cfb_forecast.api` (needs the `api` extra) |
| Dashboard | `streamlit run dashboard/app.py` (needs `dashboard/requirements.txt`) |

## 8. Command reference

`python -m cfb_forecast <command> --help` is authoritative. Options marked * are required.

| Command | Options (defaults) |
| --- | --- |
| `fetch` | `--seasons*`, `--data` (data), `--refresh`, `--raw-pbp` |
| `backtest` | `--data`, `--out` (artifacts/training), `--archive`, `--warmup` (2021), `--train-start` (2022), `--validation` (2024), `--test` (2025) |
| `forecast` | `--as-of*`, `--through*`, `--season*`, `--data`, `--model`, `--out` (artifacts/forecast), `--market`, `--market-provider` |
| `forecast-advance` | `--previous*`, `--out*`, `--as-of*`, `--through*`, `--season*`, `--data`, `--model`, `--market`, `--market-provider` |
| `weekly-forecast` | `--season*`, `--week`, `--as-of`, `--data`, `--model`, `--out` (artifacts), `--market` |
| `sunday-run` | `--season*`, `--scheduled`, `--no-refresh`, `--timezone` (America/Chicago), `--data`, `--model`, `--out` (artifacts) |
| `results-refresh` | `--season*`, `--week`, `--start`, `--end`, `--data`, `--out` (artifacts/results) |
| `season-backtest` | `--season*`, `--lead-hours` (48), `--data`, `--model`, `--out` (artifacts/backtest) |
| `distribution-backtest` | `--predictions`, `--models`, `--out` (all under artifacts/training) |
| `holdout-freeze` | `--forecast*`, `--data`, `--model`, `--out` (artifacts/holdouts), `--batch-id`, `--freeze-at` |
| `holdout-settle` | `--holdout*`, `--data` |
| `holdout-verify` | `--holdout*` |
| `chronology-update` | `--season*`, `--ledger` (artifacts/chronological_holdout), `--now`, `--lead-hours` (24), `--market`, `--market-provider`, `--primary-lateness-minutes` (60), `--data`, `--model` |
| `chronology-verify` | `--ledger` |
| `chronology-settle` | `--ledger`, `--data`, `--settled-at` |
| `chronology-bootstrap` | `--frozen-batch*`, `--ledger`, `--lead-hours`, `--primary-lateness-minutes` |
| `prospective-run` | as `chronology-update` plus `--skip-market-capture`; `--market` defaults to data/market_history.csv |
| `market-init` / `market-import` | `--store`; import also `--input*`, `--provider`, `--source` |
| `market-snapshot-sdv` | `--season*`, `--store`, `--observed-at` |
| `market-backfill-sdv` | `--seasons*`, `--out` (data/market_closing_history.csv) |
| `preflight` | `--data`, `--artifacts`, `--season` |
| `probability-audit` | `--input*`, `--out*`, `--probability-column` (score_p), `--outcome-column` (home_win), `--min-games` (100), `--regularization` (0.01), `--max-change`, `--stress-radius` (0.1), `--model` |
| `publish-update` | `--run*`, `--artifacts`, `--database`, `--run-id`, `--fetch-sources` |

Relative `data/...` and `artifacts/...` option values resolve against the configured directories (section 4).

## 9. Outputs

```text
data/                               sources and market history (section 5)
artifacts/
  cfb_forecast.sqlite               the database (section 6)
  training/                         backtest outputs and the production model (model/, models/, predictions.csv)
  runs/<season-date>/               one Sunday run: forecasts.csv/json, inputs/, run_manifest.json,
                                    validation.csv/json
  latest/                           most recent completed run, read by the dashboard
  weekly/<season>-week-N/           weekly-forecast output; latest-weekly/ is a copy of the newest
  results/                          live score pulls
  backtest/                         season-backtest output
  holdouts/, chronological_holdout/ frozen batches and the prospective ledger
  learning/                         reinforcement versions
  logs/                             forecast_updates.jsonl, email_outbox/, source_ingest.json
  work/                             a failed run, kept for retry
```

A completed run publishes one board (`publisher.publish_forecast_update`) to the `forecast_log` table,
`artifacts/latest/forecast_board.json`, `artifacts/logs/forecast_updates.jsonl` and the email outbox. Each
board row has kickoff, teams, expected score, winner and confidence. Runs record the SHA-256 of their
inputs, model and predictions in `run_manifest.json`; `holdout-verify` and `chronology-verify` re-check them.

## 10. Scheduling

Any scheduler works because the service decides whether it is the right time. Schedule it at both
candidate UTC hours (15:00 and 16:00 on Sundays); with `--scheduled` only the one that is 10:00
America/Chicago executes, so daylight-saving changes and duplicate triggers are harmless.

```cron
0 15,16 * * 0  cd /path/to/cfb-forecast && .venv/bin/python -m cfb_forecast sunday-run --season 2026 --scheduled
```

Keep `data/` and `artifacts/` between runs; the next run reads the model, prior predictions and database
from them. Back them up before moving the project.

## 11. Tests

```bash
python -m pytest -q            # whole suite, 88 tests
python -m pytest tests/test_forecast_model.py -q
```

Configured in `pyproject.toml` (`testpaths=tests`, `pythonpath=src`, temp directory `.pytest_tmp`). Tests use
small explicit fixtures and never reach the network.

| File | Covers |
| --- | --- |
| `test_engine.py` | Score heads, ratings, market cutoffs, Gaussian win probability, analytic spread/total, CRPS, covariance validation |
| `test_forecast_model.py` | Forecasting model, pregame history, rest days, known-future inputs |
| `test_live_service.py` | The Sunday service order: inputs before validation before inference, run preservation |
| `test_learning.py` | Continual reinforcement, temporal evidence, independent metric check |
| `test_layered_model.py` | Layered empirical-Bayes model |
| `test_pipeline_wiring.py`, `test_paths.py` | Runtime paths, journal, audits |
| `test_publisher.py`, `test_football_database.py` | Publishing the board, database persistence |
| `test_external_sources.py`, `test_data_watchdog.py`, `test_watchdog.py` | Source collection and integrity watchdogs |

## 12. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `No schedule files in data` | Run `fetch` first, or point `--data` / `CFB_DATA_DIR` at the folder holding `schedule_*.csv` |
| `Model training must end before the forecast season` | The model was fitted through the forecast season; retrain with `backtest` (test season must be earlier) |
| `A source marks future games completed` | The schedule source is inconsistent; refetch with `fetch --refresh` and retry |
| `Historical warmup season missing` | The warmup season files are missing; fetch the earlier seasons |
| A run sits in `artifacts/work/` | It failed; read the journal in that folder, fix the cause and rerun (completed runs are never overwritten) |
| Run exits immediately with `outside_sunday_schedule` | Expected with `--scheduled` outside 10:00 America/Chicago |
| Another run is active | A lock file prevents concurrent writers; it releases when the process exits |
| No email arrives | SMTP variables are unset; the message is in `artifacts/logs/email_outbox/` |
| CollegeFootballData sources report not configured | Set `CFBD_API_KEY` in the environment |
| `pandas` import or dtype errors | Use `pandas>=2.2,<3` as pinned |

## 13. Maintaining

- **Versioning:** the version is in `pyproject.toml` (`0.4.0`). Model files record their warmup and training seasons.
- **Schema changes:** edit `sql/schema.sql` and keep the embedded `SCHEMA` in `database.py` equal to it; for a new column on an existing table add it to `_ensure_forecast_columns` style migration so old databases upgrade.
- **Adding a source:** register its URL in `sources.SOURCE_REGISTRY`, collect it with `sources.collect` so a receipt is kept, and add a test with a recorded response.
- **Adding a command:** add a subparser and branch in `cli.py`; path options such as `--data`, `--out`, `--model` are resolved against the configured directories by `paths.cli_path` (add a new one to the list in `cli.main`).
- **Before committing:** `python -m pytest -q` must pass. Runtime directories `data/` and `artifacts/` are git-ignored and must stay out of the repository.
- **Reproducibility:** pin the interpreter and the versions in `requirements-tested.txt`; set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
