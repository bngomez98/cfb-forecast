# FieldQuant FBS Forecast Desk

A real-data forecasting application for the 2026 FBS season. Each Sunday at 10 AM America/Chicago it gathers game information, validates previously saved predictions, learns from new results, and calculates forecasts for the coming Monday–Sunday slate. Every day it re-forecasts every remaining upcoming game from the current and historical data and reports what moved. A built-in scheduler, an API, and a dashboard make the whole thing run by itself under Docker Compose. Predictions, actual results, and hypothetical betting returns remain separate records.

## Actual project configuration

| Field | Value |
|---|---|
| Project root | `/mnt/c/Users/brian/cfb-forecast` |
| Python virtual environment | `/home/brian/.venvs/cfb-forecast` |
| Windows environment | `C:\Users\brian\miniforge3\envs\cfb-python` |
| Python version | `3.12.3` |
| pip version | `26.2.1` |
| Package name | `cfb-forecast` |
| Package version | `0.4.0` |
| Python source | `/mnt/c/Users/brian/cfb-forecast/src/cfb_forecast` |
| Runtime data directory | `/mnt/c/Users/brian/cfb-forecast/data` |
| Runtime artifacts directory | `/mnt/c/Users/brian/cfb-forecast/artifacts` |
| Docker application directory | `/app` |
| Main Docker image | `cfb-forecast:local` |
| Dashboard Docker image | `cfb-forecast-dashboard:local` |
| Docker base image | `python:3.12-slim` |
| Compose service | `cfb-forecast` |
| Dashboard service | `cfb-dashboard` |
| Dashboard port | `8501` |
| Dashboard address | `http://localhost:8501` |
| Compose network | `cfb-forecast_default` |
| Forecast container UID | `10001` |
| Forecast container GID | `10001` |
| Dashboard container UID and GID | `10002` |
| API service port | `8000` |

The editable install is the git checkout. On Linux that environment is `/home/brian/.venvs/cfb-forecast`. On Windows it is `C:\Users\brian\miniforge3\envs\cfb-python`.

## Architecture, pipelines, workflows, and sources

Forecasts come from one model. A hierarchical team state sets the score location. Pregame team and player rates from the parquet seasons on disk (2000 through 2025, plus completed games after that) and known-future information (market line, total, rest, weather, and reported injuries) shift that location together. Completed games teach that shift with their real closing spread and total, downloaded from the SportsDataverse ESPN betting files into `data/market_closing_history.csv`; an upcoming game uses only a line captured before the forecast time. A game with no captured line uses a second fit that never saw lines, so a missing line is not filled with an average one. Weather and injuries have no downloadable history, so they enter only through the context model below once enough recorded weeks exist. The same location is the Gaussian mean and the Poisson rate. The win probability is the logistic of the margin. Spread and total chances are calculated directly from that Gaussian; no scores are simulated. Those rates are stored in `team_game_facts` and `player_game_facts`, and the next weekly forecast reads them back into the same hierarchical state, Poisson rates, Gaussian scores, and logistic win probability. The prediction board shows kickoff, teams, expected score, winner, and confidence. Spreads and moneylines, team and player statistics, and method notes are separate pages. Weather comes from the ESPN game summary, and from Open-Meteo for the venue when that summary has none. CollegeFootballData lines, weather, team stats, and player stats are collected when `CFBD_API_KEY` is set. FootballDB college offense, defense, scoring, and player pages are collected with the other sources. cfbtrack.com supplies current team and player season stats. A saved model keeps the warmup season it was fit with. Missing history is left missing; it is not replaced with a made-up season.

```mermaid
flowchart TD
  sources[Public source catalog] --> ingest[ingest_catalog]
  schedules[Schedules and play-by-play] --> sunday[Sunday and weekly calculation]
  ingest --> sqlite[(artifacts/cfb_forecast.sqlite)]
  sunday --> head[Forecasting model]
  head --> board[Scores, winner, confidence]
  board --> log[forecast_log]
  log --> sqlite
  log --> react[React board and Vercel]
  log --> streamlit[Streamlit forecast log]
  log --> datadog[artifacts/logs/datadog.jsonl]
  log --> mail[Email to bngomez@ieee.org]
```

```mermaid
flowchart LR
  refresh[Refresh schedule and results] --> validate[Validate saved predictions]
  validate --> features[Replay prior results into team state]
  features --> scores[Expected scores and win probability]
  scores --> publish[publish_forecast_update]
  publish --> sqlite[(SQLite)]
  publish --> apps[React, Streamlit, Datadog, email]
```

| Workflow | What it runs |
|---|---|
| Application | `cfb-forecast app` (or the `scheduler` and `cfb-dashboard` Compose services) runs the scheduler and the dashboard together. |
| Sunday service | `cfb-forecast sunday-run`, run by the scheduler on Sundays from 10:00 America/Chicago. It validates the previous slate, learns, and writes the next one. |
| Daily forecast | `cfb-forecast pipeline forecast`, run by the scheduler every day in season. It refreshes data, results and player ratings, then forecasts every remaining upcoming game and writes `changes.csv`. |
| Weekly forecast | `run_weekly_forecast` calculates one week and publishes that board. |
| Publish update | `python -m cfb_forecast publish-update --run DIR --run-id ID` stores an existing calculation in SQLite, the apps, the Datadog log, and the email outbox. |
| Source collection | `ingest_catalog` fetches the catalog and keeps each response or failure receipt; `cfb-forecast sources-collect` reads the registered web sites. |
| Data watchdog | `cfb-data-watchdog` records file checksums in `data_files` and `watchdog_events`. |
| Probability audit | `python -m cfb_forecast probability-audit` checks calibration on exported predictions without touching the production model. |
| GitHub Actions | Not active. The earlier workflow files are kept in `.github/workflows.disabled/`; move them back to `.github/workflows/` to enable tests, image builds, source collection and the Sunday run on GitHub. |

The SQLite file is `artifacts/cfb_forecast.sqlite`. `sql/schema.sql` is the schema applied on connect. The published board is `forecast_log`. Learning rows are `learning_versions` and `training_examples`. Source provenance is `source_registry`, `source_retrievals`, and `source_files`.

Datadog reads `artifacts/logs/datadog.jsonl` through `datadog/conf.yaml`. Intake runs only when `DD_API_KEY` is set. Email is sent only when `CFB_SMTP_HOST` and `CFB_SMTP_PASSWORD` are set.

| Source | URL |
|---|---|
| cfbfastR | `https://cfbfastr.sportsdataverse.org/` |
| cfbfastR models | `https://github.com/sportsdataverse/cfbfastR-cfb-data/blob/main/models/REGISTRY.md` |
| SportsDataverse | `https://github.com/sportsdataverse/sportsdataverse-data` |
| radsportsanalytics | `https://radsportsanalytics.com/blog/introduction-to-cfb-analytics/` |
| cfbtrack | `https://cfbtrack.com/` |
| cfbtracker | `https://cfbtracker.com/` |
| cfbresults | `https://cfbresults.com/` |
| college-football-results | `https://college-football-results.com/` |
| collegefootballdb | `https://collegefootballdb.com/` |
| cfbstats | `https://cfbstats.com/` |
| collegefootballdata | `https://collegefootballdata.com/` |
| mrtimo schedule | `https://mrtimo.github.io/cfb-games/games-{season}.html` |
| CollegeFootballData key | `https://collegefootballdata.com/key` |
| CollegeFootballData ratings | `https://api.collegefootballdata.com/ratings/core` |
| cfbtxt validation | `https://cfbtxt.com/schedule/` |
| FanDuel comparison | `https://www.fanduel.com/research/ncaaf` |
| FootballDB team stats | `https://www.footballdb.com/college-football/stats/teamstat.html?yr={season}&lg=FBS` |
| FootballDB stats index | `https://www.footballdb.com/college-football/stats/index.html` |
| FootballDB player stats | `https://www.footballdb.com/college-football/stats/stats.html?lg=FBS&yr={season}` |
| Open-Meteo | `https://api.open-meteo.com/v1/forecast` |
| ESPN scoreboard | `https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard` |

## How the application runs

```mermaid
flowchart LR
  subgraph scheduler[Scheduler: cfb-forecast scheduler]
    sun[Sunday 10:00 Chicago] --> sunday[sunday-run]
    day[Every day in season] --> daily[pipeline forecast]
    tue[Tuesday] --> bt[season backtest]
  end
  daily --> data[Refresh data] --> players[Player ratings] --> market[Market capture] --> results[Final scores] --> web[Web sources]
  web --> fc[Live-refit forecast] --> publish[publish_forecast_update]
  sunday --> publish
  publish --> latest[(artifacts/latest, SQLite, web board, logs, email)]
  latest --> dash[Dashboard and API]
```

**One model.** The Sunday service and the daily pipeline call the same `forecast()`. It replays every result available before the cutoff into the Kalman team state, refits the ridge score head on those results (with exponential reinforcement decay), maps known-future information onto the score location, and returns Gaussian scores with Poisson rates, a logistic win probability, and analytic spread and total probabilities. No scores are simulated.

**Player layer.** The selected variant, `pbp_player`, adds a hierarchical per-play player model to the team state (`players.py`): each player's yards per play is shrunk toward the team and then the league, adjusted for the defense faced, recency weighted, and rolled up to a team offense rating by recent usage share, with a matching defense rating. Ratings are refit point in time, once per week of history, and cached in `artifacts/cache/`. `cfb-forecast player-aggregates --seasons 2026` rebuilds the per-player aggregates from raw play-by-play; the daily pipeline does this for the current season. Each daily run writes named `player_ratings.csv` and `team_ratings.csv`.

**Daily changes.** Each daily run is a dated snapshot in `artifacts/daily/forecasts/` (the newest 14 are kept). `changes.csv` lists each game's win probability, margin and total against the previous snapshot. The dashboard shows the largest moves under Forecast desk.

**Web data.** `cfb-forecast sources-collect` reads the registered public web pages (`src/cfb_forecast/webdata/registry.py`: results, picks, lines and team statistics), interprets pages, JSON and PDFs, cross-checks final scores between sources and against the schedule, and writes `data/sources/`. Results that two or more sources agree on enter `data/results_overlay.csv`, with the time they were observed as their availability time, so they cannot enter any earlier forecast. External picks and lines are reference data; the market informs the forecast but never decides it.

**The database loop.** Data moves through SQLite in a closed loop (`dbflow.py`, stage `database` of the daily pipeline):

```mermaid
flowchart LR
  src[Sources: schedule, ESPN, cfbtrack and other web pages] --> db1[(SQLite: results, external_results, team and player facts)]
  db1 --> re[Python recomputes from stored rows]
  re --> db2[(db_validation)]
  db1 --> py[Python forecast model reads the stored history]
  py --> log[(forecast_log)]
  log --> val[Validate each logged pregame forecast against stored finals]
  val --> db1
```

1. Outcomes (schedule finals plus any final that two or more web sources agree on, such as cfbtrack.com) and the source cross-check are written to `results` and `external_results`.
2. Every pregame forecast ever logged to `forecast_log`, Sunday or daily, is joined to those outcomes into `forecast_validation` (forecasts logged after kickoff never count) and summarised in `weekly_validation`.
3. Python reads `forecast_validation` back out of SQLite, recomputes log loss, Brier score and margin error independently, stores them in `db_validation`, and refuses any disagreement with the metrics computed at validation time. Nothing here needs R.
4. The next forecast reads the stored team and player history from the same database and logs its output, which the next run validates.

**Scheduler.** Times are UTC unless noted. Outside the season nothing is due.

| Job | When | Action |
|---|---|---|
| `sunday` | Sunday from `CFB_SUNDAY_HOUR` (10) in `CFB_SUNDAY_TIMEZONE` (America/Chicago) | `sunday-run` |
| `forecast` | every `CFB_FORECAST_HOURS` (24), and at `CFB_FORECAST_HOUR` (9) | `pipeline forecast` |
| `backtest` | `CFB_BACKTEST_DAY` (tue) at `CFB_BACKTEST_HOUR` (11) | `pipeline backtest` |

```bash
cfb-forecast scheduler --status          # each job, whether it is due, last status
cfb-forecast scheduler --run forecast    # run one job now
cfb-forecast pipeline forecast --season 2026
docker compose up -d scheduler cfb-api cfb-dashboard
```

The production model lives in `artifacts/training/model` (`CFB_MODEL_DIR`). The first run on an empty volume trains it from the seasons on disk.

## Local commands

With the package installed (`make install`), everything runs through `cfb-forecast`; the Make targets wrap the same commands (`make help` lists them).

| Task | Command | Make |
|---|---|---|
| Whole application (scheduler + dashboard) | `cfb-forecast app` | `make app` |
| Scheduler only / status / force a job | `cfb-forecast scheduler`, `--status`, `--run forecast` | `make scheduler`, `make scheduler-status` |
| Sunday service now | `cfb-forecast sunday-run --season 2026` | `make sunday` |
| Daily pipeline now | `cfb-forecast pipeline forecast --season 2026 --refresh-data` | `make forecast` |
| Final scores | `cfb-forecast pipeline results --season 2026` | `make results` |
| Web sources | `cfb-forecast sources-collect --season 2026` | `make sources` |
| Player aggregates | `cfb-forecast player-aggregates --seasons 2026 --refresh` | `make players` |
| Database validation | `python -c "from cfb_forecast import dbflow, paths; dbflow.run(paths.database_path(), paths.data_root(), 2026)"` | `make database` |
| Season backtest | `cfb-forecast pipeline backtest --season 2026` | `make backtest-season` |
| Reproduce the 2025 held-out numbers | `cfb-forecast backtest --data data --out artifacts/reproduced` | `make reproduce` |
| Dashboard / API | `streamlit run dashboard/app.py`, `python -m uvicorn cfb_forecast.api:app` | `make dashboard`, `make api` |
| Lint and tests | `ruff check src tests dashboard`, `python -m pytest -q` | `make check` |

Docker: `make up` starts the scheduler, API and dashboard; `make docker-forecast` and `make docker-sunday` run one job; `python -m scripts.cfb daily|sunday|results|sources|backtest` does the same from the host.

## Run with the established Linux environment

From Ubuntu WSL:

```bash
cd /mnt/c/Users/brian/cfb-forecast
source /home/brian/.venvs/cfb-forecast/bin/activate
```

Verify the active interpreter:

```bash
command -v python
python --version
python -m pip --version
```

Expected values:

```text
/home/brian/.venvs/cfb-forecast/bin/python
Python 3.12.3
pip 26.2.1
```

Install the project into the established environment:

```bash
python -m pip install -r requirements-tested.txt
python -m pip install -e '.[test]'
```

Run the forecasting service:

```bash
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
python -m cfb_forecast sunday-run --season 2026
```

Run the dashboard directly from Linux:

```bash
python -m pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

The dashboard is available at:

```text
http://localhost:8501
```

## Alternative runtime paths

The project can use separate runtime directories while preserving the same source tree:

```bash
python -m cfb_forecast sunday-run \
  --season 2026 \
  --data runtime/data \
  --artifacts runtime/artifacts
```

Run the dashboard against custom artifacts:

```bash
CFB_ARTIFACTS_DIR=runtime/artifacts \
streamlit run dashboard/app.py
```

The first run downloads real SportsDataverse schedules and advanced team summaries for every season from 2000 through the season before the forecast season, then refreshes the forecast season and its play-by-play. It trains models from earlier seasons, measures historical performance, and calculates the coming week's slate.

Later runs reuse the historical seasons already on disk and the trained models, and always refresh the current season.

No sample data or canned performance numbers are included in the application. Downloaded data, trained models, and generated reports are runtime files excluded from source control.

The service preserves one completed forecast per slate. Running it twice does not overwrite earlier predictions. A failed run remains under:

```text
artifacts/work/
```

A failed run can be retried. Operating-system locking prevents concurrent writers and releases automatically if the process exits.

## Calculation order

1. Refresh the season schedule, advanced statistics, and authoritative final results.
2. Verify game IDs and team IDs.
3. Reject future final outcomes.
4. Select upcoming FBS-versus-FBS games in the coming calendar week.
5. Exclude FBS–FCS games from predictions and evaluation.
6. Retrieve weather, injuries, depth charts, rosters, player leaders, team statistics, and available sportsbook quotes.
7. Preserve response bytes, source URLs, retrieval times, and checksums before inference.
8. Validate previously saved predictions against completed results.
9. Score probability quality, winner accuracy, score errors, and hypothetical returns.
10. Use the prices recorded when predictions were issued.
11. Keep pending games pending.
12. Build game features.
13. Fit context candidates using earlier saved game inputs and later real results.
14. Reserve the latest prior week for model comparison before adoption.
15. Replay eligible prior FBS results into opponent-adjusted team states.
16. Calculate score expectations and winner, cover, and total probabilities directly from the fitted distribution.
17. Do not generate simulated score samples in the live workflow.
18. Apply the selected probability model and any validated context contribution.
19. Compare probabilities with the actual recorded prices.
20. Enforce decision and exposure limits.
21. Save the completed run.

## Data coverage and sources

### SportsDataverse

SportsDataverse supplies:

- Game schedules.
- Final scores.
- Per-game advanced team statistics.

Historical records can contain later corrections. Historical results therefore do not reconstruct every original publication timestamp.

### ESPN public endpoints

ESPN public endpoints supply, when available:

- Live final scores.
- Upcoming-game weather.
- Player leaders.
- Season team statistics.
- Rosters.
- Injury information.
- Depth-chart information.
- Currently displayed sportsbook prices.

The retrieval time is treated as the quote availability time.

Missing fields remain missing. The system does not invent a price, injury probability, lineup state, weather value, or player replacement value.

### Football Database and CFBTrack

The service queries the following sources for college team and player-stat tables:

```text
www.footballdb.com
cfbtrack.com
```

Verified-season tables enter the feature stage through:

- Explicit CFBTrack team IDs.
- Exact roster player names.

Football Database passing fields preserve their original source names. The most-used passer is identified by recorded attempts rather than assuming the confirmed starter.

Independent sources remain separate when their values differ.

Access failures and missing tables appear in the Sources view.

This address is not treated as equivalent:

```text
footbaldb.com
```

It is different from:

```text
www.footballdb.com
```

The system does not treat the addresses as equivalent unless an actual redirect confirms the relationship.

A roster or active-player flag does not confirm a starting lineup.

An empty injury response means that coverage is unavailable. It does not mean that every player is healthy.

Context fields are collected first and retained as calculation inputs.

Numeric weather, player statistics, team statistics, and reported injuries enter the context model only when:

- Enough recorded pregame examples have actual outcomes.
- The candidate improves validation log loss.
- The candidate does not worsen Brier score.
- The missingness indicators are retained.
- The latest prior week remains reserved for comparison.

Before that evidence exists, the base model remains active and the report explains why.

The system does not invent:

- Weather penalties.
- Player replacement values.
- Questionable-player probabilities.
- Injury probabilities.
- Starting-lineup assumptions.
- Missing sportsbook prices.
- Synthetic game results.

## Models and measured development

The engine fits dynamic Gaussian offense and defense states from prior FBS games.

The score-only and PBP score heads use standardized regularized regression.

The PBP head includes offense and defense interactions.

A separate time-decayed ridge strength model implements opponent-adjusted margin ratings with:

```text
Half-life: 365 days
Ridge penalty: 8 games
Non-neutral venue adjustment: 3 points
```

Its logistic probability regularization is selected using the earlier validation season.

Default historical training uses:

```text
Warmup season: 2021
Initial fitting seasons: 2022–2023
Settings and probability calibration: 2024
Model comparison: 2025
Prediction season: 2026
```

Models are subsequently refitted through 2025 for 2026 predictions.

The Sunday initializer compares the strength model with the selected score head on 2025 data to choose the 2026 winner model.

Because the 2025 comparison guides the deployment choice, 2025 is not an untouched evaluation of that final deployment choice.

Scores and winner probabilities may come from separate heads. The exported `probability_source` identifies the head used.

## Context learning

Context learning uses saved pregame:

- Weather.
- Player statistics.
- Team statistics.
- Reported availability evidence.
- Missingness indicators.

The maximum logit adjustment is bounded.

A context candidate requires at least:

```text
Three recorded weeks
150 settled games
```

The latest prior week remains reserved for comparison.

Outcome columns and betting prices cannot enter context features.

## Probability audit

Evaluate constrained probability fine-tuning and adversarial sensitivity using actual exported predictions:

```bash
python -m cfb_forecast probability-audit \
  --input my_actual_predictions.csv \
  --probability-column score_p \
  --outcome-column home_win \
  --max-change 0.05 \
  --stress-radius 0.1 \
  --out artifacts/probability-research
```

The command writes `predictions.csv`, `metrics.csv`, `calibration.csv`, `summary.json`, and, when there are enough games, `candidate_calibrator.json` to the output folder. Adding `--model artifacts/training/model` also writes a separate candidate model next to the audit; the production model is never changed.

The CSV requires:

```text
game_id
season
The named probability column
The named outcome column
as_of and kickoff
or forecast_cutoff and start_date
```

The required fields are:

- `game_id` must be unique.
- `season` must be an integer.
- Probabilities must come from earlier-trained models.
- Outcome values must be available for evaluation.
- Time fields must identify the prediction information set.

Each evaluation season receives calibration fitted only on prior seasons.

A single-season file produces a candidate for later use but cannot demonstrate later-season improvement.

The constraint bounds changes on:

- Clean fitting probabilities.
- A dense probability grid.

Stress testing maximizes loss within a coefficient-norm radius to measure local sensitivity.

Stress testing does not:

- Modify the deployed model.
- Hide model activity.
- Replace the production calibration.
- Convert research output into a deployed model automatically.

The uploaded adversarial document contains an MLP research outline, not an executable implementation.

The production head is not an MLP. The experiment operates on actual probability-calibration coefficients.

Published numbers from the uploaded methodology are not copied into application results.

## Betting decisions

Moneyline prices are converted to decimal odds.

A two-sided no-vig probability is then computed.

Edge and expected return are separate calculations.

A hypothetical candidate requires:

- At least two percentage points of edge.
- Positive expected net return at the offered price.
- Complete weather coverage.
- Complete injury coverage.
- Complete lineup coverage.
- Complete player-input coverage.
- Complete team-input coverage.

Quarter-Kelly exposure is capped at:

```text
1% per game
5% per slate
```

These are explicit operating settings. They are not evidence of profitability.

Spread and total probabilities use the score distribution with continuity corrections for integer pushes.

Validation uses the candidate and quote saved with the prediction.

Missing odds produce no ROI claim.

Reported unit returns are hypothetical. The application does not place bets.

Sunday prices can change materially before kickoff.

Unknown lineups or missing injury coverage suppress recommendations.

## Sunday automation

The Sunday service runs from the scheduler (see **How the application runs**). It executes only on Sunday from:

```text
10:00 AM America/Chicago
```

This handles daylight-saving changes. If the scheduler was down at 10:00 it runs later the same Sunday, and a slate that already exists is never recomputed.

Each run saves downloaded sources, trained models, prior predictions, validation reports and runtime artifacts under `artifacts/` and `data/`; mount both as volumes (Compose does) so the next run continues from them.

The earlier GitHub Actions schedule is kept in `.github/workflows.disabled/sunday.yml`. It is inactive; GitHub schedules become active only when the file is moved to `.github/workflows/` on the default branch.

## Outputs

| Path | Content |
|---|---|
| `artifacts/runs/<season-date>/forecasts.csv` | Pregame predictions, prices, model contributions, context contributions, and decision status |
| `artifacts/runs/<season-date>/inputs/` | Raw responses, receipts, game inputs, and feature values |
| `artifacts/runs/<season-date>/run_manifest.json` | Run timing, source checksums, model checksums, prediction checksum, and selection evidence |
| `artifacts/runs/<season-date>/validation.csv` | Later actual finals joined to unchanged predictions |
| `artifacts/runs/<season-date>/validation.json` | Probability, score, and saved-price betting metrics |
| `artifacts/latest/` | Most recent published run for the dashboard (Sunday or daily), with `changes.csv`, `player_ratings.csv` and `team_ratings.csv` after a daily run |
| `artifacts/daily/forecasts/<UTC stamp>/` | Dated daily snapshots of every upcoming game, with changes against the previous snapshot |
| `artifacts/cache/` | Cached point-in-time player features (safe to delete) |
| `data/sources/` | Web-source snapshots, interpreted picks, lines, results and the score cross-check |
| `artifacts/training/` | Recomputed historical comparisons and fitted model files |
| `web/public/forecasts.json` | Board the React app and the Vercel build serve |
| `artifacts/cfb_forecast.sqlite` | Source catalog and `forecast_log` rows |

A completed Sunday or weekly run calls `publish_forecast_update`. The same board is written to SQLite, `artifacts/latest/forecast_board.json`, the React file, the Streamlit forecast log, `artifacts/logs/datadog.jsonl`, and an email to `bngomez@ieee.org`. Email is sent only when `CFB_SMTP_HOST` and `CFB_SMTP_PASSWORD` are set. Datadog intake is used only when `DD_API_KEY` is set. Otherwise both receipts stay on disk.

Publish an existing calculation without retraining:

```bash
python -m cfb_forecast publish-update --run artifacts/weekly/2026-week-7 --run-id 2026-week-7 --fetch-sources
```

## Verification

Run the complete test suite:

```bash
cd /mnt/c/Users/brian/cfb-forecast
source /home/brian/.venvs/cfb-forecast/bin/activate
python -m pytest -q
```

On 2026-10-15 the suite reported `80 passed, 1 skipped` in a clean checkout. GitHub Actions runs the same suite on every push and pull request once the workflow is enabled.

Verify the command-line interface:

```bash
python -m cfb_forecast --help
```

The installed console command is:

```bash
cfb-forecast
```

The project has been installed editable into:

```text
/home/brian/.venvs/cfb-forecast
```

Tests use small explicit fixtures to check:

- Timing.
- Model isolation.
- Source archives.
- Pricing arithmetic.
- Calibration constraints.
- Context leakage.
- Prediction integrity.

Fixtures are not imported by production.

Source and network failures fail required ingestion or appear as unavailable optional context. They do not trigger sample-data generation.

## Docker configuration

The Docker images are:

```text
cfb-forecast:local
cfb-forecast-dashboard:local
```

The Docker base image is:

```text
python:3.12-slim
```

The forecast container installs the package with its API extras into `/app` and also installs R (`r-base-core`) for the R validation step.

The forecast container runs as:

```text
UID 10001
GID 10001
```

The dashboard service runs as UID and GID 10002 under Compose.

The dashboard exposes:

```text
8501
```

The dashboard URL is:

```text
http://localhost:8501
```

The Docker workflow builds both images on every pull request and push to `main`, then checks that:

- The forecast image answers `cfb-forecast --help` and `cfb-data-watchdog --help`.
- The forecast image can import the FastAPI service.
- The dashboard image can import Streamlit and the dashboard modules.

The dashboard image installs Streamlit from `dashboard/requirements.txt`, which allows any 1.x release from 1.40 on. A build on 2026-10-15 installed Streamlit 1.66.0.

Docker Desktop is reachable from Ubuntu through:

```text
docker.exe
```

The standard Linux `docker` command is not currently available directly inside the Ubuntu WSL distribution.

## Docker Compose

The Compose services are:

```text
cfb-forecast        one-off forecast commands (sunday-run, publish-update, and the rest)
cfb-api             FastAPI service on port 8000
cfb-data-watchdog   records data file checksums in SQLite and keeps watching
cfb-dashboard       Streamlit dashboard on port 8501
```

The Compose network is:

```text
cfb-forecast_default
```

The default bind mounts are:

```text
/mnt/c/Users/brian/cfb-forecast/data:/app/data
/mnt/c/Users/brian/cfb-forecast/artifacts:/app/artifacts
```

The dashboard mounts those directories read-only.

The forecast service mounts them read-write so it can produce runtime files.

For Docker Desktop bind mounts, the project directory must be shared with Docker Desktop:

```text
C:\Users\brian\cfb-forecast
```

The Docker Desktop file-sharing setting must allow access to:

```text
C:\Users\brian\cfb-forecast\data
C:\Users\brian\cfb-forecast\artifacts
```

`docker-compose.yml` is the source of truth for the services. These environment variables change its defaults without editing the file:

| Variable | Default |
|---|---|
| `CFB_DATA_DIR` | `./data` |
| `CFB_ARTIFACTS_DIR` | `./artifacts` |
| `CFB_IMAGE` | `cfb-forecast:local` |
| `CFB_DASHBOARD_IMAGE` | `cfb-forecast-dashboard:local` |
| `CFB_UID` and `CFB_GID` | `10001` |
| `CFB_DASHBOARD_UID` and `CFB_DASHBOARD_GID` | `10002` |
| `CFB_DASHBOARD_PORT` | `8501` |
| `CFB_API_PORT` | `8000` |
| `DD_API_KEY` and `DD_SITE` | empty and `datadoghq.com` |

The forecast and watchdog services run with the same `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`, and single-thread BLAS settings as the image.

Build the forecast and dashboard images:

```bash
docker compose build
```

When Docker Desktop is accessed from Ubuntu WSL through the available Windows-integrated CLI:

```bash
docker.exe compose build
```

Run the forecast job:

```bash
docker compose run --rm cfb-forecast sunday-run --season 2026
```

Run the dashboard:

```bash
docker compose up cfb-dashboard
```

For Compose startup with image refresh:

```bash
docker compose up --pull always
```

The dashboard should then be available at:

```text
http://localhost:8501
```

## Weekly controls and backtesting contract

The application is organized around a weekly Sunday pregame run.

The selected week is forecast from:

- The prior season.
- Eligible completed games from the selected season.
- Information available before the game-specific cutoff.

The default control is the earlier of:

```text
09:00 local venue time on Sunday
24 hours before kickoff
```

A forecast cannot consume:

- Its own final score.
- A later market quote.
- A later player report.
- A later weather report.
- A later injury report.
- A later lineup report.

### Weekly controls

The weekly controls are:

- Season.
- Week.
- Pregame cutoff.
- Training seasons.
- Market timestamp.
- Result availability.
- Data-audit status.

#### Season and week

Controls the FBS schedule slice shown to the user.

#### Pregame cutoff

Freezes the information set used for every game on the slate.

#### Training seasons

Uses the prior season and eligible completed games from the selected season.

#### Market timestamp

Only spreads and totals observed before the cutoff are eligible.

Missing prices remain missing.

#### Result availability

Only final scores with an availability timestamp before the cutoff update team states.

#### Data audit

The system quarantines rather than repairs:

- Duplicate game IDs.
- Invalid scores.
- Non-FBS matchups.
- Incomplete records.
- Unverifiable team IDs.
- Unverifiable game IDs.

The system does not repair quarantined data with invented values.

## Point-in-time backtesting contract

Each prediction is one immutable game-level row containing:

```text
game_id
kickoff_utc
cutoff_utc
home_team_id
away_team_id
pregame feature state
observed spread
spread observation timestamp
prediction
final outcome
```

The required leakage checks are:

```text
cutoff_utc < kickoff_utc
```

No training result may satisfy:

```text
result_available_at >= cutoff_utc
```

No market observation may satisfy:

```text
observed_at > cutoff_utc
```

The evaluation reports:

- Winner log loss.
- Winner Brier score.
- Margin MAE.
- Spread cover accuracy.
- Spread MAE.
- Calibration by probability bin.
- Sample counts.

These metrics remain separate from betting decisions.

A spread comparison is reported only when a real spread and price were captured.

The data layer does not include:

- Backfilled lines for upcoming games or betting decisions. Closing lines are read only for games already played, to learn how lines relate to final scores.
- Fabricated player states.
- Synthetic schedules.
- Simulated records.
- Invented injury probabilities.
- Invented weather values.
- Invented sportsbook prices.

## System architecture

```mermaid
flowchart LR
  A[SportsDataverse and ESPN-derived FBS schedules] --> B[Ingestion and source manifest]
  C[Team summaries and play-by-play fields] --> B
  D[Weather and timestamped player-state snapshots] --> B
  B --> E[Validation and quarantine]
  E --> F[Point-in-time feature builder]
  F --> G[Dynamic offense and defense states]
  G --> H[Football score forecast]
  H --> I[Gaussian margin and Poisson score check]
  I --> J[Spread and total comparison]
  H --> K[Immutable forecast ledger]
  K --> L[Weekly scoring and walk-forward backtest]
  J --> M[Forecast dashboard]
  L --> M
```

The forecasting, training, and validation code is the Python package in `src/cfb_forecast`.

The dashboard is a Streamlit application using the generated runtime artifacts and data.

The public board is a React and TypeScript app in `web/`, built with Vite. It reads `web/public/forecasts.json`, which `publish_forecast_update` writes, and `vercel.json` tells Vercel to build it from `web/` into `web/dist`.

## Complete Docker workflow

From the project root:

```bash
cd /mnt/c/Users/brian/cfb-forecast
```

Create runtime directories if they do not exist:

```bash
mkdir -p data artifacts
```

Build the images:

```bash
docker compose build
```

Run the forecast:

```bash
docker compose run --rm cfb-forecast sunday-run --season 2026
```

Start the dashboard:

```bash
docker compose up cfb-dashboard
```

With Docker Desktop accessed from Ubuntu through the integrated CLI:

```bash
docker.exe compose build
docker.exe compose run --rm cfb-forecast sunday-run --season 2026
docker.exe compose up cfb-dashboard
```

The dashboard will be available at:

```text
http://localhost:8501
```

Data and artifacts are excluded from the Docker build context and supplied through runtime mounts.

The single source project root is:

```text
/mnt/c/Users/brian/cfb-forecast
```

The single CFB Python environment is:

```text
/home/brian/.venvs/cfb-forecast
```
