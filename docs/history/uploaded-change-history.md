* Build FBS forecasting terminal dashboard

* feat: update lint script to use eslint

Replaced lint script with eslint for improved code quality checks

Co-authored-by: Brianna Gomez <112305731+bngomez98@users.noreply.github.com>

* Build full-stack FBS forecasting dashboard

* Upgrade FBS forecasting sources and probability layers

* Strengthen football forecasting architecture

* feat: streamline page.tsx with simplified data handling and UI updates

Co-authored-by: Brianna Gomez <112305731+bngomez98@users.noreply.github.com>

* Refine upcoming football forecast dashboard

* Correct forecast input disclosures

* feat: enhance forecast display and add team ratings section

Added detailed upcoming games forecast and introduced team ratings section with opponent-adjusted states.

Co-authored-by: Brianna Gomez <112305731+bngomez98@users.noreply.github.com>

* Refactor live forecast dashboard UI

* feat: rename app title to CFB PREDICTIONS and update subtitle

Co-authored-by: Brianna Gomez <112305731+bngomez98@users.noreply.github.com>

* Refine CFB prediction board interface

* Improve forecast probability validation metrics

* Add full-season forecast view

* Consolidate project into one build folder

* Remove duplicate Python environments and frontend folders

* Configure cross-platform Python runtime and validate deployments

* Update README.md

* Build

* Add extended external source collection and weekly cross-checks

* Add weekly external source collector entry point

* Collect external CFB validation sources weekly

* Test external source matching and weekly selection

* Build forecast desk and wire audited historical training pipeline

* feat: add R dependency and API endpoints for forecasts and statuses

Added R installation, API endpoints for forecast status, learning, pipeline, and watchdog.

Co-authored-by: Brianna Gomez <112305731+bngomez98@users.noreply.github.com>

* Implement continual statistical forecast reinforcement

* Run live Bayesian forecasts from current 2026 data

* Configure weekly source lineage and forecast validation

* feat: initialize new CFB forecast console project with HTML, package.json, and React app

Co-authored-by: Brianna Gomez <112305731+bngomez98@users.noreply.github.com>

* Create vercel.json

* Publish the weekly forecast board across the stack.

A completed run now writes scores, winner, and confidence to SQLite, the React board, Streamlit, and the local Datadog and email receipts. The Vercel build serves the Vite app, and the source catalog includes the public college football sites.

* Align the checkout, database, and Datadog log.

Scripts now use this repository. The API, Sunday run, watchdog, and Compose services share artifacts/cfb_forecast.sqlite and can ship logs when DD_API_KEY is set. The generated deployment that mounted the old Downloads tree is removed.

* Calculate live forecasts from the fitted distribution and match the schema.

The Sunday and weekly paths no longer draw score samples. The schema now includes the learning, watchdog, and forecast-log tables the code writes, and the README shows the architecture, pipelines, workflows, and source catalog.

* Keep the current model until thirty new results arrive.

A later cutoff with fewer than thirty new completed games returns the promoted model instead of fitting another one.

* Add one forecasting model for scores, outcomes, and spreads.

The live forecast uses the hierarchical state, a known-future Koopman map, Gaussian scores, Poisson rates, a logistic win probability, and Monte Carlo spread prices. Weekly validation and the forecast log store the spread.

* Feed recorded team and player history into the forecast

Same-season and prior-season scores and play rates from on-disk parquet
enter the one forecasting model before kickoff. Those facts are stored
in SQLite so the weekly run can keep using them. FootballDB college
stat pages and Open-Meteo venue weather are collected with the existing
sources. Overtime plays stay in the play totals.

* Read stored team facts into the weekly forecast and split the desk

SQLite team_game_facts and player_game_facts now feed the same pregame
rates the hierarchical state, Poisson scores, Gaussian intervals, and
logistic win probability already use. The prediction board shows kickoff,
score, winner, and confidence. Spreads and fair American moneylines, team
and player statistics, and method notes are separate pages.

* Use every prior-season and same-season game in pregame rates

The weekly forecast still uses the one model. Previous-season means are
the average of all recorded games that season. Same-season rates then
include every completed game. No subsample of history.

* Fix calculation bugs found in the forecast audit

- Replay team history in kickoff order. Postseason games restart week
  numbering at 1, so the season/week sort fed each team's bowl game into
  its own September and October pregame rates.
- Do not flag bowl and playoff games as early-season (weeks 1-4).
- Count rest days from every earlier game, including games against
  non-FBS opponents, as the rest_days docstring says.
- Save the score head a forecast used and reuse it when advancing a saved
  state, so an advance with no new results matches a full replay.
- Keep postseason games out of regular week 1 in the season backtest.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DJZ7TyRXjq7ALVo3j9CnAT

* Make the code, CI, and README agree

Restore the probability-audit command the README documents; the function
was still in probability.py but the command had been dropped from the CLI.

Fix the Docker workflow: the image name and tag were passed as build args,
which the bake file ignores, so pushes to main would have tagged
cfb-forecast:local and tried to push it to Docker Hub. The workflow now sets
the bake variables through the environment, builds both the forecast and
dashboard images, runs each image's commands as a smoke test, and pushes to
ghcr.io only after those checks pass.

Build the React board in CI, report the package version from pyproject, and
correct the README where it was out of date: test count, Compose services,
network name, first-run download range, source list, and the TypeScript app.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011fpBKnmsJCvuJsmCdQVKiD

* Calculate spread and total chances directly instead of simulating

The forecast drew thousands of random scores per game to price spreads
and totals, which contradicted the README's calculation order ("Do not
generate simulated score samples in the live workflow"). The same
Gaussian already gives exact cover, push, over and under chances with
continuity corrections for whole-number lines, so those are used now.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DJZ7TyRXjq7ALVo3j9CnAT

* Start each season's pregame rates from the previous season only

The README says the previous season's games set a team's starting mean and
every completed game in the current season updates it. pregame_history never
split seasons: remember() read a "_season" field, but itertuples() renames
fields that start with an underscore, so the season was always missing and
every season on disk since 2000 was pooled into one average. After PR #10
the column was not created at all, which also broke the bowl-order test on
main.

The column is now "rate_season", and a team's first game of a new season
uses only the previous season's mean. The bowl test now expects the plain
same-season average that the current rate rule produces.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011fpBKnmsJCvuJsmCdQVKiD

* Learn from closing lines for played games; fall back when a game has no line

Completed games now carry their real closing spread and total (SportsDataverse
ESPN betting files, downloaded in the Sunday refresh) so the forecast step can
learn how lines relate to final scores. Upcoming games still use only lines
captured before the forecast time. Because a fit that leans on the line is
biased when the line is missing, the forecast step also keeps a second fit that
never saw lines and uses it for games without one.

A pick'em line can no longer show a push, since a tied game goes to overtime.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DJZ7TyRXjq7ALVo3j9CnAT

* Match the bowl-order test to the plain average from the rates change

The merged rates change averages every earlier game instead of weighting recent
ones, so the bowl now sees the mean of the two regular-season games. The test
still checks that the bowl is replayed after them.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DJZ7TyRXjq7ALVo3j9CnAT

* Make the forecast website easier to read

Games are grouped by day and shown as small score cards: each team on its own
line with expected points and win chance, the pick highlighted, and a one-line
"Team by N" summary. The board opens on the forecast week instead of all 432
games, with a team search and an all-games toggle. Page tabs sit in a top bar,
so phones can switch pages (the sidebar used to disappear on small screens).
The odds page computes fair odds from the win chance when the board file has
none, and shows a sportsbook column only when a line was recorded. Summary
cards and the explanation pages use plain wording, and muted text has more
contrast.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DJZ7TyRXjq7ALVo3j9CnAT

* Fix the merge that copied the bowl assertion into the season test

Merging main into this branch left the bowl test's last assertion inside
test_pregame_history_starts_from_the_previous_season_only, which only
builds one query game, so it failed with an IndexError.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011fpBKnmsJCvuJsmCdQVKiD

* Describe the closing-line download in the README and CLI help

The Sunday refresh now downloads SportsDataverse ESPN closing spreads and
totals for every season into data/market_closing_history.csv, and completed
games learn from them. The README's first-run paragraph and SportsDataverse
source list now say so, and market-backfill-sdv's help no longer calls the
lines benchmark-only.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011fpBKnmsJCvuJsmCdQVKiD

* Rebuild the Streamlit dashboard around readable forecast pages

The dashboard now opens on Predictions: the forecast week's games grouped by
day, each a card with expected points and win chance per team, the pick
highlighted, a team search and an all-games toggle. Odds shows our spread and
fair odds per team, with a Book column only when a line was recorded. Past
results and How it works complete the Forecasts section. The run-checking pages
(overview, book prices, game inputs, statistics, forecast log, pipeline,
training, research, sources) move under Behind the scenes using st.navigation.
Text sizes and contrast are raised, the dev toolbar is hidden, and the board
falls back to web/public/forecasts.json when no run has been saved.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DJZ7TyRXjq7ALVo3j9CnAT

* Leave room under the dashboard logo on phones

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DJZ7TyRXjq7ALVo3j9CnAT

* Remove bowl-test lines left inside the season test by a merge

The merge of PR #12 copied the bowl assertion into the end of the
previous-season test, where the board has one row, so it failed with an
IndexError on main. The bowl test already has the same assertion.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DJZ7TyRXjq7ALVo3j9CnAT

* Update issue templates

* Group root files by function into scripts/, deploy/ and docs/

Setup, run, check and workspace helpers move into scripts/ subfolders.
Caddy, Datadog and Kubernetes configuration move into deploy/.
ARCHITECTURE.md and RESULTS.md move into docs/. Paths inside the moved
scripts, the public compose file and the README follow the moves, and the
README gains a section describing each folder.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YYZ3qu5pRxgL4bGduNePq4

* Make CI catch more real failures and fix the weekly data restore

- Python checks: lint job for crash-class errors and workflow mistakes,
  Python 3.13 added, a pinned-versions leg with R so the R cross-check
  runs, a clean wheel install check, concurrency and timeouts.
- External sources workflow: restore the Sunday runtime through the API;
  download-artifact could only see artifacts from its own run.
- Sunday workflow: look up the runtime artifact by name, install the
  tested library versions, warn when nothing was restored.
- Docker: separate build cache per image so they stop evicting each other.
- update_forecast.py and setup_project.py called helpers that did not exist.
- The R test fails instead of skipping when CFB_REQUIRE_R=1.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TX2ApGX6vj1EW83E5A2ieg

* Rewrite the README in plain language with two clear diagrams

The README had grown to 940 lines with repeated Docker sections, three
overlapping diagrams, and sections describing things the code does not do:
a 09:00 venue-time cutoff, weekly dashboard controls, data quarantine, and
win chances coming from a separate strength model.

The new README explains what the project produces, the five steps of a
Sunday run, and how one game is predicted, each with one diagram. Data
sources, running it, Docker, automation, outputs, betting rules and tests
each appear once. Facts were checked against sunday.py, forecast.py,
forecast_model.py, backtest.py, context_model.py, decision.py and the
workflows. The test count is updated to 89 passed, 1 skipped.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011fpBKnmsJCvuJsmCdQVKiD

* Group the forecast package's modules into folders by job

src/cfb_forecast now has ingest/, modeling/, odds/, evaluation/,
pipeline/, monitoring/ and storage/ subpackages. The CLI, API, config
and paths stay at the package top. Imports, the console entry point,
the packaged R script, the schema path, the Docker workflow check, the
dashboard, scripts and tests follow the moves. README edits are left to
the README rewrite in PR #18.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YYZ3qu5pRxgL4bGduNePq4

* Add a results and accuracy page to the dashboard

The Streamlit dashboard now grades saved forecasts against final scores.
It reads the graded Sunday runs (runs/*/validation.csv) when they exist and
otherwise the latest season backtest (backtest/season-*/graded_games.csv).

The page shows the share of winners picked correctly, the average miss on
the margin, the Brier score, accuracy by week, whether picks at each
confidence level came true that often, and every graded game marked right
or wrong. The predictions page summary also shows season accuracy so far.

CI gains a dashboard job that installs Streamlit, runs the new results
tests, and imports every dashboard page module; the Docker check imports
the new modules too.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DJZ7TyRXjq7ALVo3j9CnAT

* Improve the development setup and make weekly failures readable

- Lint rules live in pyproject.toml so a local `ruff check` matches CI.
- When publishing a forecast fails, the run manifest keeps the error
  message, not only the exception type.
- The Sunday workflow writes a short run summary to the Actions page.
- web/package.json pins the versions already in the lockfile instead of
  "latest", with build tools moved to devDependencies. No locked version
  changed.
- The dev container installs the tested library versions, R, Node and the
  web packages, so it matches CI and the Sunday run.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TX2ApGX6vj1EW83E5A2ieg

* Split the dashboard CI job back out of the package job

The merge of main into the dashboard branch left the dashboard job with
only runs-on and no steps, and moved its steps into the package job.
GitHub rejects a job with no steps, so the whole Python checks workflow
could not run. The dashboard job gets its steps back and the package job
starts with its own checkout and Python setup again.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DJZ7TyRXjq7ALVo3j9CnAT

* Add a repository layout table to the README

PR #16 moved files into src/cfb_forecast/ subpackages, scripts/, deploy/
and docs/. The README now lists each top-level folder and what it holds,
so a reader can find the code without browsing the tree.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011fpBKnmsJCvuJsmCdQVKiD

* Make the weekly checks and refits statistically sound

Calibration: each probability bin now carries the range that chance alone
allows (1.96 standard errors of a sum of independent outcomes) and flags
bins outside it. Weekly validation also reports Spiegelhalter's z-test, one
overall check that the win chances match the outcomes.

Weekly refit: the refit replaces the current model only when, fitted
without the held-out later games, it forecasts them at least as well.
Otherwise the report says "retained" and points at the current model. When
there are too few later games to hold out, the report marks the gate as not
independent. The report no longer says a saved strength model sets the
win chances; it never did.

Second stage: the line, rest and team-rate adjustments and the win-chance
curve are now fitted on score locations from a head that never saw that
game's season, instead of the head's own in-sample fits.

Results page: adds log loss, a chart of Brier and log loss by week, and
for each confidence group the range luck allows and whether it holds up.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01DJZ7TyRXjq7ALVo3j9CnAT

* Keep forecast probabilities consistent with one causal score model

* Build comprehensive forecast and moneyline UI

* Fix Vercel build path and improve forecast UX

* feat: add live game results fetching and display logic

Co-authored-by: Brianna Gomez <112305731+bngomez98@users.noreply.github.com>

* Show full 2026 season forecast board

* Add season football API route

* Use live football service data without fallbacks

* Focus forecast UI on 2026 college football

* Add season status and clearer data handling

* Fix missing season status component

* Move SQLite database out of artifacts

* Build dashboard with React app tooling

* Add missing API routes and fix React deployment config

* Fix React deployment paths and build output

* Fix Vercel React deployment configuration

* feat: add new dependencies and update package metadata

Co-authored-by: Brianna Gomez <112305731+bngomez98@users.noreply.github.com>

* Add root build manifest for Vercel

* Add dev script to web package

* Fix root test command for CI

* Validate saved score model artifacts and add regression tests

* Update the README's accuracy wording for the new weekly checks

Reflects the changes merged in PR #21: the refit's retained vs promoted
status, the per-bin 95% range flag and Spiegelhalter z on the weekly
calibration, log loss and Brier tracked by week on the results page, and
the strength model as comparison only (it never sets a win chance).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011fpBKnmsJCvuJsmCdQVKiD

* docs: sync README with FieldQuant spec and track data-source manifest

Replace README.md with the FieldQuant FBS Forecast Desk specification and
add docs/data-sources.json as the tracked provenance record for the runtime
data snapshots (2021-2026 schedules and advanced team summaries) that the
gitignored data/ directory holds.

Redacted values in the source text were resolved against the repo's real
settings: thread counts to 1 (docker-compose.yml), the dashboard artifacts
path to runtime/artifacts, and the probability-audit stress radius to 0.1
(the CLI default in src/cfb_forecast/cli.py).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZhWkc2eEHae9ozQUWkNRB

* Add news feed, season and player PPG, historical roster span; fix stale test fixtures

The saved-model validation merged in #44 rejects models without
metric_names, which broke eight tests whose fixtures wrote partial
models. Those fixtures now carry a complete valid model.

New ingest modules, each cutoff-safe and never fabricating on failure:
- news: ESPN college-football news feed with injury and roster-change flags
- season_ppg: team points scored and allowed per game strictly before a cutoff
- player_ppg: player points per game computed from touchdowns, field goals,
  extra points and two-point conversions
- rosters: SportsDataverse historical rosters over a season range, with
  per-season failure isolation
- capture_log: append-only JSONL audit trail shared by the commands

CLI: capture-news, capture-historical-rosters, season-ppg, player-ppg.
Suite: 161 passed, 2 skipped; ruff clean.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XGa61y1L67WkNgSBsy9dpC

* Merge pull request #51 from bngomez98/main (#52)

---------

Co-authored-by: v0 <it+v0agent@vercel.com>
Co-authored-by: vercel[bot] <35613825+vercel[bot]@users.noreply.github.com>
Co-authored-by: Claude <noreply@anthropic.com>
Co-authored-by: Brianna Gomez <bngomez@ieee.org>
Co-authored-by: Brianna Gomez <orangesoranges98@gmail.com>


Engineering documentation · Architecture and methodology · Based on the supplied development history through PR #52

# FieldQuant FBS Forecast Desk

## System architecture and forecasting methodology

## 1. Purpose and scope

FieldQuant FBS Forecast Desk is a college football forecasting platform designed to produce pregame estimates for Football Bowl Subdivision (FBS) games, publish predictions to dashboard interfaces, evaluate forecast quality against completed games, and improve its statistical models through a controlled weekly training process.

The supplied development history describes a system that evolved from a React-based prediction board into a broader full-stack application with Python forecasting services, SQLite persistence, historical data ingestion, automated workflows, odds analysis, statistical calibration, monitoring, and a Streamlit dashboard.

The system's central architectural principle is one causal score-forecasting model serving as the foundation for score predictions, winner probabilities, and betting-market calculations. Supporting models and data sources contribute information or provide comparison and validation, but the documented design aims to keep the published probabilities consistent with the underlying score model.

This document distinguishes three categories of information:

- Documented implementation: Features explicitly described as implemented in the supplied development history.
- Documented design intent: Rules and architectural goals stated in that history, without independent verification of the current code.
- Not established: Details that cannot be reliably determined from the changelog alone.

No repository inspection or independent execution of the test suite is claimed here.

## 2. System at a glance

External data and historical records

Schedules, scores, team and player statistics, rosters, news, weather, and closing markets

Ingestion, normalization, and provenance

Cutoff-safe collectors · source matching · historical facts · capture audit trail

Unified forecasting engine

Historical team state · pregame rates · hierarchical state · score distributions · outcome probabilities

Evaluation and decision layer

Win probabilities · score intervals · spreads and totals · fair odds · calibration · backtesting

Persistence and publication

SQLite · run artifacts · forecast JSON · API · React/Vite · Streamlit · monitoring and receipts

Conceptual architecture reconstructed from the supplied history. This is not a verified runtime call graph, and the diagram does not imply that every output passes through every component in precisely this order.

### 2.1 Architectural boundaries

The documented system can be understood as six cooperating layers:

1. Data acquisition: Collects historical and current football information from external providers.
2. Feature and state construction: Converts recorded games and associated facts into information available before kickoff.
3. Statistical forecasting: Produces score distributions and derives outcome and market probabilities.
4. Evaluation and model promotion: Measures forecast quality and controls whether a newly fitted model replaces the current one.
5. Persistence and operations: Stores forecasts, source records, model-related state, validation outputs, and operational logs.
6. Presentation and integration: Exposes predictions and supporting information through web interfaces, API routes, and automated reporting.

The Python forecasting package is the principal statistical backend. The React/Vite application and Streamlit application are separate presentation surfaces described in the history. The API, scheduled workflows, and storage layer connect forecasting, publication, and operations.

## 3. Technology stack and repository organization

The project history identifies the following technologies and their documented roles.

| Technology | Documented responsibility |
| --- | --- |
| Python | Forecasting, ingestion, model evaluation, orchestration, and command-line operations |
| R | Cross-language statistical checks and an R-dependent validation path |
| SQLite | Persistent forecasts, game facts, learning records, watchdog state, and forecast logs |
| pandas and Parquet | Tabular historical data processing and on-disk team/player history |
| React, TypeScript, Vite | Interactive prediction board and frontend application |
| Streamlit | Forecast desk with user-facing predictions and behind-the-scenes operational pages |
| FastAPI or Python API layer | Forecast, status, learning, pipeline, and watchdog endpoints; exact framework details should be confirmed against the repository |
| Docker and Compose | Containerized forecast and dashboard services |
| GitHub Actions | CI, weekly data collection, scheduled forecasting, and automated checks |
| Vercel | React/Vite frontend deployment |
| GitHub Container Registry | Container image publication following CI smoke tests |
| Datadog and email | Optional operational logging and run receipts |

The frontend, backend, and dashboard are not simply one application rendered in different ways. The development history records separate interfaces, API endpoints, deployment configuration, scheduled jobs, and data artifacts. The exact production topology may vary by deployment environment.

### 3.1 Python package structure

A refactor organized the forecasting package under `src/cfb_forecast/` by responsibility.

`src/cfb_forecast/`

Package entry points and shared configuration

`ingest/`

External data and historical facts

`modeling/`

States, rates, score models, fitting

`odds/`

Markets and betting calculations

`evaluation/`

Backtests and probability validation

`pipeline/`

Refreshes, fitting, weekly runs

`monitoring/` and `storage/`

Run health, logs, persistence

Other package-level responsibilities include the CLI, API, configuration, and shared paths. The folder names reflect the documented refactor; exact module contents are not independently verified.

The repository's supporting organization was also consolidated:

- `scripts/` contains setup, execution, validation, and workspace helpers.
- `deploy/` contains deployment configuration, including Caddy, Datadog, and Kubernetes-related files.
- `docs/` contains architecture, results, and data-source provenance documentation.
- `web/` contains the TypeScript frontend and its build tooling.
- `data/` contains downloaded or generated datasets, which the history describes as Git-ignored.
- `artifacts/`, `runtime/artifacts/`, and run-specific output directories serve different historical or runtime purposes; their current canonical roles should be confirmed from the active configuration.

The history specifically records a move of the SQLite database away from the general artifacts directory, a shared database path of `artifacts/cfb_forecast.sqlite` for certain scripts and services, and later references to `runtime/artifacts`. These changes make it important to treat path configuration as an architectural contract rather than assume that all historical path references remain current.

## 4. Data acquisition and provenance

The forecasting system relies on recorded game results and other facts that can legitimately be known before a game's kickoff. The history documents a growing source catalog and several specialized ingestion paths.

### 4.1 Source categories

| Source category | Documented purpose | Forecasting relevance |
| --- | --- | --- |
| College football schedules and results | Identify games, kickoff times, and completed scores | Training labels, chronology, upcoming slate |
| Advanced team summaries | Supply team-level historical information | Team state and scoring expectations |
| SportsDataverse ESPN betting files | Retrieve historical closing spreads and totals | Market-aware fitting and historical market evaluation |
| FootballDB college statistics pages | Collect team or player statistical information | Historical context and rate estimation |
| SportsDataverse historical rosters | Retrieve rosters across a range of seasons | Historical player/team context |
| ESPN college football news | Collect news and flag injury or roster-change information | Potential pregame context |
| Open-Meteo | Retrieve venue weather information | Potential environmental context |
| Historical team and player Parquet files | Preserve recorded facts for repeated use | Same-season and previous-season rates |

The development history explicitly documents the ingestion of these source categories. It does not establish that every collected field is consumed by the live forecasting model, nor that every provider is available for every game.

### 4.2 Cutoff-safe data handling

A core design requirement is that a forecast must use only information available at its forecast cutoff.

For a game $g$, let $t_g$ be the timestamp at which the forecast is generated. A valid feature set must satisfy:

$$
X_g = \{x : \operatorname{available\_at}(x) \leq t_g\}
$$

The availability timestamp is important: a fact describing an earlier game may still be unavailable to the model at the time of the original forecast if it was published or collected later.

The history describes the new news, season PPG, player PPG, and roster collectors as cutoff-safe and as avoiding fabricated data on failure. This supports the intended policy that missing data should remain missing rather than be silently replaced by invented observations.

A complete implementation of this policy should distinguish at least:

- Event time: when the game or underlying event occurred.
- Publication or availability time: when the fact became knowable.
- Collection time: when the pipeline acquired it.
- Forecast cutoff: the latest permitted information timestamp for a prediction.

The first three timestamps are recommended provenance fields for enforcing the policy consistently; the changelog does not establish that every source implements all four.

### 4.3 Historical fact ingestion

The history documents persistent `team_game_facts` and `player_game_facts` tables in SQLite. It also describes historical team and player records stored in Parquet and consumed by the forecasting process.

The documented ingestion additions include:

- `capture-news`: collect news and identify injury or roster-change flags.
- `capture-historical-rosters`: retrieve historical rosters across a season range.
- `season-ppg`: calculate team points scored and allowed per game strictly before a cutoff.
- `player-ppg`: calculate player points per game from touchdowns, field goals, extra points, and two-point conversions.
- `capture_log`: maintain a shared append-only JSON Lines audit trail.

Historical roster collection has per-season failure isolation, so a failed season need not invalidate the entire requested range.

The changelog also states that overtime plays remain in play totals. That matters because excluding overtime changes observed play rates and can make recorded game statistics inconsistent with final scores.

Important limitation: Collection is not equivalent to model utilization. The history confirms that these data sources and collectors exist in the described implementation, but does not prove that news flags, historical rosters, player PPG, or weather each have a fitted coefficient or a measurable effect on every published prediction.

### 4.4 Source matching and provenance records

The project includes external source matching, weekly source selection, source lineage, and a tracked manifest at `docs/data-sources.json`. The manifest is described as recording provenance for runtime snapshots, including 2021–2026 schedules and advanced team summaries.

This provides a basis for answering questions such as:

- Which provider supplied a particular dataset?
- Which historical snapshot was collected?
- Which scheduled run consumed the data?
- Did an ingestion failure occur?
- Can a result be traced to the inputs available when it was produced?

The precise schema of the provenance manifest and the degree to which individual forecast records reference source identifiers are not established by the supplied history.

## 5. The forecasting model

The documented design uses one forecasting model to connect historical team information, score distributions, winner probabilities, and market calculations. The history mentions a hierarchical state, a known-future Koopman map, Gaussian scores, Poisson rates, a logistic win-probability layer, and a causal second-stage fitting process.

These components should be understood as parts of a unified forecasting pipeline, not as evidence that several independent models vote on the winner.

### 5.1 Pregame historical rates

Historical rates provide a foundation for estimating each team's scoring behavior before kickoff.

The documented rate rule is:

1. For a team's first game of a new season, initialize its rate using the previous season's recorded games.
2. As the current season progresses, incorporate every completed game already available before the next forecast.
3. Preserve chronological ordering so that later games cannot affect earlier predictions.
4. Use all qualifying historical games rather than a subsample.
5. Count rest days from every earlier game, including games against non-FBS opponents.

The previous-season initialization was corrected after a bug caused season information to be lost through a field named `_season` when iterating over pandas tuples. The field was renamed to `rate_season`.

The history also describes an earlier correction to postseason ordering. Bowl and playoff games can restart week numbering at one, so sorting solely by season and week can incorrectly place a postseason game before regular-season games. The documented fix replays team history in kickoff order and avoids classifying postseason games as early-season weeks.

These fixes are particularly important because historical features are supposed to be available before the game being forecast.

For team $i$, a conceptual rate estimate at cutoff $t$ is:

$$
\widehat r_i(t)
=
\frac{1}{N_i(t)}
\sum_{\substack{j \in H_i\\t_j<t}}
r_{ij}
$$

where $H_i$ is the qualifying history and $r_{ij}$ is a recorded rate from an earlier game. This equation illustrates the all-prior-games averaging principle; it is not a claim that every rate in the production model is calculated with this exact formula.

The history does not establish how every rate handles opponent strength, missing games, or teams with no usable previous-season records.

### 5.2 Hierarchical state and score forecasting

The changelog identifies a hierarchical state and a known-future Koopman map in the live forecasting architecture.

At a conceptual level, a hierarchical state represents persistent team or game characteristics, while a state-transition map advances those characteristics to the forecast horizon. The phrase known-future indicates that the system intends to use information about the upcoming game that is available at prediction time, rather than future results.

The documented model then produces score-related quantities using Gaussian and Poisson components.

The available history does not specify the full state vector, the exact hierarchy, the Koopman operator's learned representation, or the mathematical fitting objective. Consequently, the following is an architectural interpretation, not a reconstruction of the exact production equations.

### 5.3 Gaussian score distributions

The history states that live forecasts are calculated from a fitted distribution rather than generated by sampling thousands of random scores.

For a game between teams $A$ and $B$, a conceptual score representation is:

$$
S_A \sim \mathcal N(\mu_A,\sigma_A^2),
\qquad
S_B \sim \mathcal N(\mu_B,\sigma_B^2)
$$

Here:

- $\mu_A,\mu_B$ represent expected scores.
- $\sigma_A,\sigma_B$ represent score uncertainty.
- The distributions provide a basis for score intervals and margin calculations.

The precise joint distribution, treatment of score dependence, and relationship between the Gaussian score head and Poisson rate head are not established in the supplied history. The equations above therefore describe the distributional role of the Gaussian component without asserting that the production implementation uses independent Gaussian scores in all calculations.

### 5.4 Poisson scoring rates

The Poisson component supplies rate-based scoring calculations. A conventional Poisson variable with rate $\lambda$ has probability mass function

$$
P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!},
\quad k=0,1,2,\ldots
$$

The changelog confirms that Poisson rates are part of the unified model. It does not fully specify how the rates are parameterized, whether they represent scoring events or team points directly, or how they are reconciled with the Gaussian score distribution.

It would therefore be inaccurate to claim that the production system simply uses a Poisson distribution as its final score model. The documented architecture contains both Gaussian scores and Poisson rates.

### 5.5 Logistic win probability

The model also contains a logistic win-probability layer. In a conventional logistic specification,

$$
P(A\text{ wins})
=
\frac{1}{1+\exp(-\eta)}
$$

where $\eta$ is a fitted linear predictor or another model-derived log-odds quantity.

The important architectural point is that the history explicitly describes this probability layer as part of the score-model forecasting system. A separate saved strength model is retained as a comparison, not as the source of the published win probabilities.

A later statistical refactor also trains the second-stage adjustments and win-probability curve using score locations generated by a score head that did not see the corresponding game's season during fitting. This is intended to reduce the bias that occurs when a downstream model learns from in-sample predictions that are unrealistically accurate.

The exact logistic features, regularization, fitting method, and calibration transformation require source-code verification.

### 5.6 Score, winner, and uncertainty outputs

The model supports a set of related outputs:

- Expected points for both teams.
- Predicted winner and win probability.
- Score uncertainty and score intervals.
- Predicted margin and spread-related quantities.
- Total-points and over/under probabilities.
- Fair American moneyline odds derived from win probabilities.

These outputs should remain internally consistent. For example, a predicted winner should agree with the team's model-derived win probability, and the displayed score margin should agree with the score expectations under the documented display convention.

A forecast's expected score is not a guarantee of the final score, and a win probability is not a confidence interval. They represent different properties of the model's predictive distribution.

## 6. From score distributions to betting probabilities

The history documents a specific correction to the odds calculations: the live workflow previously drew thousands of random scores per game to estimate spread and total probabilities. That simulation was removed in favor of analytical calculations from the fitted Gaussian distribution.

This is an important distinction between live forecasting and offline simulation-based analysis.

### 6.1 Spread and margin

Define the score margin as

$$
D=S_A-S_B
$$

Under a Gaussian-score formulation, if the scores are jointly normal, the margin is also normally distributed:

$$
D\sim\mathcal N(\mu_D,\sigma_D^2)
$$

where

$$
\mu_D=\mu_A-\mu_B
$$

and, in the general jointly normal case,

$$
\sigma_D^2
=
\sigma_A^2+\sigma_B^2
-2\operatorname{Cov}(S_A,S_B)
$$

This is a mathematical description of the method, not a claim that the implementation necessarily estimates a nonzero covariance. The production variance formula must be verified against the score-model code.

For a spread threshold $h$, cover probability can be computed from the margin distribution's cumulative distribution function (CDF), with the sign determined by the sportsbook's spread convention.

### 6.2 Totals

Define total points as

$$
T=S_A+S_B
$$

If the joint score distribution is normal, then $T$ is also normally distributed. Over and under probabilities can therefore be obtained directly from the total distribution rather than approximated by random score draws.

For a whole-number line, the calculation must account for the probability of landing exactly on the line. That probability contributes to a push rather than an over or under win.

### 6.3 Continuity corrections and pushes

Football scores are discrete, while Gaussian distributions are continuous. A continuity correction adjusts the boundary used for a discrete score or margin event to approximate the relevant integer outcomes.

The history explicitly states that the live spread and total calculations use continuity corrections for whole-number lines. It also records a fix preventing a pick'em line from showing a push because tied games go to overtime.

These rules should be kept distinct:

- A cover probability describes the chance a wager wins.
- A push probability describes the chance the wager is refunded under the applicable settlement rule.
- A loss probability describes the chance the wager loses.

For a market with possible pushes, these probabilities need not be interchangeable with win probability for the underlying football game.

### 6.4 Fair American moneylines

Given a modeled win probability $p$, fair American odds can be expressed as

$$
\operatorname{Odds}(p)=
\begin{cases}
-100\frac{p}{1-p}, & p>0.5\\[5pt]
+100\frac{1-p}{p}, & p<0.5
\end{cases}
$$

At $p=0.5$, the fair price is +100 or -100, depending on convention; both represent even money.

These are fair prices derived from the model, not sportsbook quotes. Actual sportsbook odds may include a margin, and the changelog states that the dashboard displays a sportsbook column only when a line was actually recorded.

### 6.5 Closing lines and missing markets

The historical market-learning workflow downloads ESPN closing spreads and totals through SportsDataverse into `data/market_closing_history.csv`. Completed games can use their recorded closing lines to learn the relationship between market prices and final scores.

Upcoming games, however, must use only lines captured before the forecast time. A future closing line cannot be fed into a pregame forecast without introducing leakage.

The changelog also documents two fitting paths:

- A market-aware fit that can learn from games with known closing lines.
- A fit that never saw market lines, used when an upcoming game's line is missing.

This fallback is a safeguard against relying on market-dependent estimates when the corresponding input is unavailable. The exact model parameters and fallback-selection implementation are not established here.

## 7. Training, historical replay, and causal integrity

The forecasting pipeline must do more than fit a model to historical data. It must reconstruct what would have been knowable at each historical kickoff and evaluate forecasts on games not used to fit the model being assessed.

### 7.1 Historical replay

The documented architecture replays team histories in kickoff order. For each forecasted game, the historical state should contain only qualifying earlier observations.

A simplified replay is:

1. Load the historical schedule and recorded results.
2. Normalize game dates, teams, seasons, and postseason status.
3. Sort games by actual kickoff chronology.
4. Build each team's pregame state from prior eligible games.
5. Produce the forecast using information available at that point.
6. Record the forecast and its inputs.
7. Advance the historical state only after the game's result becomes available.

This is a conceptual description of the causal replay process. The precise implementation order, treatment of postponed games, and handling of missing kickoff timestamps need to be confirmed from the active pipeline.

### 7.2 Previous-season initialization

A team entering a new season begins with the previous season's mean under the documented rate policy. Completed games in the new season then contribute to subsequent pregame estimates.

The development history records regression fixes to ensure:

- Previous-season means are not accidentally pooled across all seasons since 2000.
- Season identifiers survive pandas iteration.
- Every qualifying completed game contributes to the current-season mean.
- Bowl and playoff games are not incorrectly assigned to the start of the regular season.
- Postseason games do not contaminate regular-season week-one backtests.

These are not merely data-cleaning details. A chronology bug can create leakage even when the statistical equations themselves are correct.

### 7.3 Training and held-out predictions

The documented second-stage training procedure uses score locations from a head that did not see the corresponding game's season. This reduces the chance that market, rest, team-rate, or win-probability adjustments learn from unrealistically favorable in-sample score predictions.

A rigorous implementation should also preserve a strict separation between:

- Model-fitting data.
- Features available at the historical forecast cutoff.
- Held-out evaluation games.
- The eventual outcome used to grade the forecast.

The history supports the use of season-held-out score predictions in the second stage, but does not establish every split strategy used by every model component.

### 7.4 Model promotion gate

The weekly refit does not automatically replace the current model just because a new model can be fitted.

The documented promotion procedure compares a candidate model against the current model using later held-out games that were excluded from fitting. The candidate replaces the current model only when its held-out forecasts are at least as good under the configured evaluation criterion.

Otherwise, the existing model is retained and the report records that outcome.

When too few later games are available to construct an independent holdout, the report marks the promotion decision as not independent. That status must not be interpreted as successful out-of-sample validation.

A separate documented safeguard keeps the existing model until at least 30 new completed results have arrived, rather than refitting on an insufficient number of newly available results. The history does not establish that this threshold is the only eligibility condition for promotion.

### 7.5 Saved-state consistency

The forecast history records a fix to save the score head used by a forecast and reuse that same head when advancing a saved state. This is intended to ensure that advancing the state without new results produces the same result as replaying the existing history.

The underlying invariant is:

$$
\operatorname{advance}(S,\varnothing)=S
$$

where $S$ denotes the saved forecasting state and $\varnothing$ denotes an empty set of new results. This is a conceptual expression of the desired no-op behavior, not a complete specification of the state-transition implementation.

## 8. Forecast validation and statistical evaluation

A reliable forecasting system needs to assess not only whether its picks win, but whether its probabilities are accurate, whether the model is overconfident, and whether a refit genuinely improves performance.

The history describes a results and accuracy page, a weekly validation process, probability audits, and a model-promotion gate.

### 8.1 Core evaluation metrics

| Metric | What it measures | Interpretation |
| --- | --- | --- |
| Winner accuracy | Fraction of games where the selected winner is correct | Measures pick accuracy, but ignores probability quality |
| Mean absolute margin error | Average absolute difference between predicted and actual margins | Measures point-prediction error for the scoring margin |
| Brier score | Mean squared error of predicted probabilities against binary outcomes | Lower is better |
| Log loss | Penalty based on the probability assigned to the actual outcome | Strongly penalizes confident wrong predictions |
| Weekly Brier score | Brier score calculated separately by week | Reveals changes in probability quality over time |
| Weekly log loss | Log loss calculated separately by week | Helps identify periods of poor probabilistic performance |
| Spiegelhalter's z-test | Statistical check of probability calibration against observed outcomes | Tests whether observed errors are consistent with the forecast probabilities |

For binary outcome $y_i\in\{0,1\}$ and predicted probability $p_i$, the Brier score is

$$
\operatorname{BS}
=
\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
$$

The binary log-loss formula is

$$
\operatorname{LL}
=
-\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i\log(p_i)
+(1-y_i)\log(1-p_i)
\right]
$$

Probabilities used for log loss must be handled numerically so that exact zero or one does not cause an infinite logarithm.

These formulas define the standard metrics; the exact clipping, aggregation, and missing-outcome policies used by the application are not established in the changelog.

### 8.2 Calibration checks

A forecast is calibrated when events assigned probability $p$ occur approximately $p$ proportion of the time across comparable predictions.

The documented weekly calibration system:

- Groups predictions into probability bins.
- Calculates an uncertainty range for each bin.
- Uses a 95% range based on 1.96 standard errors for the sum of independent outcomes.
- Flags bins whose observed results fall outside the permitted range.
- Reports Spiegelhalter's z-test as an overall calibration check.

The intended diagnostic is not that every probability bin must match its forecast rate exactly. Random variation is expected. The uncertainty range provides context for deciding whether the observed discrepancy is unusually large under the specified model.

The stated independence assumption matters. College football games can share teams, conference environments, weather patterns, and other common factors. The described range should therefore be interpreted according to its stated statistical assumptions, rather than as a guarantee that all sources of uncertainty have been captured.

### 8.3 Confidence-group analysis

The results page also groups picks by predicted confidence and compares the fraction that actually won with the fraction the model expected.

The newer validation work adds the range of outcomes that chance permits and flags confidence groups that do not hold up against that range.

This is valuable because a model can have acceptable overall accuracy while being overconfident in its strongest picks. Conversely, a model can select winners at a reasonable rate while assigning poorly calibrated probabilities.

### 8.4 Historical backtests and saved forecasts

The documented results page chooses between two sources:

- Graded forecasts from saved Sunday runs, under `runs/*/validation.csv`, when available.
- The latest season backtest, under `backtest/season-*/graded_games.csv`, as a fallback.

This design lets the dashboard display graded outcomes without requiring a saved weekly run for every historical game.

The history does not establish that the backtest fully simulates every aspect of the live data-collection and deployment environment. Live-run evaluation and historical backtesting should therefore remain separately identifiable in reports.

### 8.5 Probability audit

The probability-audit CLI command was restored after a documentation and implementation mismatch. The history also records a stress-radius default of 0.1 in the CLI.

The supplied information does not fully define the audit algorithm or its interpretation. The radius should not be treated as a confidence level, prediction interval, or acceptable error threshold without consulting the command's implementation and help text.

## 9. Weekly pipeline and operational lifecycle

The weekly workflow coordinates data collection, historical state updates, forecasting, evaluation, publication, and operational reporting.

1. Refresh data

Collect schedules, completed results, historical facts, external validation sources, and closing markets where available.

2. Normalize and persist

Match source records, update team and player facts, preserve provenance, and prepare the chronological history.

3. Fit or retain the model

Evaluate candidate fits against the current model, enforce the documented eligibility and holdout rules, and record the decision.

4. Generate and grade forecasts

Produce score and probability estimates, compute odds-related quantities, and evaluate eligible completed games.

5. Publish and report

Write the run outputs, update the forecast board and supporting views, and emit run summaries and optional external receipts.

The five-stage view reflects the documented weekly-run description. Individual collectors, validation steps, and publishing operations may be scheduled separately or run through different entry points.

### 9.1 Scheduled automation

The development history describes several automated paths:

- A Sunday workflow that refreshes data and runs forecasts.
- Weekly external-source collection and cross-checks.
- A watchdog for monitoring pipeline status.
- A forecast API for status and operational endpoints.
- CI jobs that validate code, packages, dashboards, containers, and selected statistical dependencies.

The external-source workflow was corrected to restore the Sunday runtime through the API because an artifact download could only access artifacts from its own workflow run. The Sunday workflow subsequently looks up the runtime artifact by name, installs tested library versions, and warns when no runtime artifact is restored.

These fixes demonstrate that workflow orchestration and artifact handoff are part of the system's data integrity requirements, not merely deployment convenience.

### 9.2 Failure handling

The documented operational changes include:

- Preserving actual error messages in run manifests when forecast publication fails.
- Emitting short weekly run summaries in GitHub Actions.
- Isolating historical roster failures by season.
- Warning when expected runtime artifacts are missing.
- Requiring the R validation path to fail rather than silently skip when `CFB_REQUIRE_R=1`.
- Running container smoke tests before pushing images.

These measures improve failure visibility. They do not, by themselves, establish that every data source has retries, that all pipeline steps are idempotent, or that automatic recovery is guaranteed.

## 10. Persistence and data contracts

The system uses SQLite alongside files generated by data ingestion, training, validation, and publication workflows.

### 10.1 SQLite responsibilities

The development history explicitly identifies the following stored information:

| Store or table | Documented role |
| --- | --- |
| `team_game_facts` | Recorded team-level historical game facts used by pregame rates |
| `player_game_facts` | Recorded player-level historical facts |
| Forecast log tables | Persisted forecast records and associated operational history |
| Learning tables | Support learning or refit records |
| Watchdog tables | Persist pipeline monitoring state |
| Run artifacts and manifests | Store forecast outputs, validation records, and execution metadata |

The exact names and columns of all learning, watchdog, and forecast-log tables are not supplied. The history states that the schema was brought into alignment with the tables written by the code, but a complete database schema cannot be reconstructed from the changelog.

### 10.2 Database and artifact consistency

An earlier alignment change established a shared SQLite path for scripts, API, Sunday run, watchdog, and Compose services. The history also describes a later relocation of the database out of the artifacts directory.

These two facts should be treated chronologically. The former describes a specific shared path; the latter indicates subsequent storage reorganization. The current canonical database path must be taken from the active configuration, not inferred from the oldest changelog entry.

A sound operational invariant is that the API, scheduled run, dashboard, and watchdog must agree on the active database and artifact locations. If they point to different copies, a forecast may appear in the backend but remain absent from the dashboard.

### 10.3 Forecast record contract

The documented forecast outputs include kickoff, expected score, winner, and confidence or win probability. Other pages use spread estimates, fair American moneylines, and team/player statistics.

A normalized forecast record would logically distinguish:

- Game identity and scheduled kickoff.
- Forecast generation time and data cutoff.
- Predicted scores and uncertainty.
- Win probability and predicted winner.
- Market line and its capture timestamp, if available.
- Model version or identifier.
- Forecast run identifier.
- Eventual final score and evaluation status.

This is a recommended contract for reliable traceability. The supplied history does not establish that every listed field exists in every current forecast artifact or database table.

### 10.4 Published JSON

The React frontend has a documented fallback to `web/public/forecasts.json` when no saved run has been published. The history also describes a full-season forecast view, a season-specific API route, and a focus on the 2026 college football season.

The fallback is a static data-delivery mechanism, not evidence that live API data and the saved JSON are always synchronized. The frontend should identify the freshness and origin of the data it displays.

## 11. API and service architecture

The project history records API endpoints for:

- Forecasts and season schedules.
- Forecast status.
- Learning operations.
- Pipeline status.
- Watchdog status.
- Forecast logging or related operational functions.

The history also mentions a live football service integration and a requirement to use live data without silent fallbacks for certain endpoints.

This implies two distinct interface responsibilities:

1. Prediction delivery: Return the published forecast dataset or current season view.
2. Operational control and observability: Expose the status of forecast generation, learning, pipeline execution, and monitoring.

These responsibilities should remain distinct. A successful HTTP response from a forecast endpoint does not necessarily mean that the latest scheduled run completed successfully. Likewise, a healthy watchdog does not prove that the model is calibrated.

The exact API framework, endpoint paths, request and response schemas, authentication rules, and consistency guarantees are not established by the supplied history.

## 12. Frontend and dashboard design

The history documents two substantial presentation surfaces: a React/Vite board and a Streamlit forecast desk.

### 12.1 React/Vite prediction board

The React application evolved through several UI and deployment changes, including a focus on 2026 college football, a full-season forecast view, live results fetching, a season API route, and improvements to forecast readability.

The documented board capabilities include:

- Games grouped by day.
- Expected scores displayed separately for each team.
- Win probabilities and highlighted picks.
- A concise predicted-margin summary.
- A team search and an all-games toggle.
- A default view focused on the forecast week.
- A full-season view.
- A moneyline and spread interface.
- A team ratings section.
- Season status and clearer handling of missing data.

The history also records a rebranding to CFB PREDICTIONS and multiple deployment-path fixes. These frontend changes should be understood as presentation and delivery improvements; they do not, by themselves, demonstrate improved predictive accuracy.

### 12.2 Streamlit Forecast Desk

The Streamlit application was reorganized around readable forecast pages.

The user-facing Forecasts section includes:

- Predictions: Upcoming games grouped by day, with expected points and win probabilities.
- Odds: Model-derived spreads and fair odds, plus recorded sportsbook prices where available.
- Past results: Historical forecasts and results.
- How it works: Explanations of the model and its outputs.

The behind-the-scenes section contains operational and research pages, including overview, book prices, game inputs, statistics, forecast logs, pipeline, training, research, and sources.

The dashboard uses `st.navigation` to separate the primary forecasting experience from operational inspection. The history also records improved text contrast, larger text, mobile layout adjustments, and a fallback to the published forecast JSON.

### 12.3 Results and accuracy page

The results page connects saved forecasts to final scores and reports the documented evaluation metrics. It also supports weekly trends and confidence-group calibration diagnostics.

This separation is architecturally useful: a prediction board is optimized for consuming forecasts, while the results page is optimized for determining whether the forecasting system deserves trust.

The two views should share the same underlying forecast identity and grading definitions so that a prediction shown to users can be traced to the result later reported by the evaluation system.

## 13. Deployment, CI, and release controls

The development history records a multi-environment deployment setup with separate frontend and dashboard concerns.

### 13.1 Deployment targets

- Vercel: Serves the React/Vite application through the configured build output.
- Docker: Packages the forecasting and dashboard services.
- Docker Compose: Coordinates local or deployment services.
- GitHub Container Registry: Receives validated container images.
- GitHub Actions: Runs automated checks and scheduled forecasting workflows.
- Datadog: Receives logs when `DD_API_KEY` is configured.
- Email: Receives completed-run receipts, according to the documented publication workflow.

The history describes the removal of a generated deployment that mounted an old Downloads directory. This illustrates the importance of deployment configuration matching the actual checked-out repository rather than a machine-specific development path.

### 13.2 CI responsibilities

The documented CI improvements include:

- Python linting and crash-class error checks.
- Python 3.13 coverage.
- A pinned-dependency test path with R installed.
- Clean wheel installation checks.
- A dashboard job that imports page modules and runs dashboard tests.
- A React build.
- Container builds and command-level smoke tests.
- Package-version reporting from `pyproject.toml`.
- Concurrency and timeout controls.
- Workflow fixes for image naming and publication.

A specific container workflow bug was corrected by passing Docker Bake variables through the environment instead of build arguments that the Bake configuration ignored. The workflow now builds both forecast and dashboard images, tests them, and pushes them only after the checks pass.

The history reports an earlier test snapshot of 89 passed and 1 skipped, followed by a later changelog entry reporting 161 passed and 2 skipped after additional ingestion and audit features. These are historical statements from the supplied development history, not a claim about the current test status of PR #52.

### 13.3 Cross-platform runtime

The project has documented efforts to consolidate Python environments, remove duplicate frontend folders, configure a cross-platform Python runtime, and align local development with CI and the Sunday workflow.

The development container is described as installing tested library versions, R, Node, and the web packages. The package manifest pins the versions already represented in the lockfile rather than relying on `latest`.

These measures reduce environment drift, but reproducibility still depends on the completeness of version pins, system-level dependencies, downloaded datasets, and model artifacts.

## 14. End-to-end data flow

The following sequence brings together the documented architecture into a single operational narrative.

mermaid
```

flowchart TD
    A["External football sources"] --> B["Ingestion and source matching"]
    B --> C["Historical facts and provenance"]
    C --> D["SQLite and Parquet"]
    D --> E["Chronological pregame state"]
    E --> F["Unified score forecasting model"]
    G["Pre-cutoff market lines"] --> F
    H["Current forecast context"] --> F
    F --> I["Scores and uncertainty"]
    I --> J["Win probability and winner"]
    I --> K["Spread and total probabilities"]
    J --> L["Fair moneyline odds"]
    K --> M["Forecast artifacts and logs"]
    L --> M
    J --> M
    M --> N["API and published JSON"]
    N --> O["React/Vite board"]
    N --> P["Streamlit dashboard"]
    M --> Q["Final-score grading"]
    Q --> R["Calibration and accuracy reports"]
    R --> S["Candidate model evaluation"]
    S --> T{"Pass promotion gate?"}
    T -->|Yes| U["Promote candidate"]
    T -->|No| V["Retain current model"]
    U --> F
    V --> F
```

Conceptual flow only. The diagram shows logical dependencies, not verified execution order or every branch in the production code.

Three contracts are especially important for this architecture:

1. Temporal contract: Forecast inputs must respect the cutoff, and historical replay must not expose later results to earlier forecasts.
2. Statistical contract: Score estimates, winner probabilities, spreads, totals, and fair prices must use consistent model definitions.
3. Publication contract: The forecast that is graded later must be identifiable and traceable to the forecast that was originally published.

These contracts provide a useful framework for auditing future changes.

## 15. Architectural strengths and engineering trade-offs

### 15.1 Strengths supported by the history

A unified forecasting approach. The model is designed to connect score expectations, winner probabilities, and betting-market quantities rather than relying on unrelated predictions that can contradict one another.

Explicit controls against data leakage. Cutoff-safe ingestion, chronological replay, season-held-out second-stage predictions, and historical market-line handling address several common forecasting failure modes.

Controlled model replacement. A candidate model must be compared against the current model on later held-out games, with an explicit status when an independent holdout is unavailable.

Direct probability calculations. Removing Monte Carlo sampling from the live spread and total calculations makes the documented calculation path more direct and reproducible, provided the assumed distribution and boundary rules are correct.

Persistent operational records. SQLite, manifests, logs, and validation artifacts provide a foundation for diagnosing failures and tracking forecasts over time.

Multiple presentation and operational surfaces. React serves the interactive prediction board, while Streamlit supports both forecasting and deeper operational inspection.

Improved deployment discipline. The documented CI work covers frontend builds, package installation, container smoke tests, and workflow corrections before image publication.

### 15.2 Trade-offs and remaining risks

The history also highlights areas that warrant ongoing attention.

- Complexity: A unified model containing hierarchical state, Koopman dynamics, Gaussian scores, Poisson rates, logistic probabilities, and market adjustments requires clear mathematical documentation and carefully controlled interfaces between components.
- Source coverage: A larger catalog does not guarantee that every source is fresh, correctly matched, available before kickoff, or used by the model.
- Probability assumptions: Gaussian approximations and continuity corrections need validation against discrete football scores, particularly for low-probability events and unusual score margins.
- Market dependence: Closing lines are useful for learning from completed games, but must never enter an earlier forecast as future information. Missing-line fallback behavior needs independent testing.
- Validation uncertainty: A passing promotion gate on a small or unrepresentative sample does not guarantee future improvement.
- State consistency: Multiple workflows and services can diverge if they use different database paths, stale JSON, or mismatched model artifacts.
- Deployment complexity: Vercel, Docker, scheduled GitHub Actions, APIs, SQLite, and optional monitoring integrations create several places where a deployment can succeed technically but publish stale data.
- Feature utilization: The presence of news, roster, player-stat, and weather collectors does not establish that those features improve forecast performance. Their contribution should be measured rather than assumed.

## 16. Recommended engineering controls

The following recommendations are derived from the documented architecture. They are proposed controls, not claims that the project currently lacks them or that they have already been implemented.

### 16.1 Make forecast lineage explicit

Every published forecast should be traceable to:

- A stable game identifier.
- A forecast timestamp and data cutoff.
- A model version or immutable model-artifact identifier.
- The source snapshot or feature-state version.
- The forecast run and publication status.
- The eventual grading record.

This would make it easier to investigate discrepancies between React, Streamlit, SQLite, and saved JSON.

### 16.2 Enforce cutoff safety with automated tests

Create tests that deliberately insert a later result, a late-published news item, or a market line captured after kickoff. Verify that these additions do not change the historical forecast.

The tests should cover regular-season games, bowls, playoffs, postponed games, and games against non-FBS opponents. The history documents fixes for several chronology errors, making these particularly valuable regression tests.

### 16.3 Validate every probability transformation

For the Gaussian score calculations, market adjustments, and logistic probability layer, test:

- Probabilities remain in the interval $[0,1]$.
- The outcome probabilities satisfy the appropriate normalization rules.
- Spread and total win, push, and loss probabilities reconcile.
- Whole-number lines apply the correct settlement convention.
- Missing market data activates the intended fallback.
- Fair odds correspond to the published probabilities.
- Extreme inputs do not produce non-finite values.

Where the implementation uses approximations, compare them with exact or independently computed reference results under the same assumptions.

### 16.4 Separate data availability from model impact

For every source or feature, record whether it was:

1. Successfully collected.
2. Available before the forecast cutoff.
3. Matched to the correct team or game.
4. Included in the model input.
5. Used by a fitted model component.

Then evaluate the incremental predictive contribution of the feature using a leakage-safe comparison. This is particularly important for newer news, roster, weather, and player-stat ingestion paths.

### 16.5 Strengthen promotion reporting

A model promotion report should identify the candidate and incumbent, holdout period, eligible game count, evaluation metrics, uncertainty, promotion decision, and reason for retaining the current model.

The existing documented retained/promoted status and independent-holdout gate provide a basis for this. Additional reporting would make it easier to distinguish a genuine improvement from an inconclusive comparison.

### 16.6 Make freshness visible to users

The dashboard should distinguish:

- The scheduled game date.
- The forecast generation time.
- The latest completed data refresh.
- The model version used.
- Whether the displayed data came from the API or a saved artifact.
- Whether the latest scheduled run succeeded.

These fields would help prevent an older but valid forecast from being mistaken for a current one.

## 17. What the supplied history does not establish

A rigorous architecture document should not imply more certainty than the evidence supports. The following details remain unresolved from the changelog alone.

| Area | Information requiring source inspection |
| --- | --- |
| Model mathematics | Exact state vector, Koopman implementation, score-head parameterization, and interactions between Gaussian and Poisson components |
| Dependence structure | Whether team scores are independent, jointly modeled, or adjusted for shared variance |
| Hierarchical fitting | Exact hierarchy, priors, regularization, and optimization procedure |
| Feature usage | Which collected news, roster, weather, and player fields reach each live model component |
| API contracts | Exact routes, schemas, error codes, authentication, and concurrency behavior |
| Database schema | Complete table definitions, migrations, keys, and transaction boundaries |
| Model registry | Artifact versioning, promotion atomicity, rollback behavior, and reproducibility guarantees |
| Source provenance | Whether each forecast has a complete, immutable input manifest |
| Deployment state | Which historical paths and configuration values are currently authoritative |
| Current verification | Current test results, CI status, production health, and actual forecast accuracy |

The supplied development history is evidence of documented engineering work, not independent proof that every feature is currently deployed or functioning.

## 18. Reference glossary

| Term | Meaning in this document |
| --- | --- |
| FBS | Football Bowl Subdivision, the NCAA Division I football subdivision covered by the project |
| Pregame rate | A statistic calculated from qualifying historical games before the forecast cutoff |
| Hierarchical state | A model representation of persistent or structured game/team characteristics |
| Koopman map | A state-transition representation used in the documented forecasting architecture |
| Score head | The model component producing score-related predictions or score locations |
| Cutoff-safe | Uses only information available by the specified forecast time |
| Leakage | Use of information unavailable at the time the forecast would have been made |
| Calibration | Agreement between predicted probabilities and observed event frequencies |
| Brier score | Mean squared error of predicted probabilities against observed binary outcomes |
| Log loss | Probability scoring rule that penalizes assigning low probability to the event that occurs |
| Closing line | A betting-market spread or total recorded near the market's close |
| Fair odds | Odds implied by a probability without an added sportsbook margin |
| Promotion gate | The rule determining whether a candidate model replaces the current model |
| Backtest | Historical simulation used to evaluate forecasting performance |
| Provenance | Information identifying the source, version, timing, and lineage of data |

## 19. Conclusion

FieldQuant FBS Forecast Desk is documented as a forecasting platform whose engineering has progressively focused on more than producing a winner for each game. Its architecture encompasses historical data collection, causal pregame state construction, a unified score and probability model, analytical market calculations, statistical validation, controlled model promotion, persistent forecast records, multiple dashboard interfaces, and automated deployment.

Its most important design commitments are the separation of pregame information from future outcomes, the consistency of score and probability calculations, the evaluation of candidate models on later held-out games, and the ability to trace published predictions into their eventual results.

The next stage of engineering maturity should prioritize making these commitments independently verifiable: immutable forecast lineage, automated leakage tests, explicit probability invariants, measured feature contributions, transparent model-promotion reports, and clear data-freshness indicators.

Those controls would make the system easier to audit, maintain, and evaluate without confusing the existence of sophisticated model components with demonstrated forecasting skill.