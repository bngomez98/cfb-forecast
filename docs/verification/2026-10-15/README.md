# Verification on October 15, 2026

This record belongs to the update based on main commit `b9976e17936d47b0d0dbbeccbef3b6fbbb22e079`. Numerical results are retained beside this file. They are measurements of the existing team-state engine and its football-only adjustment, not performance evidence for the unimplemented expanded player/drive hierarchy.

## Reproduction and current forecasts

- Downloaded SportsDataverse schedules and advanced team summaries for 2021–2026. Exact URLs, retrieval times and SHA-256 values are retained in the runtime source manifest; the backtest and forecast manifests here retain source hashes.
- Reproduced the 2025 historical test population: **808 games**. PBP: **588 correct, 72.7723%**, margin MAE **12.4129**, total MAE **12.9585**, Brier **0.183260**, log loss **0.540986**. Score-only: **582 correct, 72.0297%**, Brier **0.181414**, log loss **0.537749**. PBP was selected using 2024 validation; score-only has better probability scores on the recorded 2025 test set. Neither result establishes improved accuracy from this repository update.
- Reconstructed **333 completed games across seven 2026 weeks** using the current football-only path: **254 correct, 76.2763%**, margin MAE **12.6583**, total MAE **12.8075**, team-score MAE **9.0269**, Brier **0.159219**. These were generated today with reconstructed past cutoffs and today's source versions. They are not original forecasts issued before those historical games.
- Generated **55 current forecasts**, information cutoff **2026-10-15T20:59:26.461113Z**, through **2026-10-19T05:00:00Z**. Current-season source capture check passed. There were **4,277** eligible prior FBS games and **4,199** games with advanced summaries. No market file or player-availability adjustments were supplied to this run.
- Froze the current batch as `20261015-prospective-football` before kickoff, retaining **21 input artifacts** including the exact source files and production model. Published to an isolated runtime board and SQLite; no deployed website was changed.
- Re-applied the saved numerical inputs through the saved adjustment. Maximum difference in expected scores or win probability: **7.11e-15**.
- Verified all frozen input hashes. Current settlement: **0 final, 55 pending**. Retrospective reconstructed records are stored separately and explicitly marked as reconstructed.

| Tonight's game | Expected away score | Expected home score | Predicted winner | Win probability |
| --- | ---: | ---: | --- | ---: |
| East Carolina at UAB | 24.44 | 18.45 | East Carolina | 64.48% |
| Georgia Southern at Old Dominion | 21.83 | 21.08 | Georgia Southern | 51.84% |
| Colorado State at Texas State | 28.77 | 44.08 | Texas State | 82.89% |

## Software checks

The Python suite, Ruff checks, and wheel/source-package build were executed. The final pass count is recorded in `checks.txt`. The two skipped scopes require Streamlit and R, which are absent in this execution environment. No frontend components were changed. A remote Datadog scan, deployment build, and browser interaction test were not run.

Regression coverage includes full-precision preservation, identical retries, changed-prediction rejection before board replacement, changed/corrupted input rejection, immutable database records, corrected outcome versions, relocated evidence, football versus market covariates, Gaussian CRPS, and team-total push/over/under probabilities.

The supplied Datadog configuration is included verbatim. Datadog logs remained local; SMTP delivery was disabled for this execution. Connected-service data were used only for coverage/schedule checks: StatsHawk returned tonight's three fixtures; Optimal returned no NCAAF events for October 15.

## Files

`forecasts.csv` contains the full-precision current batch. `reconstruction-weekly.csv` and `reconstruction-2026.json` describe the retrospective 2026 calculation. `backtest-2025.csv` and `backtest-manifest.json` describe the earlier-season fit/selection/test roles and results. `replay-verification.json` retains the saved-input replay and separate reconstructed-run evaluations. `attachment-sha256.json` identifies the attachments used for reconciliation.

The reproducible runtime bundle supplied with the update retains data, fits, forecasts, evidence objects, and a SQLite backup. Retrieve source code from the update's Git commit. After moving the runtime bundle, use `evaluate-run --evidence-dir runtime/artifacts/evidence` with its restored database; absolute paths originally recorded in that database refer to this execution workspace.
