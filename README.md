# College football forecasting engine

A runnable Python package that downloads free college-football data, learns changing offense and defense states, produces score and win-probability forecasts, simulates joint scores, and evaluates predictions chronologically. Every delivered forecast has an explicit information cutoff, source hashes, model settings, and saved state features.

This package was built from the earlier CFB historical backtests and extended into a reusable engine. The 13 judgment-based forecasts discussed in chat are not training inputs. The historical performance here is computed from this code, not inferred from those forecasts.

## Run in the cloud

The `.devcontainer/devcontainer.json` file supports GitHub Codespaces. Put the extracted project in a repository and open a Codespace. Its setup installs the package and testing dependencies. It does not require creating a virtual environment on your Windows computer.

From the project directory in any Python 3.11+ environment:

```bash
python -m pip install -e ".[test]"
python -m pytest -q

# Reproduce the bundled 2025 historical evaluation from the exact saved inputs.
python -m cfb_forecast backtest --data data --out artifacts/reproduced --archive data/archive/matched_predictions.csv

# Download or refresh the current season. No paid API key is needed.
python -m cfb_forecast fetch --data data --seasons 2026 --refresh

# Generate forecasts using the supplied production model.
# Change both timestamps to the actual desired forecast window, with UTC offsets.
python -m cfb_forecast forecast --data data --model artifacts/backtest/model --out artifacts/new_forecasts --as-of 2026-10-02T13:45:43Z --through 2026-10-05T05:00:00Z --season 2026
```

The first historical run can take several minutes, depending on CPU and BLAS configuration. Single-threaded BLAS (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`) avoids overhead on small matrix updates in Linux. It does not change the model.

Saved 2021–2025 inputs are bundled so the historical evaluation can run offline after dependency installation. The refreshed 2026 inputs and a forecast run for October 2–4 are also included. Forecasts are research outputs; no profitable betting edge was established by the evaluation. Read `RESULTS.md` for the measured results.

## What the engine implements

| Component | Implementation |
|---|---|
| Free inputs | SportsDataverse schedules and ESPN-derived per-game advanced team summaries |
| Dynamic ratings | Metric-specific Gaussian offense and defense states, full covariance, opponent adjustment, weekly drift and offseason shrinkage |
| Pooling | Shared zero-centered Gaussian priors shrink teams with little evidence toward the population mean |
| Scoring model | A regularized, standardized regression head fitted on earlier seasons |
| PBP extension | Prior-game yards per play, yards per pass, yards per rush, first-down creation rate, and offense/defense interactions |
| Probability calibration | Joint score residual covariance from an out-of-fit validation season |
| Joint simulation | Correlated score draws, nonnegative integer rounding, approximate overtime, margin/total intervals and push probabilities |
| Market diagnostics | Timestamp-filtered spread, total, moneyline and odds input; cover probabilities and expected net return calculations |
| Baselines | Chronological calibrated Elo and retained public pregame forecast snapshots |
| Verification | Tests for future-outcome exclusion, state covariance, time ordering, market timestamps, deterministic simulations, score-head serialization and invalid data |

The engine does not yet estimate player ability or injury effects, train its own EPA model, simulate individual drives, or learn information incremental to the market. Provider EPA, win probabilities, Elo and betting fields are deliberately excluded from the current statistical inputs. Raw-play cleaning and success/explosive-rate aggregation are available as a separate extension; the current fitted PBP head uses the deterministic per-game yardage and first-down fields in the advanced summaries, not that optional raw-play aggregate.

## Historical design

Default chronology:

| Season | Role |
|---|---|
| 2021 | Warmup; estimate normalization centers/scales; initialize ratings |
| 2022–2023 | Fit candidate score/probability mappings |
| 2024 | Select state drift, offseason carry, regularization, and score-versus-PBP variant; estimate out-of-fit score covariance |
| 2025 | Evaluate frozen fitted heads and settings; refresh team states only after earlier games become available |

The released **heldout** models train their heads through 2024. The **production** models are separately refitted through 2025 after the 2025 test predictions are written. Refitting does not overwrite the held-out probabilities or metrics. Production forecasting rejects a model trained through the season being forecast.

These years have been studied in prior work. The implementation respects chronological fitting, but this is a reproducible historical evaluation rather than a newly blind experiment. A later prospective season is needed to assess whether the improvements hold.

FBS-versus-FBS games form both the rating and evaluation population. Missing PBP summaries do not remove a game: its final score still updates score states. Missing metric observations skip that metric's update, leaving its state shrunk with greater uncertainty. FBS–FCS games currently do not update these states.

## Information cutoff and source quality

Historical forecasts use Monday 00:00 UTC before kickoff. For the archived comparison games, the model instead uses the retained Git commit timestamp of that game's competitor forecast. A result can update the states only when its availability timestamp is **strictly earlier** than the forecast cutoff. The code asserts this condition and records the latest contributing result.

When actual completion timestamps are absent, result availability is conservatively approximated as kickoff plus eight hours. That is not an assertion that a game lasted eight hours. It prevents the model from taking a final score immediately at kickoff. A known `completed_at` field overrides the proxy.

Schedules supply final outcomes. The largest score appearing in PBP is never treated as a final result. Duplicate game IDs and duplicate team-game summary rows fail validation. Rows marked completed without usable scores are explicitly quarantined, counted in `data_audit.csv`, and excluded from rating updates. Scores are not fabricated.

Historical datasets can be revised after games. Chronological result exclusion prevents later games entering earlier ratings; it does not reconstruct every historical revision or when each injury/stat correction was first published. Source hashes identify the exact record versions used. Source manifests distinguish the prior saved snapshots from newly downloaded files. Refreshing historical data changes reproducibility; use bundled snapshots for matching this run.

The archive's commit timestamp and SHA are checked against retained metadata, and probabilities/lines are checked against its original CSV snapshots. This verifies the package's internal archive consistency. The retained metadata comes from the earlier archive verification; the backtest does not authenticate GitHub commits anew on each offline run.

## Model equations

For each metric, offense `o` and defensive prevention `d` are Gaussian latent states. Positive defense strength means fewer points/yards allowed. Each side's standardized observation follows:

```text
home metric = home offense - away defense + venue adjustment + observation error
away metric = away offense - home defense - venue adjustment + observation error
```

The venue adjustment in score-state updates is half the configured 2.5-point home margin on each side. PBP metric updates have no fixed venue term; the fitted scoring head can learn their predictive relationship.

During the season, `theta_next = rho * theta + eta`, where `rho = 0.995 ** elapsed_weeks` and `Var(eta) = q * elapsed_weeks`. Across the offseason, the chosen carry factor shrinks the mean, and a separate variance increment represents uncertainty from roster/coaching changes. It does not individually model those changes.

Conditioning uses the Gaussian Kalman formulas:

```text
S = H C H.T + R
K = C H.T inverse(S)
mean_new = mean + K (observation - H mean)
C_new = C - K H C
```

Each metric uses separate states/covariance; both teams' observations in a game are conditioned jointly. The observation residual correlation is fixed at 0.10. Shared zero-centered priors identify offense/defense decompositions and produce partial pooling. State hyperparameters are selected empirically on 2024; this is not full Bayesian integration over unknown hyperparameters.

The scoring head receives pregame offense, defense, their differences/products, uncertainty summaries, venue and early-season terms. It predicts both sides with the same standardized ridge regression. Output score means are clipped at zero. The score-only head uses a restricted six-column subset. The PBP head uses all 26 columns.

The 2024 residual covariance contains game variation and realized estimation error. It is used as the forecast covariance without adding state uncertainty again. This empirical joint Gaussian approximation does not explicitly integrate every posterior parameter uncertainty through the nonlinear scoring head.

Historical win probabilities integrate the continuous Gaussian margin analytically. Simulations additionally round/clamp scores and resolve tied draws with a generic 3- or 7-point overtime increment. Both the analytical probability and the bounded simulation probability are exported so the approximation is inspectable. These simulated integers are not a football possession/rules engine. Intervals include substantial game uncertainty; the two score means are expectations, not an assertion of an exact final score.

## Market input

`examples/market_template.csv` documents the optional input columns. Supply actual observed lines, with `observed_at` including a timezone. The latest snapshot at or before the cutoff and strictly before kickoff is selected. A later snapshot never enters an earlier forecast. Missing odds are not invented.

Use standard **home-spread** signs: a home favorite giving 3.5 points has `home_spread=-3.5`. A simulated home cover is `home_score - away_score + home_spread > 0`; an equality is a push. `home_decimal` and `away_decimal` are the two spread prices. Moneyline fields are separate and are used only for a no-vig winner-probability comparison.

```bash
python -m cfb_forecast forecast --data data --model artifacts/backtest/model --out artifacts/with_market --as-of 2026-10-02T13:45:43Z --through 2026-10-05T05:00:00Z --season 2026 --market examples/my_actual_market.csv
```

Net expected return per unit is `P(win)*(decimal_odds-1)-P(loss)`; pushes return the stake. A positive calculated value depends on the validity of the model probability and does not establish an edge by itself. There is no automatic stake-sizing or bet-placement component.

The old forecast archive uses `lineca` as a **home margin target**, which is the opposite sign from a conventional home spread. Its `lineavg` is an aggregate public forecast margin, not a market spread. Archived spread diagnostics are descriptive: original sportsbook prices and exact line availability are not established, so the package reports no historical betting ROI.

## Outputs

| File | Contents |
|---|---|
| `artifacts/backtest/predictions.csv` | Every held-out prediction and final outcome, cutoff and latest contributing result |
| `artifacts/backtest/metrics.csv` | Winner accuracy, log loss, Brier score, margin/total/score error |
| `artifacts/backtest/validation_trials.csv` | All score/PBP settings tried on 2024 |
| `artifacts/backtest/elo_validation.csv` | Elo calibration selection on 2024 |
| `artifacts/backtest/paired_comparisons.csv` | Season-week block-bootstrap comparisons |
| `artifacts/backtest/archive_comparison.csv` | Matched pregame external forecasts and model outputs |
| `artifacts/backtest/archive_spread_diagnostics.csv` | Descriptive spread directions at fixed edge thresholds |
| `artifacts/backtest/run_manifest.json` | Fit/evaluation boundaries, source hashes, model choices and limitations |
| `artifacts/backtest/model/model.json` | Selected production model, normalization, coefficients and covariance |
| `artifacts/backtest/models/*/heldout/model.json` | Frozen heads used for 2025 evaluation |
| `artifacts/forecast_2026-10-02/forecasts.json` | New computed score/probability forecasts and simulation intervals |
| `artifacts/forecast_2026-10-02/forecast_manifest.json` | Forecast cutoff, every contributing game ID and source/model hashes |
| `artifacts/forecast_2026-10-02/state_snapshot.npz` | Exact state means/covariance and feature vectors for that run |

Source links:

- https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_schedules
- https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_team
- https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_pbp
- https://github.com/ssmarathontrading/ncaa_predictions

## Extending the package

The current modules separate data ingestion, metric observations, dynamic state updates, score heads, simulation, market evidence and evaluation. Add a new deterministic past-game metric through `features.py`, with an explicit missing-data policy. Add a different predictive head through `model.py`, retaining chronological fitting. Add new generative simulations through `simulate.py` and validate their probabilities against held-out outcomes.

The next substantial forecasting improvement should be evaluated prospectively or on an additional untouched period. Candidate work includes calibrated player/availability inputs, possession/drive modeling, home-field heterogeneity and predicting residuals against timestamped market forecasts. None should be promoted into production solely because it sounds more sophisticated.

## Prospective holdouts, market history, and full distributions

Version 0.3 separates **slate forecasts** from the actual **chronological prospective holdout**. A slate can still be generated from one shared as-of state, but prospective evaluation uses a per-game event-time ledger. See `CHRONOLOGY.md`.

The chronological default is a fixed 24-hour pregame cutoff. A game cannot enter the ledger before its own cutoff, forecast timestamps are never backdated, the model hash is fixed for the life of the ledger, and every appended run freezes its exact data/market/model/distribution inputs.

The market and distribution tools remain available:

```bash
# Point-in-time free spread/total snapshot. Repeat on a schedule to build line movement.
python -m cfb_forecast market-snapshot-sdv --season 2026 --store data/market_history.csv

# Historical resolved/closing benchmark only; never used as an earlier forecast input.
python -m cfb_forecast market-backfill-sdv --seasons 2021 2022 2023 2024 2025 2026 --out data/market_closing_history.csv

# Forecasts now save joint score, margin and total probability distributions.
python -m cfb_forecast distribution-backtest

# Start the chronological ledger from an existing frozen slate, keeping only games already due.
python -m cfb_forecast chronology-bootstrap --frozen-batch artifacts/holdouts/prospective_20261002T185600Z --ledger artifacts/chronological_holdout_2026 --lead-hours 24

# Append only newly due games at later real cutoffs.
python -m cfb_forecast chronology-update --data data --model artifacts/backtest/model --ledger artifacts/chronological_holdout_2026 --season 2026 --lead-hours 24 --market data/market_history.csv

# Verify the append-only chain and every immutable run.
python -m cfb_forecast chronology-verify --ledger artifacts/chronological_holdout_2026

# After games finish, settle the chronological ledger.
python -m cfb_forecast chronology-settle --ledger artifacts/chronological_holdout_2026 --data data
```

The public SportsDataverse `espn_cfb_betting` release is keyed by ESPN game ID and contains resolved home spread and total fields. `market-snapshot-sdv` treats the retrieval time as the availability timestamp and can therefore build a genuine prospective history from the moment collection begins. The historical backfill is kept separate because a resolved/closing line cannot be treated as if it were known at an earlier forecast cutoff.

Forecast output includes a discrete joint score PMF plus derived margin and total PMFs. Settlement evaluates margin and total distributions with CRPS in addition to winner log loss/Brier score and point forecast error.

## One-command prospective collection

Run this at each real collection time. It stamps one execution time across the market capture and forecast cutoff, appends only games whose individual 24-hour cutoff is due, and writes an audit receipt. A market-source failure is recorded but never replaced with an invented quote.

```bash
python -m cfb_forecast prospective-run \
  --data data \
  --model artifacts/backtest/model \
  --ledger artifacts/chronological_holdout_2026 \
  --season 2026 \
  --lead-hours 24 \
  --market data/market_history.csv
```

Between newly available final results, the chronological updater advances the exact frozen state and verifies prior source hashes without rereading Parquet inputs. Once a new final result becomes eligible, it deliberately requires a full replay so the state update uses the game-level advanced summaries.
