# October 12 design: implementation and remaining estimation work

This document maps the supplied design to the current Python repository. The original design is retained in [2026-10-12-football-system.md](2026-10-12-football-system.md). The weekly record and earlier product-development document alongside it are historical evidence; their product versions and test counts do not describe today's checkout.

The base of this update is `b9976e17936d47b0d0dbbeccbef3b6fbbb22e079` on `main`. The attached ZIP identifies commit `241c9d8329e0dda8f79ced5f6e2fddc73fd7c81d`. Current main already has later package restructuring and forecast corrections. Copying the ZIP over main would discard those changes. The attached `README.md` begins with accumulated commit messages; it is not used to replace the maintained README. The attached Datadog configuration is retained verbatim at the repository root.

| Design section | Implemented path | What remains |
| --- | --- | --- |
| Source acquisition | `ingest/data.py`, `ingest/sources.py`, `ingest/capture_log.py` | Source-specific adapters and timestamped coverage for every requested field; a catalog URL is not an implemented parser. |
| Database | `storage/database.py`, `storage/football_database.py`, `storage/evidence.py` | Canonical play, drive, athlete identity, roster membership, unit and availability observation tables with enforced provenance. |
| Observed variables | `ingest/features.py`, schedules and advanced summaries | Provider-model/definition versions for EPA and all play filters; pressure, depth chart, opportunity and personnel coverage. |
| Football states | `modeling/ratings.py` | Team offense/defense states exist for the configured five metrics. The design's position-specific player and unit vectors are not fitted. |
| Hierarchical inference | Conditional Gaussian/Kalman team-state updates | A jointly fitted team/player/unit hierarchy and learned population covariances. |
| Evolution | Weekly diffusion and offseason carry | Learned roster, coach, transfer and recruiting transition effects. |
| Observation equations | Team-game score and efficiency observations | Student-t play EPA, Bernoulli success/explosive/sack likelihoods with jointly identified player and unit effects. |
| Matchups | Existing fitted score-head state products | The expanded jointly estimated interaction matrix and shrinkage distribution. |
| Drives/possessions | Experimental `modeling/joint_forecast.py` | Audited drive extraction, estimated shared-clock opportunities, terminal outcomes, field position and overtime. The experiment's fixed coefficients are not learned evidence and are not used for this run. |
| Parameters | Saved base score head, residual adjustment and covariance | Posterior samples and fit diagnostics for the expanded hierarchy. |
| Game distribution | `modeling/forecast.py`, `modeling/analytic.py` | Existing calibrated joint Gaussian; no claim of integrating an expanded player/drive posterior. Team-total probabilities and individual score intervals are now exposed from its marginal distributions. |
| Forecast preservation | `storage/evidence.py`, publisher | Full-precision predictions and content-addressed input files are frozen. Database triggers reject UPDATE/DELETE on evidence tables. Existing pre-update runs require their retained artifacts before they can be frozen. |
| External ratings | Catalog and existing source collectors | Historical fitted source bias/scaling/error and independent-information tests. |
| Market observations | `odds/market.py`; explicit forecast mode | `football` is now the default and excludes market covariates in both adjustment training and inference. `market_conditioned` explicitly opts in. Quotes still price spreads/totals and supply comparisons in both modes. Matched evaluation must establish any incremental value. |
| Completion/validation | Existing validation plus `evaluation/evidence.py` | Frozen-run evaluation now stores separate versioned outcomes, score/margin/total errors, Brier, log loss, Gaussian CRPS and 80% interval coverage. Live outcomes must actually be final and available. |
| Operating cycle | Existing Sunday and weekly orchestration | Full expanded statistical method cannot be promoted until its missing inputs, estimators and held-out comparisons exist. |

## Executable changes in this update

`forecast` writes `forecast_inputs.json`, the full fitted production model under `inputs/`, original source files under `inputs/sources/`, its selected location head, the fitted adjustment, and `state_snapshot.npz`. Known inputs retain missingness as JSON null, with explicit column order. Full precision is retained independently of rounded UI values.

Publication freezes those artifacts before replacing the latest board. Retrying an identical run succeeds. Changing predictions, game population, input contents or the input file set under the same run ID fails. Use a new run ID for a new calculation. The publication guard also checks existing operational forecast-log rows, so a legacy saved prediction cannot silently be revised on its first post-update publication.

The evidence database keeps paths to content-addressed files. Back up both the database and its `artifacts/evidence` directory. After moving the bundle, pass `evaluate-run --evidence-dir PATH` to resolve its content-addressed files at their new location. A hash verifies integrity; it does not establish historical source publication time. Forecast records retain the declared cutoff and the actual recording time separately. Evaluations distinguish historical reconstruction from records actually frozen before kickoff. Corrected later outcomes create new evaluation versions; they do not rewrite predictions.

`forecast-advance` inherits the frozen mode. A caller changing mode must do a complete replay. Old manifests without a mode retain their former market-conditioned semantics.

## Run and evaluate

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m cfb_forecast fetch --data runtime/data --seasons 2021 2022 2023 2024 2025 2026
python -m cfb_forecast backtest --data runtime/data --out runtime/artifacts/training
```

Use the actual current UTC time for `AS_OF_UTC`; retain a separate directory for every completed calculation:

```bash
python -m cfb_forecast forecast \
  --data runtime/data --model runtime/artifacts/training/model \
  --out runtime/artifacts/new-football-run --season 2026 \
  --as-of "$AS_OF_UTC" --through "$WINDOW_END_UTC" \
  --forecast-mode football
```

For a market-conditioned comparison, supply `--forecast-mode market_conditioned` and a timestamped `--market` file, writing to another output directory. Both runs must use the same information cutoff, historical source versions and game population. Historical closing lines alone do not prove a quote was available at an earlier forecast cutoff.

After publication freezes a run, settlement verifies retained input hashes and writes clear result/error columns:

```bash
python -m cfb_forecast evaluate-run \
  --database runtime/data/cfb_forecast.sqlite --run-id RUN_ID \
  --data runtime/data --as-of "$EVALUATION_UTC" \
  --out runtime/artifacts/evaluation-RUN_ID
```

Error columns use prediction minus actual. Pending games are counted separately. A reconstructed historical cutoff is never evidence that a forecast was actually issued at that time.

## Connected-service checks on October 15

StatsHawk returned three NCAAF fixtures for October 15: East Carolina–UAB, Georgia Southern–Old Dominion and Colorado State–Texas State. These agree with the corresponding source schedule. Its opaque contest IDs and ESPN numeric game IDs need an explicit identity crosswalk before automated merging. No StatsHawk predictions or player adjustments are substituted into the fitted engine.

Optimal's NCAAF event request for October 15 returned an empty result. No Optimal probabilities or missing player statistics were invented. A connected ChatGPT plugin is not automatically an authenticated service available to scheduled Python jobs.

The Datadog skill and supplied SAST ruleset configuration were inspected. The root configuration selects the supplied Python best-practice/style/framework/security rulesets. The local publication receipt records a Datadog-shaped event. This update does not claim that a remote SAST scan ran, that an agent is deployed, or that logs reached Datadog intake.

## What prevents completion of the expanded fit

The supplied Kaggle ZIPs contain the same `cfb_box-scores_2002-2025.csv` file, despite one archive's 2002–2026 label. They are one data source and contain no 2026 play, drive, lineup or unit observations. The downloaded advanced team summaries support the existing team-state method. They cannot identify all of the proposed quarterback, offensive-line and defensive-front terms simultaneously.

Completing the full design requires real timestamped play/drive/personnel observations, a defined reference lineup and replacement comparison, identifiability constraints separating team/player/unit effects, and an implemented estimator whose saved posterior passes convergence and predictive checks. Adding empty tables or assigning arbitrary coefficients would not complete that estimation. These remain unfinished development work, explicitly separate from the tested executable changes above.
