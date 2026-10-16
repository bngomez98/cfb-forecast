# College Football Forecasting Project Development Documentation

Product and mathematical implementation as of October 6 2026

This document starts with the private CFB Forecast Lab that exists today, explains the calculations behind its forecasts, preserves the evidence accumulated during development, and specifies the work needed to continue building the established method. The immediate development objective is to connect the newly available football data to reproducible improvements in current-season predictions.

The current product is private Site version 3. It contains historical and current team statistics, player histories, a 2026 play-by-play viewer, dynamic opponent-adjusted ratings, saved forecasts, interactive score simulations, and historical validation. Its executable forecasting method estimates changing offensive and defensive team states, translates those states into expected scores with ridge regression, and derives probabilities from correlated score errors. The broader developed methodology includes player contributions, availability, richer play-level efficiency, and possession-based scoring. Those parts have different implementation states, documented below.

## 1 Current product

### Product identity and version

| Item | Current value |
| --- | --- |
| Product | CFB Forecast Lab |
| Published application | https://cfb-forecast-lab.bngomez772.chatgpt.site |
| Published Site version | 3 |
| Publication completed | October 6 2026 at 22:28:09 UTC |
| Published source commit | `acab8c7f07ced19d0df943206fa16d463acb80c8` |
| Previous Site version 2 commit | `93e840069b0928b53d60c0583aacdf52bb3e0148` |
| Access | Private to the owner |
| Retained chronological engine record | Engine 0.3.1 |
| Python package version in the source | `cfb-forecast` 0.1.0 |

Site version, Python package version, and engine-record version identify different things. Site version 3 identifies the deployed application. Package version 0.1.0 comes from `pyproject.toml`. Engine 0.3.1 identifies the separately retained chronological evidence package. A change to the interface or statistics dataset does not establish a new fitted forecasting method.

### Available application views

| View | Delivered behavior | Data scope |
| --- | --- | --- |
| Team statistics | Team comparisons, game logs, scoring trends, season histories, offensive and allowed efficiency | Observed records from 2021–2026 |
| Player statistics | Passing, rushing, receiving, defense, interceptions, fumbles, kicking, punting and return categories; player game logs and season histories | Box scores from 2021–2026; advanced passing/rushing from 2021–2026; advanced receiving for 2026 |
| Play by play | Individual play records and filters for team, offense/defense, game, down, period, play type, score difference and offensive player | 2026 raw plays |
| Dynamic ratings | Estimated offensive and defensive performance against an average opponent | Five metrics used by the executable forecast |
| Saved 2026 forecasts | Existing predicted scores, margins, totals, probabilities and associated information times | Retained forecast snapshot |
| Historical forecasts | Select a historical game and inspect its prediction, outcome and pregame ratings | Retained 2025 evaluation |
| Forecast validation | Accuracy, probability error and score error for historical methods | Historical evaluation populations |

Direct statistics view: https://cfb-forecast-lab.bngomez772.chatgpt.site/?view=1&tab=statistics

The statistics expansion was prepared on October 6 at 22:17:31 UTC. It contains 10,048 team-game rows, 335,932 player-game-category rows, 76,160 advanced player-game rows, and 67,255 play records from 390 games in the 2026 release. The latest covered 2026 kickoff is October 4 at 03:59 UTC. Player-category rows count separate statistical categories, so they are not 335,932 unique appearances.

Raw historical play detail for 2021–2025 is not loaded into the new viewer. Historical play-derived team and player summaries are available. Coverage tables distinguish completed games involving an FBS team from the broader game counts in source box-score files. These populations must remain separate when reporting coverage or forecasting results.

### What feeds predictions today

The active forecasting variant is `pbp`. Its observations are points, yards per play, yards per pass, yards per rush, and first-down creation rate. The newly delivered EPA, success, explosive-play and player statistics support inspection; they do not automatically enter the existing fitted prediction calculation. The statistics preparation script explicitly checks that the previous forecast queries remain unchanged.

The data in the statistics views are newer than the retained forecast snapshot. Displaying a team's October 4 statistics alongside a forecast made October 2 must preserve the dates of both. The later statistics cannot be attributed to the earlier prediction.

## 2 Development objective and established direction

The project predicts current-season college-football performance and outcomes using real football records. Historical seasons provide estimation and testing evidence. Each new game updates the information about team ability; repeated forecast errors guide revisions to the method. The product provides access to that analysis and makes its predictions inspectable.

The established direction combines dynamic hierarchical team abilities, opponent adjustment, play-level performance, player roles and availability, regularized matchup effects, and a joint probability distribution for the two teams' scores. Monte Carlo computation approximates probabilities and score ranges generated by the method. Drawing future scores is distinct from estimating unknown model parameters and from generating historical predictions under an earlier information cutoff.

The active method remains one applied forecasting method. Score-only and Elo variants are retained comparison methods. Earlier discussions proposed ensembles, market inputs, coaching models and other extensions; those proposals do not become part of the established production specification merely because they appeared in a conversation.

### Implementation status of the developed components

| Component | Current status |
| --- | --- |
| Dynamic offense and defense | Implemented as conditional Gaussian states with Kalman updates |
| Opponent adjustment | Implemented through the offensive-team and defensive-opponent observation equation |
| Partial pooling | Implemented through shared zero-centered Gaussian priors and fixed shared controls |
| Five team-performance measures | Integrated into the forecast |
| EPA, positive-EPA success and yardage success | Available in statistics; fuller forecasting integration remains to be fitted and tested |
| Explosiveness | Available under explicitly different provider and yardage definitions; current forecast has no separate explosiveness state |
| Player histories | Delivered for analysis |
| Player ability and availability effects | Defined in the developed methodology; fitted forecast integration remains unfinished |
| Matchup effects | Current ridge head contains metric products and pass/rush differences; richer situational effects remain development work |
| Joint score variation | Implemented using a calibrated two-score covariance |
| Complete propagation of parameter uncertainty | Unfinished; current score covariance is an empirical approximation |
| Possession-based scoring | Separate unfinished experiment; no saved fit or predictions |
| Historical chronological evaluation | Implemented and retained |
| Per-game 24-hour forecast records | Retained in the chronological evidence package; complete corresponding tooling must be reconciled with the published source |

The continuous development task is to improve these specific components and measure their effect. A new architectural description cannot substitute for a fitted component, its saved values, its prediction output, and its evaluation.

## 3 Source architecture

All paths in this section are relative to the published source commit. The forecasting package root is `src/content/engine/`. Its Python module directory is `src/content/engine/src/cfb_forecast/`.

| Source path | Responsibility |
| --- | --- |
| `src/content/engine/src/cfb_forecast/data.py` | Free-source ingestion, schema checks, input hashes, outcome eligibility and availability |
| `features.py` in the Python module directory | Observation arrays, normalization, optional raw-play cleaning and feature-column selection |
| `ratings.py` in the Python module directory | Gaussian state means/covariances, time evolution, opponent adjustment and chronological features |
| `model.py` in the Python module directory | Standardized ridge score head, serialization and residual covariance |
| `simulate.py` in the Python module directory | Analytical win probability and correlated integer score draws |
| `backtest.py` in the Python module directory | Chronological selection, calibration, 2025 evaluation and separate production refitting |
| `forecast.py` in the Python module directory | Replay available prior results, generate future-game predictions and save contributing state |
| `evaluate.py` in the Python module directory | Accuracy, log loss, Brier, score errors, calibration and paired block resampling |
| `market.py` in the Python module directory | Optional timestamp-filtered external quote calculations |
| `cli.py` in the Python module directory | Supported fetch, backtest and forecast commands |
| `src/content/engine/export_current.py` | Verify and export a saved current-season forecast and ratings to the application |
| `src/content/analysis/prepare_statistics.py` | Join and package observed team, player and play records while preserving forecast queries |
| `src/content/dashboard/statistics.mjs` | Decode statistics, aggregate player histories and calculate filtered play summaries |
| `src/content/dashboard/lab-model.mjs` | Browser score simulation and summaries of retained predictions |
| `src/content/dashboard/DashboardContent.jsx` | Ratings, historical/current forecasts, simulations and validation views |
| `src/content/dashboard/SportsStatistics.jsx` | Team, player and play-by-play views |

The Python engine performs statistical estimation and prediction generation. The application consumes saved rows and exposes calculations over those rows. An interactive browser simulation uses the saved expected scores and covariance; it does not fit new team or player abilities.

The published source uses an immutable data reference instead of committing the entire statistics snapshot to Git. The version 3 reference identifies a 104,921,170-byte source snapshot and 13 queries. Some queries use lossless columnar containers, so the outer query-row count is much smaller than the number of underlying football records. Source recovery must verify and expand that data reference before an application build.

Authored application work belongs in `src/content/`; reviewed query data belong in `src/data.json` after verified recovery. The source's `AGENTS.md` protects the application runtime, source inspection, editing, export and publication behavior. Ordinary forecast or statistics changes should use the existing authored boundary and preserve the same private Site.

## 4 Data definitions and contracts

### Forecasting schedule records

The required schedule fields are `game_id`, `season`, `week`, `start_date`, `completed`, `neutral_site`, `home_id`, `away_id`, `home_team`, `away_team`, `home_points`, and `away_points`. Division fields or an explicit `fbs_game` flag identify the forecasting population. Schedule final scores are the outcome authority.

Identifiers must be numeric, a game cannot have the same team on both sides, and duplicate game IDs fail validation. A completed record with missing or negative points is quarantined. A tied score is not a usable completed outcome for the current FBS forecast. These checks produce `invalid_completed_outcome`, `result_usable`, `fbs_game`, and `result_available_at`.

The forecast updates from FBS-versus-FBS games. Observed statistics views cover a broader source population. FBS-versus-FCS performances do not currently update the forecast states, even when they are visible in a team's statistical history.

### Forecasting advanced team observations

The summary key is `(game_id, pos_team_id)`. Required fields are `yards_per_play`, `yards_per_pass`, `yards_per_rush`, and `first_downs_created_rate`. Duplicate summary keys fail validation. Points are added from the schedule to form the observation array with shape `games × 2 sides × 5 metrics`.

The exact source denominator for yards per pass and first-down creation must follow the provider field definition. A pass attempt, a dropback including sacks, and a source-defined pass opportunity are different denominators. The engine presently consumes the provider's already-calculated fields rather than rebuilding these denominators from raw plays.

Before updating the states, the engine bounds observations: yards per play to 0–15, yards per pass to 0–20, yards per rush to −2–15, and first-down creation rate to 0–1. These are control choices that limit extreme observations. Missing values remain missing and skip only the relevant metric update. A missing PBP summary does not remove an otherwise usable game from score-state updates.

### Team and player statistics

The statistics preparation script joins schedule rows to team box scores, advanced team summaries and situational summaries. It checks uniqueness for `(game_id, team_id)` and `(game_id, pos_team_id)`. Team rows preserve the opponent, game date, scoring, passing/rushing, downs, turnovers, possession time, source EPA and available advanced fields.

Player box scores use `(game_id, team_id, athlete_id, category)` as their key. Athlete IDs support player histories across seasons and teams. Advanced passing/rushing/receiving records use a separate name/team/game identity; joining them to an athlete ID requires matching game and team context. A matching name alone is insufficient.

Missing statistical categories are not zeros. Unlabelled `stat_1`–`stat_5` source values remain unclassified. Totals sum additive counts, longest plays use the maximum, and rates divide compatible summed numerators by summed denominators. Per-game QBR is retained without summing it. A covered-history total describes the available records and should not be presented as an exhaustive career total.

### Play-level definitions

The product contains several definitions that must retain their exact names.

| Quantity | Current calculation or source meaning |
| --- | --- |
| Source EPA | Provider estimate of expected points added; the project does not fit the provider's expected-points function |
| Positive-EPA success | Fraction of plays with finite EPA whose EPA is greater than zero |
| Yardage-based success | Gain at least 50% of distance on first down, 70% on second, and 100% on third/fourth |
| Yardage-based explosive play | At least 10 yards on a rush or 20 yards on a pass/sack-classified play |
| Provider explosive rate | Source `EPA_explosive` and `EPA_explosive_rate`; retained under provider semantics |
| First-down creation rate | Provider first-down creation per opportunity; a separate metric from either success definition |

The September 24 example used a 15-yard passing explosive threshold. Current implemented raw-play calculations use 20 yards. The historical example should remain part of the development history, while current outputs use the implemented definition. Changing thresholds later requires a versioned feature definition and a new comparison.

The browser play summary requires a scrimmage play, excludes kneels and no-play penalties, requires down 1–4 and finite yardage, and then applies the selected filters. EPA averages use only plays with finite EPA; yardage-success denominators require valid positive distance. The viewer can include overtime when selected.

The optional Python cleaner has a different purpose and profile. It deduplicates `(game_id, game_play_number)`, validates flags, includes rush/pass/sack plays, excludes kneels and no-play penalties, requires distance 1–99 and periods 1–4, limits absolute preplay score difference to 21, and bounds yardage to −20–80 for its yardage mean. This optional cleaner is not the source of the current fitted PBP observations. The two profiles must not silently share a metric name without recording their filters.

### Source and time provenance

Each retained source file needs its release URL, content hash, byte count and retrieval time where known. Game date, release date, retrieval time and forecast cutoff are separate quantities. A file hash identifies the record version; it does not prove that version existed at an earlier cutoff.

Primary free-source location: https://github.com/sportsdataverse/sportsdataverse-data/releases

The current Python fetcher downloads schedules and advanced team summaries, with optional raw PBP. The larger statistics preparation step expects additional team/player/situational files and an input manifest. The published preparation script is a transformation program; it is not a complete downloader for all those releases.

## 5 Implemented mathematical method

### Observations and standardized units

For metric `m`, let `y(i,g,m)` be the observed performance of team `i` in game `g`. The metric list is fixed in this order: points, yards per play, yards per pass, yards per rush, first-down creation rate.

The 2021 warmup observations determine `c(m)`, the metric mean, and `s(m)`, its standard deviation. The standardized observation is:

```text
z(i,g,m) = [y(i,g,m) - c(m)] / s(m)
```

These centers and scales are fitted preprocessing values, stored in the saved model. They are fixed when making the evaluated forecasts. A one-unit change in a standardized state corresponds to one warmup standard deviation of that metric.

### Offensive and defensive states

For each metric, `o(i,t,m)` is offensive ability and `d(i,t,m)` is defensive prevention. Positive offense increases expected performance. Positive defense reduces the opponent's expected performance.

With `T` team identities, the metric state is a vector of `2T` values: all offensive states followed by all defensive states. Its posterior mean is `a(m,t)` and its covariance is `C(m,t)`. The covariance connects teams and offense/defense within the same metric. Different metrics have separate state systems; cross-metric state covariance is not estimated.

Initial means are zero. Initial covariance is `0.5 × identity`. Teams with little evidence remain close to the common center. The first season-reset event also applies the offseason transformation, so the constructor's 0.5 is not necessarily the covariance immediately before the first observed game.

### Movement over time

For an in-season elapsed interval `w` measured in weeks:

```text
rho = 0.995 ^ w
a_before_observation = rho × a_previous
C_before_observation = rho² × C_previous + 0.02 × w × identity
```

The carry factor slowly shrinks the mean while process variance increases uncertainty. The implementation applies this diffusion only when `0 < w < 15`. Longer gaps use the explicit season-reset event instead of accumulating the weekly variance across the whole offseason.

At the offseason reset:

```text
a_after_reset = 0.80 × a_before_reset
C_after_reset = 0.80² × C_before_reset + 0.30 × identity
```

This retains prior team information while allowing uncertainty about changes between seasons. It does not calculate individual roster changes. Reset events occur 21 days before the earliest scheduled game for each season in the replay population.

### Opponent-adjusted observation equation

Let `v = 1` at an ordinary home venue and `v = 0` at a neutral venue. For the points metric, the standardized venue adjustment is `h = v × 2.5 / [2 × s(points)]`. Other metrics have zero fixed venue adjustment in the state update.

```text
z(home,g,m) = o(home,t,m) - d(away,t,m) + h(m) + error_home
z(away,g,m) = o(away,t,m) - d(home,t,m) - h(m) + error_away
```

The fixed 2.5 points is a home-margin adjustment divided equally between the two score observations. The later ridge head also contains signed venue, whose coefficient is fitted. Consequently, the final score forecast is not obtained by simply adding 2.5 to an otherwise finished predicted margin.

For both observed sides, the standardized observation-error covariance is:

```text
R = [[1.00, 0.10],
     [0.10, 1.00]]
```

If only one side has a valid observation for a metric, the corresponding one-row update uses variance 1. The current observation variance is fixed in standardized units. It is not a separately estimated `R(m)` for every metric.

### Gaussian conditioning

The observation matrix `H` selects the offense of each side and subtracts the opposing defense. Subtract the venue vector from the observed `z` before conditioning. With prior `a` and `C`:

```text
S = H × C × transpose(H) + R
K = C × transpose(H) × inverse(S)
a_updated = a + K × [z_adjusted - H × a]
C_updated = C - K × H × C
```

`S` is the uncertainty in the predicted observation. `K` is the Kalman gain; it is calculated from the uncertainties and is not a manually selected team weight. The code solves the corresponding linear system rather than explicitly forming an inverse, then symmetrizes the covariance numerically.

This is Bayesian conditioning for Gaussian latent states with empirically selected fixed controls. A fully hierarchical Bayesian extension would additionally estimate and propagate uncertainty in the shared controls and richer team/player structure. Current numerical simulation draws scores; it does not sample unknown state controls from their posterior.

### All 26 pregame score inputs

The same feature construction is used for each scoring side. For home, offense is the home state and defense is the away state; away reverses those identities.

| Zero-based columns | Inputs | Count |
| --- | --- | --- |
| 0–4 | Offensive states in the fixed five-metric order | 5 |
| 5–9 | Opponent defensive-prevention states | 5 |
| 10–14 | Offense minus opponent defense for each metric | 5 |
| 15–19 | Offense multiplied by opponent defense for each metric | 5 |
| 20 | Posterior variance of the points offense-minus-defense contrast | 1 |
| 21 | Mean contrast variance for the four other metrics | 1 |
| 22 | Signed venue: +1 home, −1 away, 0 neutral | 1 |
| 23 | Early-season indicator: scheduled week at most 4 | 1 |
| 24 | Offensive pass state minus offensive rush state | 1 |
| 25 | Opponent defensive pass state minus defensive rush state | 1 |

For each contrast, variance is `C(off,off) + C(def,def) - 2 × C(off,def)`. Thus covariance between the offensive and opposing defensive estimate contributes to the uncertainty input.

The score-only comparison uses columns `[0, 5, 10, 20, 22, 23]`. The active PBP variant uses all 26. Offense, defense and their difference contain exact linear redundancy. Ridge produces a unique penalized solution, but the three coefficients cannot be read as three independent causal football effects. The redundancy also affects how the penalty distributes coefficient weight.

### Translation to expected points

Each input column is standardized using a `StandardScaler` fitted on the score-head training rows. These feature means/scales differ from the earlier metric centers/scales. One ridge model is fitted to both sides' scoring rows.

```text
x_scaled(j) = [x(j) - feature_mean(j)] / feature_scale(j)
raw_expected_points = beta0 + sum_j beta(j) × x_scaled(j)
expected_points = max(0, raw_expected_points)
```

The ridge objective is squared prediction error plus `alpha × sum_j beta(j)²`; the intercept is handled separately. The retained active PBP setting is `alpha = 100`. The class constructor defaults to 10, so invoking a constructor without restoring the saved model does not reproduce the selected fitted head.

The coefficients, intercept and scaler arrays are estimated values. Their definitions and storage contract are precise; their numerical values belong to the selected `model.json`. They must be exported from that file rather than guessed or copied from a different fit.

### Fixed controls and estimated values

| Quantity | Meaning | Retained value or source |
| --- | --- | --- |
| `process_variance` | Added standardized state variance per week | 0.02 |
| `weekly_carry` | Weekly mean carry | 0.995 |
| `offseason_carry` | Offseason mean carry | 0.80 |
| `prior_variance` | Constructor prior variance | 0.50 |
| `offseason_variance` | Variance added at the reset | 0.30 |
| `home_margin` | Home-margin adjustment in points-state observations | 2.5 points |
| `observation_correlation` | Same-game two-side residual correlation | 0.10 |
| `R` diagonal | Standardized observation variance | 1 per valid side |
| `alpha` | Active ridge penalty | 100 |
| `c(m)`, `s(m)` | Observed-metric normalization | Saved `center`, `scale` arrays |
| Feature normalization | Score-head input normalization | Saved head `mean`, `scale` arrays |
| `beta0`, `beta(j)` | Fitted scoring intercept and coefficients | Saved head `intercept`, `coef` |
| `a`, `C` | Current latent-state mean and covariance | Saved state snapshot |
| `K` | Gain for a particular observation | Calculated from `H`, `C` and `R` |

## 6 Scores and probabilities

### Joint score-error distribution

The mean vector contains expected home and away points. Error covariance is estimated from 2024 residuals produced by a head trained on 2022–2023. The covariance calculation requires at least 50 calibration games and floors its eigenvalues at 1 point squared.

The retained PBP score covariance is:

```text
Sigma = [[145.8080347241,   4.5372972853],
         [  4.5372972853, 122.9965336957]]  points²
```

These values come from the retained modelled-product parameter record. The marginal home/away errors are correlated by about 0.03388. The implied margin standard deviation is 16.11614 points, and total standard deviation is 16.66971 points. Positive covariance increases total variance and decreases margin variance.

This residual covariance includes actual game variation and realized estimation error in that calibration period. Current forecasts do not explicitly draw every state and ridge coefficient before predicting the scores. Adding a second generic state-uncertainty term to the existing covariance would risk double counting; fuller propagation requires separating those sources and recalibrating the complete predictive calculation.

### Analytical probability

Let `muH` and `muA` be expected scores, `M = home score - away score`, and `T = home score + away score`.

```text
expected_margin = muH - muA
expected_total = muH + muA
margin_variance = SigmaHH + SigmaAA - 2 × SigmaHA
home_win_probability = normal_CDF(expected_margin / sqrt(margin_variance))
```

The continuous Gaussian home-win probability is bounded to `[1e-9, 1-1e-9]`. Historical log loss and Brier use this analytical probability. The saved score means describe expectations, rather than an asserted exact final score.

### Numerical score draws

The Python forecast defaults to 50,000 correlated Gaussian score pairs. It rounds to integers and bounds each score at zero. Tied draws are resolved by selecting either winner with equal probability and adding 3 points with probability 0.6 or 7 points with probability 0.4. This is the current overtime approximation, with fixed controls; it has no fitted possession or NCAA overtime sequence.

The Python seed is a SHA-256-derived value from base seed `20261002` and `game_id`. The browser has its own deterministic generator, seeded from the entered seed, game ID and variant. Its interactive default is 10,000 draws, with a permitted range of 1,000–100,000. The two implementations are reproducible independently, but identical displayed seed numbers do not promise identical draws across Python and JavaScript.

The exported results distinguish analytical home-win probability, simulated home-win frequency, expected scores and simulated average scores. Rounding, bounding and tie resolution can make the simulated averages differ from the original means. Central 80% intervals use the 10th and 90th percentiles. The Python and browser quantile implementations also differ slightly.

More draws reduce numerical noise, without improving the estimated football abilities. At probability 0.5, 50,000 independent draws have about 0.00224 standard error in a simulated probability, or 0.224 percentage points. That is computational approximation error, separate from forecasting error.

### Supported model-derived outcomes

One coherent joint score distribution supports win probabilities, margin probabilities, game-total probabilities and team-total probabilities. For a home handicap `s`, home cover means `M + s > 0`; equality is a push. For game-total threshold `l`, over means `T > l`, under means `T < l`, and equality is a push. Team totals apply the same comparisons to each team's score.

The deployed lab presently provides score/probability analysis and optional scenario calculations. Separate conversations requested an own-model NCAA sportsbook display with moneyline, spread, total and team totals. That product request should consume the same prediction outputs; it does not justify changing the estimator or inserting unrelated bookmaker controls. Full team-total support and the final sportsbook product are separate implementation work. Existing external benchmark evidence remains in the historical record.

## 7 Chronology and fitting

### Statistical training roles

| Period | Role in the retained engine |
| --- | --- |
| 2021 | Metric normalization and rating warmup |
| 2022–2023 | Initial score-head estimation for candidate settings |
| 2024 | Select settings and variant; estimate out-of-fit residual covariance |
| 2025 | Historical held-out evaluation using a head fitted through 2024 |
| Production after evaluation | Refit the score head through 2025 for 2026 forecasting |

Selection tries state process variance `{0.005, 0.02, 0.06}`, offseason carry `{0.5, 0.8}`, ridge penalty `{10, 100, 1000}`, and score/PBP variants. This produces 36 candidate combinations. Validation ranking uses 2024 log loss and a covariance estimated from the candidate's training residuals. The released forecast covariance is then calculated from out-of-fit 2024 residuals.

Team states can update during 2025 from earlier eligible results. The score head and selected controls remain fixed for that historical evaluation. Production refitting happens separately and does not replace the evaluation predictions. Forecasting rejects a model whose training-through season is the forecast season or later.

The 2025 period had already been studied during earlier development. The chronological calculation is reproducible historical evidence; it should not be described as a newly blind test of the entire research process.

### Historical and current forecast times

Historical forecasts default to Monday 00:00 UTC before kickoff. Where a retained comparison forecast has a verified archived timestamp, that timestamp overrides the default so the methods share the same cutoff. These historical cutoffs differ from the later per-game target of 24 hours before kickoff.

Only results with `result_available_at < cutoff` may affect a forecast. Equality is excluded. A verified completion time overrides the fallback; otherwise the source adapter uses kickoff plus eight hours. This is an availability proxy, not a record of the actual final whistle.

A reconstruction made later can enforce earlier-game exclusion while using subsequently revised records. It does not prove that every statistic was available in that exact version at the historical cutoff. Prospective records therefore need actual capture/execution times and their input copies.

### Current-season record counts

| Record | Count | Meaning |
| --- | --- | --- |
| Original saved slate | 54 forecasts | Original batch of pregame predictions |
| Chronological records | 30 | 17 imported due games plus 13 later due games |
| Within the 60-minute target tolerance | 14 | One imported game plus the later 13 |
| Later pregame records | 16 | Beyond the target tolerance; retained separately |
| Games not yet due at bootstrap time | 37 | Omitted from that bootstrap import; 13 subsequently added at their own due time |

The 14-game weekly record reports Toledo–Ball State at 56 minutes after its T-24 target and the other 13 at one minute after their targets. They are pregame forecasts with accurately recorded timing; none should be relabelled as made at the exact target time.

The retained October 2 verification record lists zero settled games at verification. The October 6 weekly document separately contains later final scores. The app's older saved-state verification and the newer written results review must be synchronized through an explicit settlement operation; an old zero-settled count is not today's completion status.

## 8 Accumulated evidence

### Earlier component study

The September component study contains 3,136 games from 2021–2024. It used an earlier implementation, and its labels must remain attached to that study.

| Earlier component | Winner accuracy | Log loss | Brier |
| --- | --- | --- | --- |
| Elo | 67.09% | 0.59593 | 0.20528 |
| Dynamic Strength | 68.81% | 0.57641 | 0.19736 |
| Situational PBP | 68.97% | 0.58416 | 0.20028 |
| Matchup Interactions | 68.14% | 0.59006 | 0.20239 |

Dynamic Strength improved probability scoring over that Elo baseline. The then-tested PBP and matchup additions did not improve probability scoring over Dynamic Strength. This is evidence about those feature definitions, fit and population. It does not establish that every possible play-level or matchup feature lacks predictive value.

### Later 2025 evaluation

The later Kalman/ridge engine evaluated 808 FBS-versus-FBS games with a different architecture and historical sample.

| Method | Correct winners | Accuracy | Log loss | Brier | Margin MAE | Total MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Elo | 562 | 69.55% | 0.5750 | 0.1965 | Unavailable | Unavailable |
| Score only | 582 | 72.03% | 0.5377 | 0.1814 | 12.50 | 12.98 |
| PBP | 590 | 73.02% | 0.5406 | 0.1830 | 12.41 | 12.99 |

The selected PBP method gained eight correct winners and slightly lower margin error than score-only, while score-only had slightly better probability scores. Winner accuracy alone would conceal that difference. The PBP-minus-score log-loss difference was +0.0029 with a reported 95% block-resampling interval of −0.0040 to +0.0105, leaving that comparison unresolved.

PBP margin CRPS was 8.8989 versus 8.9384 for score-only. The central 80% PBP margin interval covered 82.05% of outcomes. Coverage describes the frequency with which outcomes landed inside the interval; CRPS evaluates the forecast distribution against the outcome and rewards both useful concentration and calibration.

On 763 matched pregame public forecasts, PBP log loss was 0.5379 and margin MAE 12.43, compared with 0.5200 and 12.00 for the retained external forecast archive. Those matched comparisons are inherited benchmark evidence. They do not define the present product objective or supply its prediction inputs.

### Metrics used to assess development

For outcome `y = 1` when home wins, probability `p`, predicted margin `m` and actual margin `a`:

```text
winner accuracy = mean[(p >= 0.5) equals y]
log loss = mean[-y × log(p) - (1-y) × log(1-p)]
Brier = mean[(p-y)²]
margin MAE = mean[abs(m-a)]
margin RMSE = sqrt(mean[(m-a)²])
margin bias = mean[m-a]
```

Team-score and total MAE compare the corresponding predictions with observed scores. Negative margin bias means the predicted home-minus-away margin was lower on average than the observed margin; it does not, by itself, establish compressed team-strength differences.

The retained comparison procedure uses 3,000 resamples of season-week blocks with seed `20261002`. This allows same-week dependence while leaving some repeated-team dependence across weeks and selection uncertainty unrepresented. Every new comparison should report the common game population and the paired uncertainty, alongside the point difference.

### What the weekly evidence should change

The October 6 weekly record identifies ordering errors, large margin misses, confidence questions and turnover-heavy results. Those are diagnosis targets. A single high-confidence loss is insufficient to establish probability miscalibration, and a single large win is insufficient to establish a permanent strength gap.

The useful next analysis connects each error to the exact pregame feature vector and fitted scoring contributions, then examines whether the same error repeats across games. For possible margin compression, compare actual and predicted margins over a sufficiently broad common population, with calibration intercept and slope. Preserve selection boundaries when fitting any corrective mapping. Football evidence such as sustained efficiency, field position, unusual returns or personnel changes helps explain individual errors without replacing the numerical test.

Existing archived spread-direction diagnostics were close to 50/50 and lack enough verified quote-price information for an ROI claim. That historical finding is retained; it does not redirect current development toward an external sportsbook comparison.

## 9 Broader developed method and mathematical work remaining

### Opportunity volume and scoring value

The originating expression linked expected points to minutes, opportunities per minute and points per opportunity. It is a useful decomposition to investigate. Multiplying three marginal expectations is not generally an exact expectation of their product when the quantities depend on each other.

The direct scoring identity for team `i` is:

```text
Y(i) = sum from k=1 to N(i) of V(i,k) + separately allocated scores
```

`N(i)` is the number of scoring opportunities and `V(i,k)` the points allocated to opportunity `k`. If `Vbar(i)` is realized average opportunity value, then for that component:

```text
E[N × Vbar] = E[N] × E[Vbar] + Cov(N, Vbar)
```

Pace, long drives, quick scores and score state can affect both volume and value. The developed method must represent their dependence, or state and test an approximation. The two teams also share a clock and possession sequence, which links their opportunity counts.

### Hierarchical state extension

The developed state-space idea allows ability to change over time and shares information among related teams or players. The current Gaussian implementation supplies a concrete starting point. The next fitted hierarchy must specify which abilities vary, their time units, their population/position grouping, their observation likelihoods and the unknown shared quantities being estimated.

The following are development quantities, rather than additional numerically fitted production parameters.

| Quantity | Definition | Numerical status |
| --- | --- | --- |
| Team offensive/defensive states | Current opponent-adjusted ability for each integrated metric | Current five-metric states implemented; richer states unfinished |
| Metric process variance `Q(m)` | Variation in latent ability per specified time interval | Current shared q fixed; metric-specific inference unfinished |
| Metric observation variance `R(m)` | Conditional noise of the observed performance | Current standardized diagonal fixed at 1; richer estimation unfinished |
| Player state `P(p,t,k)` | Player contribution in role/skill component k at time t | Defined; no saved fitted player parameters |
| Player prior mean and variance | Sharing of information within a justified position/role population | Not numerically fixed |
| Availability probability `A(p,g)` | Pregame probability of participation under captured evidence | Input contract unfinished |
| Role weight `w(p,g)` | Expected share of relevant opportunities/snaps | Estimation unfinished |
| Replacement differential | Starter contribution minus expected replacement contribution in comparable units | Estimation unfinished |
| Matchup coefficient `gamma(j)` | Effect of a defined offense-versus-defense interaction | Current ridge products fitted; expanded interactions unfinished |
| Shared opportunity parameters | Expected possession volume and its dependence between teams | Separate experiment unfinished |
| Possession scoring probabilities | Conditional probabilities of explicitly allocated scoring outcomes | Separate experiment unfinished |
| Starting-position and late-half effects | Changes in opportunity value given field position and time | Experiment specified; no saved fit |

Player information should adjust the difference between personnel represented in the team-state evidence and personnel expected in the next game. Adding the full player production on top of a team rating already created by those players would double count their contribution. The build must define that reference personnel, the replacement comparison and how uncertain participation enters the score calculation.

Rest, travel, weather and other context were discussed as possible contributors. They need captured input definitions and measured future-period value before becoming fixed prediction adjustments. Schedule connectivity is also a source of uncertainty in cross-team comparisons; no separate fitted schedule-connectivity parameter is present in the current engine.

### Unfinished possession experiment

The October 3 recovery record reports 1,022 examined completed FBS-versus-FBS games, of which 912 were accepted and 110 excluded. The accepted table contains 21,081 reconstructed possessions: 718 games from 2025 and 194 from 2026, covering 138 team identities. Input cutoff was October 3 at 13:49:08 UTC.

Preparation reconciled allocated points against regulation scores and, for regulation endings, schedule final scores. Some source drive IDs combined an offensive touchdown and a following kickoff-return touchdown. The extraction separated recognizable kickoff events and excluded unexplained score allocations or discrepancies. Excluded games may differ systematically from accepted games; their missingness needs to be analyzed before fitting.

The experimental specification used outcome probabilities for no score, field goal, touchdown and rare events; separate offensive/defensive effects; starting field position; venue; and late-half effects. Time variation used four-week knots with linear interpolation. Shared possession counts used a continuous approximation to logged counts, and starting field position used five groups. These are experimental choices requiring evaluation, not numerical facts established by the historical engine's validation.

The attempted fit failed while saving samples because neither required NetCDF writer was available. No posterior samples, parameter summary, convergence report or predictions were saved. The scoring predictor, completed overtime component and referenced requirements file were absent in the recovery record. Reaching a save instruction did not prove a valid fit.

The current publication checkout does not contain those experiment files. Recovery of the extraction code, audit tables, model specification and original inputs is a prerequisite to continuing that branch. Results from the 808-game engine cannot be used as performance evidence for this experiment.

### Full uncertainty propagation

A complete Bayesian predictive calculation averages future outcomes over uncertain team/player states and fitted parameters:

```text
P(Yhome,Yaway | information)
  = integral P(Yhome,Yaway | theta, information)
             × P(theta | information) d(theta)
```

Here `theta` contains the unknown quantities of the specified fitted method. Parameter draws approximate the second distribution; future-game score draws approximate the conditional outcome distribution. The implemented Kalman/ridge engine uses state posterior summaries and a separately calibrated error covariance. The next extension must define which uncertainty is already represented and which is being added, then evaluate the resulting probabilities and intervals.

## 10 Build and reproduction instructions

### Environment and retained prerequisites

Use a cloud Python environment, consistent with the established preference to avoid requiring a Windows-local project. The retained Python package requires Python 3.11 or later, NumPy 1.26+, pandas 2.2+, SciPy 1.12+, scikit-learn 1.4+ and pyarrow 15+. Testing additionally requires pytest 8+.

The earlier proposed stack included statsmodels, DuckDB and requests. They are not dependencies of the current retained `pyproject.toml`. Add them when an implemented component needs them rather than treating the original suggestion as today's installed environment.

The publication checkout contains package source and tests but does not include its historical `data/`, fitted `artifacts/`, or the complete chronological-record tooling. Its README describes a fuller engine bundle. Reproducing those saved results requires that bundle or the exact retained inputs and model outputs. The README's Codespaces setup file is also not present in the inspected publication checkout.

Required retained files for an exact historical replay are the saved 2021–2025 schedules and advanced summaries, the matched archive CSV and its raw snapshots/commit metadata when using that comparison, and the historical manifests. The production forecast additionally needs its selected `model.json`. Regenerating a model from refreshed historical releases constitutes a new run, not a byte-identical replay.

### Supported Python commands

Run these from `src/content/engine/` in the cloud environment. Installation does not download the football data.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

For exact reproduction, restore the saved inputs first and write to a new output directory:

```bash
python -m cfb_forecast backtest \
  --data data \
  --out artifacts/reproduced \
  --archive data/archive/matched_predictions.csv
```

For a new chronological run using current releases, fetch the required seasons. This changes the input versions relative to the retained evaluation:

```bash
python -m cfb_forecast fetch \
  --data data \
  --seasons 2021 2022 2023 2024 2025 2026
```

After restoring or fitting a production model trained through 2025, use the actual UTC execution time and the chosen forecast-window end:

```bash
python -m cfb_forecast forecast \
  --data data \
  --model artifacts/backtest/model \
  --out artifacts/new_forecasts \
  --as-of "$FORECAST_UTC" \
  --through "$WINDOW_END_UTC" \
  --season 2026 \
  --draws 50000
```

Set `FORECAST_UTC` and `WINDOW_END_UTC` to timezone-aware ISO timestamps before running. The ordinary forecast command creates a batch at the supplied information time. It does not itself implement the separate append-only per-game 24-hour record policy. That distinction must remain explicit until the chronological runner is restored and verified.

### Prediction artifacts

| File | Required contents |
| --- | --- |
| `model.json` | Schema version, selected variant, controls, metric normalization, head/scaler values, score covariance, training/calibration boundaries and source hashes |
| `forecasts.csv` and `forecasts.json` | Game IDs, teams, kickoff, information time, expected scores, margin/total, probabilities, intervals and simulation settings |
| `state_snapshot.npz` | Team IDs, state means, full within-metric covariance and exact pregame features |
| `forecast_manifest.json` | Information time, window, contributing game IDs and source/model hashes |
| `predictions.csv` for evaluation | Historical probabilities/scores, outcomes, cutoffs and latest contributing results |
| `metrics.csv` | Common-population accuracy, probability scores and point errors |
| `validation_trials.csv` | Every candidate tried on the selection period |
| `run_manifest.json` | Roles of periods, selected settings, population and known availability assumptions |

The score-head serialization fields are `variant`, `alpha`, `cols`, `mean`, `scale`, `coef`, and `intercept`. Exporting them provides every numerical coefficient needed to reproduce its expected-score calculation. Saving a coefficient vector without its feature order and scaler is insufficient.

### Statistics preparation and app build

The existing preparation command is:

```bash
python src/content/analysis/prepare_statistics.py \
  --project "$PROJECT_ROOT" \
  --data "$STATISTICS_INPUT_ROOT" \
  --history "$RETAINED_HISTORY_ROOT"
```

The paths refer to the full application root, the directory of additional statistics releases and `manifest.json`, and the retained history directory. The application snapshot must already be recovered. Required filenames and fields are enforced in the script. The script asserts that `games`, `ratings`, `model_meta`, `calibration`, `audit`, `current_games`, and `current_ratings` are unchanged.

Use the installed Data application's supported preparation/build process described in the checkout's `AGENTS.md`. Recover the pinned source snapshot before building; a blank substitute would discard the evidence. Publish the resulting reviewed app to the existing private Site only as part of an authorized product change. This documentation task does not change the product.

## 11 Concrete development work

The order below follows the dependencies of the actual product. It preserves the current prediction method while making each new contribution measurable. It specifies development work; it does not claim those changes have been implemented.

### First recover one reproducible product state

Restore the fitted production and held-out models, exact historical data, complete chronological runner, forecast records and settlement files into a single documented cloud project. Associate them with their content hashes and the published source. Retain the source snapshot reference instead of replacing it with a mutable dataset.

The deliverable is a runnable project with one command path that reproduces the 808 historical predictions and another that reproduces a saved current forecast from its contributing state. Completion requires matching game populations, probabilities and scores within declared numerical tolerance. It also requires an exported parameter record containing centers, scales, coefficients, covariance and controls. No statistical redesign is needed for this task.

### Then synchronize current results and explain errors

Preserve the original forecast rows and write later outcomes into separate result/settlement records keyed by forecast ID and game ID. Reconcile the 14-game weekly document with the chronological records, rather than overwriting forecasts or using the 54-game slate as a single evaluation population.

Add a forecast detail export containing the 26 raw inputs, their standardized values, the fitted coefficient contributions, state uncertainty, contributing prior-game IDs and the actual score errors. Because the current ridge features overlap, present these as an accounting of the scoring calculation rather than independent causal explanations.

Completion requires that a selected forecast's displayed expected scores sum from its stored intercept and contributions, and that later statistics are visibly dated as postforecast evidence. This produces a direct basis for investigating repeated ordering, margin and confidence errors.

### Integrate play efficiency with controlled definitions

Choose and version the actual forecast definitions for EPA, success and explosiveness, retaining the current alternate definitions in descriptive analysis. Record the denominator, eligible play profile, score/time filters, missingness and source estimator for each.

Extend the observation schema and state feature construction together; changing `METRICS` alone would break hard-coded 26-column assumptions. Fit the new observations using the same opponent-adjusted chronological structure. Use an explicit feature schema instead of positional constants for new versions.

The deliverables are a feature-definition file, audited team-game observations, saved fitted parameters and paired predictions from the unchanged and extended methods on the same future-period games. Keep the extension when probability/score-distribution evidence supports it. A descriptive EPA table alone does not complete this integration.

### Integrate player role and availability

Create a stable athlete identity table, player-game performance/usage table, and timestamped pregame availability observations. Preserve unknown status. Define the reference personnel behind each team estimate, expected participation and replacement quality.

Begin with a bounded contribution whose role and opportunity denominator can be measured, such as the passing personnel component. Estimate player effects with position/role pooling and propagate uncertain participation. Avoid a fixed injury-point deduction that ignores replacement quality or duplicates the existing team effect.

Completion requires saved player-state estimates, the mathematical lineup adjustment, pregame input provenance and paired predictions. Statistics inspection remains available for every category while the forecast integration is developed in a measurable sequence.

### Complete or revise the possession branch

Recover and audit the experiment before resuming estimation. Review the 110 exclusions by scoring type, team and game characteristics; validate scoring allocation, starting field position and regulation/overtime treatment. Define the common-clock relationship between the teams' opportunities and the dependence of pace on scoring value.

Save parameter samples atomically, retain diagnostics and restart from saved computation where supported. The deliverables include posterior summaries, convergence/fit diagnostics, a complete game prediction function, saved score probabilities and held-out evaluation. Installing a missing NetCDF writer fixes the serialization dependency only; it does not establish statistical validity.

The possession method becomes a production replacement only after its prediction path is complete and comparative evidence supports that change. Keep the previous engine and evidence addressable by their original versions.

### Connect the chosen predictions to product outputs

Publish refreshed forecasts with a visible information time and method version. Keep observed histories, ratings and prediction contributions accessible from the same selected game or team. If continuing the requested own-model sportsbook product, derive its supported outcomes from the same score distribution and record which calculations are active. Team totals need both an output contract and an implementation check.

A new interface version must preserve the statistical version that generated its rows. Refreshing statistics, fitting coefficients, updating team states, drawing scores and settling results are separate operations with separate timestamps.

## 12 Proposed durable record contracts

These contracts are specifications for the next development pass. Existing files supply many fields, but the entire schema below is not yet implemented.

| Record | Key | Essential fields |
| --- | --- | --- |
| Source version | Source ID and content hash | URL, source release time when known, retrieval time, bytes, schema, eligibility notes |
| Feature definition | Definition ID and version | Field meaning, unit, numerator/denominator, filters, missingness, transformations |
| Statistical fit | Fit ID | Code commit, input hashes, period roles, feature schema, controls, fitted values, diagnostics |
| Forecast | Forecast ID | Game ID, actual creation time, target cutoff, fit ID, inputs/state reference, score/probability outputs |
| Observed outcome | Game ID and outcome version | Final score, completion/result availability, source version |
| Forecast evaluation | Forecast ID and evaluation version | Outcome reference, errors, probability scores, timing eligibility |
| Development change | Change ID | Specific error/question, implemented change, paired evidence, decision and resulting version |

Store actual forecast time separately from its intended target time. Preserve prior forecast rows when states or coefficients change. Add observations and evaluation records without rewriting what the method predicted. Document one concrete method change at a time so later results can be attributed to the relevant feature or calculation.

## 13 Source record and verification boundaries

The current-product sections are grounded in the deployed version 3 metadata, its source commit, Python modules, authored React/JavaScript views, statistics preparation script and validation records. The earlier method and experiment sections also use the September 24–October 6 project conversations and the retained project/weekly documents.

| Evidence source | What it supplies |
| --- | --- |
| Published source commit `acab8c7f07ced19d0df943206fa16d463acb80c8` | Current code, authored views, controls and statistics transformations |
| `src/content/analysis/statistics-validation.json` | Statistics row counts, per-season coverage and preserved forecast-query list |
| `src/content/analysis/final-checks.json` | Recorded preservation, player-rate and packed-container checks |
| `src/content/engine/README.md` and `RESULTS.md` | Engine contracts, historical design, computed 808/763-game results and paired uncertainty |
| `reviewed_snapshot.json` from October 2 | Chronological engine record, 30/14/16 counts, distribution validation and recorded 25-test verification |
| `CFB_Project_Record.md` from October 3 | Possession extraction, save failure, missing fitted/prediction work and preserved prior implementation |
| `CFB_Forecasting_Weekly_Record.md` updated October 6 | Parameter-category definitions, 14 saved pregame predictions and later written results review |
| September component study and project conversations | 3,136-game earlier comparison, original development direction and decisions |
| Retained modelled-product `index.html` | Exact active controls and PBP score covariance |

The publication source contains the older engine test suite; its recorded result is 20 tests. The separate chronological record reports 25. Those counts belong to different verification scopes. This documentation pass inspected the source and recorded results but did not rerun the historical fit or reproduce those pass counts. The available runtime lacked pytest, and immutable data recovery did not complete successfully. Numerical scaler/coefficient arrays remain a required recovery/export item.

Site version 3 publication was verified as succeeded. A complete browser interaction test was not performed during this documentation pass. Earlier source records also distinguish unavailable browser checks from successful numerical checks. None of those verification limits changes the implemented equations or the historical values recorded above; they define the checks needed when preparing the next product version.

The first build task is therefore concrete: recover the exact fitted/data/chronological artifacts, export all fitted values, reproduce the existing predictions, and connect their later outcomes and calculation details. The new team, player and play data then have a stable basis for testing improvements to the established current-season method.
