# Methodology



This is one statistical model whose output is **predictive probability**: for every upcoming FBS game it produces a win probability and full margin/total distributions, kept free of any information that was not available before kickoff. The named methods are a Gaussian linear state-space model updated by the **Kalman filter**, **Bayesian hierarchical partial pooling** (shared zero-centred priors; player ratings pooled toward team and league), **ridge regression** for the points mapping, an analytic **Gaussian predictive win probability**, and **Monte-Carlo integration** for the discrete margin, total and cover probabilities. Everything below is one model with these components; for the measured results see [RESULTS.md](RESULTS.md).

## Components

| Component | Implementation |
|---|---|
| Free inputs | SportsDataverse schedules and ESPN-derived per-game advanced team summaries; ESPN play-by-play for the player layer |
| Dynamic team ratings | Metric-specific Gaussian offense and defense states, full covariance, opponent adjustment, weekly drift and offseason shrinkage, updated by the Kalman filter |
| Hierarchical pooling | Shared zero-centred Gaussian priors shrink teams with little evidence toward the population mean |
| Player layer | Per-play rush/pass yardage modelled with partial pooling (player → team → league) and an opponent-defense term, recency-weighted; usage-weighted into a team player-offense rating (`players.py`) |
| Points mapping | A regularized, standardized ridge regression head fitted on earlier seasons |
| Feature set | Pregame offense, defense, their differences/products, uncertainty summaries, venue and early-season terms; the player variant adds two point-in-time player columns |
| Predictive probability | Analytic Gaussian win probability from the margin, plus Monte-Carlo margin/total/cover probabilities from an out-of-fit validation residual covariance |
| Market diagnostics | Timestamp-filtered spread, total, moneyline and odds input; cover probabilities and expected net return — informative only, never an input to the win probability |
| Baselines | Chronological calibrated Elo and retained public pregame forecast snapshots |
| Verification | Tests for future-outcome exclusion, state covariance, time ordering, market timestamps, deterministic simulations, head serialization, player pooling/point-in-time safety, and invalid data |

The model estimates player offensive ability from play-by-play (`players.py`) but does not model injuries or declared availability directly — it infers expected usage from recent play shares. It does not train its own EPA model, simulate individual drives, or treat the market as information. Provider EPA, win probabilities, Elo and betting fields are deliberately excluded from the statistical inputs.

## Historical design

Default chronology:

| Season | Role |
|---|---|
| 2021 | Warmup; estimate normalization centers/scales; initialize ratings |
| 2022–2023 | Fit candidate score/probability mappings |
| 2024 | Select state drift, offseason carry, regularization, and the variant (score / pbp / pbp_player); estimate out-of-fit score covariance |
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

The score head receives pregame offense, defense, their differences/products, uncertainty summaries, venue and early-season terms. It predicts both sides with the same standardized ridge regression. Output score means are clipped at zero. The `score` head uses a restricted six-column subset; the `pbp` head uses all 26 state columns; the `pbp_player` head uses those 26 plus two player columns (28 total).

## Player layer

The player layer is the same probabilistic idea applied one level down. On standardized per-play yards for role r (pass or rush), a play by ball-carrier j on team t against defense d is

```text
y = alpha_j + beta_d + error,   error ~ N(0, sigma_r^2)
alpha_j ~ N(theta_t, tau_player^2)   # players pooled toward their team
theta_t ~ N(0,       tau_team^2)     # teams pooled toward the league
beta_d  ~ N(0,       tau_def^2)      # defenses pooled toward the league
```

The posterior means are obtained in closed form by conjugate Gaussian shrinkage — a group's estimate is `(sum w*y + kappa*prior) / (sum w + kappa)` with prior pseudo-count `kappa = 1/tau^2` — iterated a few times between the offense (player/team) and defense blocks (coordinate ascent). Each play is weighted `decay ** weeks_before_cutoff`, an exponential-forgetting state filter on a slowly varying latent. Only plays from games that kicked off strictly before the cutoff contribute. A team's pre-kickoff player-offense rating is the usage-weighted average of its expected players' `alpha`, with expected usage taken from each player's share of recent plays, so a team whose high-rated player has no recent usage is rated lower. The rating enters the score head as two columns: the team's own player offense and the opponent's player defense. This variant was selected on 2024 validation and modestly improves held-out calibration (see RESULTS.md).

The 2024 residual covariance contains game variation and realized estimation error. It is used as the forecast covariance without adding state uncertainty again. This empirical joint Gaussian approximation does not explicitly integrate every posterior parameter uncertainty through the nonlinear scoring head.

Historical win probabilities integrate the continuous Gaussian margin analytically. Simulations additionally round/clamp scores and resolve tied draws with a generic 3- or 7-point overtime increment. Both the analytical probability and the bounded simulation probability are exported so the approximation is inspectable. These simulated integers are not a football possession/rules engine. Intervals include substantial game uncertainty; the two score means are expectations, not an assertion of an exact final score.

## Market input

`examples/market_template.csv` documents the optional input columns. Supply actual observed lines, with `observed_at` including a timezone. The latest snapshot at or before the cutoff and strictly before kickoff is selected. A later snapshot never enters an earlier forecast. Missing odds are not invented.

Use standard **home-spread** signs: a home favorite giving 3.5 points has `home_spread=-3.5`. A simulated home cover is `home_score - away_score + home_spread > 0`; an equality is a push. `home_decimal` and `away_decimal` are the two spread prices. Moneyline fields are separate and are used only for a no-vig winner-probability comparison.

```bash
python -m cfb_forecast forecast --data data --model artifacts/backtest/model --out artifacts/with_market --as-of 2026-10-02T13:45:43Z --through 2026-10-05T05:00:00Z --season 2026 --market data/market_history.csv
```

Net expected return per unit is `P(win)*(decimal_odds-1)-P(loss)`; pushes return the stake. A positive calculated value depends on the validity of the model probability and does not establish an edge by itself. There is no automatic stake-sizing or bet-placement component.

The old forecast archive uses `lineca` as a **home margin target**, which is the opposite sign from a conventional home spread. Its `lineavg` is an aggregate public forecast margin, not a market spread. Archived spread diagnostics are descriptive: original sportsbook prices and exact line availability are not established, so the package reports no historical betting ROI.

## Extending the package

The current modules separate data ingestion, metric observations, dynamic state updates, score heads, simulation, market evidence and evaluation. Add a new deterministic past-game metric through `features.py`, with an explicit missing-data policy. Add a different predictive head through `model.py`, retaining chronological fitting. Add new generative simulations through `simulate.py` and validate their probabilities against held-out outcomes.

The next substantial forecasting improvement should be evaluated prospectively or on an additional untouched period. Candidate work includes calibrated player/availability inputs, possession/drive modeling, home-field heterogeneity and predicting residuals against timestamped market forecasts. None should be promoted into production solely because it sounds more sophisticated.

## Sources

- https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_schedules
- https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_adv_team
- https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_cfb_pbp
- https://github.com/ssmarathontrading/ncaa_predictions
