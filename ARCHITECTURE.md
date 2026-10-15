# CFB forecasting system

The application collects football data, stores it, runs one statistical model over it, and
displays forecasts. The application itself has no statistical state. The time-varying state
representations belong to the football system being modeled: teams, units, players,
availability, possessions and the conditions of a game.

```
sources -> retrieval records -> database -> observations -> ONE joint hierarchical Bayesian model
        -> state estimates + parameters -> database -> forecast (posterior predictive) -> database
        -> game played -> results + evaluation -> database -> next calculation
```

## Layers and modules

| Layer | Module | What it does |
| --- | --- | --- |
| Acquisition | `ingest.py` | Records every retrieval (source id, time, schema version, content hash, stored file) and loads normalized records. No forecasting calculation. |
| Database | `schema.sql`, `database.py`, `warehouse.py`, `db_cli.py` | 48 tables; chronological record with provenance (`source_id`, `retrieval_id`, `available_at`). Existing databases are migrated on connect. `cfb-db` is the SQL entry point. |
| Observations | `observations.py` | Derives per-game team observations from plays and drives into `derived_team_observations` (value + sample size), schedule conditions, and assembles the arrays the model conditions on as the database stood at a cutoff. |
| Model | `model_spec.py`, `football_model.py` | The specification of the football system and the single joint probability model. |
| Forecast | `forecasting.py` | Posterior-predictive draws, exact score distribution per draw, storage, evaluation. |
| Cycle | `model_cli.py` (`cfb-model`) | ingest, derive, fit, forecast, evaluate, backtest, cycle. |

## Sources

SportsDataverse/cfbfastR (schedules, play-by-play, drives, team and player statistics, EPA from the
registered models), CollegeFootballData (games, plays, drives, statistics, players, rosters, recruiting,
transfers, usage, returning production, ratings, weather, venues), FootballDB and CFBTrack (independent
team/player tables, kept separate), FanDuel (timestamped quotes), CFB TXT (schedule/result reconciliation),
mrtimo (secondary game metadata). Each source is collected independently; a missing source stays missing.
The CollegeFootballData client needs `CFBD_API_KEY`; without it collection is skipped, never invented.

## The joint model (`football_model.py`)

One NumPyro model; one joint posterior over everything unknown, fitted at once by stochastic variational
inference. Nothing is estimated in stages.

**State representations** (`model_spec.py`)

* Team: offense (20 components), defense (18), offensive line (3), defensive front (4), special teams (6).
* Players by position: quarterback (10), running back (6), receiver (6) - including usage capacity.
* Each group has a population distribution (multivariate normal), a season-to-season persistence matrix
  `A`, roster-change effects `B R`, a season-change covariance `Q`, and weekly within-season drift.
* Not estimated because no source provides the data: offensive-line continuity; separate scramble vs
  designed-run efficiency (one `rushing_efficiency` component); quarterback decision quality.

**Observation equations** - 37 streams. Each stream is a mean (normal) or a rate (beta-binomial, the
over-dispersed Bernoulli aggregate) whose link-scale mean is

```
alpha_s + eta_s * home + loading * (offense-side states - defense-side states)
        + gamma_s * (O' W D) + sum_p share_p * player_state_p + source bias
```

tying one play-level quantity at once to the states of the offense, the opposing defense, the units
involved and the players on the field. `W` is the matchup interaction matrix with a shrinkage prior.
Player shares are the observed shares of dropbacks, carries and targets.

**Availability and usage** - participation of each player in each team game is Bernoulli with a
player effect; opportunities among the participants are multinomial in the player's usage capacity.

**Possessions** - the outcome of every possession (touchdown, field goal, missed field goal, turnover,
turnover on downs, end of period, rare score, punt) is categorical in the same stream-level predictors,
field position, home field and schedule conditions; possession totals are negative binomial in tempo and
sustainability; points per outcome have estimated distributions.

**External ratings** measure a composite team strength: `R = a_r + b_r * theta + e`, with `a_r, b_r,
sigma_r` estimated per rating source. They enter only when rating rows exist.

**Market quotes** are not part of the football model. A separate market-conditioned calculation can combine
the football forecast with a quote so the information in football data beyond the market stays measurable.

## Forecast

For each posterior draw of states and parameters the exact joint distribution of the two scores is
computed from the possession model (convolutions of per-possession point distributions mixed over
possession counts, with sampled lineups). The forecast is the average over draws:

```
P(Y | D, X) = integral P(Y | Theta, Phi, Psi, X) P(Theta, Phi, Psi | D) dTheta dPhi dPsi
```

From the joint distribution: win probabilities, expected points per team, margin and total
distributions with variances, spread and total probabilities with pushes, team totals, the full score
distribution. Each forecast is stored with its inputs, information cutoff and parameter set; the
distribution artifact is `forecast_distributions/<run>.npz`.

## Evaluation

When a game is final its result joins the stored forecast: winner accuracy, Brier score, log loss,
margin MAE/RMSE/bias, total MAE, team-score MAE, CRPS of the margin distribution, interval coverage and
calibration bins. Completed games become eligible for the next calculation according to their recorded
availability time.

## Operating cycle

```
cfb-model cycle --season 2026
```

ingest -> derive -> fit (jointly, at the current cutoff) -> forecast the coming week -> evaluate finals.

## What the model does not impose

The model is the football system and the probability model of its observations, nothing else.

* No filtering of data: every completed game with recorded plays informs the states, whatever the
  division of the teams; every player that appears is a node. Missing values stay missing: an unrecorded
  drive start position is marginalised over the empirical start distribution, an unclassified possession
  contributes no outcome term, a quantity with no observations has no likelihood term.
* No constraints on parameters beyond what a probability requires (scales are positive). Population
  means, persistence matrices, roster-change effects, loadings and matchup interactions all carry weakly
  informative priors centred on zero (loadings on one) and nothing is clipped, floored, capped or
  clamped; the score distribution is exact, with support sized so the neglected probability is below 1e-12.
* No fitted point estimators (ridge/logistic/regression maps): everything unknown is a random variable
  in the one joint model and is inferred together.
* No inputs that are not football: market lines are not features of the football forecast (they can
  condition a separate calculation); betting decisions and sportsbook logic are outside the model.
* Overtime is part of the model: a regulation tie is resolved by the same possession-outcome model
  evaluated at the overtime start position, round by round.

## Inference

`football_model.infer(data, method=...)`:

* `nuts` - exact posterior sampling (Hamiltonian Monte Carlo) of the whole model.
* `svi` - a variational approximation of the same posterior, used at full scale because exact sampling
  of ~10^6 latent variables is not practical per weekly cycle. It is mean-field, so its uncertainty is
  likely too narrow. `scripts/validate_inference.py` runs both on the same data and reports how far the
  approximation is from the exact posterior.

## Remaining approximations, stated plainly

* Observation likelihoods use game-level sufficient statistics (means and counts) of play-level
  quantities: exact for rates, a central-limit approximation for EPA-type means.
* Player states are per season (no within-season drift); team states drift weekly.
* Forecast lineups use the posterior participation/usage model; the split of carries and targets
  between position groups uses each team's recent shares (league shares when a team has none).
* Not estimated because no source provides the data: offensive-line continuity; separate scramble vs
  designed-run efficiency; quarterback decision quality.

The legacy `sunday-run` service and its Gaussian score head remain in the repository for comparison and
are not used by `cfb-model`.
