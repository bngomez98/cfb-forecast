# Production score model contract

These notes refine the existing Sunday/weekly forecasting path. They do not introduce a neural network, a new database, or a new deployment stack.

## Bayesian team state

`modeling/ratings.py` maintains a Gaussian offense/defense state with Kalman updates. For observation matrix **H** and observation covariance **R**:

```
K = P H' (H P H' + R)^-1
m_new = m + K (y - H m)
P_new = P - K H P
```

Only results available strictly before the forecast cutoff enter the state. Off-season shrinkage and weekly evolution remain unchanged. The selected ridge score head maps this state and its matchup features to expected points.

## Adjust the location, keep one distribution

The existing regularized residual map uses recorded pregame rates, rest, and eligible market quotes. Its normalized inputs now include missingness indicators:

```
X = [normalized observed/imputed inputs, input-is-missing indicators]
B = (X'X + ridge_penalty I)^-1 X' (actual_scores - prior_score_locations)
mu = selected_head(state) + X B
```

Imputation is a numerical convenience inside the estimator; it does not manufacture or write missing source statistics. The no-line branch remains a conditional parameterization within the same forecast, not a separate winner vote.

Historical locations used to fit **B** are generated from heads trained on strictly earlier seasons. Initial seasons that lack 50 earlier eligible games are excluded from adjustment fitting. If fewer than 30 such locations remain, the identity adjustment is used. There is no fallback to in-sample locations.

The score approximation is:

```
(home_score, away_score) ~ Normal(mu, Sigma)
```

With margin **M** and total **T**:

```
E[M] = mu_home - mu_away
E[T] = mu_home + mu_away
Var[M] = Sigma_hh + Sigma_aa - 2 Sigma_ha
Var[T] = Sigma_hh + Sigma_aa + 2 Sigma_ha
P(home wins) = Phi(E[M] / sqrt(Var[M]))
```

Spreads and totals use the same moments, including continuity corrections for integer betting lines and explicit push mass. At a zero spread, home-cover probability equals home-win probability. No independent logistic winner model overwrites this result. No shrinkage toward other games in the same request occurs.

Expected scores are also the non-negative Poisson rate parameters exposed in each row. This retains the existing Gaussian/Poisson mean relationship. It does **not** claim that a Gaussian is an exact Poisson distribution, nor that a Poisson-lognormal posterior is integrated. The calibrated Gaussian residual covariance already accounts for scoring overdispersion, correlation, and state-estimation error; adding Poisson variance or state variance again would double-count uncertainty.

When the weather/injury context layer passes its validation gate, its bounded log-odds adjustment is converted back to a margin using the Gaussian inverse CDF. The two score means move equally in opposite directions, preserving the total and non-negative scores. All probabilities are then regenerated from the adjusted scores. This remains one conditional score forecast rather than conflicting score and winner forecasts.

## Learning and compatibility

Only the learning stage may promote a score-head refit. Promotion requires at least 30 non-overlapping gate games and no worse gate log-loss. Forecasting must not silently refit and bypass that decision. Team states continue to update regardless of head promotion. The gate evaluates the score-head component, not a guaranteed improvement of the complete pipeline. New learning-cache identities use algorithm version 3 to avoid reusing decisions made under the former promotion rule.

Adjustment artifacts now use schema 3 and store missingness coefficients. Schemas 1 and 2 remain readable. Legacy logistic and slate-reconciliation fields are retained for parsing compatibility but no longer control production inference. Previously published predictions are never rewritten. Existing CLI commands, output fields, storage schemas, and deployment configuration are preserved; Poisson rate fields are additive.

Legacy standalone experimental and comparison modules remain available. They do not participate in the default Sunday/weekly score distribution.

## Relation to CRAFT

The IJCAI-25 paper *CRAFT: Time Series Forecasting with Cross-Future Behavior Awareness* motivates distinguishing partial known-future information from observed values and combining it with historical trends. Missingness-aware inputs refine the existing residual map without replacing the Bayesian backbone.

This is **not** a reproduction of CRAFT’s encoder/decoder Koopman dynamics. Existing `koopman` parameter names are retained for compatibility, but the implementation is a regularized covariate-to-score-residual map. Football scores are not additive hotel-inventory hierarchies, so neither request-wide mean shrinkage nor the paper’s hotel-demand constraints are justified here.

## Verification limits

Tests establish algebraic consistency, causal fitting and build compatibility.
They do not establish better forecasting accuracy. Compare unchanged code and
candidate code on identical historical cutoffs and games before reporting gains.
Historical source revisions and model-selection history still limit how closely
a replay represents a genuinely prospective forecast.


## Player-rate implementation

The historical matrix includes 18 observed player-role inputs in addition to the
20 existing team inputs. For each team and each passing/rushing/receiving role:

```
yards_per_play = sum(recorded yards) / sum(opportunities with observed yards)
epa = sum(observed player EPA * opportunities) / sum(opportunities with observed EPA)
usage_concentration = sum(player opportunities squared) / sum(opportunities) squared
```

These per-game values enter the prior-season/current-season running rates and
the same ridge residual map as team statistics. A query event precedes final
results at exactly its timestamp. Historical team state, team/player rates and
captured market quotes share that query's information cutoff. Backfilled closing
quotes are not substituted for captured historical quotes.

The contribution export decomposes the fitted linear map, including learned
missingness terms. A player's-group contribution is statistical attribution
within the fitted model, not an injury replacement value or a causal estimate.
Team and player point contributions plus known-future contributions and the
explicit score-bound term add back to the published expected score. Gated live
context can make a further margin adjustment while preserving the total.

## Causal lag and window features

The [Feature-engine forecasting guide](https://feature-engine.trainindata.com/en/1.8.x/user_guide/timeseries/forecasting/index.html)
provides the lag, rolling-window and expanding-window feature families used here.
The football adaptation operates per team, in result-availability order, rather
than treating all teams as one regularly spaced series. It uses the prior and
current season only, preserving missing game slots.

For each of the 19 historical team/player metrics on each side, production now
calculates lag 1, lag 2, 3-game and 5-game rolling means and population standard
deviations (`ddof=0`), and an expanding mean. Rolling windows require a full window
of observed values. A missing lag stays missing; it is not replaced by the next
available earlier game. Finals at the query cutoff are excluded before every
transformation. This adds 266 temporal columns to the 38 season-rate columns.

These inputs participate in the existing joint ridge score-residual fit, with
training-only normalization, per-field coverage checks and missingness terms.
Their coefficients are saved with the selected input names. No `feature-engine`
dependency is required because its regular-index transforms cannot directly
express the game's per-team availability events and individual query cutoffs.

Use `--no-temporal-features` to fit an ablation retaining the season-rate inputs,
or `--temporal-features` to enable them explicitly. The effective
`adjustment_config.use_temporal_features` and feature policy are saved in model
and forecast artifacts. The defaults are engineering choices, not measured
optimal settings. Legacy 20- and 38-column feature arrays are padded with missing
new columns; saved name-based adjustment coefficients remain readable.

## Two inspectable forecast series

The model exports football-only and market-conditioned expected scores, margin,
total and home win probability separately. Football-only inference uses the fit
that never consumed market columns. Market-conditioned values are unavailable
unless a fitted market adjustment and eligible target quote exist. A legacy
market artifact lacking its no-line fit cannot claim a football-only prediction.
Neither series changes the saved primary prediction after publication.

Validation reports each series separately and compares them on the intersection
of games with both saved probabilities. This avoids comparing different game
populations and does not use later quotes to fill missing market predictions.
The full player/unit posterior and possession likelihoods in the supplied design
remain separate implementation work; these additions do not claim those
likelihoods have been fitted.
Tests establish algebraic consistency, causal fitting, and build compatibility. They do **not** establish better forecasting accuracy. Compare unchanged code and candidate code on identical historical cutoffs and games before claiming gains. Historical source revisions and model-selection history still limit how closely a replay represents a genuinely prospective forecast.
