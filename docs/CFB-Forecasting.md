CFB Forecasting 

These notes update the existing Sunday and weekly forecasting process without adding a neural network, new database, or deployment stack.

Bayesian team state

modeling/ratings.py maintains a Gaussian offense/defense state with Kalman updates. For observation matrix H and observation covariance R:

K = P H' (H P H' + R)^-1
m_new = m + K (y - H m)
P_new = P - K H P


Only results available before the forecast cutoff are used. Off-season shrinkage and weekly updates remain as before. The selected ridge score head translates this state and relevant matchup features into expected points.

Adjust the location, keep one distribution

The regularized residual map incorporates recorded pregame rates, rest, and eligible market quotes. Normalized inputs now also contain indicators for missing data:

X = [normalized observed/imputed inputs, input-is-missing indicators]
B = (X'X + ridge_penalty I)^-1 X' (actual_scores - prior_score_locations)
mu = selected_head(state) + X B

is performed within the estimator for computational convenience and does not create or record missing input statistics. The no-line branch is a conditional parameterization within the same forecast, rather than a separate winner prediction.

Historical locations used to fit B come from heads trained on earlier seasons. Seasons wWe exclude initialer than 50 prior eligible games are excluded from tting. If there are fewer than 30 such locations, the identity adwe use justment is applied. Tho fallback to in-sample locations.

The score approximation is:

(home_score, away_score) ~ Normal(mu, Sigma)


With margin M and total T:

E[M] = mu_home - mu_away
E[T] = mu_home + mu_away
Var[M] = Sigma_hh + Sigma_aa - 2 Sigma_ha
Var[T] = Sigma_hh + Sigma_aa + 2 Sigma_ha
P(home wins) = Phi(E[M] / sqrt(Var[M]))

Spreads and totals use the same statistical parameters, with continuity corrections for integer betting lines and explicit push probability. At zero spread, the probability of a home cover equals the probability of a home win. No separate logistic winner model overrides this result, and there is no shrinkage toward other games in the same request.

Expected scores serve as the non-negative Poisson rate parameters for each row, preserving the Gaussian/Poisson mean relationship. This does not mean the Gaussian is an exact Poisson or that a Poisson-lognormal posterior is integrated. The calibrated Gaussian residual covariance already addresses overdispersion, correlation, and state-estimation error, so incorporating Poisson or state variance again would double-count uncertainty.

When the weather or injury context layer passes validation, it translates its bounded log-odds adjustment into a margin using the Gaussian inverse CDF. The two score means are adjusted equally but in opposite directions, keeping both the total and the absence of negativity in scores. All probabilities are recalculated from these adjusted scores. This process produces a single conditional score forecast, not separate or conflicting score and winner forecasts.

Learning and compatibility

Only the learning stage can promote a score-head refit. Promotion requires at least 30 non-overlapping gate games and no worse gate log-loss. Forecasting must not silently refit or bypass this step. Team states continue updating regardless of head promotion. The gate evaluates only the score-head component, not the entire pipeline. New learning-cache identities use algorithm version 3 to prevent reusing decisions from the previous promotion rule.

Adjustment artifacts now use schema 3 and include missingness coefficients. Schemas 1 and 2 remain readable. Legacy logistic and slate-reconciliation fields are kept for parsing compatibility but no longer influence production inference. Previously published predictions are never altered. Existing CLI commands, output fields, storage schemas, and deployment configurations remain unchanged. Poisson rate fields are additive.

Legacy standalone experimental and comparison modules are still available, but do not participate in the default Sunday or weekly score distribution.

Relation to CRAFT

The IJCAI-25 paper, CRAFT: Time Series Forecasting with Cross-Future Behavior Awareness, stresses the importance of distinguishing partially known future information from recorded values and integrating it with historical trends. Missingness-aware inputs refine the residual map without replacing the Bayesian backbone.

This is not a reproduction of CRAFT’s encoder or decoder Koopman dynamics. We retain existing Koopman parameter names for compatibility, but the implementation is a regularized covariate-to-score-residual map. Football scores are not additive hotel-inventory hierarchies, so neither request-wide mean shrinkage nor the paper’s hotel-demand constraints apply here.

Verification limits

Tests confirm consistency, causal fitting, and build compatibility.
They do not attain enhanced forecast precision. Compare unchanged code and candidate code on identical historical cutoffs and games before reporting gains.
candidate code on identical historical cutoffs and games before reporting gains.
Historical source revisions and model-selection history still limit how closely a replay represents a genuinely prospective forecast.
A replay represents a genuinely prospective forecast.

Player-rate implementation

The historical matrix includes 18 observed player-role inputs in addition to the
20 existing team inputs. For each team and each rushing/rushing/receiving role:

yards_per_play = sum(recorded yards) / sum(opportunities with recorded yards)
epa = sum(observed player EPA * opportunities) / sum(opportunities containing observed EPA)
usage_concentration = sum(player opportunities squared) / sum(opportunities) squared

These per-game values enter the prior-season/current-season running rates and
the same ridge residual map as team statistics. A query event precedes final
results at exactly its timestamp. Historical team state, team/player rates and
captured market quotes share that query’s information cutoff. Backfilled closing
quotes are not substituted for captured historical quotes.

The contribution export, including learned
terms. A player’s-group contribution is statistical attribution
within the fitted model, not an injury replacement value or a causal estimate.
Team and player point contributions plus known-future contributions and the
explicit score-bound term add back to the published expected score. Gated live
context can make a further margin adjustment while preserving the total.

Causal lag and window features

The Feature-engine forecasting guide delivers the lag, rolling-window and expanding-window feature families used here.
The football adaptation operates per team, in result-availability order, rather
than treating all teams as one regularly spaced series. It uses the prior and
current season only, preserving missing game slots.

For each of the 19 historical team/player metrics on each side, production now
calculates lag 1, lag 2, 3-game and 5-game rolling means and population standard
deviations (ddof=0), and an expanding mean. Rolling windows require a full window
of observed values. A missing lag stays missing; it is not replaced by the next
available earlier game. Finals at the given query cutoff are excluded before every
transformation. This adds 266 temporal columns to the 38 season-rate columns.

These inputs participate in the existing joint ridge score-residual fit, with
training-only normalization, per-field coverage checks and missingness terms.
Their coefficients are saved with the selected input names. No feature-engine
dependency is required because its regular-index transforms cannot directly
express the game’s per-team availability events and individual query cutoffs.

Use --no-temporal-features to fit an ablation retaining the season-rate inputs,
or --temporal-features to enable them explicitly. The effective
adjustment_config.use_temporal_features and feature policy are saved in model
and forecast artifacts. The defaults are engineering choices, not measured
optimal settings. Legacy 20- and 38-column feature arrays are padded with missing
new columns; saved name-based adjustment coefficients remain readable.

Two inspectable forecast series

The model exports football-only and market-conditioned expected scores, margin,
total and home win probability separately. Football-only inference uses the fit
that never consumed market columns. Market-conditioned values are unavailable
unless a fitted market adjustment and eligible target quote exist. A legacy
market artifact lacking its no-line fit cannot claim a football-only prediction.
Neither series changes the saved primary prediction after publication.

Validation reports each series separately and compares them on the intersection
of games with both saved probabilities. This avoids comparing different game
populations and does not use later quotes to fill missing market outlooks.
The full player/unit posterior and possession likelihoods in the supplied design
remain separate implementation work; these additions do not claim those
likelihoods have been fitted.
Tests establish code integrity, causal fitting, and build compatibility. Compare unchanged code and uncommitted code on identical historical cutoffs and games before updating to /main. Historical source revisions and model-selection history still limit how closely a replay represents a genuinely prospective forecast.
