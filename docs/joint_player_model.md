# Joint team/player/opponent forecast

The default `forecast`, weekly, Sunday, and season-reconstruction paths now use
`PlayerHierarchy`. The previous Kalman/ridge family is retained through
`--legacy-score` for explicit reproduction. Previously saved forecasts are not
rewritten. Existing experimental possession simulators remain experimental;
the default does not consume their fixed scoring or weather constants.

## One posterior

Player-game rate observations and game scores are rows of ONE Gaussian working
likelihood, with ONE sparse posterior precision matrix. The score head,
player-only predictions, and a residual correction are not fitted and combined
as separate predictors.

For player i, opponent j, role r and season t, standardized observed performance is

    z[i,g,r] = b[r] + a[i,r,t] - d[j,r,t] + gamma[i,j,r] + error[i,g,r].

The role observation is mean EPA per recorded opportunity when that role has at
least three variable EPA observations. Otherwise it is yards per recorded
opportunity, provided that rate has sufficient variable observations. A role
without usable rate history receives no fitted player-rate output. Yardage is
not a count likelihood. Each player-game rate receives one observation, rather
than treating every opportunity as an independent measurement.

Standardized team score has the equation

    z_score[team,g] = intercept + venue_coef * signed_venue
                   + team_offense[team,t] - team_defense[opponent,t]
                   + x[team,g]' context_coef
                   + SUM_role SUM_player opportunity_share[player,g,role]
                       * (a[player,role,t] - d[opponent,role,t]
                          + gamma[player,opponent,role]) + score_error.

The SAME a, d and gamma appear in both equations. Role intercepts center player
outcomes; the separate score intercept centers team points. Scores and player
rates use training-only empirical centers and scales. Proper Gaussian priors
regularize the overlap between team and player effects; coefficients do not
have a causal interpretation.

Team and role-defense states have season transitions. Player transitions follow
an explicit identity, including transfers when a stable source ID or verified
crosswalk supports it. Transition carry is 0.8 per season; innovation variance
is 0.3 per elapsed season. Root team/player/role-defense variances are 0.5,
matchup variance is 0.1, context variance is 1, intercept variance is 10.
These are conditional configuration defaults, not measured optimal values or
posterior distributions over hyperparameters. Root priors are applied only at
the first season; later states have conditional transition priors, avoiding
spurious backward conditioning when an unrelated future matchup is queried.

For stacked observations y, design X, likelihood precision W and prior precision Q:

    C = inverse(Q + X' W X)
    m = C X' W y.

Sparse LU solves the precision system; no dense all-player covariance is stored.
For a two-score query design X*, the conditional predictive covariance is

    Sigma* = score_noise + score_scale^2 * X* C X*'.

The score working noise is the empirical training-score variance on each
side, with zero conditional residual cross-covariance. It is deliberately
separate from posterior parameter covariance. The older forecast family's
residual covariance is not added, which would risk counting parameter error
again. This noise specification has NOT been chronologically calibrated.

Within-season posterior estimates change as new observations enter. Latent
state transitions are currently by season, not weekly player state transitions.
Pregame scoring, allowed scoring, YPP, success, explosive rate, turnover rate,
and EPA history also enter the common context coefficients. Numeric inputs and
missingness indicators are included only when their training coverage supports
them; an entirely unobserved weather column cannot acquire a fabricated effect.

## Availability and usage

A timestamped lineup supplies player identity, role, opportunity share and,
optionally, an explicitly supplied availability probability. Shares sum to one
within each recorded role. Unknown availability stays unknown. The model never
assigns a made-up probability to a questionable designation or an empty injury
response.

For supplied probabilities, all participation scenarios are enumerated exactly
(up to ten uncertain players across a matchup). Each scenario reallocates the
remaining role opportunities. If a whole role is unavailable, an explicitly
unidentified, team-scoped replacement contributes its prior uncertainty.

The two scores share one conditional Gaussian distribution per scenario.
Win/cover/total probabilities integrate the finite Gaussian mixture. Score
covariance uses the law of total variance, including differences between lineup
scenarios. Reported 80% intervals use moment-matched Gaussian quantiles and are
labelled accordingly; they are not exact mixture quantiles.

Without a timestamped future lineup, the adapter uses the team's latest observed
role usage and labels it `last_observed_usage_unconfirmed`. This is not a verified
future roster or evidence of health. If no lineup/history exists for a selected
team, the joint forecast fails explicitly instead of silently dropping players.

Player outputs are conditional rate means, rate variances and expected role
shares. They do not claim fitted attempt totals, touchdown counts, or coherent
passing/receiving yard totals. Those require a fitted opportunity/event model.
Shared-play dependence among passer/receiver measurements is not estimated in
this initial Gaussian working likelihood.

## Database and source records

Both `sql/schema.sql` and the installed-package fallback schema create:

- `player_identities`: explicit identities.
- `player_source_ids`: provider-specific identity crosswalks, with collision checks.
- `player_observations`: versioned player/game/opponent/role observations and provenance.
- `player_lineups`: complete supplied role-lineup snapshots with observation times.
- `game_context_snapshots`: timestamped pregame weather, rest, injury-count and eligible market features.

`player_game_facts` gains a nullable `source_player_id` through a nondestructive
migration. Historical parquet processing preserves passer/rusher/receiver IDs.
A verified crosswalk takes precedence. Otherwise a supplied SportsDataverse ID
is source-scoped. A name without an ID is scoped to team and season and cannot
establish a transfer. No fuzzy name matching merges players.

Import normalized evidence with:

    python -m cfb_forecast import-player-evidence --input evidence.json --database data/cfb_forecast.sqlite

The JSON object accepts `players`, `observations`, `lineups`, and `contexts` lists.
Every observation/lineup/context requires provider, source_url and source_sha256.
Observations additionally require game_id, team_id, opponent_id, player_id, role,
opportunities and available_at; yards and mean EPA are optional. Lineups require
game_id, team_id, observed_at and players; each player has player_id, role,
opportunity_share and optional availability_probability. Contexts require game_id,
observed_at and a `features` object whose keys are the allowlisted KNOWN_COLUMNS.
Imports are atomic, invalid identities/opponents/probabilities are rejected,
and alternate provider measurements are not counted as additional games.

The registered collegefootballdb, cfbtracker, cfbresults and other sources may
supply normalized records through this contract. Merely collecting a homepage
is not a player observation. This change does not claim new verified HTML
extractors for sites whose field layouts have not been established. Existing
raw source captures remain available; the importer requires resolved IDs,
statistic definitions and provenance before those captures enter estimation.
Sunday's existing numeric context capture now archives its allowlisted fields
in the same database before inference. It never applies the old independent
context probability adjustment to a joint-player forecast.

## Cutoffs, artifacts and evaluation

A result and its player observation must both be available strictly before the
forecast cutoff. Team context for each historical game is queried before that
game's kickoff and excludes results that were not yet available then. Future
context and lineup snapshots are excluded, including equality at cutoff.
Legacy parquet records lack original publication receipts and are explicitly
retrospective; they cannot establish original point-in-time availability.

Each run saves posterior means/normalization/configuration in
`joint_player_model.json`, sparse precision in `joint_player_precision.npz`,
and exact selected observations/lineups/team facts/query context in
`joint_player_inputs.json`. The forecast manifest hashes these artifacts and
raw source files. Prospective holdout freezing verifies and copies all three.
Advancing a joint run refits from newly cutoff-filtered evidence rather than
reusing a team-only snapshot that would discard new player reports.

Outputs retain expected home/away score, margin, total, winner probability,
market comparison and player_forecasts. The publisher preserves player outputs
on its JSON board. Sunday database model registration identifies the joint
artifact, rather than mislabelling it as the previous ridge head.

The season-backtest command reconstructs the NEW family through the same
forecast entry point. Use --legacy-score to reproduce prior-family scores.
The older 73% historical accuracy and all prior comparison numbers belong to
those earlier implementations. This change has behavioral and integration
tests; no new out-of-sample accuracy, calibration or source-coverage result is
asserted. A full chronological evaluation on actual connected data remains
necessary before treating it as a performance improvement.
