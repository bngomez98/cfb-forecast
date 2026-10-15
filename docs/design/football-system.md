The corrected design is this:

The application itself does not have a statistical “state.” It is software that collects football data, stores it, runs statistical calculations, and displays forecasts.

The state representations belong to the football system being modeled: teams, players, units, matchups, availability, possession conditions, and other time-varying football quantities.

The new architecture should therefore be organized around the football data and the mathematical representation of that data.

## CFB forecasting system

### 1. Source acquisition

Each source is collected independently and preserved with retrieval time, source identifier, schema version, and content hash.

Primary sources:

- SportsDataverse / cfbfastR: play-by-play, schedules, drives, team and player statistics, EPA-related variables.
- cfbfastR model registry: expected-points, win-probability, completion-probability and related published statistical models used to derive specific observations.
- CollegeFootballData API: games, plays, drives, advanced statistics, players, rosters, recruiting, transfers, usage, returning production, ratings, weather and venues.
- FootballDB: independent team and player statistical tables.
- CFBTrack: independent team/player statistics and contextual records.
- FanDuel: timestamped spreads, totals, moneylines and other market observations.
- CFB TXT: schedule/result reconciliation and weekly forecast validation.
- `mrtimo` game data: secondary game metadata and cross-checking.

The acquisition layer writes source records into the database. It performs no forecasting calculation.

## 2. Database

The database is the chronological record of the football information available to the forecasting process.

It contains raw source records, normalized football observations, derived statistical observations, Bayesian parameters, estimated football-state representations, forecast inputs, forecasts, market observations, actual outcomes and forecast evaluations.

Core relational tables:

```text
sources
source_retrievals
source_files

teams
players
coaches
venues

games
game_teams
plays
drives

team_game_statistics
player_game_statistics
player_usage

rosters
player_team_membership
transfers
recruiting
returning_production
player_availability

weather
schedule_conditions

external_ratings
market_quotes

derived_play_observations
derived_drive_observations
derived_team_observations
derived_player_observations

statistical_parameter_sets
team_state_estimates
player_state_estimates
unit_state_estimates
availability_estimates

forecast_runs
forecast_inputs
forecasts

game_results
forecast_evaluations
```

Large play-by-play and posterior-draw datasets can be stored in Parquet or another columnar format while the relational database retains their IDs, provenance, timestamps and metadata.

## 3. Observed football variables

These are values actually supplied by a source or deterministically calculated from source records.

### Game and schedule

```text
game_id
season
week
kickoff_datetime
home_team_id
away_team_id
neutral_site
venue_id
conference
home_points
away_points
game_status
```

Sources: SportsDataverse schedules, CFBD games, CFB TXT for validation.

### Play conditions

```text
offense_team_id
defense_team_id
period
clock
down
distance
yard_line
yards_to_goal
score_difference
play_type
```

Sources: cfbfastR PBP, CFBD plays.

### Play outcomes

```text
yards_gained
first_down
touchdown
completion
incompletion
interception
fumble
fumble_lost
sack
tackle_for_loss
penalty
penalty_yards
field_goal_attempt
field_goal_result
punt
kickoff
return_yards
```

Sources: cfbfastR PBP, CFBD plays.

### Derived play quantities

```text
expected_points_before
expected_points_after
EPA
expected_completion_probability
completion_over_expected
success_indicator
explosive_play_indicator
negative_play_indicator
```

Sources: cfbfastR PBP and cfbfastR registered statistical models.

These derived quantities must retain:

```text
derivation_model
derivation_model_version
input_definition
calculation_timestamp
```

### Drive observations

```text
drive_start_field_position
drive_end_field_position
drive_plays
drive_yards
drive_duration
drive_first_downs
drive_result
drive_points
```

Sources: CFBD drives and cfbfastR/ESPN drive data.

### Team observations

```text
pass_attempts
rush_attempts
passing_yards
rushing_yards
yards_per_play
yards_per_pass
yards_per_rush
first_downs
third_down_attempts
third_down_conversions
fourth_down_attempts
fourth_down_conversions
red_zone_entries
red_zone_touchdowns
turnovers
sacks_allowed
sacks_created
tackles_for_loss
penalties
possession_time
```

Sources: SportsDataverse, CFBD, FootballDB, CFBTrack.

### Player observations

```text
player_id
team_id
position
games_played
starts
pass_attempts
completions
passing_yards
interceptions
carries
rushing_yards
targets
receptions
receiving_yards
touchdowns
sacks
tackles
interceptions_made
field_goal_attempts
field_goal_makes
punt_attempts
punt_yards
return_attempts
return_yards
```

Sources: SportsDataverse player data, CFBD player/game data, FootballDB, CFBTrack.

### Personnel observations

```text
roster_membership
position
class_year
height
weight
previous_team
transfer_date
recruiting_rating
recruiting_position
returning_production
historical_usage
current_usage
depth_chart_position
reported_availability
confirmed_inactive
```

Sources: CFBD roster, recruiting, transfer portal, player usage and returning-production endpoints, plus available timestamped injury/depth-chart data.

### Environment and schedule observations

```text
temperature
wind_speed
precipitation
humidity
venue
surface
elevation
home_away_neutral
days_rest
short_week
bye_week
travel_distance
timezone_change
```

Sources: CFBD weather, venues and schedules; travel quantities are derived from location data.

## 4. Football state representations

These are time-indexed statistical representations of football ability or condition.

They belong to teams, players, units or games.

### Team offense

\[
\mathbf O_{i,t}
\]

contains:

```text
pass_efficiency
rush_efficiency
success_creation
pass_explosiveness
rush_explosiveness
early_down_efficiency
late_down_efficiency
short_yardage_efficiency
red_zone_efficiency
goal_to_go_efficiency
drive_sustainability
field_position_conversion
sack_avoidance
passing_turnover_tendency
rushing_ball_security
penalty_tendency
tempo
fourth_down_execution
scoring_opportunity_creation
scoring_opportunity_conversion
```

### Team defense

\[
\mathbf D_{i,t}
\]

contains:

```text
pass_suppression
rush_suppression
success_suppression
explosive_pass_prevention
explosive_rush_prevention
early_down_defense
late_down_defense
short_yardage_defense
red_zone_defense
goal_line_defense
drive_termination
pressure_creation
sack_creation
interception_creation
fumble_creation
penalty_tendency
field_position_control
tempo_influence
```

### Quarterback

\[
\mathbf Q_{p,t}
\]

contains:

```text
passing_efficiency
completion_quality
short_pass_efficiency
intermediate_pass_efficiency
deep_pass_efficiency
decision_quality
interception_tendency
sack_avoidance
scramble_efficiency
designed_rush_efficiency
ball_security
```

### Running back

```text
rushing_efficiency
explosive_run_ability
short_yardage_ability
receiving_efficiency
ball_security
usage_capacity
```

### Receiver / tight end

```text
target_efficiency
catch_efficiency
deep_receiving_efficiency
yards_after_catch_efficiency
red_zone_receiving_efficiency
usage_capacity
```

### Offensive line unit

```text
pass_protection
rush_blocking
penalty_tendency
continuity
```

### Defensive front unit

```text
pass_pressure
rush_disruption
sack_creation
tackle_for_loss_creation
```

### Special teams

```text
field_goal_ability
kickoff_ability
punt_ability
kick_coverage
punt_coverage
return_efficiency
```

### Availability and expected participation

For each player and upcoming game:

\[
A_{p,g}=P(\text{player participates})
\]

and

\[
U_{p,g}=E(\text{share of relevant opportunities})
\]

Those estimates are derived from roster status, historical participation, usage, depth chart and timestamped availability information.

## 5. Hierarchical Bayesian statistical model

The hierarchy connects these state representations instead of estimating every team and player independently.

Team states are drawn from common population distributions:

\[
\mathbf O_{i,t}
\sim
MVN(\boldsymbol{\mu}_O,\Sigma_O)
\]

\[
\mathbf D_{i,t}
\sim
MVN(\boldsymbol{\mu}_D,\Sigma_D)
\]

Player states are grouped by position:

\[
\mathbf Q_{p,t}
\sim
MVN(\boldsymbol{\mu}_{QB},\Sigma_{QB})
\]

\[
\mathbf R_{p,t}
\sim
MVN(\boldsymbol{\mu}_{RB},\Sigma_{RB})
\]

and similarly for receivers and other identifiable player groups.

The hierarchy allows teams and players with less information to borrow statistical strength from the relevant population while teams and players with substantial observations are determined more strongly by their own data.

## 6. Evolution through time

Football abilities evolve.

For team offense:

\[
\mathbf O_{i,t}
\sim
MVN(
A_O\mathbf O_{i,t-1}
+
B_OR_{i,t},
Q_O
)
\]

For defense:

\[
\mathbf D_{i,t}
\sim
MVN(
A_D\mathbf D_{i,t-1}
+
B_DR_{i,t},
Q_D
)
\]

where \(R_{i,t}\) contains observed changes such as:

```text
returning production
transfers
recruiting
roster turnover
quarterback change
coaching change
player availability
```

The numerical values in:

```text
A_O
A_D
B_O
B_D
Q_O
Q_D
```

are estimated from historical seasons.

## 7. Observation equations

Historical football observations determine the numerical state estimates.

Passing plays:

\[
EPA_p
\sim
t_{\nu}
(
\mu_p,
\sigma^2_{pass}
)
\]

with

\[
\mu_p =
\alpha_{pass}
+
O^{pass}_{i,t}
-
D^{pass}_{j,t}
+
Q_{qb,t}
+
OL^{protection}_{i,t}
-
Front^{pressure}_{j,t}
+
X_p\beta_{pass}
+
M^{pass}_{ij,t}
\]

where \(X_p\) contains the observed play conditions.

Rushing:

\[
EPA_p^{rush}
\sim
t_{\nu}
(
\alpha_{rush}
+
O^{rush}_{i,t}
-
D^{rush}_{j,t}
+
RB_{p,t}
+
OL^{rush}_{i,t}
-
Front^{rush}_{j,t}
+
X_p\beta_{rush},
\sigma^2_{rush}
)
\]

Success:

\[
Success_p
\sim
Bernoulli(\pi_p)
\]

\[
logit(\pi_p)
=
\alpha_s
+
O^{success}_{i,t}
-
D^{success}_{j,t}
+
X_p\beta_s
\]

Explosiveness:

\[
Explosive_p
\sim
Bernoulli(\rho_p)
\]

\[
logit(\rho_p)
=
\alpha_e
+
O^{explosive}_{i,t}
-
D^{explosive}_{j,t}
+
X_p\beta_e
\]

Sacks:

\[
Sack_p
\sim
Bernoulli(s_p)
\]

\[
logit(s_p)
=
\alpha_{sack}
-
O^{sackavoid}_{i,t}
-
OL^{protection}_{i,t}
+
D^{pressure}_{j,t}
+
Front^{pressure}_{j,t}
+
X_p\beta_{sack}
\]

Those equations tie the same play simultaneously to team, player, unit and opponent representations.

## 8. Matchup relationships

Matchup effects are estimated jointly.

\[
M_{ij,t}
=
\mathbf O_{i,t}^{T}
W
\mathbf D_{j,t}
\]

The interaction matrix \(W\) contains relationships such as:

```text
pass efficiency × pass suppression
rush efficiency × rush suppression
pass explosiveness × explosive-pass prevention
rush explosiveness × explosive-rush prevention
sack avoidance × pressure creation
short-yardage offense × short-yardage defense
red-zone offense × red-zone defense
drive sustainability × drive termination
tempo × tempo influence
```

The historical data estimate the entries in \(W\).

Hierarchical shrinkage controls weak interaction terms:

\[
W_{ab}\sim N(0,\tau_W^2)
\]

## 9. Drives and possessions

The forecast uses the football-state representations to determine possession opportunity and possession outcome distributions.

Possession count is modeled jointly for both teams:

\[
(N_h,N_a)
\sim
P(
N_h,N_a
\mid
Tempo_h,
Tempo_a,
DriveSustainability_h,
DriveSustainability_a,
Context
)
\]

For each possession, the model calculates probabilities for:

```text
touchdown
field goal
turnover
punt
turnover on downs
end of half/game
rare scoring event
```

conditioned on:

```text
starting field position
offensive state
defensive state
expected participating players
special teams
down-distance tendencies
game conditions
```

Points emerge from the sequence and value of possessions.

## 10. Parameters

The model estimates:

```text
population means
population covariance matrices

team-state persistence matrices
player-state persistence parameters
state-change covariance matrices

play observation variances
drive observation variances

home-field coefficient
weather coefficients
rest coefficients
travel coefficients
venue coefficients

roster-transition coefficients
recruiting coefficients
transfer coefficients
returning-production coefficients

matchup interaction matrix
interaction shrinkage parameters

position-group means
position-group covariance matrices

availability coefficients
usage coefficients

source-specific measurement error
source-specific systematic bias
```

These numerical parameters are stored with:

```text
parameter_set_id
calculation_version
data_cutoff
source_manifest
estimated_value
uncertainty
```

## 11. Forecast calculation

For an upcoming game, the system retrieves:

```text
current team state estimates
current player state estimates
current unit estimates
player availability probabilities
expected player usage
venue
weather
rest
travel
matchup
schedule conditions
```

The statistical calculation combines uncertainty in all of those quantities:

\[
P(Y_g\mid D,X_g)
=
\int
P(Y_g\mid \Theta,\Phi,\Psi,X_g)
P(\Theta,\Phi,\Psi\mid D)
\,d\Theta\,d\Phi\,d\Psi
\]

The forecast output is a probability distribution over the game.

From that distribution the application calculates:

```text
home win probability
away win probability

expected home points
expected away points

expected margin
margin distribution

expected total
total distribution

probability of either side covering a specified spread
probability above/below a specified total

team-total probabilities
score distribution
```

## 12. Forecast storage

Every forecast is stored exactly as generated:

```text
forecast_id
game_id
creation_timestamp
information_cutoff
parameter_set_id

home_expected_points
away_expected_points
home_win_probability
away_win_probability

expected_margin
margin_variance
expected_total
total_variance

forecast_distribution_artifact

team_estimate_reference
player_estimate_reference
availability_reference
weather_reference
market_quote_reference
```

The corresponding inputs are also preserved.

## 13. External ratings

CORE, FPI, SP+, Elo and similar ratings enter as external statistical measurements when enabled.

For source \(r\):

\[
R_{i,t}^{(r)}
\sim
N(
a_r+b_r\theta_{i,t},
\sigma_r^2
)
\]

The historical record estimates:

```text
a_r
b_r
sigma_r
```

for each rating source.

This allows external ratings to contribute according to the amount of independent predictive information they contain.

## 14. Market observations

FanDuel observations are stored with:

```text
game_id
sportsbook
market_type
line
price
retrieved_at
```

The primary football forecast is produced from football data.

A separately evaluated market-conditioned calculation can incorporate the market through an additional statistical observation relationship.

This preserves the ability to measure whether football information contributes information beyond the market.

## 15. Game completion and validation

CFB TXT supplies weekly result validation.

Once a game is final:

```text
game_id
actual_home_points
actual_away_points
actual_margin
actual_total
actual_winner
result_retrieved_at
```

are associated with the existing forecast.

The evaluation calculates:

\[
AE_{margin}=|\hat M-M|
\]

\[
AE_{total}=|\hat T-T|
\]

\[
AE_{home}=|\hat S_h-S_h|
\]

\[
AE_{away}=|\hat S_a-S_a|
\]

\[
Brier=(p-y)^2
\]

\[
LogLoss=-y\log(p)-(1-y)\log(1-p)
\]

as well as:

```text
winner accuracy
margin MAE
margin RMSE
margin bias
total MAE
team-score MAE
Brier score
log loss
calibration
CRPS
coverage of predictive intervals
```

The completed game's football observations then become eligible for subsequent calculations according to their recorded availability time.

## 16. Continuous operating cycle

The complete system therefore operates as:

```text
SportsDataverse
CFBD
FootballDB
CFBTrack
weather
rosters
availability
recruiting
transfers
market data
        ↓
source retrieval records
        ↓
database
        ↓
canonical games / plays / drives /
players / roster / environment
        ↓
derived football observations
        ↓
hierarchical Bayesian statistical calculation
        ↓
team football-state representations
player football-state representations
unit football-state representations
estimated statistical parameters
        ↓
database
        ↓
upcoming game + current conditions
        ↓
joint predictive calculation
        ↓
game probability distribution
        ↓
forecast
        ↓
database
        ↓
game is played
        ↓
CFB TXT + detailed game observations
        ↓
outcome + forecast evaluation
        ↓
database
        ↓
new completed football evidence becomes
available to the next calculation
```

There is no statistical “application state” in this design. The application executes this system. The time-varying state representations describe football entities and football conditions.