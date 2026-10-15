# CFB Prediction Model — Ongoing Accuracy Record

Updated October 6, 2026

This is the persistent weekly record for the developed college-football prediction model. Its purpose is to improve prediction accuracy. The record preserves what the model predicted before each game, what actually happened, why the prediction was accurate or inaccurate, which information or adjustment most likely caused the difference, and whether the same error appears again in later games.

The developed prediction model uses historical and current team data, historical and current player data, opponent-adjusted football performance, dynamic hierarchical Bayesian updating, matchup-specific effects, player availability and role information, and the pregame adjustments we have established for turning those inputs into a prediction. Those components exist to improve the prediction; the weekly review is not organized around the internal mechanics for their own sake.

A completed game is therefore reviewed from the prediction outward. We first ask what we predicted and how far it was from reality. We then trace the error through the information and adjustments that produced the prediction. Only after repeated evidence identifies a specific weakness do we change the prediction method itself.

## What counts as prediction accuracy

Accuracy is not one number. A prediction can get the winner right while badly missing the score, or miss the winner while still describing a genuinely close game well. Each weekly review therefore evaluates several concrete parts of the prediction.

**Winner accuracy** asks whether the team we assigned the higher win probability actually won. This is the simplest measure, but it throws away how confident the prediction was and how close the game was expected to be.

**Predicted score accuracy** compares each team's predicted points with the points it actually scored. If we predict 31-21 and the game ends 28-24, the prediction was three points high for one team and three points low for the other. Looking at the two team errors separately helps identify whether the problem came mainly from the offensive expectation for one side, the defensive expectation for the other, or both.

**Margin accuracy** compares the predicted difference between the teams with the actual difference. If Ohio State is predicted to win by 4.5 and wins by 17, the winner is correct but the expected separation between the teams was 12.5 points too small. This is important because a method can consistently identify the better team while still compressing or exaggerating differences in strength.

**Combined-score accuracy** compares the predicted total points with the actual total points. If the model predicts 27-24, the predicted total is 51. If the game ends 31-21, the actual total is 52. This tells us whether we were wrong about the overall scoring environment even if the points were allocated to the wrong teams.

**Probability accuracy** asks whether the win probabilities match how often those events actually occur over time. A 52% prediction that loses is very different from an 82% prediction that loses. One close miss is ordinary uncertainty. Repeated high-confidence misses indicate that the prediction probabilities are too confident, that the underlying ordering is wrong, or both.

These measures are descriptive tools. They tell us where a prediction failed. They do not by themselves explain why it failed.

## The information and parameters used to create a prediction

Not every number in the system is a parameter. Some quantities are observed football data, some are changing estimates of underlying team or player ability, some are fixed control settings, some are matchup adjustments, and some are final prediction outputs. Keeping those categories separate prevents us from calling every statistic a parameter.

### Observed team information — inputs, not parameters

**Points scored and allowed.** The actual scoring output of previous games. It is useful but noisy because touchdowns can come from short fields, turnovers, returns and other events that are not equally persistent from game to game.

**Yards per play.** Total offensive yards divided by offensive plays. It measures how much field position an offense gains per snap. It is more informative than total yards when teams play at different tempos, but it still mixes rushing, passing, explosives and game-state effects.

**Yards per pass.** Passing yards relative to passing opportunities. It helps isolate passing efficiency from overall offense. The exact denominator used must remain consistent with the source implementation.

**Yards per rush.** Rushing yards relative to rushing attempts. It describes rushing efficiency, but a single long run can distort the average, which is why explosiveness and success information are also relevant.

**First-down creation rate.** How frequently an offense creates first downs relative to its opportunities. It captures drive sustainability that raw yardage can miss.

**EPA / expected-points contribution.** Play-level change in expected scoring value. EPA is intended to distinguish a five-yard gain on third-and-four from the same five-yard gain on first-and-ten because the two plays change scoring prospects differently. The broader developed specification uses this information; the October 2 executable engine did not yet implement the full intended EPA layer.

**Success rate.** The share of plays meeting the established down-and-distance success requirement. It describes how consistently an offense stays on schedule rather than how many yards it averages. It is complementary to explosiveness.

**Explosiveness.** Frequency and/or magnitude of unusually large gains. This separates an offense that repeatedly gains useful yardage from one whose average is driven by occasional huge plays.

**Negative-play rate.** Frequency of sacks, tackles for loss or other damaging offensive plays. These plays can end drives even when average yardage looks acceptable.

**Turnovers and turnover opportunities.** Actual interceptions and lost fumbles are observed outcomes. They are not treated as pure team strength because turnover totals contain substantial game-to-game randomness. Persistent quarterback interception tendency, ball-security problems or defensive pressure can still carry predictive information, but the realized turnover margin of a single game should not be copied directly into the next forecast.

**Drive and possession information.** Number of possessions, starting field position, drive length, scoring result and related play sequence information. These are used to understand opportunity volume and how efficiently teams turn opportunities into points.

### Observed player information — inputs, not parameters

**Player historical production.** Previous passing, rushing, receiving and other position-relevant performance. Historical production provides a starting point for what a player can contribute but must be interpreted in the context of role, teammates, opponent quality and age/season progression.

**Current-season production.** What the player has done this season under the current coaching staff, teammates and role. It normally receives more relevance for the current prediction than old production once enough current evidence exists.

**Usage / role.** How much of the team's offense, defense or specialist work is actually flowing through the player: attempts, carries, targets, snaps, quarterback role, kicking role and similar responsibilities. Production without expected usage should not be projected forward unchanged.

**Availability.** Whether a player is expected to play and, when uncertain, how likely or limited participation appears to be from information available before the cutoff. Availability changes which historical team performance is transferable to the upcoming game.

**Replacement quality.** The expected contribution of the player who assumes the role if the usual player is unavailable or limited. Losing a starter is not a fixed penalty; the effect depends on the difference between the starter and the replacement and on how important that role is to the matchup.

### Dynamic team-state parameters

These are changing quantities intended to represent current football ability after accounting for prior evidence, opponent quality and uncertainty.

**Offensive state for metric m, O(i,t,m).** The current underlying offensive ability of team *i* at time *t* for a specific component *m*. The component may be scoring, yards per play, passing efficiency, rushing efficiency, first-down creation or another integrated metric. It is not the team's raw average. It is the opponent-adjusted current level inferred from the games available before the prediction.

Why it matters for accuracy: if this state is too high, the team will generally be projected to score or move the ball too efficiently. If it is too low, the prediction will systematically suppress that team's offense.

**Defensive state for metric m, D(i,t,m).** The current underlying defensive ability of team *i* for the same component. It represents how the defense changes an opponent's expected performance after accounting for the quality of offenses it has faced.

Why it matters for accuracy: an underrated defense can make us repeatedly overpredict opposing offenses; an overrated defense can make us underpredict them.

**State uncertainty, V(i,t,m).** How uncertain the method is about the current offensive or defensive state. Early in a season, after major personnel change, or against weakly connected schedules, uncertainty should be larger. After more informative games, it should generally decrease unless the team's performance is changing rapidly.

Why it matters for accuracy: uncertainty controls how strongly we should trust a current rating and how widely the possible game outcomes should be spread. Treating an uncertain team as precisely known makes win probabilities too confident.

**State-change variance, Q(m).** The amount a team is allowed to change from one time point to the next for component *m*. This is the mathematical expression of the fact that current ability is not fixed for an entire season.

Why it matters for accuracy: if Q is too small, the prediction reacts too slowly to genuine improvement, decline or personnel change. If Q is too large, recent games cause excessive movement and the prediction becomes unstable.

**Observation variance, R(m).** The amount of game-to-game noise expected in the observed value of metric *m* after accounting for team and opponent ability.

Why it matters for accuracy: R determines how much a single game should move the current state. A noisy statistic should not cause the same update as a highly stable one.

**Kalman gain, K.** In the currently implemented dynamic update, K is calculated from prior uncertainty and observation uncertainty. It determines how much the current state moves toward the newest opponent-adjusted observation.

In simplified form:

K = prior uncertainty / (prior uncertainty + observation uncertainty)

A larger K means the new game has more influence. A smaller K means prior information remains more influential. K is calculated from the uncertainty terms; it is not a football statistic and is not chosen independently for each team by hand.

### Player-state parameters in the developed method

**Player ability state, P(p,t,k).** The current contribution of player *p* at time *t* for role or skill component *k*—for example quarterback passing, rushing contribution, receiving contribution or another position-specific component.

Why it matters for accuracy: team history generated by one personnel group should not automatically carry forward after meaningful personnel changes. Player states determine how much historical team performance remains applicable to the lineup expected for the next game.

**Player-state uncertainty.** Uncertainty about that player's current contribution. A long-term starter with stable usage can be known more precisely than a new quarterback with one start.

**Expected participation / role weight.** The expected share of the relevant role the player will perform in the upcoming game. A player who is active but expected to play 30% of snaps should not receive the same weight as a full-time starter.

**Replacement differential.** The difference between the expected contribution of the normal player and the expected contribution of the replacement. This is the actual football quantity that determines the prediction effect of an absence; the injury label itself is only information used to infer it.

These player terms are part of the developed prediction model. The October 2 executable implementation did not yet contain the complete player-availability layer, so current weekly reviews must explicitly distinguish what the developed model calls for from what that specific executable prediction actually used.

### Opponent-adjustment quantities

**Opponent-adjusted observation.** The performance observed in a game after removing the part expected from the opponent's current strength. For offense, strong production against a strong defense carries more positive information than the same raw production against a weak defense. For defense, suppressing a strong offense carries more information than suppressing a weak one.

Why it matters for accuracy: college teams play radically different schedules. Without opponent adjustment, raw statistics systematically overrate teams with weak schedules and underrate teams that have faced strong opponents.

**Schedule-connection uncertainty.** Additional uncertainty caused by how well a team's opponents connect it to the broader FBS schedule network. Early-season groups that mostly play among themselves provide weak evidence for cross-group comparisons.

Why it matters for accuracy: the same number of games can provide different amounts of information depending on who played whom.

### Matchup parameters and adjustments

**Pass matchup adjustment.** The change to the offensive passing expectation produced by the relationship between the offense's passing strengths and the opposing defense's pass-defense strengths.

**Rush matchup adjustment.** The corresponding change for rushing performance.

**Success-rate matchup adjustment.** Adjustment based on the offense's ability to stay on schedule against the defense's ability to prevent successful plays.

**Explosive-play matchup adjustment.** Adjustment based on the offense's tendency to create large gains and the defense's tendency to allow or suppress them.

**Situational matchup adjustment.** Adjustment for down, distance, red-zone, passing-down, short-yardage or other game situations when those data are integrated and sufficiently supported.

Why these matter for accuracy: a team is not equally strong against every opponent. A single overall team rating can miss a game where one team's specific strength attacks the opponent's specific weakness.

**Regularization strength, lambda.** A control parameter used in the current regression-based score translation. It penalizes excessively large coefficients when several inputs overlap.

Why it matters for accuracy: too little regularization can make the prediction chase noise; too much can flatten meaningful differences between teams. Because the current score layer contains overlapping information, lambda affects how strongly the final predicted score responds to those differences.

### Context adjustments

**Home-field term, H.** The expected change in performance associated with playing at home rather than away, after controlling for the teams themselves. Neutral-site games do not receive an ordinary home-field adjustment.

**Rest / travel / schedule context.** Pregame information about preparation time, travel and schedule circumstances when included and supported by the data. These should only enter if their predictive contribution has been established rather than assumed.

**Game-state policy terms.** The way expected behavior changes with score differential, quarter and remaining time. Teams do not use the same pace and play mix while leading by 17 in the fourth quarter as they do in a tie game in the first quarter.

### Possession and scoring parameters in the developed method

These quantities describe the possession-based part of the developed forecasting method. The separate possession experiment did not complete numerical estimation and is not silently treated as the current executable prediction engine.

**Expected shared possession opportunity.** The expected number and sequence of possessions created by both teams within the same game. The key point is that the two teams' possession counts are dependent because they share one clock and possession sequence.

**Possession-count covariance.** The dependence between the opportunity counts of the two teams. A fast game tends to increase opportunities for both teams; a slow game tends to reduce both.

**Possession value.** Expected scoring value of a possession given the offense, defense, field position and game situation.

**Possession-count / possession-value dependence.** The relationship between how many possessions occur and how valuable those possessions are. The method cannot safely assume these are independent because quick scores, long drives, score state and play style affect both.

**Starting-field-position effect.** The change in scoring expectation associated with where the possession begins. A possession beginning near midfield does not have the same scoring expectation as one beginning at the offense's own ten-yard line.

**Field-goal scoring probability.** Conditional probability that a possession results in a field goal under the current offense-defense-field-position situation.

**Touchdown scoring probability.** Conditional probability that a possession results in a touchdown under the same situation.

**No-score probability.** Conditional probability that the possession produces no points.

**Rare-score probabilities.** Probabilities for safeties, defensive/return scoring and other uncommon outcomes when explicitly represented. These events require careful treatment because they can distort ordinary offensive-possession accounting.

**Late-half effect.** Change in possession or scoring behavior near halftime or the end of regulation, when clock constraints alter play calling and scoring opportunities.

### Score-production parameters in the October 2 executable implementation

The October 2 executable engine converts the dynamic opponent-adjusted team components into expected team scores through a regularized regression layer.

**Intercept.** Baseline scoring level before the team and matchup inputs move the prediction up or down.

**Regression coefficients, beta(j).** Weights applied to each input entering the score calculation. A positive coefficient means larger values of that input increase the predicted score, holding the other inputs fixed; a negative coefficient decreases it.

Why they matter for accuracy: the coefficients determine how strongly the current team information is translated into points. If the translation is too weak, large football differences can be compressed into small predicted margins. If it is too strong, ordinary differences can become exaggerated.

**Score-error variance for each team.** Historical amount by which actual team scores vary around the expected score after the deterministic part of the calculation.

**Home/away score-error covariance or correlation.** Historical tendency for the two teams' score errors to move together within the same game. For example, a game environment that is unexpectedly fast or slow can affect both teams.

These error terms are used to represent uncertainty around the expected scores. In the October 2 implementation, they are estimated separately from the uncertainty in the underlying team ratings and score coefficients, so the entire uncertainty chain is not yet fully propagated through one calculation.

### Prediction outputs — not parameters

**Expected home score.** The average home-team scoring result implied by the information and adjustments used for the game.

**Expected away score.** The corresponding away-team result.

**Predicted margin.** Expected home score minus expected away score. A negative value means the away team is favored.

**Predicted total.** Expected home score plus expected away score.

**Win probability.** Probability that a team finishes with more points than its opponent under the uncertainty represented in the prediction calculation.

**Cover / total probability.** Probability of finishing on the relevant side of a sportsbook spread or total. These are only meaningful when a genuine point-in-time market number is available.

**Simulation count.** Number of random draws used when numerical simulation is used to approximate outcome probabilities. This is a computational setting, not a football parameter. Increasing it reduces Monte Carlo sampling noise but does not improve the underlying football information.

## How each completed game will now be reviewed

The weekly review is centered on prediction accuracy. For every important forecast, the record will explain the following in full sentences rather than merely naming a statistical category.

### 1. What we predicted

The exact score, margin, total and win probability actually produced before kickoff, along with the timestamp and the version of the executable calculation that produced it.

### 2. What actually happened

The final score plus the football evidence needed to understand the result: team efficiency, passing and rushing performance, success and explosive plays where available, turnovers, possession/drive information, player usage and performance, and availability changes that occurred before or during the game.

### 3. Exactly where the prediction was right or wrong

The review will not stop at "winner correct" or "margin missed." It will explain whether the prediction error came primarily from:

- expecting the wrong team to be stronger;
- correctly ordering the teams but understating or overstating the strength gap;
- getting the overall scoring environment wrong;
- allocating a reasonable total to the wrong teams;
- assigning too much or too little confidence;
- missing an important player or availability effect;
- misreading a specific matchup;
- overreacting or underreacting to recent games;
- or encountering a game outcome heavily affected by high-variance events such as turnovers, return scores or unusual field position.

### 4. Why the game evidence supports that conclusion

This is where the weekly review must explain the reasoning rather than merely say "update offense" or "update defense."

For example, Missouri's 45-17 win over Florida is much stronger evidence of a pregame strength-ordering problem than a 28-point win created by a +4 turnover margin would have been. Missouri had no turnover advantage, gained 560 yards to Florida's 325 and averaged 8.5 yards per play to Florida's 5.1. The result therefore contains substantial direct evidence that Missouri's offense and/or Florida's defense were materially mischaracterized before kickoff. The review must then determine which pregame inputs created the Florida-favored prediction.

By contrast, Louisville's loss to NC State cannot be interpreted the same way simply because the favored team lost. Louisville produced 8.8 yards per play to NC State's 5.9 but lost three fumbles. That result gives us evidence about both teams, but the winner alone is weak evidence that Louisville was actually the inferior team that day. The forecast error has to be separated into underlying performance and turnover realization before it informs the next prediction.

### 5. What this means for the next prediction

The review will state concretely how the new evidence changes the information entering the next game. For example: stronger evidence that Missouri's current offense is better than previously represented; weaker evidence for Florida's defensive ability; increased uncertainty if a quarterback change makes old passing history less transferable; or a larger matchup adjustment if a repeated pass-protection weakness is likely to be attacked by the next opponent.

This is not the objective by itself. It is the bridge from what we learned about prediction accuracy this week to a more accurate next prediction.

### 6. Whether the prediction method itself needs revision

One game does not trigger a redesign. A method-level change requires a repeated, measurable error that can be traced to a particular part of the prediction process.

If, for example, future weeks repeatedly show that predicted margins are too small even after current team/player information is updated correctly, then we would inspect the score-translation coefficients, regularization and interaction structure. If high-confidence favorites repeatedly lose despite the expected-strength ordering being sound, we would investigate probability calibration and uncertainty. If errors cluster around quarterback replacements, we would inspect how player information is entering the prediction.

The change is justified by the repeated accuracy failure and its identified cause, not by a desire to make the system more elaborate.

## Week ending October 3, 2026

### Prediction timing

The October 2 executable record contains 14 pregame predictions relevant to this review. Thirteen were produced one minute after their exact T-24 timestamp. Toledo-Ball State was produced 56 minutes after its exact T-24 timestamp. They remain genuine pregame predictions, but none is relabeled as exactly on time.

### Saved predictions and actual results

| Game | Predicted score (away-home) | Favored team | Home win probability | Final score (away-home) | Actual winner | Timing |
|---|---:|---|---:|---:|---|---|
| Toledo at Ball State | 37.2-16.3 | Toledo | 9.7% | 39-24 | Toledo | 56 min after T-24 |
| Florida at Missouri | 29.5-27.0 | Florida | 43.9% | 17-45 | Missouri | 1 min after T-24 |
| Auburn at Tennessee | 22.6-26.5 | Tennessee | 59.6% | 14-24 | Tennessee | 1 min after T-24 |
| California at UNLV | 20.1-27.1 | UNLV | 66.9% | 31-39 | UNLV | 1 min after T-24 |
| Louisville at NC State | 31.3-26.3 | Louisville | 37.9% | 28-31 | NC State | 1 min after T-24 |
| Virginia at Florida State | 25.8-23.4 | Virginia | 44.2% | 7-38 | Florida State | 1 min after T-24 |
| Ohio State at Iowa | 23.6-19.1 | Ohio State | 39.1% | 31-14 | Ohio State | 1 min after T-24 |
| Memphis at Charlotte | 31.3-19.5 | Memphis | 23.1% | 59-8 | Memphis | 1 min after T-24 |
| Wyoming at North Dakota State | 13.2-27.3 | North Dakota State | 80.8% | 0-28 | North Dakota State | 1 min after T-24 |
| Akron at Central Michigan | 22.2-30.1 | Central Michigan | 68.9% | 17-41 | Central Michigan | 1 min after T-24 |
| Eastern Michigan at Massachusetts | 20.2-29.5 | Massachusetts | 72.0% | 38-14 | Eastern Michigan | 1 min after T-24 |
| Ohio at Kent State | 30.7-21.5 | Ohio | 28.4% | 13-10 | Ohio | 1 min after T-24 |
| Bowling Green at Miami (OH) | 17.9-32.7 | Miami (OH) | 82.1% | 24-20 | Bowling Green | 1 min after T-24 |
| Old Dominion at Georgia State | 25.4-27.3 | Georgia State | 54.7% | 10-42 | Georgia State | 1 min after T-24 |

### What this week's predictions actually tell us

The first job is to describe the error precisely before assigning a cause. The October 3 games contain several different kinds of misses, and those cannot all be explained by the same adjustment.

Florida-Missouri: the prediction was Florida 29.5, Missouri 27.0, so Florida was projected by about 2.5. Missouri won 45-17. The margin error was therefore about 30.5 points. Missouri gained 560 yards to Florida's 325 and 8.5 yards per play to 5.1, with no turnover advantage. Those postgame facts tell us that the final score was supported by a large difference in ordinary offensive production rather than being created mainly by turnover luck. They do **not** tell us which pregame component caused the prediction to favor Florida. That diagnosis requires the actual pregame decomposition: the team ratings used for each side, player/availability information, opponent adjustments, matchup transformations and the score-translation contribution that produced 29.5-27.0. Until that decomposition is recovered, the source of the prediction error is unresolved.

Virginia-Florida State: the prediction was Virginia 25.8, Florida State 23.4; Florida State won 38-7. The winner ordering was wrong and the margin miss was about 33.4 points. Florida State produced 7.1 yards per play while Virginia produced 2.8, and Florida State did not benefit from a positive turnover margin. This tells us the realized football performance was far from the forecasted relationship. It still does not establish whether the pregame error came from team-state values, player information, opponent adjustment, matchup effects or the final score translation. The next accuracy step is to recover those actual contributions rather than infer the cause from the box score.

Eastern Michigan-Massachusetts: Massachusetts was predicted around 29.5-20.2 with roughly a 72% home win probability, but Eastern Michigan won 38-14. This was simultaneously a winner error, a large score-allocation error and a confidence error. The useful question is what pregame inputs created such a strong Massachusetts advantage. Until the contributing team/player and matchup values are recovered, the result establishes the size and direction of the miss, not its cause.

Bowling Green-Miami (OH): Miami was projected around 32.7-17.9 with about an 82% win probability and lost 24-20. Because the actual game was close, this is especially useful for evaluating probability confidence. An 82% prediction implies the model believed a Miami loss was relatively uncommon; one close upset cannot show that the underlying team ordering was badly wrong, but repeated losses or near-losses among similarly strong favorites would indicate that confidence is too concentrated.

Ohio State-Iowa: Ohio State was projected to win by about 4.5 and won by 17. The winner was correct, while the magnitude was too small by roughly 12.5 points. The predicted total was close to the actual total, so the larger error was how those points were divided between the teams. If this same pattern repeats across games, the decomposition should test whether the score translation is shrinking true matchup separation; one game alone cannot establish that.

Memphis-Charlotte: Memphis was projected by about 11.8 and won by 51. Memphis also dominated total offense and possession, while three turnovers increased the final separation. The review should therefore separate the part of the margin supported by repeatable offensive/defensive performance from the part amplified by high-variance events. The cause of the original 11.8-point prediction still has to be traced through the pregame calculation.

Louisville-NC State: Louisville was projected to win by about five and lost 31-28, but Louisville gained 8.8 yards per play to NC State's 5.9 and lost three fumbles. This is a good example of why a wrong winner does not automatically mean the pregame team-strength ordering was wrong. The final score was materially influenced by turnovers, so the accuracy review must compare the expected relationship with the underlying play as well as the final outcome.

### Accuracy question carried into the next week

The main unresolved accuracy question is whether the October 2 executable calculation systematically understated large team differences. Several games show correct or plausible overall scoring environments but much larger actual team separations than predicted. That is now a hypothesis to test against future predictions, not a conclusion already built into the system.

If the same pattern persists after new team and player information is incorporated, we will trace it specifically through the score translation, regularization, matchup interactions and uncertainty treatment. If it disappears, this week was simply an unusually blowout-heavy group of games.

## Earlier evidence retained for comparison

The 2025 evaluation contains 808 FBS games. The PBP implementation selected before that test season chose 590 winners, or 73.0%. Its average absolute margin error was about 12.4 points. The simpler score-only implementation chose 582 winners, or 72.0%, and produced slightly better probability scoring. These are historical accuracy measurements for those implementations, not definitions of the current prediction model.

The earlier 2021-2024 component study covered 3,136 games and showed that the dynamic opponent-adjusted strength component improved probability quality over the tested Elo baseline, while the PBP and matchup additions used in that earlier implementation did not improve probability error. That evidence remains useful because it demonstrates that added complexity only earns a place when it improves future prediction accuracy.

## Weekly update standard

Each week this same artifact is extended rather than replaced.

For every new forecast, preserve the actual pregame prediction, exact timestamp, executable version and the team/player information available at the time.

For every completed game, explain in readable language what the prediction got right or wrong, how large the error was, what happened on the field, which pregame inputs or adjustments most likely produced the error, why the game evidence supports that interpretation, and what this implies for the next prediction.

Do not write statements such as "update the offense," "adjust the defense," or "evidence suggests recalibration" without explaining the football evidence, the mathematical connection and the expected effect on future predictions.

Do not change the prediction method because of one result. Preserve repeated error patterns and only alter a component when the accumulated accuracy evidence identifies a specific, reproducible problem.

Past predictions are never rewritten with later information.

## Source state used for the October 6 update

- Saved October 2 executable state: `reviewed_snapshot.json`, engine version 0.3.1.
- Existing project continuity record: `CFB_Project_Record.md`, updated October 3, 2026.
- October 3 final scores and game statistics were verified against current college-football game records on October 6, 2026.
- Previously inspected project source: commit `93e840069b0928b53d60c0583aacdf52bb3e0148`.
- The unfinished possession experiment remains separate from the executable prediction calculation until its estimation, validation and prediction path are completed.


## Chronological point-in-time workflow — October 7, 2026

The purpose of this section is operational: make the actual point-in-time prediction at each game's exact T-24 timestamp, preserve it unchanged, and later compare that exact prediction with the completed game. Reaching the cutoff does not itself create a prediction.

### Cutoffs that were missed

| Game | Kickoff | Exact T-24 cutoff | Status |
|---|---|---|---|
| Jacksonville State at Kennesaw State | Oct. 7, 7:00 PM ET / 6:00 PM CT | Oct. 6, 7:00 PM ET / 6:00 PM CT | Cutoff passed; no T-24 point-in-time prediction was produced |
| New Mexico State at FIU | Oct. 7, 7:30 PM ET / 6:30 PM CT | Oct. 6, 7:30 PM ET / 6:30 PM CT | Cutoff passed; no T-24 point-in-time prediction was produced |

These games may still receive a clearly timestamped late pregame prediction if useful, but that prediction cannot enter the exact T-24 prospective sample.

### Next exact point-in-time prediction times

| Game | Kickoff | Exact T-24 prediction time |
|---|---|---|
| Sam Houston at Liberty | Oct. 8, 7:00 PM ET / 6:00 PM CT | Oct. 7, 7:00 PM ET / 6:00 PM CT |
| Missouri State at Western Kentucky | Oct. 8, 7:00 PM ET / 6:00 PM CT | Oct. 7, 7:00 PM ET / 6:00 PM CT |
| South Florida at UTSA | Oct. 8, 7:30 PM ET / 6:30 PM CT | Oct. 7, 7:30 PM ET / 6:30 PM CT |
| South Alabama at Arkansas State | Oct. 8, 7:30 PM ET / 6:30 PM CT | Oct. 7, 7:30 PM ET / 6:30 PM CT |

At those timestamps the prediction record must contain the actual outputs produced by the developed prediction calculation: both projected team scores, margin, total, win probability, information timestamp, executable/version identity, and the contemporaneous market observation if a genuine same-time market quote is available. If the executable prediction path cannot be run at that time, the record must state that the T-24 prediction was missed rather than fill in a later reconstruction.
