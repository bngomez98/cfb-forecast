# Joint player model: season-by-season holdout results

Run on 2026-10-16 with `cfb-forecast joint-holdout`, using 2021-2025 schedules,
team stats and player stats (770 to 808 completed FBS games per season, with
player stats for 742 to 807 of them). Each season was replayed week by week.
Before each week, the model was refitted from results available 48 hours
before that week's first kickoff, and then it predicted that week's games.

The previous team-only model was trained with `run_backtest` on seasons before
each test season. Closing spreads come from the SportsDataverse betting files.
They have no moneylines, so the betting market has no Brier score. Every row
below covers the same games.

## Untouched test seasons, original settings

| Season | Model | Games | Winners right | Avg margin miss | Brier | Log loss |
|---|---|---|---|---|---|---|
| 2023 | Joint, opponent-adjusted summaries | 761 | 69.6% | 13.45 | 0.195 | 0.574 |
| 2023 | Joint, plain season averages | 761 | 69.1% | 13.54 | 0.195 | 0.575 |
| 2023 | Previous team-only model | 761 | 73.2% | 12.64 | 0.180 | 0.534 |
| 2023 | Closing spread | 761 | 72.7% | 12.18 | n/a | n/a |
| 2024 | Joint, opponent-adjusted summaries | 793 | 70.6% | 13.09 | 0.193 | 0.567 |
| 2024 | Joint, plain season averages | 793 | 69.5% | 13.20 | 0.194 | 0.568 |
| 2024 | Previous team-only model | 793 | 71.1% | 12.75 | 0.188 | 0.553 |
| 2024 | Closing spread | 793 | 71.5% | 11.78 | n/a | n/a |
| 2025 | Joint, opponent-adjusted summaries | 806 | 71.5% | 12.81 | 0.187 | 0.551 |
| 2025 | Joint, plain season averages | 806 | 71.2% | 12.81 | 0.187 | 0.551 |
| 2025 | Previous team-only model | 806 | 72.7% | 12.30 | 0.182 | 0.538 |
| 2025 | Closing spread | 806 | 73.3% | 11.85 | n/a | n/a |

The joint model's Brier and log loss use score noise measured on earlier
replayed seasons only.

## Score noise measured on unseen games

The model's own uncertainty about its team and player estimates already
exceeded the actual errors. Average parameter variance was 163 to 196
points squared per team score. The observed squared error was 137 to 155.
So the measured leftover noise hit its floor of 1 point squared in every fold.
Home and away errors were essentially uncorrelated (correlation 0.02).
Errors were somewhat larger in weeks 1-4 (158 and 148) than later (148 and 126).

## Tuning the fixed settings

Tuning used 2022 and 2023 replays, refitting every four weeks, and was
confirmed on 2024. 2025 was not used for tuning. The best setting on the
tuning seasons cut all six prior variances to a quarter of their original values:

| Check | Settings | Winners right | Avg margin miss | Brier | Log loss |
|---|---|---|---|---|---|
| 2024 confirmation | original | 68.8% | 13.52 | 0.198 | 0.583 |
| 2024 confirmation | tuned | 69.3% | 13.24 | 0.193 | 0.565 |
| 2025 final check | original | 71.5% | 12.81 | 0.187 | 0.551 |
| 2025 final check | tuned | 72.1% | 12.55 | 0.183 | 0.542 |

With the tuned settings, the measured noise is no longer at its floor
(61 and 46 points squared, with a slight negative home/away correlation).
The tuned settings are now the defaults in `HierarchyConfig`. A quarter was
the smallest value in the grid, so smaller values have not been checked.

On 2025, the tuned joint model is still behind the previous team-only model:
72.1% against 72.7% of winners, a margin miss of 12.55 against 12.30, and a
Brier score of 0.183 against 0.182.

## Limitations

- Final-result times are kickoff plus eight hours, because the schedule records no end time.
- Historical team and player stats were downloaded after the fact and may include later corrections. Their original publication times are unknown, so they enter only after each game's result time.
- Historical lineups are the last observed participation before the cutoff, not confirmed pregame lineups.
- Games where a team had no player history before the cutoff were skipped: 1 or 2 per season.
