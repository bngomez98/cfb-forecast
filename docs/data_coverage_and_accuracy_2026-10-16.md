# Data coverage and forecast accuracy, 2021 to 2026

Checked on 2026-10-16 against main plus the repair in PR #80 (main alone does
not import, because `sql/schema.sql` and `pipeline/publisher.py` are broken).

## Database

The shared SQLite file had been created before the player evidence and run
evidence tables existed. Opening it with `cfb_forecast.storage.database.connect`
adds the missing tables without touching existing rows:
`forecast_evaluations`, `forecast_evidence`, `forecast_inputs`,
`game_context_snapshots`, `player_identities`, `player_lineups`,
`player_observations`, `player_source_ids`. Integrity check: ok. All 23
existing tables kept their row counts. The new player tables are empty, so the
joint model reads player history from `player_game_facts`, as it did before.

## Schedule and result coverage

Counts come from a fresh `cfb-forecast fetch` of 2021-2026 and are compared
with the `results` table in the database.

| Season | FBS games | Completed with scores | In database | Missing | Postseason |
|---|---|---|---|---|---|
| 2021 | 770 | 770 | 770 | 0 | 38 |
| 2022 | 776 | 776 | 776 | 0 | 42 |
| 2023 | 792 | 792 | 792 | 0 | 42 |
| 2024 | 799 | 798 | 798 | 0 | 46 |
| 2025 | 808 | 808 | 808 | 0 | 46 |
| 2026 | 761 | 333 | 333 | 0 | 0 |

The one 2024 game without a score is App State vs Liberty (game 401640992),
which the schedule lists as canceled. 2026 has 761 scheduled FBS games; weeks
1-6 are complete and 4 of 59 week-7 games are final. No game ID is duplicated.
cfbresults.com, cfbtrack.com and collegefootballdb.com were not needed: there
are no gaps. The collegefootballdb receipt shows that fetch failed (DNS error).

## Accuracy

Winner picks, margin miss (points) and Brier score. Each game was predicted
week by week from results available 48 hours before that week's first kickoff.

| Season | Games | Winners right | Margin miss | Brier | Source |
|---|---|---|---|---|---|
| 2023 | 761 | 69.6% | 13.45 | 0.195 | joint, original settings (docs/joint_holdout_results.md) |
| 2024 | 793 | 70.6% | 13.09 | 0.193 | same |
| 2025 | 806 | 72.1% | 12.55 | 0.183 | joint, tuned defaults |
| 2026 (weeks 1-7) | 331 | 75.5% (250) | 13.03 | 0.173 | joint, tuned defaults, replayed 2026-10-16 |

2026 by week: week 1 81.6%, week 2 79.6%, week 3 77.2%, week 4 77.6%, week 5
80.4%, week 6 62.1%, week 7 (4 games) 1 of 4. Two week-1 games were skipped
because a team had no player history. The 2026 Brier uses the saved score noise
from `joint_noise_calibration.json`, which was measured with the original
settings; winners right and margin miss do not depend on it.
