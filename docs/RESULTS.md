# Computed results

Selected variant: **pbp**, selected using 2024 validation only.

Initial score heads fit 2022–2023; normalization and rating warmup use 2021. 2024 selects settings and calibrates covariance. These held-out 2025 predictions use heads fitted through 2024; only earlier completed games update the ratings during 2025.

## All 808 held-out FBS-versus-FBS games in 2025

| Model | Correct | Winner accuracy | Log loss | Brier | Margin MAE | Total MAE |
|---|---:|---:|---:|---:|---:|---:|
| elo | 562/808 | 69.55% | 0.5750 | 0.1965 | — | — |
| score | 582/808 | 72.03% | 0.5377 | 0.1814 | 12.50 | 12.98 |
| pbp | 590/808 | 73.02% | 0.5406 | 0.1830 | 12.41 | 12.99 |

Lower log loss, Brier score and MAE are better. The selected PBP variant gets more winners correct than score-only, but its probability error is slightly worse in this test year. Variant selection is retained from 2024; this test is not used to switch the released model.

## Matched pregame public forecasts: 763 games

| Model | Correct | Winner accuracy | Log loss | Brier | Margin MAE |
|---|---:|---:|---:|---:|---:|
| archive | 549/763 | 71.95% | 0.5200 | 0.1753 | 12.00 |
| score_matched | 551/763 | 72.21% | 0.5345 | 0.1801 | 12.51 |
| pbp_matched | 557/763 | 73.00% | 0.5379 | 0.1820 | 12.43 |

The archived tracker has lower probability error and lower margin error than either implementation, despite slightly fewer correct winner picks. It is a public ensemble/forecast tracker, not a sportsbook moneyline baseline. Model and archive use the same retained snapshot cutoff for these games.

## Paired uncertainty

| Comparison | Log-loss difference | 95% interval |
|---|---:|---|
| pbp_p minus score_p | +0.0029 | [-0.0040, +0.0105] |
| score_p minus elo_p | -0.0373 | [-0.0639, -0.0137] |
| pbp_p minus elo_p | -0.0344 | [-0.0650, -0.0056] |
| score_p minus archive_p | +0.0145 | [+0.0039, +0.0247] |
| pbp_p minus archive_p | +0.0179 | [+0.0076, +0.0278] |

Negative differences favor the first model. Resampling uses 3,000 season-week block draws, with seed 20261002. Blocks capture same-week dependence but not all repeated-team dependence or model-selection uncertainty. The two state models improve on calibrated Elo in this sample; the PBP-versus-score difference is unresolved. Both are worse than the retained external tracker on probability loss.

## Descriptive archived spread directions

| Variant | Minimum projected margin advantage | Wins | Losses | Pushes | Win rate without pushes |
|---|---:|---:|---:|---:|---:|
| score | >0 points | 366 | 379 | 18 | 49.13% |
| score | >2 points | 228 | 228 | 9 | 50.00% |
| score | >4 points | 116 | 126 | 7 | 47.93% |
| pbp | >0 points | 371 | 374 | 18 | 49.80% |
| pbp | >2 points | 215 | 218 | 9 | 49.65% |
| pbp | >4 points | 113 | 116 | 6 | 49.34% |

The selected PBP variant is essentially 50/50 on recorded spread directions. That does not establish an advantage after betting costs. These lines lack verified sportsbook prices and exact line-availability timestamps, so no historical ROI is calculated. Fixed edge thresholds are diagnostics, not settings selected for deployment.

## Checks and limits

- 20 automated tests passed. They include future-outcome exclusion for both state and Elo ratings, strict result availability at cutoff, covariance stability, neutral-site symmetry, frozen model serialization, simulation determinism, market-timestamp filtering and invalid-score quarantine.
- Every evaluated forecast cutoff is before kickoff, and every latest contributing result is before its forecast cutoff.
- Saved input hashes were verified against the prior source package. Original external snapshot CSVs and commit metadata are retained; the archive adapter checks their internal consistency.
- The source records may have been corrected after games. No immutable as-known statistical archive or historical injury model is reconstructed.
- Final score simulation is an approximate joint Gaussian model, not a drive-level rules engine.
- The historical test season had been studied in prior work. A prospective holdout is still needed.

The delivered result is a working, testable forecasting engine. Its historical performance does not justify a claim that it beats the market.
