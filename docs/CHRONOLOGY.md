# Chronological prospective holdout

The prospective validation path is event-time, not slate-time.

A slate forecast answers, “What would the model say about every known future game from one shared moment?” That is useful operationally, but it is not a chronological holdout. A chronological holdout gives each game its own fixed pregame cutoff and appends forecasts only after that cutoff has actually arrived.

## Policy

The default target is **24 hours before kickoff**.

For each game:

1. `scheduled_cutoff = kickoff - 24 hours`.
2. The game is ineligible before `scheduled_cutoff`.
3. When the runner executes at or after the cutoff but before kickoff, the real execution time is saved as `actual_forecast_at`.
4. All final results with `result_available_at < actual_forecast_at` may update team state. Later results cannot.
5. Only market observations with `observed_at <= actual_forecast_at` and `observed_at < kickoff` may enter the forecast.
6. The exact model, source files, market file, state snapshot, forecast rows and score/margin/total distributions are frozen in an immutable run.
7. The model hash cannot change within the ledger. A new model starts a new prospective ledger.
8. Forecast times are never backdated. If collection runs late, `cutoff_delay_minutes` records the delay.

The primary 24-hour analysis includes rows where `cutoff_delay_minutes <= 60` by default. Late-but-pregame rows remain auditable but are not allowed to masquerade as on-time 24-hour observations.

## Corrected October 2 bootstrap

The earlier `prospective_20261002T185600Z` batch forecast 54 games from one common 1:56 PM CDT state. That batch remains preserved as a slate snapshot, but it is **not** the chronological holdout.

The chronological bootstrap keeps only the 17 games whose individual 24-hour cutoffs had already arrived by 1:56 PM CDT. The other 37 games are excluded until their own cutoffs occur.

Of those 17 bootstrap games:

- 1 is within the one-hour primary cutoff tolerance.
- 16 are pregame but late relative to the 24-hour target and are flagged accordingly.
- 0 had a captured 2026 market observation available in the current point-in-time market store at that forecast timestamp.

This means the first primary 24-hour sample is intentionally tiny. It grows only as future game cutoffs actually occur.

## Commands

Bootstrap the ledger from the already-frozen batch:

```bash
python -m cfb_forecast chronology-bootstrap \
  --frozen-batch artifacts/holdouts/prospective_20261002T185600Z \
  --ledger artifacts/chronological_holdout_2026 \
  --lead-hours 24
```

At each later collection time, first append the market snapshot and refresh permitted current data, then append only newly due games:

```bash
python -m cfb_forecast market-snapshot-sdv \
  --season 2026 \
  --store data/market_history.csv

python -m cfb_forecast chronology-update \
  --data data \
  --model artifacts/backtest/model \
  --ledger artifacts/chronological_holdout_2026 \
  --season 2026 \
  --lead-hours 24 \
  --market data/market_history.csv
```

Verify every frozen run and the ledger chain:

```bash
python -m cfb_forecast chronology-verify \
  --ledger artifacts/chronological_holdout_2026
```

After final scores become available:

```bash
python -m cfb_forecast chronology-settle \
  --ledger artifacts/chronological_holdout_2026 \
  --data data
```

Settlement reports all forecasts separately from the fixed-cutoff primary subset. When contemporaneous no-vig moneyline probabilities exist, the same settled games can compare model probability scores with the market rather than comparing against a later closing line.

## October 2 live continuation

At 19:31 UTC (2:31 PM CDT), thirteen additional FBS games reached their exact 24-hour cutoff. They were appended from the previous frozen state at 23 hours 59 minutes before kickoff, so all thirteen qualify for the primary cutoff cohort. The ledger therefore contains 30 forecasts: 14 primary-cutoff-eligible and 16 preserved late bootstrap forecasts.

The public SportsDataverse market capture could not resolve DNS inside this sandbox at 19:31 UTC. The collection receipt records that failure and the forecasts remain marked `missing_at_forecast`; no later line was backfilled. The one-command runner is designed to continue forecasting when the optional market source is temporarily unavailable while retaining the failure as audit evidence.
