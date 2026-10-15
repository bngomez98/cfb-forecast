# Application architecture

The `sunday` service coordinates source collection, feature preparation, validation, model selection, inference, and persistence. Its CLI and the built-in scheduler call the same service. The dashboard reads completed runs and never performs inference or creates results.

`data` handles schedules, per-game advanced observations, source manifests, and schema validation. `sources` handles public game context (weather, injuries, provider tables) and preserves original responses with receipts. `sunday.sync_results` verifies game/team identity before updating authoritative outcomes; `capture_prices` uses actual displayed quotes with retrieval-time availability.

`features`, `ratings`, and `model` implement opponent-adjusted dynamic states and fitted score heads. `strength` supplies the time-decayed strength probability candidate. `analytic` calculates distribution probabilities directly. `probability` supplies prior-season calibration and bounded research experiments. `context_model` learns bounded context effects from saved pregame features and later results, evaluating them on a later week before selection.

`decision` converts prices, computes no-vig probabilities, expected returns and constrained hypothetical exposure. `validation` joins unchanged predictions to real final outcomes and evaluates forecast quality and recorded-price returns. Hash checks detect changed prediction files.

Generated sources, reports, and fitted models belong to runtime storage, not source control. Missing external facts retain explicit coverage statuses. Empty reports are not health or lineup assertions. The first run cannot manufacture past Sunday predictions; validation begins with actual saved runs.

Earlier library modules remain for compatibility and offline experimentation. The current CLI, scheduled workflow, README, and desk expose the Sunday application flow.

## Application layer

`scheduler` decides from the game calendar when each job is due (the Sunday service, the daily forecast, the weekly backtest) and runs it through `pipeline`, whose stages (data, preflight, players, market, results, web sources, forecast, verify) are recorded in a receipt under `artifacts/pipeline/`. `app` runs the scheduler and the dashboard in one process; the Compose services run them separately next to `api` and `data_watchdog`. `logs` gives every entry point one console format and per-run log files.

`players` is the hierarchical per-play player layer. It feeds two extra columns to the `pbp_player` score head, which `forecast`, `learning` and `backtest` all build the same way: point-in-time ratings per week, cached in `artifacts/cache/`. `weekly.run_daily_forecast` runs the whole remaining slate, compares it with the previous snapshot (`changes.csv`), writes the named player and team ratings, and publishes through `publisher`.

`webdata` reads the public web pages in its registry (HTML, JSON, PDF), maps team names onto schedule identities, cross-checks final scores between providers and the schedule, and exposes agreed finals through the results overlay in `data`. Its picks and lines are reference data and never set the forecast.
