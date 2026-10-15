# Application architecture

The `sunday` service coordinates source collection, feature preparation, validation, model selection, inference, and persistence. Its CLI and GitHub workflow call the same service. The dashboard reads completed runs and never performs inference or creates results.

`data` handles schedules, per-game advanced observations, source manifests, and schema validation. `sources` handles public game context and preserves original responses with receipts. `sunday.sync_results` verifies game/team identity before updating authoritative outcomes; `capture_prices` uses actual displayed quotes with retrieval-time availability.

`features`, `ratings`, and `model` implement opponent-adjusted dynamic states and fitted score heads. `strength` supplies the time-decayed strength probability candidate. `analytic` calculates distribution probabilities directly. `probability` supplies prior-season calibration and bounded research experiments. `context_model` learns bounded context effects from saved pregame features and later results, evaluating them on a later week before selection.

`decision` converts prices, computes no-vig probabilities, expected returns and constrained hypothetical exposure. `validation` joins unchanged predictions to real final outcomes and evaluates forecast quality and recorded-price returns. Hash checks detect changed prediction files.

Generated sources, reports, and fitted models belong to runtime storage, not source control. Missing external facts retain explicit coverage statuses. Empty reports are not health or lineup assertions. The first run cannot manufacture past Sunday predictions; validation begins with actual saved runs.

Earlier library modules remain for compatibility and offline experimentation. The current CLI, scheduled workflow, README, and desk expose the Sunday application flow.
