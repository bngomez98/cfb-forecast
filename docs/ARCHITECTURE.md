# Application architecture

The `sunday` service is the central coordinator. It handles source collection, feature preparation, validation, model selection, inference, and persistence. Both the CLI and the GitHub Actions workflow invoke the same service. The dashboard only reads completed runs; it never performs inference or writes results.

## Module responsibilities

| Area | Responsibility |
| --- | --- |
| `data` | Schedules, per-game advanced observations, source manifests, schema validation |
| `sources` | Public game context; preserves original responses with retrieval receipts |
| `sunday.sync_results` | Verifies game/team identity before updating authoritative outcomes |
| `capture_prices` | Records actual displayed quotes with retrieval-time availability |
| `features`, `ratings`, `model` | Opponent-adjusted dynamic states and fitted score heads |
| `strength` | Time-decayed strength-probability candidate (comparison only) |
| `analytic` | Direct calculation of distribution probabilities |
| `probability` | Prior-season calibration and bounded research experiments |
| `context_model` | Bounded context effects from saved pregame features; evaluated on a later week before selection |
| `decision` | Price conversion, no-vig probabilities, expected returns, constrained exposure |
| `validation` | Joins unchanged predictions to final outcomes; evaluates quality and recorded-price returns |

Hash checks detect any alteration of prediction files after they were issued.

## Runtime vs. source control

Generated sources, reports, and fitted models belong in runtime storage, not source control. Missing external facts keep explicit coverage statuses. An empty report is never treated as an assertion that every player is healthy or that a full lineup is known.

The first run cannot manufacture past Sunday predictions; validation begins only with actually saved runs.

Earlier library modules remain available for compatibility and offline experimentation. The current CLI, scheduled workflow, README, and desk expose the Sunday application flow.
