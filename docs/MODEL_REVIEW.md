# Model implementation review

The following model defects were found and fixed in the working source tree.
YAML files were left unchanged.

| Finding | Model consequence | Implemented change |
| --- | --- | --- |
| Player game facts were loaded into an unused variable | Recorded player data could not affect expected scores | Passing/rushing/receiving yards per opportunity, weighted EPA, and usage concentration now produce 18 home/away features in the residual model |
| Team rate snapshots advanced by kickoff order | A final that was not yet available could influence a historical query | Rates now replay by result availability; each query precedes results at exactly its cutoff |
| A missing prior season could retain an older season | Initial rates could represent the wrong roster year | Only the immediately preceding season and eligible current-season games contribute |
| Historical rate and market inputs used different cutoffs from the state | Adjustment fitting did not use a consistent information set | Historical state, rates and captured quotes now share the state cutoff; closing backfills are excluded from adjustment fitting |
| Stored database rows replaced more complete current source rows | Fresh values or unsupported database columns could disappear | Current disk values take precedence and stored facts fill absent fields |
| Advanced summaries and PBP measurements were averaged together | The same game could combine incompatible source measurements | Per-metric PBP values take precedence when observed, without summing duplicate measurements |
| Score changes lacked an inspectable decomposition | The effect of player rates was not visible in forecasts | Rows export base scores, recorded features, feature counts, and learned point contributions; the board exposes them |
| Adjustment settings were fixed module constants | Runs could not explicitly select and retain these fitting controls | Validated ridge, minimum sample and coverage settings flow from CLI/model JSON into fitting and saved artifacts |
| Insufficient adjustment samples could still apply score bounds | An unfitted adjustment could move the base forecast | Unfitted adjustments retain the base location and zero feature contributions |

Verification exercises the actual replay, out-of-fold adjustment fitting, model
serialization and forecast writer. Tests also demonstrate that player efficiency
can change scores and the resulting Gaussian win probability, that contribution
terms reconstruct expected scores, and that missing or future player information
cannot create a populated feature. These are implementation checks, not evidence
of improved accuracy on real held-out games.

The uploaded `cfb-forecast-version-2.zip` was inspected as reference material. Its
flat package layout differs from the working source tree. Its model includes
legacy logistic and request-wide reconciliation behavior, and its player facts
are likewise not used in the historical matrix. It was not extracted over the
working project. No instructions inside the archive were executed, and the
attached Datadog YAML was not modified.
