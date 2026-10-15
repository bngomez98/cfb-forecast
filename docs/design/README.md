# October 15 supplied design and repository update

[Football system design](football-system.md) preserves the supplied mathematical
design. It is the target architecture; it is not a description of every feature
currently implemented. [MODEL_CONTRACT](../MODEL_CONTRACT.md) describes the
production score calculation.

## Integration of supplied files

| Input | Repository treatment |
| --- | --- |
| `cfb 10-12-26 model designnn.txt` | Preserved in `football-system.md`, with line endings normalized. |
| `code-security.datadog.yaml` | Installed at the repository root with all supplied rulesets. |
| `README.md` | Preserved as `../history/uploaded-change-history.md`; its contents are commit history, so the application README remains the project entry point. |
| `cfb-forecast-version-2.zip` | Compared against current source, including relocated modules, scripts, and deployment files. File hashes and comparison results are in `upload-manifest.json`. |

The ZIP predates the current package organization and subsequent model fixes.
Its Python modules now live under `ingest`, `modeling`, `evaluation`, `storage`,
`pipeline`, `monitoring`, and `odds`. Deployment files live under `deploy`, and
shell helpers under `scripts`. The web app uses `react-scripts`; restoring its
old Vite entry points would mix incompatible build setups. An `archive_only`
entry in the manifest means there is no mapped current file, not a missing
production dependency. The archive is retained by the supplied file's hash
rather than committing a second source tree.

## Implementation status against the supplied design

| Design requirement | Current implementation or remaining work |
| --- | --- |
| Time-varying football states | Gaussian team offense/defense states with Kalman updates exist in `modeling/ratings.py`. The full position-specific player and unit states in sections 4–7 remain to be implemented. |
| Chronological forecasting | Forecast cutoffs, prior-season head fitting, saved snapshots, and evaluation exist. Source revisions still limit historical point-in-time reconstruction. |
| Joint game distribution | The production score head uses a joint Gaussian approximation. Full posterior integration over player, unit, availability, and possession states is not implemented. |
| Play observation likelihoods | The design's Student-t EPA and Bernoulli success, explosive-play, and sack likelihoods are target equations; the production model does not jointly fit those likelihoods. |
| Possession-based scoring | Section 9's joint possession count and conditional drive outcomes remain to be implemented. The current expected score is calculated by the score head. |
| External ratings | Source collection exists; joint source-specific measurement-error estimation described in section 13 remains to be implemented. |
| Football-only and market-conditioned forecasts | Production now saves football-only and market-conditioned scores/probabilities separately, using the no-line fit for the football-only series. Validation reports both series and their common-game comparison; missing quotes leave the conditioned series unavailable. |
| Forecast evaluation | Saved forecasts are joined to final results. Performance improvements must be measured on held-out games; importing this design supplies no accuracy evidence. |

This update imports the supplied design, records its implementation boundaries,
and installs the supplied security configuration. It does not promote the
unimplemented equations into the production forecast.

## Datadog configuration

The root `code-security.datadog.yaml` selects the supplied Python analysis
rulesets. Datadog must run an analysis to produce findings. This file alone does
not enable a connected Datadog account, install an agent, or establish a clean
security scan.
