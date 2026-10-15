#!/usr/bin/env bash
set -euo pipefail
exec python -m cfb_forecast sunday-run \
  --data "${CFB_DATA_DIR:-data}" \
  --artifacts "${CFB_ARTIFACTS_DIR:-artifacts}" \
  --season "${CFB_FORECAST_SEASON:-2026}"
