#!/usr/bin/env bash
set -euo pipefail
exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/artifacts:/app/artifacts" \
  "${CFB_IMAGE:-cfb-forecast:local}" sunday-run \
  --data /app/data --artifacts /app/artifacts \
  --season "${CFB_FORECAST_SEASON:-2026}"
