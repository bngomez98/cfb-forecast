#!/usr/bin/env bash
set -euo pipefail

SEASON="${CFB_FORECAST_SEASON:-2026}"
DRAWS="${CFB_FORECAST_DRAWS:-50000}"
IMAGE="${CFB_IMAGE:-cfb-forecast:local}"

mkdir -p data artifacts

if [[ ! -f "data/schedule_${SEASON}.csv" || ! -f "data/adv_team_${SEASON}.parquet" ]]; then
  docker run --rm \
    -v "$PWD/data:/app/data" \
    "$IMAGE" fetch \
    --data /app/data \
    --seasons "$SEASON"
fi

docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/artifacts:/app/artifacts" \
  "$IMAGE" preflight \
  --data /app/data \
  --artifacts /app/artifacts \
  --season "$SEASON"

docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/artifacts:/app/artifacts" \
  "$IMAGE" results-refresh \
  --data /app/data \
  --out /app/artifacts/results \
  --season "$SEASON"

docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/artifacts:/app/artifacts" \
  "$IMAGE" weekly-forecast \
  --data /app/data \
  --model /app/artifacts/backtest/model \
  --out /app/artifacts \
  --season "$SEASON" \
  --draws "$DRAWS"

docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/artifacts:/app/artifacts" \
  "$IMAGE" season-backtest \
  --data /app/data \
  --model /app/artifacts/backtest/model \
  --out /app/artifacts/backtest \
  --season "$SEASON" \
  --draws "$DRAWS" \
  --lead-hours 48
