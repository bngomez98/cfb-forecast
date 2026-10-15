# syntax=docker/dockerfile:1

# Forecast engine image: the cfb_forecast library (src/), the Sunday service, the daily
# pipeline and scheduler, the API, the data watchdog, source collectors, logging, the SQL
# schema and web board output folder, bundled seed data and the committed backtest evidence.
#
#   docker compose build
#   docker compose run --rm cfb-forecast sunday-run --season 2026
#   docker compose run --rm cfb-forecast pipeline forecast
#
# R is installed for the independent learning validation (validate_learning.R). Build with
# --build-arg INSTALL_R=0 to skip it (about 300 MB); learning then reports the R check as
# "unavailable" unless CFB_REQUIRE_R=1, in which case it refuses to promote a model.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS runtime

ARG INSTALL_R=1

LABEL org.opencontainers.image.title="cfb-forecast" \
      org.opencontainers.image.description="Real-data FBS forecasts: Sunday slates, daily updates, betting diagnostics and weekly validation" \
      org.opencontainers.image.source="https://github.com/bngomez98/cfb-forecast"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    CFB_PROJECT_ROOT=/app \
    CFB_DATA_DIR=/app/data \
    CFB_ARTIFACTS_DIR=/app/artifacts \
    CFB_DATABASE=/app/artifacts/cfb_forecast.sqlite \
    CFB_LOG_DIR=/app/artifacts/logs \
    CFB_LOG_LEVEL=INFO

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && if [ "$INSTALL_R" = "1" ]; then apt-get install -y --no-install-recommends r-base-core; fi \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so code-only changes reuse the cached layer.
# requirements-tested.txt pins the versions the bundled results were verified with.
COPY requirements-tested.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --requirement requirements-tested.txt "httpx>=0.27" "fastapi>=0.115" "uvicorn>=0.30"

# The library is installed in editable mode, so /app/src is the code that runs.
# Mount ./src over it (see docker-compose.yml) to try edits without rebuilding.
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --editable . \
    && python -c "import cfb_forecast, cfb_forecast.cli, cfb_forecast.sunday, cfb_forecast.pipeline, cfb_forecast.scheduler, cfb_forecast.api, cfb_forecast.webdata"

# Everything else the services read: SQL schema, web board output, seed data, the model
# evidence, dashboard code, docs and tests. Runtime state is excluded by .dockerignore.
COPY . .

RUN useradd --create-home --uid 10001 cfb \
    && mkdir -p /app/artifacts/logs /app/data/sources \
    && chown -R cfb:cfb /app
USER cfb

VOLUME ["/app/data", "/app/artifacts"]

ENTRYPOINT ["cfb-forecast"]
CMD ["--help"]
