# syntax=docker/dockerfile:1

# Forecast engine image: the cfb_forecast library (src/), the pipeline, external
# source collectors, logging, bundled data snapshots and the production model.
#
#   docker compose build
#   docker compose run --rm pipeline                     # weekly pipeline
#   docker compose run --rm cfb-forecast sources-list    # any engine command

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="cfb-forecast" \
      org.opencontainers.image.description="Reproducible college football forecasting engine and pipeline" \
      org.opencontainers.image.source="https://github.com/bngomez98/cfb-forecast"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    CFB_DATA_DIR=/app/data \
    CFB_ARTIFACTS_DIR=/app/artifacts \
    CFB_LOG_DIR=/app/artifacts/logs \
    CFB_LOG_LEVEL=INFO

WORKDIR /app

# NumPy, SciPy and scikit-learn wheels bundle their own BLAS/OpenMP, so no apt packages are needed.
# Dependencies first so code-only changes reuse the cached layer.
# requirements-tested.txt pins the versions the bundled results were verified with.
COPY requirements-tested.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --requirement requirements-tested.txt

# The library is installed in editable mode, so /app/src is the code that runs.
# Mount ./src over it (see docker-compose.yml) to develop without rebuilding.
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --editable . \
    && python -c "import cfb_forecast, cfb_forecast.pipeline, cfb_forecast.sources, cfb_forecast.logs"

# Pipeline inputs, fitted production model and blend, dashboard code, docs, tests and task runner.
COPY data ./data
COPY artifacts ./artifacts
COPY examples ./examples
COPY dashboard ./dashboard
COPY docs ./docs
COPY tests ./tests
COPY Makefile ./

RUN useradd --create-home --uid 10001 cfb \
    && mkdir -p /app/artifacts/logs /app/artifacts/pipeline /app/data/sources \
    && chown -R cfb:cfb /app
USER cfb

VOLUME ["/app/data", "/app/artifacts"]

ENTRYPOINT ["cfb-forecast"]
CMD ["--help"]
