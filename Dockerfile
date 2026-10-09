# syntax=docker/dockerfile:1

# Forecast engine image: CLI, pipeline, bundled data snapshots and the production model.
#   docker build -t cfb-forecast:local .
#   docker run --rm cfb-forecast:local pipeline weekly --season 2026

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="cfb-forecast" \
      org.opencontainers.image.description="Reproducible college football forecasting engine" \
      org.opencontainers.image.source="https://github.com/bngomez98/cfb-forecast"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    CFB_DATA_DIR=/app/data \
    CFB_ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

# NumPy, SciPy and scikit-learn wheels bundle their own BLAS/OpenMP, so no apt packages are needed.
# Dependencies first so code-only changes reuse the cached layer.
# requirements-tested.txt pins the versions the bundled results were verified with.
COPY pyproject.toml README.md requirements-tested.txt ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --constraint requirements-tested.txt .

# Bundled inputs and the fitted production model (see .dockerignore for exclusions).
COPY data ./data
COPY artifacts ./artifacts
COPY examples ./examples

RUN useradd --create-home --uid 10001 cfb \
    && chown -R cfb:cfb /app
USER cfb

ENTRYPOINT ["cfb-forecast"]
CMD ["--help"]
