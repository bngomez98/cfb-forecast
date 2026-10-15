# syntax=docker/dockerfile:1

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 r-base-core \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install '.[api]'

COPY . .

RUN useradd --create-home --uid 10001 cfb \
    && chown -R cfb:cfb /app
USER cfb

ENTRYPOINT ["cfb-forecast"]
CMD ["--help"]
