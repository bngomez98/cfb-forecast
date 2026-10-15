# syntax=docker/dockerfile:1

# Streamlit dashboard. Reads forecasts from a mounted artifacts directory.
#   docker build -f dashboard/Dockerfile -t cfb-forecast-dashboard:local .

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim

LABEL org.opencontainers.image.title="cfb-forecast-dashboard" \
      org.opencontainers.image.description="Streamlit dashboard for weekly CFB forecasts and backtests" \
      org.opencontainers.image.source="https://github.com/bngomez98/cfb-forecast"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CFB_DATA_DIR=/app/data \
    CFB_ARTIFACTS_DIR=/app/artifacts

WORKDIR /app
COPY dashboard/requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -r requirements.txt
COPY dashboard ./dashboard
COPY src ./src
COPY .streamlit ./.streamlit
ENV PYTHONPATH=/app/src

RUN useradd --create-home --uid 10001 cfb && chown -R cfb:cfb /app
USER cfb

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
