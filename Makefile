# Common tasks. Run `make help` for the list.
SEASON ?= 2026
DRAWS  ?= 50000
PYTHON ?= python3
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1

.DEFAULT_GOAL := help
.PHONY: help install lint test check reproduce weekly results backtest-season dashboard \
        docker-build docker-weekly up down clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package with dev tools (editable)
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run ruff
	ruff check src tests dashboard

test: ## Run the test suite
	$(PYTHON) -m pytest -q

check: lint test ## Lint and test

reproduce: ## Reproduce the 2025 held-out evaluation from bundled inputs
	cfb-forecast backtest --data data --out artifacts/reproduced --archive data/archive/matched_predictions.csv

weekly: ## Full weekly pipeline: refresh data, live scores, forecast, grade season
	cfb-pipeline weekly --season $(SEASON) --draws $(DRAWS) --refresh-data

results: ## Refresh live ESPN scores only
	cfb-pipeline results --season $(SEASON)

backtest-season: ## Re-forecast and grade every completed week of SEASON
	cfb-pipeline backtest --season $(SEASON) --draws $(DRAWS)

dashboard: ## Run the Streamlit dashboard locally
	streamlit run dashboard/app.py

docker-build: ## Build the engine and dashboard images
	docker compose build

docker-weekly: ## Run the weekly pipeline in Docker as the current user
	CFB_UID=$$(id -u) CFB_GID=$$(id -g) docker compose run --rm pipeline

up: ## Start the dashboard container
	docker compose up -d dashboard

down: ## Stop all containers
	docker compose --profile cli --profile public down

clean: ## Remove caches and build outputs
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
