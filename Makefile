# Common tasks. Run `make help` for the list.
SEASON ?= 2026
DRAWS  ?= 50000
PYTHON ?= python3
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1

.DEFAULT_GOAL := help
.PHONY: help install lint test check reproduce sunday forecast results sources players database app scheduler scheduler-status \
        backtest-season dashboard api docker-build docker-forecast docker-sunday up down logs clean

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

sunday: ## Sunday service now: sources, validation, learning, the weekend slate, publication
	cfb-forecast sunday-run --season $(SEASON)

forecast: ## Ongoing daily pipeline: refresh data/results/players, forecast every upcoming game
	cfb-forecast pipeline forecast --season $(SEASON) --draws $(DRAWS) --refresh-data

results: ## Final scores of completed games from ESPN and result sites
	cfb-forecast pipeline results --season $(SEASON)

sources: ## Collect every external source once
	cfb-forecast sources-collect --season $(SEASON)

players: ## Rebuild the current season's per-player aggregates from play-by-play
	cfb-forecast player-aggregates --seasons $(SEASON) --refresh

database: ## Store outcomes in SQLite and validate every logged forecast against them
	$(PYTHON) -c "from cfb_forecast import dbflow, paths; print(dbflow.run(paths.database_path(), paths.data_root(), $(SEASON)))"

app: ## Run the whole application (scheduler + dashboard) in the foreground
	cfb-forecast app

scheduler: ## Run the automation loop in the foreground
	cfb-forecast scheduler

scheduler-status: ## Show each automated job's last run and whether it is due
	cfb-forecast scheduler --status

backtest-season: ## Point-in-time backtest of every completed week of SEASON
	cfb-forecast pipeline backtest --season $(SEASON) --draws $(DRAWS)

dashboard: ## Run the Streamlit dashboard locally
	streamlit run dashboard/app.py

api: ## Run the JSON API locally on :8000
	$(PYTHON) -m uvicorn cfb_forecast.api:app --host 127.0.0.1 --port 8000

docker-build: ## Build the engine and dashboard images
	docker compose build

docker-forecast: ## Run the daily forecast pipeline in Docker as the current user
	CFB_UID=$$(id -u) CFB_GID=$$(id -g) docker compose run --rm cfb-forecast pipeline forecast --season $(SEASON)

docker-sunday: ## Run the Sunday service in Docker as the current user
	CFB_UID=$$(id -u) CFB_GID=$$(id -g) docker compose run --rm cfb-forecast sunday-run --season $(SEASON)

up: ## Start the scheduler (automation) and dashboard containers
	CFB_UID=$$(id -u) CFB_GID=$$(id -g) docker compose up -d scheduler cfb-api cfb-dashboard

logs: ## Follow the scheduler container's logs
	docker compose logs -f scheduler

down: ## Stop all containers
	docker compose --profile cli --profile public down

clean: ## Remove caches and build outputs
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
