# ─────────────────────────────────────────────────────────────────────────────
# Him Cab AI Travel Assistant — task runner.
#
# `make install` puts the package on the path with `pip install -e .`, which is why
# nothing here needs a PYTHONPATH= prefix.
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install dev test test-unit test-cov test-integration lint format \
        type-check check lock docker-up docker-down docker-logs docker-build \
        seed deploy clean

help: ## Show this help message
	@echo "Him Cab AI Travel Assistant — Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────

install: ## Install the package and dev tooling into the current environment
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e . --no-deps
	pre-commit install

lock: ## Regenerate requirements*.txt from pyproject.toml (the source of truth)
	python scripts/gen_requirements.py

# ── Development ──────────────────────────────────────────────────────────────

dev: ## Start the dev server with hot reload
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

dev-docker: docker-up ## Start full stack via Docker Compose
	@echo "Stack is running at http://localhost:8000 — API docs at /docs"

# ── Testing ──────────────────────────────────────────────────────────────────

test: ## Run the default suite (everything except integration)
	pytest -m "not integration"

test-unit: ## Run unit tests only
	pytest tests/unit

test-cov: ## Run tests with a coverage report
	pytest -m "not integration" --cov --cov-report=term-missing

test-integration: ## Run tests that hit real backends (needs credentials)
	pytest -m integration

# ── Code Quality ─────────────────────────────────────────────────────────────

lint: ## Run the linter
	ruff check app tests scripts

format: ## Auto-format and apply safe fixes
	ruff format app tests scripts
	ruff check --fix app tests scripts

type-check: ## Run the type checker
	mypy app

check: lint type-check ## Everything CI checks, minus the tests
	python scripts/gen_requirements.py --check

# ── Docker ───────────────────────────────────────────────────────────────────

docker-up: ## Start the Docker Compose stack
	docker-compose up -d --build

docker-down: ## Stop the Docker Compose stack
	docker-compose down

docker-logs: ## Tail Docker Compose logs
	docker-compose logs -f api

docker-build: ## Build the production image
	docker build -t himcab-ai-assistant:latest .

# ── Data / Deployment ────────────────────────────────────────────────────────

seed: ## Seed Firestore with test data
	python scripts/seed_data.py

deploy: ## Deploy to Cloud Run
	powershell -ExecutionPolicy Bypass -File deploy.ps1

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
