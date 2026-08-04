# ─────────────────────────────────────────────────────────────────────────────
# Him Cab AI Travel Assistant — Makefile
# Enterprise-grade task runner for development, testing, and deployment.
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help dev test lint format type-check docker-up docker-down clean deploy seed

# Default target
help: ## Show this help message
	@echo "Him Cab AI Travel Assistant — Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Development ──────────────────────────────────────────────────────────────

dev: ## Start the dev server with hot reload
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

dev-docker: docker-up ## Start full stack via Docker Compose
	@echo "Stack is running at http://localhost:8000"

# ── Testing ──────────────────────────────────────────────────────────────────

test: ## Run all tests
	python -m pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	python -m pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests only
	python -m pytest tests/integration/ -v --tb=short

test-cov: ## Run tests with coverage report
	python -m pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

# ── Code Quality ─────────────────────────────────────────────────────────────

lint: ## Run linter (ruff)
	ruff check app/ tests/ workers/

format: ## Auto-format code (ruff)
	ruff format app/ tests/ workers/
	ruff check --fix app/ tests/ workers/

type-check: ## Run type checker (mypy)
	mypy app/ --ignore-missing-imports --no-error-summary

check: lint type-check ## Run all static checks (lint + type-check)

# ── Docker ───────────────────────────────────────────────────────────────────

docker-up: ## Start Docker Compose stack (API + Redis)
	docker-compose up -d --build

docker-down: ## Stop Docker Compose stack
	docker-compose down

docker-logs: ## Tail Docker Compose logs
	docker-compose logs -f api

docker-build: ## Build production Docker image
	docker build -t himcab-ai-assistant:latest .

# ── Database / Seeding ───────────────────────────────────────────────────────

seed: ## Seed Firestore with test data
	python scripts/seed_data.py

# ── Deployment ───────────────────────────────────────────────────────────────

deploy: ## Deploy to Cloud Run (production)
	powershell -ExecutionPolicy Bypass -File deploy.ps1

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
