# ─────────────────────────────────────────────────────────────────────────────
# Him Cab AI Travel Assistant — Dev Setup Script (Windows PowerShell)
# Run this once after cloning the repo to set up your local environment.
# Usage: .\scripts\dev_setup.ps1
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

Write-Host "`n═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Him Cab AI — Development Setup" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════`n" -ForegroundColor Cyan

# 1. Check Python version
Write-Host "[1/6] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "3\.\d+") {
    Write-Host "  ✓ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python 3.11+ required" -ForegroundColor Red
    exit 1
}

# 2. Create virtual environment
Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "  ✓ Created .venv" -ForegroundColor Green
} else {
    Write-Host "  ✓ .venv already exists" -ForegroundColor Green
}

# 3. Activate venv and install dependencies
Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
Write-Host "  ✓ Dependencies installed" -ForegroundColor Green

# 4. Install dev tools
Write-Host "[4/6] Installing dev tools (ruff, mypy, pre-commit)..." -ForegroundColor Yellow
pip install ruff mypy pre-commit --quiet
Write-Host "  ✓ Dev tools installed" -ForegroundColor Green

# 5. Set up pre-commit hooks
Write-Host "[5/6] Setting up pre-commit hooks..." -ForegroundColor Yellow
if (Test-Path ".pre-commit-config.yaml") {
    pre-commit install --quiet
    Write-Host "  ✓ Pre-commit hooks installed" -ForegroundColor Green
} else {
    Write-Host "  ⚠ No .pre-commit-config.yaml found — skipping" -ForegroundColor Yellow
}

# 6. Copy .env.example if .env doesn't exist
Write-Host "[6/6] Checking environment config..." -ForegroundColor Yellow
if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  ✓ Created .env from .env.example — please fill in your API keys" -ForegroundColor Green
} else {
    Write-Host "  ✓ .env already exists" -ForegroundColor Green
}

Write-Host "`n═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor White
Write-Host "  1. Edit .env with your API keys" -ForegroundColor White
Write-Host "  2. Start Redis: docker-compose up redis -d" -ForegroundColor White
Write-Host "  3. Run the dev server: make dev" -ForegroundColor White
Write-Host "  4. Open http://localhost:8000/docs`n" -ForegroundColor White
