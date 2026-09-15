# ─────────────────────────────────────────────────────────────────────────────
# Him Cab AI Travel Assistant — task runner for Windows.
#
# The Makefile beside this file is the same set of tasks for anyone on a Unix
# shell. `make` is not installed on Windows and is not worth requiring, so this
# mirrors it target for target. Change one, change the other — `.\make.ps1 check`
# fails if they drift.
#
#   .\make.ps1              list the tasks
#   .\make.ps1 dev          start the dev server
#   .\make.ps1 test         run the suite
#   .\make.ps1 check        everything CI checks, minus the tests
#
# One difference from the Makefile, and it is deliberate: every task runs through
# the project's own .venv when one exists, so nothing here depends on having
# activated it first. Forgetting to activate is the single most common reason a
# command "works for me" and not for somebody else.
# ─────────────────────────────────────────────────────────────────────────────

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Task = 'help',

    # Anything after the task name is passed straight through, so
    # `.\make.ps1 test -k rental -x` reaches pytest intact.
    #
    # KNOWN LIMIT: a single argument containing spaces does not survive. Windows
    # PowerShell 5.1 re-splits it on the way to a native executable, so
    # `-k "a or b"` arrives as three arguments and pytest matches only the first.
    # For an expression with spaces, call pytest directly:
    #     .venv\Scripts\python.exe -m pytest -k "a or b"
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [object[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# ── Which Python ─────────────────────────────────────────────────────────────
$VenvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = 'python'
    Write-Host "No .venv found - falling back to the python on PATH." -ForegroundColor Yellow
    Write-Host "Run  .\make.ps1 install  to set the project up properly.`n" -ForegroundColor Yellow
}

# Strip a leading `--` so both `.\make.ps1 test -k foo` and
# `.\make.ps1 test -- -k foo` behave the same.
$Extra = @()
if ($Rest) { $Extra = @($Rest | Where-Object { $_ -ne '--' }) }

function Invoke-Step {
    <#
        .SYNOPSIS
        Run a command, echo it first, and stop the task if it fails.

        Native executables do not set $ErrorActionPreference, so a failing tool
        would otherwise be reported as success and the next step would run on top
        of it - which is how a red check gets mistaken for a green one.
    #>
    param(
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    Write-Host "> $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nFailed (exit $LASTEXITCODE): $Exe $($Arguments -join ' ')" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

function Invoke-Py {
    <#
        .SYNOPSIS
        Run the project's Python with the given argument array.

        Takes one array parameter rather than ValueFromRemainingArguments: that
        binder flattens a [string[]] into a single space-joined string, so
        `-m ruff check app` reached Python as the module named " ruff check app".
    #>
    param([Parameter(Mandatory, Position = 0)][string[]]$Arguments)
    Invoke-Step -Exe $Python -Arguments $Arguments
}

function Show-Help {
    Write-Host ''
    Write-Host 'Him Cab AI Travel Assistant - tasks' -ForegroundColor Cyan
    Write-Host ''
    $tasks = [ordered]@{
        'install'          = 'Install the package and dev tooling into .venv'
        'lock'             = 'Regenerate requirements*.txt from pyproject.toml'
        'dev'              = 'Start the dev server with hot reload (port 8000)'
        'dev-docker'       = 'Start the full stack via Docker Compose'
        'test'             = 'Run the default suite (everything except integration)'
        'test-unit'        = 'Run unit tests only'
        'test-cov'         = 'Run tests with a coverage report'
        'test-integration' = 'Run tests that hit real backends (needs credentials)'
        'lint'             = 'Run the linter'
        'format'           = 'Auto-format and apply safe fixes'
        'type-check'       = 'Run the type checker'
        'check'            = 'Everything CI checks, minus the tests'
        'docker-up'        = 'Start the Docker Compose stack'
        'docker-down'      = 'Stop the Docker Compose stack'
        'docker-logs'      = 'Tail Docker Compose logs'
        'docker-build'     = 'Build the production image'
        'seed'             = 'Seed Firestore with test data'
        'deploy'           = 'Deploy to Cloud Run'
        'clean'            = 'Remove caches and build artifacts'
    }
    foreach ($name in $tasks.Keys) {
        Write-Host ('  {0,-18}' -f $name) -ForegroundColor Green -NoNewline
        Write-Host $tasks[$name]
    }
    Write-Host ''
    Write-Host 'Arguments after the task are passed through, e.g.' -ForegroundColor DarkGray
    Write-Host '  .\make.ps1 test -k rental -x' -ForegroundColor DarkGray
    Write-Host ''
    Write-Host 'Once the server is up, the browser test clients are at:' -ForegroundColor DarkGray
    Write-Host '  http://localhost:8000/outstation   http://localhost:8000/local   http://localhost:8000/rental' -ForegroundColor DarkGray
    Write-Host ''
}

switch ($Task.ToLowerInvariant()) {

    # ── Setup ────────────────────────────────────────────────────────────────
    'install' {
        if (-not (Test-Path $VenvPython)) {
            Write-Host 'Creating .venv...' -ForegroundColor Cyan
            Invoke-Step -Exe 'python' -Arguments @('-m', 'venv', '.venv')
            $Python = $VenvPython
        }
        Invoke-Py @('-m', 'pip', 'install', '--upgrade', 'pip')
        Invoke-Py @('-m', 'pip', 'install', '-r', 'requirements.txt', '-r', 'requirements-dev.txt')
        Invoke-Py @('-m', 'pip', 'install', '-e', '.', '--no-deps')
        Invoke-Py @('-m', 'pre_commit', 'install')
        Write-Host "`nReady. Start the server with  .\make.ps1 dev" -ForegroundColor Green
    }

    'lock' { Invoke-Py @('scripts/gen_requirements.py') }

    # ── Development ──────────────────────────────────────────────────────────
    'dev' {
        Write-Host ''
        Write-Host 'Test clients once this is up:' -ForegroundColor Cyan
        Write-Host '  http://localhost:8000/outstation  - visitor, day-by-day itinerary' -ForegroundColor Green
        Write-Host '  http://localhost:8000/local       - resident, cab and fare' -ForegroundColor Green
        Write-Host '  http://localhost:8000/rental      - car by the hour' -ForegroundColor Green
        Write-Host '  http://localhost:8000/docs        - API reference' -ForegroundColor DarkGray
        Write-Host ''
        Write-Host 'Paste a real Him Cab rider JWT into a page to get live fares;' -ForegroundColor DarkGray
        Write-Host 'the placeholder dev token gets you the conversation but no prices.' -ForegroundColor DarkGray
        Write-Host ''
        Invoke-Py (@('-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000',
                     '--reload', '--log-level', 'info') + $Extra)
    }

    'dev-docker' {
        Invoke-Step -Exe 'docker-compose' -Arguments @('up', '-d', '--build')
        Write-Host 'Stack is running at http://localhost:8000' -ForegroundColor Green
    }

    # ── Testing ──────────────────────────────────────────────────────────────
    'test'             { Invoke-Py (@('-m', 'pytest', '-m', 'not integration') + $Extra) }
    'test-unit'        { Invoke-Py (@('-m', 'pytest', 'tests/unit') + $Extra) }
    'test-cov'        {
        Invoke-Py (@('-m', 'pytest', '-m', 'not integration',
                     '--cov', '--cov-report=term-missing') + $Extra)
    }
    'test-integration' { Invoke-Py (@('-m', 'pytest', '-m', 'integration') + $Extra) }

    # ── Code quality ─────────────────────────────────────────────────────────
    'lint' { Invoke-Py (@('-m', 'ruff', 'check', 'app', 'tests', 'scripts') + $Extra) }

    'format' {
        Invoke-Py @('-m', 'ruff', 'format', 'app', 'tests', 'scripts')
        Invoke-Py @('-m', 'ruff', 'check', '--fix', 'app', 'tests', 'scripts')
    }

    'type-check' { Invoke-Py (@('-m', 'mypy', 'app') + $Extra) }

    'check' {
        # The same order CI runs them, so a failure here is the failure there.
        Invoke-Py @('-m', 'ruff', 'check', 'app', 'tests', 'scripts')
        Invoke-Py @('scripts/gen_requirements.py', '--check')
        Invoke-Py @('-m', 'mypy', 'app/domain', 'app/schemas')
        Write-Host "`nAll checks passed." -ForegroundColor Green
    }

    # ── Docker ───────────────────────────────────────────────────────────────
    'docker-up'    { Invoke-Step -Exe 'docker-compose' -Arguments @('up', '-d', '--build') }
    'docker-down'  { Invoke-Step -Exe 'docker-compose' -Arguments @('down') }
    'docker-logs'  { Invoke-Step -Exe 'docker-compose' -Arguments @('logs', '-f', 'api') }
    'docker-build' { Invoke-Step -Exe 'docker' -Arguments @('build', '-t', 'himcab-ai-assistant:latest', '.') }

    # ── Data and deployment ──────────────────────────────────────────────────
    'seed'   { Invoke-Py @('scripts/seed_data.py') }
    'deploy' { Invoke-Step -Exe 'powershell' -Arguments @('-ExecutionPolicy', 'Bypass', '-File', 'deploy.ps1') }

    # ── Cleanup ──────────────────────────────────────────────────────────────
    'clean' {
        $patterns = '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '*.egg-info'
        foreach ($pattern in $patterns) {
            Get-ChildItem -Path $PSScriptRoot -Filter $pattern -Recurse -Force -Directory `
                -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -notmatch '\\\.venv\\' } |
                ForEach-Object {
                    Write-Host "  removing $($_.FullName)" -ForegroundColor DarkGray
                    Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
                }
        }
        Get-ChildItem -Path $PSScriptRoot -Filter '*.pyc' -Recurse -Force -File `
            -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notmatch '\\\.venv\\' } |
            Remove-Item -Force -ErrorAction SilentlyContinue
        Write-Host 'Clean.' -ForegroundColor Green
    }

    { $_ -in 'help', '-h', '--help', '/?' } { Show-Help }

    default {
        Write-Host "Unknown task '$Task'." -ForegroundColor Red
        Show-Help
        exit 1
    }
}
