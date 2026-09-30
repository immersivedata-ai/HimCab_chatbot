# ── Builder stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# The browser test clients, served by FastAPI itself (see the frontend block in
# app/main.py) — no web server of their own. They MUST land at /app/frontend, one
# level up from the package, because that mount resolves the path relative to app/.
# Every page calls the same rider-authenticated endpoints any client would; the token
# is pasted in by whoever is testing. Served only when APP_ENV is not production.
COPY frontend/ ./frontend/

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Cloud Run expects PORT env var (default 8080)
ENV PORT=8080
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8080

EXPOSE 8080

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/v1/health')" || exit 1

# ONE worker, deliberately. This ran with four, and on Cloud Run that is four separate
# processes each importing the whole application — roughly seven seconds of Python
# apiece — while contending for the single vCPU an instance gets by default. It was the
# largest part of a ~40s cold start, during which the browser cannot even fetch the
# page, because this container serves the page too.
#
# Nothing is lost by dropping to one. The app is async end to end (FastAPI, httpx, the
# async Firestore client), so concurrency comes from the event loop rather than from
# processes, and Cloud Run scales by adding INSTANCES, not by the workers inside one.
# Measured locally, on a machine with cores to spare: 9.58s to first served request
# with four workers, 5.52s with one. On a single vCPU the gap is wider still.
CMD ["sh", "-c", "uvicorn app.main:app --host $APP_HOST --port $APP_PORT --workers 1 --loop uvloop"]
