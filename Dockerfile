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

# Copy application source (endpoints only — frontend/ is served locally, not deployed)
COPY app/ ./app/

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

# Use uvicorn with uvloop for production performance
CMD ["sh", "-c", "uvicorn app.main:app --host $APP_HOST --port $APP_PORT --workers 4 --loop uvloop"]
