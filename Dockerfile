# Agent OS Docker Image
# Multi-stage build for optimized production image

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml .
COPY src/ src/

# Install with all optional dependencies for production
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[redis,observability]"

# =============================================================================
# Stage 2: Production Image
# =============================================================================
FROM python:3.11-slim as production

# Labels
LABEL org.opencontainers.image.title="Agent OS"
LABEL org.opencontainers.image.description="Constitutional Operating System for Local AI"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/kase1111-hash/Agent-OS"

# Create non-root user
RUN groupadd --gid 1000 agentos && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home agentos

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=agentos:agentos . .

# Create data directories
RUN mkdir -p /app/data /app/logs && \
    chown -R agentos:agentos /app/data /app/logs

# Switch to non-root user
USER agentos

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AGENT_OS_WEB_HOST=0.0.0.0
ENV AGENT_OS_WEB_PORT=8080
ENV AGENT_OS_DATA_DIR=/app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "-m", "uvicorn", "src.web.app:get_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

# =============================================================================
# Stage 3: Development Image (optional)
# =============================================================================
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Install additional dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER agentos

# Override command for development
CMD ["python", "-m", "uvicorn", "src.web.app:get_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--reload"]
