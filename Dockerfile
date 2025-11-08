# Multi-stage Dockerfile for GitForAI
# Optimized for production use with minimal image size

# ============================================================================
# Stage 1: Builder - Install dependencies and build wheels
# ============================================================================
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy dependency files and source code
COPY pyproject.toml ./
COPY src/ ./src/

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies and package with full support
# Includes: OpenAI API + sentence-transformers + torch + ChromaDB
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir ".[llm,local-embeddings,vectordb]"

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.10-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash gitforai

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Create directories for data
RUN mkdir -p /repos /output /cache /data /vectordb && \
    chown -R gitforai:gitforai /repos /output /cache /data /vectordb /app

# Switch to non-root user
USER gitforai

# Create state directory for incremental updates
RUN mkdir -p /home/gitforai/.gitforai

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import gitforai; print(gitforai.__version__)" || exit 1

# Set default command
ENTRYPOINT ["gitforai"]
CMD ["--help"]

# ============================================================================
# Stage 3: Development - Full development environment
# ============================================================================
FROM runtime AS development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER gitforai

# Install development Python packages with full support
RUN pip install --no-cache-dir -e ".[dev,llm,local-embeddings,vectordb]"

# Override entrypoint for development
ENTRYPOINT []
CMD ["/bin/bash"]

# ============================================================================
# Metadata
# ============================================================================
LABEL maintainer="GitForAI Team"
LABEL description="Git History Mining for AI Agents"
LABEL version="0.1.0"

# Document exposed ports (for future API service)
EXPOSE 8000

# Document volumes
VOLUME ["/repos", "/output", "/cache", "/data", "/vectordb", "/home/gitforai/.gitforai"]
