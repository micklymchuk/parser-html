# Multi-stage build for HTML RAG Pipeline
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r htmlrag && useradd -r -g htmlrag htmlrag

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

# Development stage
FROM base as development

# Install development dependencies
RUN pip install -e ".[dev,docs]"

# Copy additional development files
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY .pre-commit-config.yaml ./

# Note: pre-commit hooks are set up in local development, not in containers

USER htmlrag
EXPOSE 8000

CMD ["python", "-m", "html_rag.cli.main", "--help"]

# Production stage
FROM base as production

# Copy only necessary files
COPY examples/ ./examples/
COPY README.md LICENSE ./

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R htmlrag:htmlrag /app

# Switch to non-root user
USER htmlrag

# Create volume mount points
VOLUME ["/app/data", "/app/logs"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import html_rag; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "html_rag.cli.main", "stats"]

# Jupyter notebook stage for development
FROM development as jupyter

# Install Jupyter
RUN pip install jupyter jupyterlab notebook

# Expose Jupyter port
EXPOSE 8888

# Copy example notebooks
COPY notebooks/ ./notebooks/

USER htmlrag

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Testing stage
FROM development as testing

# Copy test configuration
COPY pytest.ini ./
COPY .coveragerc ./

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=html_rag"]