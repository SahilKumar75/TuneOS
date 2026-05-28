FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files first for better caching
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies (no dev, no desktop extras)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy application code
COPY . .

# Default command (overridden by docker-compose per service)
CMD ["reflex", "run", "--env", "prod"]
