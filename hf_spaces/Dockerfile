# TuneOS — Hugging Face Spaces Docker image
# Runs Redis + the Celery training worker inside a single container.
#
# Deploy as a "Docker" Space on Hugging Face:
#   https://huggingface.co/spaces/<your-org>/tuneos-worker
#
# The local PyQt6 desktop app should point its REDIS_URL and CELERY_BROKER_URL
# at the public URL of this Space (via an SSH tunnel or HF Spaces persistent URL).

FROM python:3.11-slim

# ── System deps ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        redis-server \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────
WORKDIR /app

# ── Python deps ──────────────────────────────────────────────────
# Copy only the dependency manifest first so Docker can cache this layer.
COPY pyproject.toml ./

RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction

# ── App source ───────────────────────────────────────────────────
COPY workers/ ./workers/
COPY trainer/ ./trainer/
COPY storage/ ./storage/
COPY .env.example .env

# ── Persistent storage dirs ──────────────────────────────────────
RUN mkdir -p /app/models_cache /app/outputs /app/storage/datasets

# ── Environment defaults (override via Space secrets) ────────────
ENV REDIS_URL=redis://localhost:6379/0 \
    HF_HOME=/app/models_cache \
    CELERY_BROKER_URL=redis://localhost:6379/0 \
    CELERY_RESULT_BACKEND=redis://localhost:6379/0 \
    PYTHONUNBUFFERED=1

# ── HF Spaces: port 7860 must be exposed ────────────────────────
EXPOSE 7860 6379

# ── Entrypoint: start Redis then Celery worker ───────────────────
# We use a small shell script so both processes run and their logs
# are visible in the Space logs panel.
COPY hf_spaces/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
