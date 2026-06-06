# syntax=docker/dockerfile:1
# TuneOS — Single container (Reflex UI + FastAPI + Celery worker)
# For HF Spaces: nginx on 7860 proxies to Reflex on 3000.
# For docker-compose: override CMD per service.

FROM python:3.11-slim

# ── System deps ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ curl git unzip \
        redis-server \
        nginx \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Install Poetry ───────────────────────────────────────────────
RUN pip install --no-cache-dir poetry

WORKDIR /app

# ── Python deps ─────────────────────────────────────────────────
# BuildKit cache mounts keep the wheel download cache across builds so a small
# dependency bump doesn't re-download the multi-GB torch wheel. In CI these are
# persisted run-to-run by buildkit-cache-dance (see .github/workflows/ci.yml).
COPY pyproject.toml poetry.lock* ./
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# ── App source ───────────────────────────────────────────────────
COPY . .

# ── nginx config ─────────────────────────────────────────────────
COPY hf_spaces/nginx.conf /etc/nginx/sites-available/default

# ── Storage dirs ─────────────────────────────────────────────────
RUN mkdir -p /app/models_cache /app/outputs /app/storage/datasets /app/uploaded_files

# ── Environment defaults (override via Space secrets) ────────────
ENV REDIS_URL=redis://localhost:6379/0 \
    HF_TOKEN="" \
    HF_HOME=/app/models_cache \
    PYTHONUNBUFFERED=1

EXPOSE 7860

COPY hf_spaces/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
