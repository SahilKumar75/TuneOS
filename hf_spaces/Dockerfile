# TuneOS — Single Space (Reflex UI + FastAPI + Celery worker)
# nginx listens on 7860 (required by HF Spaces) and proxies to Reflex on 3000.

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
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --without dev

# ── App source ───────────────────────────────────────────────────
COPY app/ ./app/
COPY workers/ ./workers/
COPY trainer/ ./trainer/
COPY storage/ ./storage/
COPY rxconfig.py ./

# ── nginx: proxy HF's required port 7860 → Reflex 3000 ──────────
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
