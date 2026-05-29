# TuneOS — Hugging Face Spaces Docker image
# Runs Redis + the Celery training worker inside a single container.

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

# ── Python deps (inline — no pyproject.toml needed) ─────────────
# Install torch (CPU-only) first with its own index, then everything else
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers \
    peft \
    bitsandbytes \
    datasets \
    accelerate \
    trl \
    "celery[redis]" \
    redis \
    huggingface-hub \
    python-dotenv \
    evaluate \
    nltk \
    pandas

# ── App source ───────────────────────────────────────────────────
COPY workers/ ./workers/
COPY trainer/ ./trainer/
COPY storage/ ./storage/

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
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
