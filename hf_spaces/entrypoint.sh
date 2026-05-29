#!/bin/bash
set -e

# ── Redis: start bundled only if REDIS_URL points to localhost ───
if echo "${REDIS_URL:-redis://localhost:6379/0}" | grep -qE "localhost|127\.0\.0\.1"; then
  echo "[tuneos] Starting bundled Redis..."
  redis-server --daemonize yes --loglevel notice
  until redis-cli ping | grep -q PONG; do
    echo "[tuneos] Waiting for Redis..."
    sleep 1
  done
  echo "[tuneos] Redis is up."
else
  echo "[tuneos] Using external Redis: ${REDIS_URL}"
fi

# ── nginx ────────────────────────────────────────────────────────
echo "[tuneos] Starting nginx..."
nginx

# ── Celery worker (background) ───────────────────────────────────
echo "[tuneos] Starting Celery worker..."
celery -A workers.celery_app worker \
  --loglevel=info \
  --concurrency=1 \
  -Q celery &

# ── Reflex app (foreground — keeps container alive) ──────────────
echo "[tuneos] Starting Reflex app..."
exec reflex run --env prod
