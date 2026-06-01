#!/bin/bash
# TuneOS HF Spaces entrypoint — starts Redis, Celery worker, FastAPI, Reflex UI, and nginx.
set -e

echo "[tuneos] Starting Redis..."
redis-server --daemonize yes --loglevel notice

until redis-cli ping | grep -q PONG; do
    echo "[tuneos] Waiting for Redis..."
    sleep 1
done
echo "[tuneos] Redis is up."

echo "[tuneos] Starting Celery worker..."
celery -A workers.celery_app worker --loglevel=info --concurrency=1 -Q celery &

echo "[tuneos] Starting Reflex app (UI + API)..."
reflex run --env prod --backend-port 8000 &

echo "[tuneos] Waiting for Reflex to be ready..."
until curl -sf http://127.0.0.1:8000/api/health > /dev/null 2>&1; do
    sleep 2
done
echo "[tuneos] Reflex is up."

echo "[tuneos] Starting nginx on port 7860..."
nginx -g "daemon off;"
