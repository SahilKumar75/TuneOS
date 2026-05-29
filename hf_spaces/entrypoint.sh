#!/bin/bash
# Start Redis in the background, then start the Celery worker in the foreground.
set -e

echo "[tuneos] Starting Redis..."
redis-server --daemonize yes --loglevel notice

# Wait for Redis to be ready
until redis-cli ping | grep -q PONG; do
  echo "[tuneos] Waiting for Redis..."
  sleep 1
done
echo "[tuneos] Redis is up."

echo "[tuneos] Starting Celery worker..."
exec celery -A workers.celery_app worker \
  --loglevel=info \
  --concurrency=1 \
  -Q celery
