---
title: TuneOS Worker
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# TuneOS — Training Worker Space

This Space runs the **Redis + Celery worker** backend for TuneOS.

The local TuneOS desktop app submits fine-tuning jobs here via Celery.
Set `REDIS_URL` and `CELERY_BROKER_URL` in your local `.env` to point at this Space.
