# TuneOS Deployment Guide

TuneOS runs as two Hugging Face Spaces connected by an external Redis broker (Upstash).

## Step 1 — Get a free Upstash Redis

1. Go to https://upstash.com and create a free account.
2. Create a Redis database (select any region).
3. Copy the **Redis URL** — it looks like `rediss://default:PASSWORD@HOST:PORT`.

## Step 2 — Deploy the Worker Space

1. Create a new HF Space: **SahilKumar75/TuneOS-Worker**
   - SDK: Docker
2. Push the `hf_spaces/` directory as the Space repo root:
   ```bash
   cd hf_spaces
   git init && git remote add origin https://huggingface.co/spaces/SahilKumar75/TuneOS-Worker
   git add . && git commit -m "init worker" && git push -u origin main
   ```
3. In the Space → Settings → Repository secrets, add:
   - `REDIS_URL` = your Upstash Redis URL
   - `HF_TOKEN` = your Hugging Face token

## Step 3 — Deploy the App Space

1. Create a new HF Space: **SahilKumar75/TuneOS**
   - SDK: Docker
2. The app Space needs the full repo (so Reflex can build). Push from the project root:
   ```bash
   git remote add hf-app https://huggingface.co/spaces/SahilKumar75/TuneOS
   git push hf-app main
   ```
3. In the Space → Settings → Repository secrets, add:
   - `REDIS_URL` = same Upstash Redis URL
   - `HF_TOKEN` = your Hugging Face token

## Architecture

```
Browser → HF Space (TuneOS App, port 7860)
              nginx → Reflex frontend (3000)
                    → FastAPI /api (8000)
                          ↓ Celery task
              Upstash Redis (external)
                          ↓
          HF Space (TuneOS Worker)
              Celery worker → trainer/
```

## Local dev

```bash
cp .env.example .env
# Set REDIS_URL to your Upstash URL (or leave as localhost to use docker-compose Redis)
docker-compose up
```

## Cloud GPU via Modal (optional)

If you have no local GPU, route training to a free Modal T4 instead of running
it on the worker's local device.

1. Sign up at [modal.com](https://modal.com) and create an API token at
   **Settings → Tokens**.
2. Install the optional dependency: `poetry install --with modal`.
3. Add the credentials to `.env`:
   ```bash
   MODAL_TOKEN_ID=ak-...
   MODAL_TOKEN_SECRET=as-...
   ```
4. In Step 4 (Configure), choose **Modal** under *Compute backend* before
   starting training.

The local worker stays the orchestrator: it serializes the dataset, runs the
identical `trainer.finetune` pipeline on a Modal T4, and writes the resulting
adapter + eval metrics back to `OUTPUT_DIR`. Job status and metrics persistence
are identical to a local run. Modal's free tier provides ~$30/month of compute
(roughly 10–15 T4 hours). Without the tokens set, selecting Modal fails fast
with a clear message and no job is queued.
