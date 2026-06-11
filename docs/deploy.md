# Deployment Guide

TuneOS runs in two configurations from one codebase: a local Docker Compose stack for
development, and two Hugging Face Spaces connected by an external Redis broker for
cloud hosting.

---

## Local Development

The quickest path is Docker Compose, which starts Redis and a Celery worker alongside
the application.

```bash
cp .env.example .env
# Add HF_TOKEN if you plan to use gated models (Llama 3, Mistral-instruct)
docker-compose up
```

The Reflex UI is available at `http://localhost:3000` and the FastAPI service at
`http://localhost:8000`.

### Without Docker

```bash
# terminal 1 — Redis broker
redis-server

# terminal 2 — Celery worker (runs the actual training)
celery -A workers.celery_app worker --loglevel=info

# terminal 3 — Reflex application
poetry run reflex run
```

---

## Hugging Face Spaces

The cloud topology uses two Spaces. The App Space runs nginx, Reflex, and FastAPI.
The Worker Space runs the Celery worker. Both connect to a shared Upstash Redis
instance.

```
Browser
  --> App Space (port 7860)
        nginx --> Reflex frontend (3000)
               --> FastAPI /api (8000)
                        |
                  Upstash Redis (external)
                        |
        Worker Space --> Celery worker --> trainer/
```

### Step 1 — Provision Upstash Redis

1. Create a free account at [upstash.com](https://upstash.com).
2. Create a Redis database (any region).
3. Copy the Redis URL — it looks like `rediss://default:PASSWORD@HOST:PORT`.

### Step 2 — Deploy the Worker Space

Create a Space with the Docker SDK, then push the `hf_spaces/` directory as the
Space repository root:

```bash
cd hf_spaces
git init
git remote add origin https://huggingface.co/spaces/<username>/TuneOS-Worker
git add .
git commit -m "init worker"
git push -u origin main
```

In the Space settings add these repository secrets:

| Secret | Value |
|---|---|
| `REDIS_URL` | Your Upstash Redis URL |
| `HF_TOKEN` | Your Hugging Face token |

### Step 3 — Deploy the App Space

Create a second Space with the Docker SDK. The App Space needs the full repository
so Reflex can build:

```bash
git remote add hf-app https://huggingface.co/spaces/<username>/TuneOS
git push hf-app main
```

Add the same secrets in the App Space settings.

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `REDIS_URL` | Redis broker URL | Yes |
| `HF_TOKEN` | Hugging Face token for gated models and Hub push | Recommended |
| `OPENROUTER_API_KEY` | OpenRouter key for intent flow and synthetic data | Optional |
| `MODAL_TOKEN_ID` | Modal.com token ID for cloud GPU training | Optional |
| `MODAL_TOKEN_SECRET` | Modal.com token secret | Optional |
| `EXPERIMENTS_DB_URL` | PostgreSQL DSN; defaults to local SQLite | Optional |
| `OUTPUT_DIR` | Where adapter weights are stored | Optional |

---

## Cloud GPU via Modal

If no local GPU is available, individual jobs can route to a free Modal.com T4.

1. Sign up at [modal.com](https://modal.com) and create an API token under
   Settings → Tokens.
2. Install the optional dependency: `poetry install --with modal`
3. Add credentials to `.env`:

```bash
MODAL_TOKEN_ID=ak-...
MODAL_TOKEN_SECRET=as-...
```

4. In Step 4 (Configure), choose **Modal** under Compute backend before submitting.

The local Celery worker stays the orchestrator. It serializes the dataset, runs the
training pipeline remotely on a T4, and streams the adapter and eval metrics back to
local disk. Loss progress is published to the shared Redis broker so the loss chart
updates live — identical to a local run.

Modal's free tier provides roughly $30/month of compute (~10-15 T4 hours). If
credentials are absent when Modal is selected, the job fails fast with a clear message
and nothing is queued.

---

## PostgreSQL for Multi-Worker Deployments

When multiple worker machines share one experiment store, switch from SQLite to
PostgreSQL:

```bash
EXPERIMENTS_DB_URL=postgresql://user:password@host:5432/tuneos
poetry install --with postgres  # installs psycopg2-binary
```

All upsert statements use portable `ON CONFLICT ... DO UPDATE` syntax compatible with
both SQLite 3.24+ and PostgreSQL.

---

## Desktop Application

The macOS desktop build packages the full stack into a native `.app` bundle.

```bash
poetry install --with desktop
poetry run python build_desktop.py
open dist/TuneOS.app
```

The PyQt6 shell starts Reflex, FastAPI, Redis, and the Celery worker automatically.
When Docker is available it uses Docker Compose; otherwise services start as local
subprocesses. Windows and Linux packaging is planned.
