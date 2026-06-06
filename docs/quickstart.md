# Quickstart

Get from a clone to a running fine-tune in a few minutes.

## 1. Clone & configure

```bash
git clone https://github.com/SahilKumar75/TuneOS
cd TuneOS
cp .env.example .env
# Add HF_TOKEN — required for gated models (Llama 3, Mistral-instruct, …)
```

## 2. Start everything (Docker)

```bash
docker-compose up -d
```

This brings up the Reflex UI, the FastAPI service, Redis, and a Celery worker.
Open <http://localhost:3000>.

### 2b. Run without Docker

No Docker? Start the pieces yourself:

```bash
# terminal 1 — Redis (or point REDIS_URL at a remote broker in .env)
redis-server

# terminal 2 — Celery worker (this is what actually runs training)
celery -A workers.celery_app worker --loglevel=info

# terminal 3 — the app
reflex run
```

The desktop launcher does this for you: when Docker is unavailable it falls back
to starting Redis and the Celery worker as local subprocesses.

## 3. Start training in ~5 minutes

1. Paste a model id — `EleutherAI/pythia-410m` is tiny and great for a first run.
2. Walk the 7-step wizard: **Model → Intent → Data → Configure → Train → Results → Deploy**.
3. Upload a small instruction CSV (or generate a synthetic one in step 3), keep
   the defaults, and submit.

Before a job is queued the UI checks `GET /api/health/workers`; if no worker is
alive, the job is rejected with a clear message instead of hanging forever.

## 4. Optional: free cloud GPU (Modal)

No local GPU? In **Step 4 → Compute backend**, pick **Modal** to train on a free
T4. It needs `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` in `.env` — see
[DEPLOY.md](DEPLOY.md#cloud-gpu-via-modal-optional).

## Gated models

Models like Llama 3 and Mistral-instruct require accepting the license on the
Hugging Face Hub and a valid `HF_TOKEN` in `.env`. A missing token fails the job
fast with a clear message rather than a cryptic download hang.
