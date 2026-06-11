# Quickstart

Get from a clone to a running fine-tune in a few minutes.

---

## 1. Clone and Configure

```bash
git clone https://github.com/SahilKumar75/TuneOS
cd TuneOS
cp .env.example .env
# Add HF_TOKEN — required for gated models (Llama 3, Mistral-instruct)
```

---

## 2. Start Everything

### With Docker (recommended)

```bash
docker-compose up -d
```

This brings up the Reflex UI, the FastAPI service, Redis, and a Celery worker.
Open `http://localhost:3000`.

### Without Docker

```bash
# terminal 1 — Redis
redis-server

# terminal 2 — Celery worker (this is what runs training)
celery -A workers.celery_app worker --loglevel=info

# terminal 3 — Reflex app
reflex run
```

The desktop launcher does this for you: when Docker is unavailable it starts Redis
and the Celery worker as local subprocesses.

---

## 3. Run Your First Fine-Tune

1. Paste a model ID — `EleutherAI/pythia-410m` is small (~2 GB VRAM) and good for
   a first run.
2. Walk the 7-step wizard: **Model → Intent → Data → Configure → Train →
   Results → Deploy**.
3. Upload a small instruction CSV (or generate a synthetic one in step 3), keep the
   defaults, and submit.

Before a job is queued the UI checks `GET /api/health/workers`. If no worker is
alive, the job is rejected with a clear message instead of sitting in the queue
indefinitely.

---

## 4. Training Modes

The wizard supports three training modes selected in step 1:

| Mode | Dataset shape | Endpoint |
|---|---|---|
| SFT (default) | `instruction`, `output` columns | `POST /api/jobs` |
| DPO | `prompt`, `chosen`, `rejected` columns | `POST /api/jobs/dpo` |
| Knowledge Distillation | `instruction`, `output` + teacher model | `POST /api/jobs/distill` |

---

## 5. Cloud GPU via Modal (Optional)

No local GPU? In Step 4 → Compute backend, pick **Modal** to train on a free T4. You
need `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `.env` — see
[docs/deploy.md](deploy.md#cloud-gpu-via-modal) for setup.

---

## Gated Models

Models like Llama 3 and Mistral-instruct require accepting the license on the
Hugging Face Hub and a valid `HF_TOKEN` in `.env`. A missing token fails the job
immediately with a clear message rather than a silent download hang.
