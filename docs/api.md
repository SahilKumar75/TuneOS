# TuneOS API Documentation

This document outlines the HTTP API exposed by the TuneOS backend (FastAPI),
mounted under the `/api` prefix. It backs the Reflex UI and manages background
training jobs via Celery + Redis.

> *Note: This API is intended for use by the TuneOS UI. Endpoint stability is
> not yet guaranteed for external integrations.*

## System

### `GET /api/health`
Liveness probe. Returns `{ "status": "ok", "version": "<semver>" }`.

### `GET /api/gpu`
Detects the available accelerator (CUDA, Apple Metal/MPS, or CPU).

## Models

### `GET /api/models`
Lists the curated set of supported base models.

### `POST /api/models/validate`
Validates that a model id is loadable. Body: `{ "model_id": "...", "hf_token": "" }`.

## Datasets

### `GET /api/datasets/search?q=<query>`
Searches the Hugging Face Hub for datasets.

### `GET /api/datasets/{dataset_id}/preview`
Returns the first few rows and column names of a Hub dataset.

### `POST /api/datasets/generate`
Generates a synthetic instruction dataset (self-instruct or template-based).

## Jobs

### `POST /api/jobs`
Creates and enqueues a fine-tuning job.

**Request body** (`JobConfig`, abbreviated):
```json
{
  "model_id": "mistralai/Mistral-7B-v0.1",
  "model_source": "hub",
  "dataset_path": "string",
  "technique": "qlora",
  "lora_rank": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "learning_rate": 2e-4,
  "epochs": 3,
  "batch_size": 4,
  "max_seq_length": 512,
  "eval_split_ratio": 0.1,
  "early_stopping_patience": 0,
  "resume_from_checkpoint": ""
}
```
LoRA `target_modules` are auto-detected from the model architecture, so they
are not part of the request.

Phase 2 training controls:

| Field | Default | Meaning |
| --- | --- | --- |
| `eval_split_ratio` | `0.1` | Fraction of the dataset held out for in-training validation. `0` disables the eval loop. |
| `early_stopping_patience` | `0` | Stop after this many evals with no `eval_loss` improvement. `0` disables early stopping. Requires a non-zero `eval_split_ratio`. |
| `resume_from_checkpoint` | `""` | Path to a checkpoint dir to resume from; empty starts fresh. |

If training hits a CUDA out-of-memory error, the job status reports
`status: "failed"` with a `suggestion` field describing how to reduce memory.

**Response:** `{ "job_id": "string", "status": "queued" }`

### `GET /api/jobs`
Lists all runs from the durable SQLite store, most-recent first. Each item is a
`JobStatus` (`job_id`, `status`, `output_path`). Job status is persisted to
SQLite at start/completion/failure, so this endpoint works even if Redis is
unavailable.

### `GET /api/jobs/{job_id}`
Returns live status for one job (`status`, `progress`, `output_path`, `error`).

### `DELETE /api/jobs/{job_id}`
Cancels a running job (Celery revoke).

### `GET /api/jobs/{job_id}/eval`
Returns evaluation metrics computed after training. Metrics come from the
pluggable registry in `trainer/metrics.py`: `perplexity` (loss-based, default)
plus `rouge1` and `bleu` (reference-based). Loss-based metrics run on the
held-out validation split; reference-based metrics compare generated text
against the dataset's reference outputs.

### `GET /api/jobs/{job_id}/download` · `GET /api/jobs/{job_id}/download-merged`
Streams the adapter (or merged model) as a ZIP archive.

### `POST /api/jobs/{job_id}/infer`
Runs inference against the fine-tuned model. Body: `{ "prompt": "...", "max_new_tokens": 300, "temperature": 0.7 }`.

### `POST /api/jobs/{job_id}/merge` · `/export-gguf` · `/push-github` · `/push_hub`
Deployment actions: merge the adapter into the base model, export GGUF,
push to a GitHub repo, or push to the Hugging Face Hub.

### `POST /api/jobs/{job_id}/commentary`
Returns templated progress commentary based on the loss trajectory.

## Experiments

### `GET /api/experiments`
Lists all recorded runs (hyperparameters, final metrics, status).

### `DELETE /api/experiments/{experiment_id}`
Deletes a recorded run.

## Storage Model

Run history is persisted in `storage/experiments.db` (SQLite):

| Table | Purpose |
| --- | --- |
| `runs` | One row per run: config, final loss/perplexity, status, output path |
| `run_metrics` | Step-level metrics `(run_id, key, value, step, timestamp)` — queryable training curves |
| `run_params` | Immutable hyperparameter snapshot `(run_id, key, value)` |

The pure-SQLite persistence layer lives in `app/state/experiments_db.py` and has
no Reflex dependency, so the headless Celery worker can write to it directly.

## Internal Architecture

`POST /api/jobs` enqueues a Celery task onto Redis; a background worker
(`workers/train_task.py`) runs the training stack (`trainer/`) and streams
step-level metrics back over a Redis channel, which the UI consumes live.
