# API Reference

The TuneOS backend is a FastAPI service mounted under the `/api` prefix. It manages
training jobs via Celery and Redis, exposes experiment history from a durable SQLite
(or PostgreSQL) store, and handles model and dataset operations for the Reflex UI.

All job submission endpoints return `{ "job_id": "<uuid>" }` on success and a `4xx`
JSON error body on validation failure.

---

## System

### GET /api/health

Liveness probe. Returns the operational status of the service, Redis broker, and
worker pool.

```json
{
  "status": "ok",
  "redis": true,
  "worker_count": 2
}
```

### GET /api/health/workers

Reports whether at least one Celery worker is alive and ready. The UI calls this
before queuing a job so it can warn when no worker is running rather than letting
the job sit in the queue indefinitely.

```json
{ "workers_alive": true, "workers": ["celery@hostname"] }
```

### GET /api/gpu

Detects the available accelerator and reports VRAM.

```json
{
  "device": "cuda",
  "device_count": 1,
  "vram_total_gb": 16.0,
  "vram_free_gb": 14.2,
  "cuda_version": "12.1"
}
```

---

## Models

### GET /api/models

Lists the curated set of supported base models.

### POST /api/models/validate

Validates that a model ID is loadable from the Hugging Face Hub.

Request body:

```json
{ "model_id": "mistralai/Mistral-7B-v0.1", "hf_token": "" }
```

---

## Datasets

### GET /api/datasets/search?q=\<query\>

Searches the Hugging Face Hub for datasets matching the query string.

### GET /api/datasets/\{dataset_id\}/preview

Returns the first few rows and column names of a Hub dataset.

### POST /api/datasets/generate

Generates a synthetic instruction dataset. Uses OpenRouter first, falls back to
HuggingFace, then rule-based templates. Body fields: `intent` (string), `n` (count),
`seed_examples` (optional list).

Response:

```json
{
  "samples": [{ "instruction": "...", "output": "..." }],
  "dataset_path": "storage/datasets/generated_abc123.jsonl",
  "stats": {
    "total_generated": 10,
    "final_count": 10,
    "generation_method": "openrouter"
  }
}
```

---

## Jobs

### POST /api/jobs

Submit a supervised fine-tuning (SFT) job. Placed on the `sft` Celery queue.

| Field | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | yes | Hugging Face model ID |
| `dataset_path` | string | one of | Path to a local `.jsonl` dataset file |
| `hub_dataset_id` | string | one of | Hugging Face dataset ID |
| `technique` | string | no | Adapter technique: `lora`, `qlora`, `adalora`, `ia3`, `prefix`, `prompt` (default `qlora`) |
| `lora_rank` | int | no | LoRA rank `r` (default `8`) |
| `lora_alpha` | int | no | LoRA scaling alpha (default `16`) |
| `lora_dropout` | float | no | LoRA dropout (default `0.05`) |
| `epochs` | int | no | Training epochs (default `3`) |
| `batch_size` | int | no | Per-device train batch size (default `4`) |
| `learning_rate` | float | no | AdamW learning rate (default `2e-4`) |
| `bf16` | bool | no | Enable bfloat16 mixed precision |
| `seed` | int | no | Global random seed (default `42`) |
| `eval_split_ratio` | float | no | Fraction held out for validation; `0` skips eval |
| `eval_steps` | int | no | Run evaluation every N steps |
| `early_stopping_patience` | int | no | Stop after N evals with no `eval_loss` improvement; `0` disables |
| `resume_from_checkpoint` | string | no | Path to checkpoint dir; empty starts fresh |
| `prompt_template` | string | no | One of `alpaca`, `chatml`, `llama3`, `phi3`, `zephyr` |
| `packing` | bool | no | Enable SFTTrainer sample packing |
| `use_torch_compile` | bool | no | Enable `torch.compile()` acceleration |
| `compute_backend` | string | no | `local`, `modal`, or `hf_spaces` (default `local`) |
| `experiment_id` | string | no | Tag this run under an existing experiment |
| `compose_adapters` | bool | no | Stack a second adapter after training |
| `overlay_technique` | string | no | Technique for the overlay adapter: `lora`, `adalora`, or `ia3` |

LoRA `target_modules` are auto-detected from the model architecture and are not part
of the request.

After training, the held-out evaluation sample is scored for perplexity, ROUGE-1,
and BLEU. These surface in Step 6 (Results) and are persisted per run. If training
hits an OOM error, the job status reports `status: "failed"` with a `suggestion`
field describing how to reduce memory.

Response: `{ "job_id": "string", "status": "queued" }`

---

### POST /api/jobs/dpo

Submit a Direct Preference Optimization job. Placed on the `dpo` Celery queue.

| Field | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | yes | Base model Hugging Face ID |
| `dataset_path` / `hub_dataset_id` | string | one of | Preference dataset |
| `prompt_col` | string | no | Column name for the prompt (default `"prompt"`) |
| `chosen_col` | string | no | Column name for the chosen response (default `"chosen"`) |
| `rejected_col` | string | no | Column name for the rejected response (default `"rejected"`) |
| `beta` | float | no | KL-penalty coefficient (default `0.1`) |
| `max_length` | int | no | Maximum total sequence length (default `1024`) |
| `max_prompt_length` | int | no | Maximum prompt length (default `512`) |
| `lora_rank` | int | no | LoRA rank `r` (default `8`) |
| `epochs` | int | no | Training epochs (default `1`) |
| `batch_size` | int | no | Per-device train batch size (default `4`) |
| `seed` | int | no | Random seed (default `42`) |

---

### POST /api/jobs/distill

Submit a knowledge distillation job. The student model is fine-tuned to match the
teacher model's output distribution. Placed on the `kd` Celery queue.

| Field | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | yes | Student model Hugging Face ID |
| `teacher_model` | string | yes | Teacher model Hugging Face ID |
| `dataset_path` / `hub_dataset_id` | string | one of | Training dataset |
| `temperature` | float | no | Softmax temperature for distillation (default `2.0`) |
| `alpha` | float | no | Weight on distillation loss vs CE loss (default `0.5`) |
| `lora_rank` | int | no | LoRA rank `r` (default `8`) |
| `epochs` | int | no | Training epochs (default `3`) |

---

### POST /api/jobs/vision

Submit a vision-language model fine-tuning job. Images are preprocessed via
`AutoProcessor`. Placed on the `sft` queue by default.

| Field | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | yes | Multimodal model Hugging Face ID |
| `dataset_path` / `hub_dataset_id` | string | one of | Image-text dataset |
| `image_col` | string | no | Column name for images (default `"image"`) |
| `instruction_col` | string | no | Instruction text column (default `"instruction"`) |
| `output_col` | string | no | Target output column (default `"output"`) |
| `use_4bit` | bool | no | Enable 4-bit quantization via bitsandbytes |
| `lora_rank` | int | no | LoRA rank `r` (default `8`) |
| `epochs` | int | no | Training epochs (default `3`) |
| `batch_size` | int | no | Per-device batch size (default `2`) |

---

### GET /api/jobs

Lists all runs from the durable SQLite store, most-recent first. Supports
`limit` (default 50, max 500) and `offset` pagination. Returns `job_id`,
`status`, `output_path` per item. Works even if Redis is unavailable.

### GET /api/jobs/\{job_id\}

Returns live status for one job: `status`, `progress`, `output_path`, `error`,
and optionally `suggestion` when an OOM failure is detected.

### DELETE /api/jobs/\{job_id\}

Cancels a running job via Celery revoke.

### GET /api/jobs/\{job_id\}/eval

Returns evaluation metrics computed after training. Includes perplexity (loss-based),
rouge1, rouge2, rougeL, bleu, and meteor. Falls back to the durable SQLite copy when
the Redis key has expired.

### GET /api/jobs/\{job_id\}/download

Streams the adapter weights as a ZIP archive.

### GET /api/jobs/\{job_id\}/download-merged

Streams the merged model (base + adapter) as a ZIP archive.

### POST /api/jobs/\{job_id\}/infer

Runs inference against the fine-tuned model.

```json
{ "prompt": "Explain gradient descent", "max_new_tokens": 300, "temperature": 0.7 }
```

### POST /api/jobs/\{job_id\}/merge

Merges the adapter into the base model and writes the result to disk.

### POST /api/jobs/\{job_id\}/export-gguf

Exports the merged model to GGUF format.

### POST /api/jobs/\{job_id\}/push_hub

Pushes the adapter to the Hugging Face Hub.

### POST /api/jobs/\{job_id\}/push-github

Pushes the adapter to a GitHub repository.

---

## Experiments

### GET /api/experiments

Lists all recorded runs with hyperparameters, final metrics, and status.

### DELETE /api/experiments/\{experiment_id\}

Deletes a recorded run.

### GET /api/experiments/compare?ids=run1,run2&metric=loss

Returns step-level metric data for up to 10 runs, suitable for overlaid loss curves.

Query params:

| Param | Description |
|---|---|
| `ids` | Comma-separated run IDs |
| `metric` | `loss` (default), `eval_loss`, `learning_rate`, or `epoch` |

```json
{
  "metric": "loss",
  "runs": {
    "run_id_1": [{ "step": 0, "value": 1.23 }, { "step": 10, "value": 1.05 }],
    "run_id_2": [{ "step": 0, "value": 1.31 }]
  }
}
```

### GET /api/experiments/models

Lists all entries in the model registry.

### POST /api/experiments/models

Registers (or updates) a named model pointing to a training run. This is the
Register action in Step 6 of the wizard.

```json
{
  "name": "my-chatbot",
  "run_id": "b3f2a1c0-...",
  "alias": "latest",
  "metric_snapshot": { "perplexity": 4.2, "final_loss": 1.1 }
}
```

### DELETE /api/experiments/models/\{name\}

Removes a named model from the registry.

---

## OpenRouter Integration

Three wizard operations call the OpenRouter API. All use the
`deepseek/deepseek-v4-flash:free` model and fail gracefully if the key is absent
or the call times out.

| Operation | Trigger | Tokens | Fallback |
|---|---|---|---|
| Question generation | Phase A Continue | ~700 | Default 5 questions |
| Live plan update | Each question answer (up to 5x) | ~200 | Silent — no plan shown |
| Synthetic data generation | User clicks Generate | ~2000 | HuggingFace, then templates |

Required environment variable: `OPENROUTER_API_KEY=sk-or-v1-...`

To test that your key works:

```bash
curl -X POST https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek/deepseek-v4-flash:free", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}'
```

---

## Error Reference

| HTTP status | Meaning |
|---|---|
| 200 | Success |
| 400 | Validation error — check request body |
| 401 | Missing or invalid API key |
| 404 | Job or resource not found |
| 429 | Rate limit — wait and retry, or use fallback |
| 503 | Service temporarily unavailable |

Failed jobs include `error` and optionally `suggestion` in the status blob:

```json
{
  "status": "failed",
  "job_id": "b3f2a1c0-...",
  "error": "CUDA out of memory",
  "suggestion": "Reduce batch_size from 4 to 2, or lower max_seq_length."
}
```
