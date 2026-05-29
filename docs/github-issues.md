# GitHub Issues to File

Copy-paste each block into GitHub → Issues → New Issue.

---

## Issue 1 — Wire API job creation to Celery

**Labels:** `bug`, `backend`, `priority: high`

**Title:** `POST /jobs` does not enqueue a Celery task

**Body:**
`app/api.py` `create_job()` generates a UUID but never calls `workers.train_task`. The fine-tuning pipeline is never actually triggered from the UI.

**Fix:**
```python
from workers.train_task import run_training
result = run_training.delay(config.model_dump())
return JobCreated(job_id=result.id)
```

Also update `GET /jobs/{job_id}` to read real status from Redis via `workers/status.py`.

---

## Issue 2 — Implement trainer/evaluate.py

**Labels:** `feature`, `ml`, `good first issue`

**Title:** `evaluate_model()` returns placeholder `None` values

**Body:**
`trainer/evaluate.py` loads perplexity and BLEU metrics but the evaluation loop is not implemented — it always returns `{"perplexity": None, "bleu": None}`.

Implement the actual evaluation loop:
- Run model inference on `test_dataset`
- Compute perplexity from cross-entropy loss
- Compute BLEU on decoded outputs vs. references
- Return populated metrics dict

---

## Issue 3 — Scientific Data Generator

**Labels:** `feature`, `dataset`, `enhancement`

**Title:** Feature: Scientific dataset generator

**Body:**
One of the 8 core features of TuneOS. No code exists yet.

Allow users to generate synthetic training datasets from:
- A topic / domain description
- Seed examples (few-shot)
- Format templates (instruction-response, QA, etc.)

Could leverage a local model or a template-based approach.

---

## Issue 4 — Model Conversion (HuggingFace ↔ GGUF ↔ SafeTensors)

**Labels:** `feature`, `model-conversion`, `enhancement`

**Title:** Feature: Convert model weights between formats

**Body:**
One of the 8 core features. No code exists yet.

Support:
- HuggingFace → GGUF (via `llama.cpp` convert scripts)
- HuggingFace → SafeTensors (via `safetensors` library)
- GGUF → HuggingFace

Expose via `POST /convert` API endpoint and a UI page.

---

## Issue 5 — Model Analysis page

**Labels:** `feature`, `ui`, `ml`

**Title:** Feature: Model analysis and performance metrics dashboard

**Body:**
One of the 8 core features. Add a `/analysis` page that shows:
- Model parameter count
- Memory footprint estimate
- Perplexity on a held-out set
- Loss curve post-training

Hook into `trainer/evaluate.py` once that is implemented (#2).

---

## Issue 6 — Model Understanding / Architecture Visualizer

**Labels:** `feature`, `ui`, `ml`, `enhancement`

**Title:** Feature: Visualize model architecture and tokenization

**Body:**
One of the 8 core features. No code exists yet.

Show:
- Layer-by-layer architecture tree (transformer blocks, attention heads)
- Tokenizer behaviour: live tokenization preview for user input
- Attention weight heatmaps for a sample prompt (optional)

---

## Issue 7 — Jupyter Notebook Integration

**Labels:** `feature`, `enhancement`, `modular`

**Title:** Feature: Jupyter notebook integration for exploration workflows

**Body:**
One of the 8 core features. Allow users to open a pre-seeded Jupyter notebook from within TuneOS that has the trained adapter already loaded, for interactive evaluation and experimentation.

---

## Issue 8 — Transfer Learning workflow

**Labels:** `feature`, `ml`, `enhancement`

**Title:** Feature: Transfer learning from existing fine-tuned adapter

**Body:**
One of the 8 core features. Allow users to start a new training run from a previously saved LoRA adapter (continue training / domain adaptation) rather than always starting from the base model.

---

## Issue 9 — Commit poetry.lock for reproducible installs

**Labels:** `chore`, `devex`

**Title:** `poetry.lock` not committed — installs are not reproducible

**Body:**
`poetry.lock` is currently absent (or gitignored). Contributors who run `poetry install` may get different dependency versions.

Run `poetry lock` and commit the result.

---

## Issue 10 — Add unit tests for workers/ and app/api.py

**Labels:** `testing`, `chore`

**Title:** No tests for Celery workers or REST API

**Body:**
`tests/` only covers `trainer/config.py`. Add:
- `tests/test_api.py` — FastAPI TestClient tests for all `/api` endpoints
- `tests/test_workers.py` — Mock-Celery tests for `train_task.py` and `status.py`

---

## Issue 11 — Add screenshots and demo GIF to README

**Labels:** `documentation`, `good first issue`

**Title:** README has no screenshots or demo

**Body:**
The README lacks visuals. Add:
1. A screenshot of the main TuneOS window
2. A screenshot of the training progress page
3. A short demo GIF (start a training run, watch loss curve update)

Place images in `docs/assets/`.

---

## Issue 12 — Enable GPU training on Apple Silicon (MPS)

**Labels:** `feature`, `ml`, `apple-silicon`

**Title:** MPS (Apple Silicon) backend support for training

**Body:**
`api.py` detects Apple Silicon MPS but the training pipeline in `trainer/finetune.py` uses `device_map="auto"` which maps to CUDA if available and CPU otherwise. It does not explicitly enable MPS.

Add an MPS code path in `trainer/config.py` and `trainer/finetune.py`:
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
```
Note: `bitsandbytes` 4-bit quantization is not supported on MPS — the MPS path should fall back to full-precision LoRA.

---

## Issue 13 — Implement real job status tracking

**Labels:** `backend`, `feature`

**Title:** `GET /jobs/{job_id}` always returns `"unknown"` status

**Body:**
`api.py` `get_job()` is a stub. It should query `workers/status.py` (which reads from Redis) to return real progress, loss, and state for a running or completed job.

Implement:
```python
from workers.status import get_job_status
state = get_job_status(job_id)
return JobStatus(job_id=job_id, status=state["status"], progress=state.get("progress", 0))
```

---

## Issue 14 — Dataset upload endpoint

**Labels:** `feature`, `backend`

**Title:** No file upload endpoint in the API

**Body:**
`app/pages/upload.py` exists but there is no `POST /api/upload` endpoint in `app/api.py`. Add a FastAPI file upload endpoint that saves the dataset to `storage/datasets/` via `storage/dataset_store.py` and returns the saved filename for use in job creation.
