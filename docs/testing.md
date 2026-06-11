# Testing Guide

TuneOS has a pytest-based test suite covering unit tests for the training pipeline,
dataset utilities, and experiment database, plus an opt-in integration test that runs
a real training step on a tiny model.

---

## Running the Suite

```bash
# Full test suite (no GPU required)
poetry run pytest

# With coverage report
poetry run pytest --cov=app --cov=trainer --cov=workers

# Verbose output
poetry run pytest -v
```

Most tests are mocked and pass without a GPU. The integration test loads a tiny model
and runs an actual training step — opt in explicitly:

```bash
TUNEOS_INTEGRATION_TESTS=1 poetry run pytest tests/test_trainer_integration.py
```

---

## Lint and Type Checks

CI enforces Ruff formatting, Ruff linting, and mypy type checking. Run the same
checks locally before pushing:

```bash
# Lint
poetry run ruff check .

# Format check (what CI runs)
poetry run ruff format --check .

# Apply formatting
poetry run ruff format .

# Type checking (pure-logic backend modules)
poetry run mypy
```

---

## Test Files

| File | What it covers |
|---|---|
| `tests/test_dataset.py` | `load_and_tokenize`, `load_preference_pairs`, `load_multimodal`, `detect_dataset_type` |
| `tests/test_metrics.py` | Perplexity, rouge1, rouge2, rougeL, bleu, meteor registry |
| `tests/test_dpo.py` | DPO dataset loading and `train_dpo` smoke test |
| `tests/test_experiments_db.py` | All DB helpers, upsert idempotency, SQL placeholder conversion |
| `tests/test_trainer_integration.py` | Full training step on a tiny model — opt-in only |

---

## Environment Setup

Copy the example env file and set the variables you need for the scenarios you want
to exercise:

```bash
cp .env.example .env
```

| Variable | Required for |
|---|---|
| `HF_TOKEN` | Loading gated models (Llama 3, Mistral-instruct) in integration tests |
| `OPENROUTER_API_KEY` | Testing the intent flow question generation and plan updates |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | Testing the Modal cloud GPU backend |
| `REDIS_URL` | Defaults to `redis://localhost:6379/0` |

---

## Manual Test Scenarios

### SFT Golden Path

1. Start the application (`reflex run` or `docker-compose up`).
2. Open `http://localhost:3000` and navigate to the Fine-tune section.
3. Complete the intent flow (Phase A → B → C) with any context.
4. Paste `EleutherAI/pythia-410m` as the model ID — small enough to run locally
   with ~2 GB VRAM.
5. Upload a small instruction CSV or generate a synthetic one in step 3.
6. Accept defaults in step 4 and submit.
7. Watch the loss curve update live in step 5.
8. Verify that eval metrics (perplexity, ROUGE-1, BLEU) appear in step 6.

### DPO Path

1. In step 1, choose **DPO** as the training technique.
2. In step 3, upload a preference dataset with `prompt`, `chosen`, `rejected` columns.
3. In step 4, verify the DPO beta and column-mapping fields appear.
4. Submit and confirm the job routes to the `dpo` Celery queue.

### Knowledge Distillation Path

1. Choose **Knowledge Distillation** in step 1.
2. In step 4, enter a teacher model ID in the KD field.
3. Submit and confirm the job routes to the `kd` Celery queue.

### Vision-Language Model Path

1. Select a VLM model ID (e.g. a LLaVA checkpoint).
2. Upload an image-text dataset with `image`, `instruction`, `output` columns.
3. Confirm the job submits to `POST /api/jobs/vision` and processes via
   `AutoProcessor`.

### Fallback Behavior — No API Key

1. Remove `OPENROUTER_API_KEY` from `.env` and restart.
2. Go through Phase A of the intent flow and click Continue.
3. Confirm that five generic default questions appear immediately (no spinner).
4. Confirm no live plan amber card appears.
5. Confirm the rest of the wizard still completes end to end.

### Worker Health Check

1. Stop the Celery worker.
2. Try to submit a training job.
3. Confirm the UI shows a warning that no worker is alive instead of queuing
   the job silently.

### Modal Cloud GPU

1. Set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `.env`.
2. In step 4, choose **Modal** under Compute backend.
3. Submit a small job and confirm the Provisioning banner appears.
4. Confirm the loss chart updates live once training starts on the remote GPU.

---

## Common Issues

**Tests fail with import errors** — Run `poetry install` to ensure all dependencies
are installed.

**Integration test hangs on first run** — Downloading the test model can take a few
minutes. Set `HF_HOME` to a warmed cache directory if you run this test frequently.

**Ruff format fails in CI** — Run `poetry run ruff format .` locally and commit the
result.

**mypy errors on Reflex state vars** — State vars that hold lists must use
`list[ConcreteModel]` where `ConcreteModel` inherits from `pydantic.BaseModel`.
`list[dict[str, Any]]` and bare `Any` will fail to compile in `rx.foreach`.

---

## Adding a New Metric

Metrics live in `trainer/metrics.py`. Register a function with the `@register`
decorator:

```python
@register("my_metric", greater_is_better=True, kind="reference")
def compute_my_metric(predictions: list[str], references: list[str]) -> float | None:
    ...
```

Use `kind="loss"` for metrics that take `(model, tokenizer, dataset)` and
`kind="reference"` for metrics over prediction/reference string pairs. Add a test in
`tests/test_metrics.py`.
