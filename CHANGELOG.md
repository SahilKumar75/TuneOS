# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Trainer flexibility (P4-A).** `TrainingConfig.report_to` makes the HF
  experiment-tracker integration configurable (default `"none"`).
  `ModelConfig.attn_implementation` (e.g. `flash_attention_2`/`sdpa`) and
  `ModelConfig.rope_scaling` are now plumbed into model loading.
  `LoraConfig.init_lora_weights` exposes PEFT's adapter-init strategy. More
  architectures auto-detect LoRA targets (Qwen3, Phi-4, Cohere, OLMo, StableLM,
  Mixtral, MPT, StarCoder2, GPT-BigCode). `prepare_qlora_model` now honors
  `use_4bit` instead of forcing it on, and all `save_pretrained` calls use
  `safe_serialization=True`. Minimum `trl` raised to `>=0.12.0`.
- **Modal.com cloud-GPU training backend.** Jobs can now run on a free Modal T4
  GPU instead of the local device — useful when no local GPU is available. Step 4
  (Configure) has a new **Compute backend** selector (Local GPU / Modal / HF
  Spaces), threaded end-to-end through `JobConfig.compute_backend` →
  `TrainingConfig.compute_backend` → `workers/train_task.py`. The new
  `workers/modal_runner.py` serializes the dataset, runs the identical
  `trainer.finetune` pipeline remotely on a Modal T4, and streams the adapter +
  eval metrics back to local disk so the status/metrics layer is unchanged.
  Evaluation logic is shared between backends via `train_task._compute_eval`.
  Enabled by setting `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`; `modal` is an
  optional dependency (`poetry install --with modal`) so nothing changes for
  local-only users. Live loss streaming from Modal is deferred to a later phase.
- **Reproducible runs via a configurable seed.** `TrainingConfig.seed` (default
  `42`) now drives every source of randomness — the train/validation split,
  data shuffling, and weight initialization — through `transformers.set_seed`
  and `TrainingArguments(seed=, data_seed=)`. The seed is persisted with each
  run's hyperparameters, so any run can be reproduced exactly. Previously the
  seed was hardcoded in two places with no user control.
- **Optional `torch.compile()` acceleration.** `TrainingConfig.use_torch_compile`
  (default `False`) enables PyTorch 2.0 compilation via
  `TrainingArguments(torch_compile=...)` for faster training on supported GPUs.
  Off by default so CPU/CI and unsupported backends are unaffected.
- **Reference metrics (ROUGE-1 / BLEU) in results.** After training, the worker
  generates predictions on the held-out evaluation sample and computes ROUGE-1
  and BLEU against the reference outputs (via the new
  `trainer.evaluate.generate_predictions` and
  `trainer.dataset.load_instruction_pairs`). Step 6 (Results) now shows ROUGE-1
  and BLEU tiles alongside perplexity, and the metrics are persisted per run
  (`save_final_metrics`) for cross-run comparison.
- **Register button wired (Phase 4).** Step 6 (Results) now shows a "Register to
  model registry" card when training completes. Users type a name and click
  **Register** — the run is stored via `ModelRegistryState.do_register`, wiring
  up the API and state that was built in Phase 3.
- **Hyperparameter comparison table (Phase 4).** The past-runs table in Step 6
  now includes Technique, LR, LoRA r, and Batch columns alongside the existing
  Final Loss and Perplexity, making cross-run hyperparameter diffing visible at
  a glance.
- **PostgreSQL experiment backend (Phase 4).** Set `EXPERIMENTS_DB_URL` to a
  `postgresql://` DSN to have all experiment data go to Postgres instead of the
  local SQLite file — enables multi-machine worker deployments that share a
  single experiment store. Requires `psycopg2-binary` (`pip install
  psycopg2-binary` or `poetry install --with postgres`).  All upsert statements
  migrated from `INSERT OR REPLACE` to portable `ON CONFLICT … DO UPDATE`,
  compatible with both SQLite 3.24+ and PostgreSQL.
- **`ModelRegistryState.do_register` event.** Reads `register_name` from local
  state and accepts `run_id`, `perplexity`, and `final_loss` as arguments so the
  UI can bind to cross-state Vars cleanly.
- **`FinetuneState.last_train_loss` computed var.** Exposes the final training
  step loss from `loss_history` for use in the register metric snapshot.
- **`tests/test_experiments_db.py`** — 13 unit tests covering all DB helpers,
  upsert idempotency, and the `_adapt_sql` placeholder conversion.
- **Run comparison & model registry (Phase 3).**
  `GET /experiments/compare?ids=run1,run2` returns step-level metrics for up to 10
  runs. `app/components/loss_chart.py` gains `comparison_loss_chart()` for overlaid
  curves. New `registered_models` table + `POST /experiments/models` API lets you
  name and alias a training run (the "Register" action on the Results step).
- **Trainer hardening (Phase 2).** Training now supports an in-training
  validation split (`eval_split_ratio`), `EarlyStoppingCallback`
  (`early_stopping_patience`), and resuming from a checkpoint
  (`resume_from_checkpoint`). These are exposed on `JobConfig` / `POST /api/jobs`.
- **GPU OOM handling.** A CUDA out-of-memory failure is now caught and reported
  with a remediation hint (reduce batch size / sequence length / model) in the
  job status, instead of an opaque crash.
- **Pluggable evaluation metrics.** New `trainer/metrics.py` registry with
  `perplexity` (loss-based) plus `rouge1` and `bleu` (reference-based);
  metrics are requested by name via `trainer.evaluate.evaluate_model`.
- **Trainer test coverage.** Added `tests/test_metrics.py`, `tests/test_dataset.py`,
  and a GPU-free integration test (`tests/test_trainer_integration.py`, gated by
  `TUNEOS_INTEGRATION_TESTS=1`) that runs a real training step on a tiny model.
- **Static typing in CI.** `mypy` now runs in the lint job over the pure-logic
  backend modules.
- **Observability layer (Phase 1).** Step-level metrics are now
  persisted to queryable `run_metrics` and `run_params` tables in
  `storage/experiments.db`, instead of only living in a JSON blob.
- `GET /api/jobs` now returns all runs from the durable SQLite store (was a stub
  returning `[]`).
- Durable job lifecycle: the worker writes job status to SQLite at start,
  completion, and failure, so job state survives a Redis restart.
- `artifact_path(job_id, artifact)` helper centralizes output-path construction
  across the API and workers.
- Automatic LoRA `target_modules` detection per model architecture — fixes
  silent breakage on Gemma, Phi-3, Falcon, Qwen2, and GPT-NeoX.
- Basic API documentation for backend services.
- GitHub Action workflows for CI and Release.
- Issue and PR templates for standardized contribution tracking.
- Contributor Covenant Code of Conduct and Security Policy.
- Empty `tests` directory with initial pytest configuration.

### Documentation
- Refreshed docs to match the current product: quickstart now covers the
  no-Docker Celery worker fallback, gated-model `HF_TOKEN`, and the Modal cloud
  GPU option; `docs/api.md` documents `GET /api/health/workers` and
  `compute_backend`; `docs/DEPLOY.md` gains an `.env` reference table;
  `docs/supported-models.md` reflects auto `target_modules` detection (any HF
  causal LM); `docs/lora-explained.md` adds a DPO preview; README notes the
  Modal/local/ZeroGPU compute backends.

### Changed
- State data models embedded in `rx.State` now use `pydantic.BaseModel`
  (documented in `CONTRIBUTING.md`); `rx.Base` was removed in newer Reflex.
- Explicit lockfile (`poetry.lock`) pinning policy in `CONTRIBUTING.md`.
- Removed stale 'Finetune Platform' branding references and standardized to 'TuneOS'.

### Removed
- Untracked `__pycache__`, `build/`, and `dist/` from the git repository.
