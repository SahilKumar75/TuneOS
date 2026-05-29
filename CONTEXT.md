# TuneOS — Project Context

> This file exists so any new AI session can get fully up to speed on TuneOS instantly.
> Last updated: 2026-05-29

---

## What Is TuneOS

TuneOS is an open-source, full-stack **desktop application** for local LLM lifecycle management. The name "OS" is branding — not a literal operating system. The goal is to give users complete local control over models without sending data to external APIs.

It runs as a **PyQt6 frameless desktop shell** that manages all background services automatically. The user never needs to touch a terminal.

**Stack:**
- Desktop shell: PyQt6 + PyInstaller (packaged as a `.app` on macOS)
- UI: Reflex (Python-based React-like framework)
- Training workers: Celery + Redis (async job queue)
- ML: PyTorch + HuggingFace Transformers + PEFT (LoRA/QLoRA)
- Packaging: Poetry, Docker Compose

---

## The 8-Feature Vision (from founder's notes)

These are the full planned capabilities of TuneOS. Listed in rough priority order:

1. **LoRA / QLoRA Fine-tuning** — Parameter-efficient fine-tuning on local hardware using PyTorch. Primary goal of the project. Mistral 7B, Llama 3 8B, Phi-3 Mini, Gemma 2B are the target base models.
2. **Scientific Data Generator** — Generate, format, and synthesize datasets from user needs and inputs. Not yet built.
3. **Loss / Export Tracking** — Track training loss curves and export results. Partially built (RedisLossCallback exists, loss_chart.py exists).
4. **Model Conversion A↔B** — Convert weights between formats (HuggingFace ↔ GGUF ↔ SafeTensors). Not yet built.
5. **Model Analysis** — Evaluate model metrics, track performance. `trainer/evaluate.py` exists but status unknown.
6. **Model Understanding** — Visualize model architectures, tokenization behavior, internal representations. Not yet built.
7. **Modular Cohesion with Notebooks** — Jupyter notebook integration for exploration workflows. Not yet built.
8. **Transfer Learning** — Apply transfer learning workflows. Not yet built.

---

## Current Build Status

**Honest assessment: less than 10% complete.** The project is very early stage. Here is what exists vs. what is a stub:

### Built (functional or near-functional)
- `trainer/config.py` — ModelConfig, LoraConfig, TrainingConfig dataclasses. Clean and complete.
- `trainer/lora.py` — LoRA injection, adapter save, merge-and-save. Complete.
- `trainer/qlora.py` — QLoRA model prep with 4-bit quantization. Complete.
- `trainer/dataset.py` — Dataset loading and tokenization. Complete.
- `trainer/finetune.py` — Full fine-tuning pipeline wiring everything together. Complete.
- `trainer/callbacks.py` — RedisLossCallback for streaming loss to Redis. Complete.
- `workers/celery_app.py` — Celery app setup. Complete.
- `workers/train_task.py` — Celery task wrapping finetune pipeline. Complete.
- `workers/status.py` — Job status via Redis. Complete.
- `docker-compose.yml` — Redis + Celery worker + Reflex app. Functional.

### Partial / Unknown Status
- `trainer/evaluate.py` — Skeleton only. Loads `perplexity` and `bleu` metrics but returns `{perplexity: None, bleu: None}`. Not functional.
- `trainer/loader.py` — Complete. Loads model with 4-bit/8-bit quantization via BitsAndBytesConfig. Clean code.
- `app/app.py` — Complete. Reflex app with all routes wired, mounts REST API at `/api`.
- `app/api.py` — Well-structured FastAPI endpoints (health, gpu, models, job CRUD). **Key gap:** `POST /jobs` generates a UUID but never enqueues a Celery task — the trainer pipeline is not connected to the API.
- `app/pages/`, `app/components/`, `app/state/` — Exist but backend wiring to Celery/API not confirmed.
- `storage/adapter_store.py` — Functional: lists/deletes adapter directories.
- `storage/dataset_store.py` — Functional: lists/deletes dataset files.

### Not Built Yet
- `desktop/` — PyQt6 shell. Referenced in imports but folder status unclear.
- `build_desktop.py` — Referenced in README quickstart but may not exist.
- Data generator, model conversion, model understanding, notebooks, transfer learning — none of these exist in any form.

---

## Known Bugs

1. **Desktop app crashes** ✅ FIXED — Root cause: `process_manager.start()` ran `subprocess.run(..., timeout=60)` on the main thread, blocking the Qt event loop entirely. Fixed by moving it into a `_StartupThread(QThread)` in `main.py`.
2. **Duplicate window spawning** ✅ FIXED — Added `QLockFile` at app startup in `main.py`. Second instance shows a warning dialog and exits immediately.

### Critical Wiring Gap (not yet fixed)
- `app/api.py` `POST /jobs` generates a UUID but never calls `workers/train_task.py`. The UI → Celery connection is broken.

---

## Infrastructure Plan

### Current
Docker Compose runs Redis + Celery worker locally on the user's Mac. This is heavy on local hardware.

### Planned
Move the Docker-based training workers to **Hugging Face Spaces** (Docker Space type) to reduce load on the user's Mac. The Reflex UI and PyQt6 shell stay local. The Celery worker + Redis backend runs on HF Spaces.

This requires:
- A `Dockerfile` for the HF Space
- A `README.md` inside the Space (HF format with metadata header)
- Updated `.env` / config pointing the local app to the remote worker URL

---

## Repository Structure

```
tuneos/
├── app/                  # Reflex UI
│   ├── pages/            # home, landing, configure, training, results, upload
│   ├── components/       # loss_chart, config_form, model_card, sidebar, layout
│   ├── state/            # app_state, job_state, model_state
│   ├── api.py
│   └── app.py
├── trainer/              # ML training logic
│   ├── config.py         # ModelConfig, LoraConfig, TrainingConfig
│   ├── lora.py           # LoRA injection + save
│   ├── qlora.py          # QLoRA / 4-bit quantization
│   ├── dataset.py        # Data loading + tokenization
│   ├── finetune.py       # Main pipeline
│   ├── callbacks.py      # RedisLossCallback
│   ├── evaluate.py
│   └── loader.py
├── workers/              # Celery async workers
│   ├── celery_app.py
│   ├── train_task.py
│   └── status.py
├── storage/              # Local adapter + dataset storage
├── desktop/              # PyQt6 shell (status unclear)
├── docs/                 # lora-explained, supported-models, quickstart, api, roadmap, issue-labels
├── tests/                # Real unit tests for trainer/config.py (12 tests)
├── .github/
│   ├── workflows/        # ci.yml (lint + pytest), release.yml (auto-changelog)
│   └── ISSUE_TEMPLATE/   # bug_report.md, feature_request.md
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── LICENSE               # Apache 2.0
├── README.md             # Has CI/license/python badges
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── SECURITY.md
```

---

## Open Source Status

**Current score: ~85/100** (after fixes applied in this session)

### Fixed in this session
- Placeholder emails in `CODE_OF_CONDUCT.md` and `SECURITY.md` → replaced with `shekhar59324@gmail.com`
- Placeholder author in `pyproject.toml` → `Shekhar <shekhar59324@gmail.com>`
- Placeholder URL in `docs/quickstart.md` → `github.com/SahilKumar75/TuneOS`
- `__pycache__/` added explicitly to `.gitignore`
- CI, license, Python badges added to `README.md`
- Skeletal tests replaced with 12 real unit tests for `trainer/config.py`

### Still missing for 100/100
- `poetry.lock` not committed (reproducible installs broken) — run `poetry lock` locally and commit
- No screenshots or demo video in README
- `build_desktop.py` exists but has not been tested end-to-end

---

## GitHub Goals

**Owner:** Sahil Kumar Singh (@SahilKumar75), Pune. GitHub PRO member.

**Current contribution graph:** 74% commits, 12% issues, 11% PRs, 3% code reviews. Goal is to diversify.

**Strategy:**
- File issues for every planned feature → drives issues %
- Open a branch per issue, merge via PR → drives PRs %
- Request review before merging (even self-review) → drives code review %

**Badges to earn:**
- **Pull Shark** — get 2 PRs merged (open branch per issue)
- **Quickdraw** — close an issue within 5 minutes of opening
- **Galaxy Brain** — enable GitHub Discussions, get an answer marked accepted
- **Starstruck** — needs 16 stars on a repo; share TuneOS when ready

---

## Contributor Strategy

- Solo project currently, open to external contributors
- External contributors: **create issues and solve them via PRs** — no direct commits to main
- No features are locked; everything from the 8-feature vision is fair game for contributors
- Issue labels taxonomy is documented in `docs/issue-labels.md`

---

## Supported Base Models

| Model | HF ID | VRAM |
|---|---|---|
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | ~16GB |
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B` | ~18GB (needs HF token) |
| Phi-3 Mini | `microsoft/Phi-3-mini-4k-instruct` | ~8GB |
| Gemma 2B | `google/gemma-2b` | ~6GB |

---

## Session Log

### Session 1 (prior session)
- Placeholder emails/author/URLs fixed across `CODE_OF_CONDUCT.md`, `SECURITY.md`, `pyproject.toml`, `docs/quickstart.md`
- `__pycache__/` added to `.gitignore`
- CI, license, Python badges added to `README.md`
- 12 real unit tests written for `trainer/config.py`

### Session 2 (2026-05-29)

**Desktop fixes (`main.py`):**
- Fixed crash: `process_manager.start()` was blocking the Qt main thread with `subprocess.run(..., timeout=60)`. Moved into `_StartupThread(QThread)` — UI stays responsive during startup.
- Fixed duplicate windows: added `QLockFile` single-instance guard. Second launch shows a warning dialog and exits.

**Codebase review (previously unknown files):**
- `trainer/loader.py` — complete and clean
- `storage/adapter_store.py` / `storage/dataset_store.py` — functional
- `trainer/evaluate.py` — skeleton only, returns `None` for all metrics
- `app/app.py` — complete, all routes wired
- `app/api.py` — well-structured but `POST /jobs` never calls Celery (critical gap)

**New files created:**
- `hf_spaces/Dockerfile` — deploys Redis + Celery worker to Hugging Face Spaces
- `hf_spaces/entrypoint.sh` — starts Redis then Celery in one container
- `hf_spaces/README.md` — HF Spaces metadata header
- `docs/github-issues.md` — 14 ready-to-file GitHub issues covering all 8 features + known bugs
- `tests/test_api.py` — 16 tests for all REST API endpoints (FastAPI TestClient, no server needed)
- `tests/test_workers.py` — 8 tests for Celery task and status (Redis mocked)
- `tests/test_evaluate.py` — 4 tests for evaluate_model (ML deps mocked)

**GitHub connector:** Added but requires a new session to activate MCP tools.

## Next Steps (prioritised)

1. **Start new session** → file 14 GitHub issues directly via GitHub MCP from `docs/github-issues.md`
2. **Enable GitHub Discussions** on the repo (Settings → Discussions) — needed for Galaxy Brain badge
3. **Commit `poetry.lock`** — run `poetry lock` locally and commit
4. **Add screenshots/demo GIF** to README — put in `docs/assets/`
5. **Wire `POST /jobs` to Celery** — biggest functional gap; see Issue 1 in `docs/github-issues.md`
6. **Implement `trainer/evaluate.py`** — evaluation loop is a stub; see Issue 2
7. **Deep planning session** — architect the remaining 5 features (data gen, model conversion, analysis, understanding, notebooks)
