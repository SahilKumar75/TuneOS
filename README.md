---
title: TuneOS
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
app_port: 7860
---

# TuneOS

[![CI](https://github.com/SahilKumar75/TuneOS/actions/workflows/ci.yml/badge.svg)](https://github.com/SahilKumar75/TuneOS/actions/workflows/ci.yml)
[![Release](https://github.com/SahilKumar75/TuneOS/actions/workflows/release.yml/badge.svg)](https://github.com/SahilKumar75/TuneOS/actions/workflows/release.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

TuneOS is a fine-tuning workstation for large language models. You bring a model and a dataset; TuneOS handles dataset prep, training, evaluation, and pushing the adapter to the Hub. It runs as a macOS desktop app or as two Docker containers on Hugging Face Spaces — compute stays on infrastructure you own.

## Getting started

You need Python 3.10+, Poetry, and Docker Desktop.

```bash
git clone https://github.com/SahilKumar75/TuneOS
cd TuneOS
cp .env.example .env        # add HF_TOKEN for gated models
poetry install
docker compose up -d        # starts Redis + Celery worker
poetry run reflex run
```

Open `http://localhost:3000`. For your first run, try `EleutherAI/pythia-410m` — it trains on CPU and finishes in a few minutes.

No Docker? Start Redis and the worker manually:

```bash
redis-server &
celery -A workers.celery_app worker --loglevel=info
```

## Training modes

The wizard supports three paths. Pick the mode in step 1 and the rest of the form adapts.

| Mode | What it does |
|------|-------------|
| SFT | Supervised fine-tuning on instruction/chat data |
| DPO | Preference alignment from chosen/rejected pairs |
| KD | Knowledge distillation from a teacher model |

Six adapter techniques are available per run: LoRA, QLoRA, AdaLoRA, IA3, prefix tuning, and prompt tuning. Advanced mode lets you stack a second technique on top via `PeftMixedModel`.

## Supported models

Any Hugging Face causal LM works — `target_modules` are auto-detected. Well-tested defaults exist for Mistral 7B, Llama 3 8B, Phi-3 Mini, Gemma 2B, and Pythia 410M. Auto-detection also covers Qwen2/3, Falcon, StarCoder2, Mixtral, and more. See [docs/supported-models.md](docs/supported-models.md) for VRAM estimates.

## Docs

- [Quickstart](docs/quickstart.md) — first fine-tune, step by step
- [Architecture](docs/architecture.md) — how the queue, worker, and UI fit together
- [Deploy to HF Spaces](docs/deploy.md) — two-Space setup with Upstash Redis
- [API reference](docs/api-reference.md) — all REST endpoints

## Tests

```bash
poetry run pytest
poetry run ruff check .
```

The trainer integration test (real GPU, real model) is opt-in: `TUNEOS_INTEGRATION_TESTS=1 pytest tests/test_trainer_integration.py`.

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md). Security issues go to [SECURITY.md](.github/SECURITY.md). Apache 2.0 license.
