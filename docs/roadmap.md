# Roadmap

TuneOS is a model operations workspace covering the full lifecycle of language model
development — from discovery and dataset preparation through fine-tuning, evaluation,
and deployment. This document describes the core product pillars and near-term
priorities.

---

## Core Pillars

### Model Intake

Users can import models from the Hugging Face Hub, GitHub, and local storage. TuneOS
parses model links, fetches repository metadata, shows a confirmation preview, and
opens a model workspace with chat-assisted setup.

### Open Data Discovery

The Datasets area surfaces public datasets from open sources. Users can search,
filter, preview, compare, and add datasets to a project. TuneOS recommends datasets
with suitability badges based on model, task, modality, license, size, and expected
fine-tuning strategy.

### Analytics

Analytics help users understand any model with visual explanations — charts, flow
diagrams, data-to-output journey views, technique mapping, and model behavior
summaries.

### Fine-Tuning

Fine-tuning supports LoRA, QLoRA, AdaLoRA, IA3, prefix-tuning, and prompt-tuning.
DPO preference alignment and knowledge distillation are now live. Vision-language
model fine-tuning is available via AutoProcessor. An AI auto mode proposes settings
based on the model, dataset, task, and hardware profile.

### Project and State Management

TuneOS maintains durable project state across model intake, dataset selection,
analytics, configuration, training, chat, and results. Users can return to a project
and recover the complete workflow context.

---

## Shipped

The following capabilities are live as of the current release:

| Capability | Details |
|---|---|
| SFT with adapter registry | LoRA, QLoRA, AdaLoRA, IA3, prefix-tuning, prompt-tuning |
| DPO preference alignment | `(prompt, chosen, rejected)` triples via `trl.DPOTrainer` |
| Knowledge distillation | Student trained against teacher logits; configurable temperature and alpha |
| VLM fine-tuning | Image-text datasets via `AutoProcessor`; dedicated Celery task |
| Adapter composition | Stack a second adapter type via `PeftMixedModel` (advanced mode) |
| Multi-queue Celery | Separate `sft`, `dpo`, `kd` queues for independent scaling |
| Compute backend selector | Local GPU, Modal T4, or HF ZeroGPU per job |
| Eval metrics | Perplexity, ROUGE-1/2/L, BLEU, METEOR; live validation curve |
| Experiment tracking | SQLite or PostgreSQL; run comparison, hyperparameter table |
| Model registry | Name and alias a training run; register from the Results step |
| Structured observability | JSON logging, Redis-batched step metrics, durable SQLite fallback |
| AI intent flow | OpenRouter-powered personalized question generation and plan summaries |

---

## Near-Term Priorities

- Cross-platform desktop packaging — `.exe` (Windows) and `AppImage` / Snap (Linux)
  via GitHub Actions.
- Fully offline dataset processing and local Hugging Face cache management.
- Data discovery with dataset recommendation badges.
- Analytics visuals for model and dataset behavior.
- Expanded model support for additional multimodal and code-focused architectures.
