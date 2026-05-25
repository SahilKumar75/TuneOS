# Finetune Platform

Finetune Platform is an open-source, full-stack LLM fine-tuning platform. It allows users to bring their own dataset, select an open-source base model from the Hugging Face Hub, configure LoRA/QLoRA hyperparameters, and kick off a fine-tuning job asynchronously. Users can watch the training loss in real time and download the resulting adapter weights.

## Architecture

```
User (Browser) 
  <--> [ Reflex UI (App) ]
           |
      (Celery Task)
           v
[ Redis ] <--> [ Celery Worker (ML Engine) ]
                   |
             (PyTorch / PEFT)
                   |
            (Saves Output)
```

## Desktop App (New!)

TuneOS now includes a native desktop application powered by PyQt6, offering a clean, frameless interface that embeds the Reflex UI and automatically manages the background Docker services (Redis + Celery worker). 

**To build the desktop app on macOS:**
```bash
poetry install -E desktop
poetry run python build_desktop.py
open dist/TuneOS.app
```

**Upcoming Desktop Features:**
- **Cross-platform Support:** Packaging for Windows (`.exe`) and Linux (AppImage/Snap).
- **Native Notifications:** System tray alerts for when model training finishes.
- **Offline Mode:** Tools to process datasets offline and manage the Hugging Face local cache.

## Quickstart

```bash
git clone https://github.com/SahilKumar75/TuneOS
cd TuneOS
cp .env.example .env    # add HF_TOKEN
docker-compose up
# open http://localhost:3000
```

## Supported Models

| Model | HF ID | Notes |
|---|---|---|
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | Primary target, well-tested with QLoRA |
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B` | Requires HF token |
| Phi-3 Mini | `microsoft/Phi-3-mini-4k-instruct` | Fast, runs on smaller GPUs |
| Gemma 2B | `google/gemma-2b` | Good for low-VRAM environments |

## Dataset Format

**JSONL (preferred):**
```json
{"instruction": "Summarize this in a formal tone", "output": "The document presents..."}
{"instruction": "Reply to this customer complaint", "output": "Dear valued customer..."}
```

**CSV:**
```csv
instruction,output
"Summarize this","The document presents..."
```

## LoRA Configuration

| Parameter | Description |
|---|---|
| **Rank (r)** | Determines the number of parameters trained in the adapter. Higher rank allows for more capacity but takes more VRAM. |
| **Alpha** | Scaling factor. Typically set to 2x the Rank. |
| **Dropout** | Dropout probability for LoRA layers to prevent overfitting. |

## How QLoRA Works

QLoRA (Quantized Low-Rank Adaptation) works by loading the base model in a highly compressed 4-bit representation (NormalFloat4). The base model weights are frozen and only a tiny set of adapter weights (LoRA) are injected and trained in full precision (or mixed precision). This drastically reduces the VRAM required to train large models, allowing 7B models to be fine-tuned on consumer GPUs with as little as 8GB of VRAM.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started with contributing.
