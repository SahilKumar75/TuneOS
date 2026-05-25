# TuneOS

TuneOS is an open-source, full-stack **desktop application** designed to give you complete local control over the lifecycle of Large Language Models. 

Unlike traditional web-based platforms, TuneOS runs entirely as a native application on your machine, orchestrating complex machine learning tasks in the background while providing a clean, distraction-free GUI.

## Core Aims & Capabilities

TuneOS is built around five primary goals:

1. **Local Fine-Tuning (LoRA / QLoRA):** Leverage PyTorch to run parameter-efficient fine-tuning entirely on your own hardware without sending private data to external APIs.
2. **Scientific Data Generator:** Generate, format, and synthesize complex datasets tailored precisely to your specific needs and inputs.
3. **Model Conversion (A ↔ B):** Seamlessly convert model weights between different formats (e.g., Hugging Face, GGUF, SafeTensors) for deployment across various engines.
4. **Model Analysis:** Track training loss, evaluate model metrics, and analyze performance natively.
5. **Model Understanding:** Explore model architectures, tokenization behavior, and internal representations to gain a deeper understanding of how the model is processing information.

---

## Architecture

TuneOS operates as a standalone desktop application. The GUI is built using a **PyQt6 frameless shell**, which automatically manages all required background services so you never have to touch a terminal.

```text
[ Desktop App (PyQt6) ]
          |
    (Manages Lifecycle)
          |
  +-------+-------+
  |               |
[ Reflex ]    [ Docker Compose ]
  (UI)        (Redis + Celery Worker)
                  |
             [ PyTorch / PEFT ]
```

## Quickstart (Building the App)

TuneOS is currently packaged for macOS, with Windows and Linux support coming soon.

**Prerequisites:**
- Python 3.10+
- [Poetry](https://python-poetry.org/)
- Docker Desktop (for background training workers)

**Build & Run:**
```bash
# Clone the repository
git clone https://github.com/SahilKumar75/TuneOS
cd TuneOS

# Add your Hugging Face Token for gated models
cp .env.example .env    

# Install dependencies (including desktop packaging tools)
poetry install -E desktop

# Build the desktop executable
poetry run python build_desktop.py

# Launch the app!
open dist/TuneOS.app
```

## Supported Base Models

| Model | HF ID | Notes |
|---|---|---|
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | Primary target, well-tested with QLoRA |
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B` | Requires HF token |
| Phi-3 Mini | `microsoft/Phi-3-mini-4k-instruct` | Fast, runs on smaller GPUs |
| Gemma 2B | `google/gemma-2b` | Good for low-VRAM environments |

## Upcoming Desktop Features

We are actively expanding TuneOS's desktop-native capabilities:
- **Cross-platform Support:** Automatic `.exe`, `AppImage`, and `Snap` packaging via GitHub Actions.
- **Native Notifications:** System tray alerts for when model training jobs finish.
- **Offline Mode:** Tools to process datasets entirely offline and manage the Hugging Face local cache without an internet connection.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started with contributing.
