# Supported Models

TuneOS fine-tunes decoder-only causal LMs with LoRA / QLoRA. A curated set ships
as one-click presets, but **any** Hugging Face causal LM works because LoRA
`target_modules` are auto-detected from the model architecture (no manual config).

## Curated presets

| Model | HuggingFace Hub ID | VRAM (QLoRA, approx) |
|---|---|---|
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | ~16 GB |
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B` | ~18 GB |
| Phi-3 Mini | `microsoft/Phi-3-mini-4k-instruct` | ~8 GB |
| Gemma 2B | `google/gemma-2b` | ~6 GB |
| StarCoder2 3B | `bigcode/starcoder2-3b` | ~8 GB |
| Pythia 410M | `EleutherAI/pythia-410m` | ~2 GB (great for testing the pipeline) |

## Beyond the presets

Paste any Hub id and TuneOS detects the right LoRA target modules per
architecture. Auto-detection covers (among others) Mistral/Llama, Gemma,
Phi-3, Falcon, Qwen2, and GPT-NeoX/Pythia families. Rough additional guidance:

| Family | Example id | VRAM (QLoRA, approx) |
|---|---|---|
| Qwen2.5 7B | `Qwen/Qwen2.5-7B` | ~16 GB |
| Llama 3.x 8B | `meta-llama/Llama-3.1-8B` | ~18 GB |
| Gemma 2 9B | `google/gemma-2-9b` | ~20 GB |

VRAM scales with `batch_size × max_seq_length`; lower either if you hit OOM (the
trainer reports an OOM with a remediation hint).

## Gated models

For gated families (Llama, some Mistral/Gemma variants) you must accept the
license on the Hugging Face Hub **and** provide `HF_TOKEN` in `.env`. A missing
token fails the job immediately with a clear message.
