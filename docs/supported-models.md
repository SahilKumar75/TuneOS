# Supported Models

TuneOS fine-tunes any Hugging Face causal language model using LoRA/QLoRA, and adds
dedicated pipelines for DPO preference alignment, knowledge distillation, and
vision-language model fine-tuning.

---

## SFT / DPO / KD — Text Models

A curated set ships as one-click presets. Any Hugging Face causal LM also works
because `target_modules` are auto-detected from the model architecture — no manual
configuration needed.

### Curated Presets

| Model | Hugging Face ID | VRAM (QLoRA, approx) | Notes |
|---|---|---|---|
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | ~16 GB | Well tested; good general baseline |
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B` | ~18 GB | Requires HF token (gated) |
| Phi-3 Mini | `microsoft/Phi-3-mini-4k-instruct` | ~8 GB | Fast; runs on smaller GPUs |
| Gemma 2B | `google/gemma-2b` | ~6 GB | Good for low-VRAM environments |
| StarCoder2 3B | `bigcode/starcoder2-3b` | ~8 GB | Code tasks |
| Pythia 410M | `EleutherAI/pythia-410m` | ~2 GB | Best option for testing the pipeline |

### Additional Models (auto-detected)

| Family | Example ID | VRAM (QLoRA, approx) |
|---|---|---|
| Qwen2.5 7B | `Qwen/Qwen2.5-7B` | ~16 GB |
| Llama 3.x 8B | `meta-llama/Llama-3.1-8B` | ~18 GB |
| Gemma 2 9B | `google/gemma-2-9b` | ~20 GB |
| Falcon 7B | `tiiuae/falcon-7b` | ~16 GB |
| GPT-NeoX / Pythia | any `EleutherAI/` checkpoint | varies |

Auto-detection covers Mistral, Llama, Gemma, Phi-3, Falcon, Qwen2/Qwen3, Phi-4,
Cohere, OLMo, StableLM, Mixtral, MPT, StarCoder2, and GPT-BigCode architectures.
Paste any Hub ID and TuneOS will find the right LoRA target modules.

VRAM scales with `batch_size × max_seq_length`. If you hit an out-of-memory error,
the trainer reports a remediation hint (lower batch size or sequence length).

---

## DPO — Preference Datasets

DPO requires a dataset with three columns: `prompt`, `chosen`, and `rejected`. The
column names are configurable in the wizard's step 3 UI.

Any causal LM from the table above works as the base model for DPO. The adapter
configuration (rank, alpha, dropout) is the same as for SFT.

---

## Knowledge Distillation

KD trains a student model to match the output distribution of a teacher. Both student
and teacher must be causal LMs loadable from the Hugging Face Hub. A common pattern
is using a large teacher (e.g. Llama 3 70B) to improve a small student (e.g.
Mistral 7B).

---

## Vision-Language Models

VLM fine-tuning processes image-text datasets via `AutoProcessor`. Any multimodal
checkpoint that `AutoProcessor` can load should work. The pipeline has been tested
with LLaVA-style architectures.

Dataset requirements: a column containing images (PIL or byte strings), an
instruction column, and an output column. Column names are configurable in the wizard.

| Field | Default |
|---|---|
| `image_col` | `"image"` |
| `instruction_col` | `"instruction"` |
| `output_col` | `"output"` |

4-bit quantization (`use_4bit`) is available for VLM training on constrained hardware.

---

## Gated Models

For gated model families (Llama, some Mistral and Gemma variants) you must accept the
license on the Hugging Face Hub and provide `HF_TOKEN` in `.env`. A missing token
causes the job to fail immediately with a clear error message rather than hanging
during the model download.
