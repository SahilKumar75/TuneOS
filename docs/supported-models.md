# Supported Models

This platform supports fine-tuning the following base models via QLoRA:

| Model Name | HuggingFace Hub ID | VRAM Requirements (approx) |
|---|---|---|
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | ~16GB |
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B` | ~18GB |
| Phi-3 Mini | `microsoft/Phi-3-mini-4k-instruct` | ~8GB |
| Gemma 2B | `google/gemma-2b` | ~6GB |

*Note: For Llama 3, you must have accepted the agreement on the Hugging Face hub and provide your `HF_TOKEN` in the `.env` file.*
