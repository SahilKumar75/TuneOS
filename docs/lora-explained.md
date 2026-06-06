# LoRA & QLoRA Explained

## LoRA (Low-Rank Adaptation)
LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. This greatly reduces the number of trainable parameters for downstream tasks.

## QLoRA (Quantized LoRA)
QLoRA takes this a step further by quantizing the base model to 4-bit precision using NormalFloat4 (NF4) and double quantization. The adapter weights are still kept in higher precision (e.g. FP16/BF16) and trained. This drastically cuts down the VRAM usage.

**Key Configs:**
- **Rank (r):** Defines the dimension of the injected matrices. Higher values give the adapter more representation capacity but use more memory.
- **Alpha:** Scaling factor.
- **Dropout:** Used for regularization to prevent overfitting.

`target_modules` (which projections the adapter attaches to) are auto-detected
from the model architecture, so the same config works across Mistral, Llama,
Gemma, Phi-3, Falcon, Qwen2, and GPT-NeoX families.

## DPO (preview)

LoRA/QLoRA above is supervised fine-tuning (SFT) — the model learns to imitate
reference outputs. **Direct Preference Optimization (DPO)** is a planned
alternative recipe that trains on *preference* data — triples of
`(prompt, chosen, rejected)` — to push the model toward responses people prefer,
without a separate reward model. It reuses the same LoRA adapter machinery; only
the dataset shape and loss differ. A full DPO guide lands with the DPO recipe.
