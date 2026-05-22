# LoRA & QLoRA Explained

## LoRA (Low-Rank Adaptation)
LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. This greatly reduces the number of trainable parameters for downstream tasks.

## QLoRA (Quantized LoRA)
QLoRA takes this a step further by quantizing the base model to 4-bit precision using NormalFloat4 (NF4) and double quantization. The adapter weights are still kept in higher precision (e.g. FP16/BF16) and trained. This drastically cuts down the VRAM usage.

**Key Configs:**
- **Rank (r):** Defines the dimension of the injected matrices. Higher values give the adapter more representation capacity but use more memory.
- **Alpha:** Scaling factor.
- **Dropout:** Used for regularization to prevent overfitting.
