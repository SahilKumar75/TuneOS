"""Post-training INT8 dynamic quantization export.

Note on QAT: this project's quantization-aware path is **QLoRA** — the adapter is
trained over a 4-bit NF4-quantized base, so training already accounts for the
quantized weights. This module covers the deployment side: exporting a (merged)
model to dynamic INT8 for lean CPU inference. Heavy imports are lazy.
"""

from __future__ import annotations

import os


def dynamic_quantize_export(model_path: str, output_path: str) -> str:
    """Load the HF model at ``model_path``, apply PyTorch dynamic INT8
    quantization to its ``Linear`` layers, and save the quantized ``state_dict``
    under ``output_path``. Returns the saved file path."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "model_int8.pt")
    torch.save(qmodel.state_dict(), out_file)
    return out_file
