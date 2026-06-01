"""Merge a LoRA adapter into its base model and optionally export as GGUF."""

import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(
    base_model_id: str,
    adapter_path: str,
    output_path: str,
    hf_token: str = "",
) -> str:
    """
    Load base model in full precision, apply the LoRA adapter, merge weights,
    and save the resulting standalone model to output_path.

    Returns output_path on success.
    """
    from peft import PeftModel

    token = hf_token or os.getenv("HF_TOKEN") or None
    is_local = os.path.exists(base_model_id)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
        local_files_only=is_local,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        token=token,
        local_files_only=is_local,
    )
    tokenizer.save_pretrained(output_path)

    return output_path


def export_gguf(
    merged_model_path: str,
    output_dir: str,
    quant_type: str = "Q4_K_M",
) -> str:
    """
    Convert a merged safetensors model to GGUF using llama.cpp's convert script.
    Requires llama-cpp-python or the llama.cpp binary to be installed.

    Returns path to the .gguf file on success, raises RuntimeError otherwise.
    """
    import subprocess
    import sys

    # Try to find llama.cpp convert script via llama-cpp-python package
    try:
        import llama_cpp
        convert_script = os.path.join(
            os.path.dirname(llama_cpp.__file__), "convert_hf_to_gguf.py"
        )
    except ImportError:
        convert_script = shutil.which("convert_hf_to_gguf.py") or ""

    if not convert_script or not os.path.exists(convert_script):
        raise RuntimeError(
            "llama.cpp convert_hf_to_gguf.py not found. "
            "Install llama-cpp-python or add llama.cpp to PATH."
        )

    os.makedirs(output_dir, exist_ok=True)
    gguf_path = os.path.join(output_dir, f"model-{quant_type.lower()}.gguf")

    # Step 1: Convert to f16 GGUF
    f16_path = os.path.join(output_dir, "model-f16.gguf")
    subprocess.run(
        [sys.executable, convert_script, merged_model_path, "--outfile", f16_path],
        check=True,
    )

    # Step 2: Quantize (requires llama-quantize binary)
    quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
    if quantize_bin:
        subprocess.run(
            [quantize_bin, f16_path, gguf_path, quant_type],
            check=True,
        )
        os.remove(f16_path)
        return gguf_path
    else:
        # Return f16 if quantize binary not available
        return f16_path
