import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from trainer.config import ModelConfig


def _resolve_model_path(cfg: ModelConfig) -> str:
    """Return the model identifier to pass to from_pretrained."""
    if cfg.model_source == "local" and cfg.local_model_path:
        return cfg.local_model_path
    return cfg.model_name


def load_model_and_tokenizer(cfg: ModelConfig):
    """
    Load any Transformers-compatible model with optional 4-bit/8-bit quantization.
    Supports HF Hub IDs, local paths, and any string from_pretrained accepts.
    Returns (model, tokenizer).
    """
    model_path = _resolve_model_path(cfg)
    token = cfg.hf_token or os.getenv("HF_TOKEN") or None
    local_only = cfg.model_source == "local" and os.path.exists(model_path)

    bnb_config = None
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif cfg.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Optional extras: only pass when set so we don't override model defaults.
    extra: dict = {}
    if cfg.attn_implementation:
        extra["attn_implementation"] = cfg.attn_implementation
    if cfg.rope_scaling:
        extra["rope_scaling"] = cfg.rope_scaling

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=torch.float16,
        token=token,
        local_files_only=local_only,
        **extra,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=cfg.trust_remote_code,
        token=token,
        local_files_only=local_only,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer
