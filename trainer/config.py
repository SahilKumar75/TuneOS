from __future__ import annotations

import logging
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

# Enumerate the correct LoRA projection names per architecture rather than
# silently using a default that breaks Gemma / Phi-3 / Falcon.
_TARGET_MODULES_BY_ARCH: dict[str, list[str]] = {
    "mistral": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "llama": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "phi3": ["qkv_proj", "o_proj"],
    "phi": ["q_proj", "v_proj"],
    "gemma": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "gemma2": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "falcon": ["query_key_value"],
    "gpt2": ["c_attn"],
    "gpt_neox": ["query_key_value"],
    "bloom": ["query_key_value"],
    "t5": ["q", "v"],
    "qwen2": ["q_proj", "v_proj", "k_proj", "o_proj"],
}
_DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]


def get_target_modules(model_type: str) -> list[str]:
    """Return LoRA target modules for a given HF model_type string.

    Falls back to a conservative default for unknown architectures, but logs
    a warning so the misconfiguration is discoverable rather than silent.
    """
    key = (model_type or "").lower()
    if key not in _TARGET_MODULES_BY_ARCH:
        _logger.warning(
            "No LoRA target_modules mapping for model_type=%r; falling back to default %s",
            model_type,
            _DEFAULT_TARGET_MODULES,
        )
    return _TARGET_MODULES_BY_ARCH.get(key, _DEFAULT_TARGET_MODULES)


@dataclass
class ModelConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    use_4bit: bool = True
    use_8bit: bool = False
    trust_remote_code: bool = False
    max_seq_length: int = 512
    hf_token: str = ""
    local_model_path: str = ""
    model_source: str = "hub"  # "hub" | "local" | "custom_string"


@dataclass
class LoraConfig:
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # scaling factor
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # None → auto-detected from model.config.model_type in inject_lora()
    target_modules: list[str] | None = None


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 10
    save_steps: int = 100
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 0.3
