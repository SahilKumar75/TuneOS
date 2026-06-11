from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

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
    "qwen3": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "phi4": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "cohere": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "olmo": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "stablelm": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "mixtral": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "mpt": ["Wqkv"],
    "starcoder2": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "gpt_bigcode": ["c_attn"],
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
    # Attention kernel passed to from_pretrained, e.g. "flash_attention_2",
    # "sdpa", "eager". Empty lets Transformers pick its default.
    attn_implementation: str = ""
    # Optional RoPE scaling to extend context, e.g. {"type": "linear", "factor": 2.0}.
    rope_scaling: dict | None = None


@dataclass
class LoraConfig:
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # scaling factor
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # None → auto-detected from model.config.model_type in inject_lora()
    target_modules: list[str] | None = None
    # True → pass target_modules="all-linear" to PEFT, skipping arch-map lookup
    use_all_linear: bool = False
    # PEFT adapter init strategy: True (default), "gaussian", or "pissa" etc.
    init_lora_weights: str | bool = True


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
    # ── Phase 2: validation + resumption ──────────────────────────
    # Fraction of the dataset held out for in-training validation. 0 disables.
    eval_split_ratio: float = 0.1
    # Stop after this many evals with no improvement. 0 disables early stopping.
    early_stopping_patience: int = 0
    # Evaluate every N steps (instead of once per epoch) when >0 and a validation
    # split exists — gives a denser eval_loss curve.
    eval_steps: int = 0
    # Path to a checkpoint dir to resume from, or True to auto-detect the latest
    # checkpoint under output_dir. None/"" starts fresh.
    resume_from_checkpoint: str | bool | None = None
    # Metrics computed post-training; names must exist in trainer.metrics.REGISTRY.
    eval_metrics: list[str] | None = None
    # Seed for every source of randomness (split, data shuffle, init) so runs are
    # reproducible — the precondition for experiment tracking to be meaningful.
    seed: int = 42
    # Opt-in PyTorch 2.0 compilation of the model for faster training. Off by
    # default so CPU/CI and GPUs without a working dynamo backend are unaffected.
    use_torch_compile: bool = False
    # Where training runs. Routing happens in workers/train_task.py.
    compute_backend: Literal["local", "modal", "hf_spaces"] = "local"
    # Experiment-tracker integration for HF Trainer, e.g. "none" (default),
    # "wandb", "tensorboard". Passed straight to TrainingArguments(report_to=).
    report_to: str = "none"
    # Prompt template name from trainer.dataset.PROMPT_TEMPLATES
    # (alpaca/chatml/llama3/phi3/zephyr).
    prompt_template: str = "alpaca"
    # Sample packing: concatenate examples up to max_seq_length for higher GPU
    # efficiency. When True the trainer tokenizes raw text itself.
    packing: bool = False
    # Adapter technique: one of "qlora" | "lora" | "adalora" | "ia3" | "prefix" | "prompt"
    technique: str = "qlora"
    # Multi-GPU sharded training via PyTorch FSDP. Empty disables it; otherwise a
    # space-separated option string, e.g. "full_shard auto_wrap". Passed straight
    # to TrainingArguments(fsdp=, fsdp_config=).
    fsdp: str = ""
    fsdp_config: dict | None = None


@dataclass
class DistillConfig:
    """Knowledge-distillation config — trains a (LoRA) student to match a frozen
    teacher's soft logits (KL) blended with the hard-label cross-entropy."""

    output_dir: str = "./outputs"
    teacher_model: str = ""  # HF id / path of the (larger) teacher
    temperature: float = 2.0  # softens teacher/student logits
    alpha: float = 0.5  # weight on distillation loss vs. hard-label CE (0..1)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    bf16: bool = False
    seed: int = 42
    prompt_template: str = "alpaca"


@dataclass
class DPOConfig:
    """Direct Preference Optimization config (trl.DPOTrainer).

    Trains a LoRA adapter on (prompt, chosen, rejected) preference triples.
    """

    output_dir: str = "./outputs"
    # KL penalty strength — higher keeps the policy closer to the reference.
    beta: float = 0.1
    max_length: int = 1024
    max_prompt_length: int = 512
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    bf16: bool = False
    seed: int = 42
