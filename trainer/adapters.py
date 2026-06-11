"""Adapter strategy registry for TuneOS fine-tuning.

Each strategy encapsulates a PEFT adapter type and exposes a single
``prepare(model_cfg, lora_cfg) -> (model, tokenizer)`` call so that
``trainer/finetune.py`` is fully decoupled from the PEFT internals.

Supported techniques
--------------------
lora    — Standard LoRA (full-precision or 8-bit base model)
qlora   — QLoRA: 4-bit NF4 quantisation + LoRA adapters  (default)
adalora — AdaLoRA: adaptive rank allocation per-layer
ia3     — IA³: scales activations with learned vectors (very few params)
prefix  — Prefix Tuning: learned prefix tokens prepended to the input
prompt  — Prompt Tuning: soft prompt tokens only (even fewer params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from trainer.config import LoraConfig, ModelConfig


# ── Strategy protocol ──────────────────────────────────────────────────────────


class AdapterStrategy(Protocol):
    """Any object with a ``prepare`` method qualifies as an adapter strategy."""

    def prepare(
        self,
        model_cfg: ModelConfig,
        lora_cfg: LoraConfig,
    ) -> tuple:
        """Return *(model, tokenizer)* ready for SFTTrainer / DPOTrainer."""
        ...


# ── Concrete strategies ────────────────────────────────────────────────────────


@dataclass
class _LoRAStrategy:
    """Standard LoRA — base model loaded in full precision (or 8-bit if requested)."""

    def prepare(self, model_cfg: ModelConfig, lora_cfg: LoraConfig) -> tuple:
        from trainer.loader import load_model_and_tokenizer
        from trainer.lora import inject_lora

        model, tokenizer = load_model_and_tokenizer(model_cfg)
        model = inject_lora(model, lora_cfg)
        return model, tokenizer


@dataclass
class _QLoRAStrategy:
    """QLoRA — 4-bit NF4 quantisation + LoRA (most memory-efficient for 7B+ models)."""

    def prepare(self, model_cfg: ModelConfig, lora_cfg: LoraConfig) -> tuple:
        from trainer.qlora import prepare_qlora_model

        return prepare_qlora_model(model_cfg, lora_cfg)


@dataclass
class _AdaLoRAStrategy:
    """AdaLoRA — budget-constrained adaptive rank allocation across layers."""

    def prepare(self, model_cfg: ModelConfig, lora_cfg: LoraConfig) -> tuple:
        from peft import AdaLoraConfig, TaskType, get_peft_model

        from trainer.config import get_target_modules
        from trainer.loader import load_model_and_tokenizer

        # AdaLoRA always builds on top of the same quantised base as LoRA/QLoRA.
        model, tokenizer = load_model_and_tokenizer(model_cfg)

        model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "")
        target_modules = lora_cfg.target_modules or get_target_modules(model_type)

        adalora_config = AdaLoraConfig(
            init_r=lora_cfg.r * 2,  # start with 2× budget; pruned down to r
            target_r=lora_cfg.r,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            deltaT=10,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, adalora_config)
        model.print_trainable_parameters()
        return model, tokenizer


@dataclass
class _IA3Strategy:
    """IA³ — scales keys, values, and feed-forward activations with learned vectors.

    Roughly 10-100× fewer trainable parameters than LoRA for the same model.
    """

    def prepare(self, model_cfg: ModelConfig, lora_cfg: LoraConfig) -> tuple:
        from peft import IA3Config, TaskType, get_peft_model

        from trainer.loader import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(model_cfg)

        ia3_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            # feedforward_modules must be a subset of target_modules; leaving
            # None lets PEFT pick appropriate defaults for the architecture.
            target_modules=lora_cfg.target_modules or None,
            feedforward_modules=None,
        )
        model = get_peft_model(model, ia3_config)
        model.print_trainable_parameters()
        return model, tokenizer


@dataclass
class _PrefixTuningStrategy:
    """Prefix Tuning — prepends learned virtual tokens to the key/value states.

    ``lora_cfg.r`` is reused as the number of virtual prefix tokens.
    """

    def prepare(self, model_cfg: ModelConfig, lora_cfg: LoraConfig) -> tuple:
        from peft import PrefixTuningConfig, TaskType, get_peft_model

        from trainer.loader import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(model_cfg)

        prefix_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=lora_cfg.r,  # reuse rank as token count
            encoder_hidden_size=None,
        )
        model = get_peft_model(model, prefix_config)
        model.print_trainable_parameters()
        return model, tokenizer


@dataclass
class _PromptTuningStrategy:
    """Prompt Tuning — learns soft prompt token embeddings only (fewest params).

    ``lora_cfg.r`` is reused as the number of virtual prompt tokens.
    """

    def prepare(self, model_cfg: ModelConfig, lora_cfg: LoraConfig) -> tuple:
        from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model

        from trainer.loader import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(model_cfg)

        prompt_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=lora_cfg.r,
            prompt_tuning_init_text="Fine-tune this model to follow instructions:",
            tokenizer_name_or_path=model_cfg.model_name,
        )
        model = get_peft_model(model, prompt_config)
        model.print_trainable_parameters()
        return model, tokenizer


# ── Registry ───────────────────────────────────────────────────────────────────

REGISTRY: dict[str, AdapterStrategy] = {
    "lora": _LoRAStrategy(),
    "qlora": _QLoRAStrategy(),
    "adalora": _AdaLoRAStrategy(),
    "ia3": _IA3Strategy(),
    "prefix": _PrefixTuningStrategy(),
    "prompt": _PromptTuningStrategy(),
}

#: Human-readable labels and descriptions for the UI.
ADAPTER_META: dict[str, dict[str, str]] = {
    "qlora": {
        "label": "QLoRA",
        "short": "4-bit + LoRA — best VRAM/accuracy trade-off for 7B+ models",
        "params": "~0.5% trainable",
    },
    "lora": {
        "label": "LoRA",
        "short": "Standard LoRA — full-precision base, lower quantisation noise",
        "params": "~0.5% trainable",
    },
    "adalora": {
        "label": "AdaLoRA",
        "short": "Adaptive rank allocation — prunes unimportant layers automatically",
        "params": "~0.5% trainable",
    },
    "ia3": {
        "label": "IA³",
        "short": "Scale activations only — 10-100× fewer params than LoRA",
        "params": "~0.01% trainable",
    },
    "prefix": {
        "label": "Prefix Tuning",
        "short": "Learned prefix tokens prepended to attention — no weight updates",
        "params": "~0.1% trainable",
    },
    "prompt": {
        "label": "Prompt Tuning",
        "short": "Soft prompt embeddings only — fewest trainable parameters",
        "params": "<0.01% trainable",
    },
}


def get_strategy(technique: str) -> AdapterStrategy:
    """Return the adapter strategy for *technique*, raising ValueError if unknown."""
    strategy = REGISTRY.get(technique)
    if strategy is None:
        supported = ", ".join(REGISTRY)
        raise ValueError(f"Unknown adapter technique {technique!r}. Supported: {supported}")
    return strategy
