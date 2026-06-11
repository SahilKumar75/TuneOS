"""AdapterConfig — single source of truth for all PEFT adapter parameters.

Replaces the manual dict mapping in ``app/api/jobs_routes._build_finetune_kwargs``
and gives P3 adapter strategies a typed config object to build against.

Design constraints
------------------
* Zero imports from ``app/`` — trainer must stay deployable as a standalone package
  (Modal, HF Spaces) without dragging in Reflex/FastAPI.
* Duck-typed ``from_job_config`` accepts any object with the right attributes so
  callers don't need to import this module's deps to pass data in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AdapterConfig:
    # ── common ────────────────────────────────────────────────────────────────
    method: str = "qlora"  # qlora | lora | adalora | ia3 | prefix | prompt
    task_type: str = "CAUSAL_LM"  # CAUSAL_LM | SEQ_2_SEQ_LM | (vision types P6)

    # ── LoRA / QLoRA / AdaLoRA ───────────────────────────────────────────────
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] | None = None  # None → arch-map lookup in lora.py
    use_all_linear: bool = False  # True → target_modules="all-linear"
    init_lora_weights: str | bool = True

    # ── AdaLoRA-specific ─────────────────────────────────────────────────────
    adalora_init_r: int = 12  # initial rank before adaptive pruning
    adalora_target_r: int = 8  # final rank after pruning

    # ── IA³-specific ─────────────────────────────────────────────────────────
    ia3_feedforward_modules: list[str] = field(default_factory=list)

    # ── Prefix / Prompt tuning ───────────────────────────────────────────────
    num_virtual_tokens: int = 20  # virtual tokens prepended to each input

    # -------------------------------------------------------------------------

    @classmethod
    def from_job_config(cls, config: Any) -> "AdapterConfig":
        """Build from a ``JobConfig`` (or any object with matching attributes).

        Uses ``getattr`` with safe defaults so callers don't need to import
        ``JobConfig`` and this stays circular-import-free.
        """

        def _get(attr: str, default: Any) -> Any:
            return getattr(config, attr, default)

        return cls(
            method=_get("technique", "qlora"),
            task_type=_get("task_type", "CAUSAL_LM"),  # pre-computed by _detect_task_type
            r=_get("lora_rank", 16),
            lora_alpha=_get("lora_alpha", 32),
            lora_dropout=_get("lora_dropout", 0.05),
            bias="none",
            target_modules=None,
            use_all_linear=_get("use_all_linear", False),
            init_lora_weights=True,
            adalora_init_r=_get("adalora_init_r", 12),
            adalora_target_r=_get("adalora_target_r", 8),
            ia3_feedforward_modules=_get("ia3_feedforward_modules", []),
            num_virtual_tokens=_get("num_virtual_tokens", 20),
        )

    def to_lora_cfg_dict(self) -> dict:
        """Return the subset used to construct ``trainer.config.LoraConfig``."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "target_modules": self.target_modules,
            "use_all_linear": self.use_all_linear,
            "init_lora_weights": self.init_lora_weights,
        }

    def to_dict(self) -> dict:
        """Full dataclass → plain dict (useful for serialisation / logging)."""
        return asdict(self)
