"""Tests for trainer.adapter_config.AdapterConfig.

All tests are CPU-only / no GPU required — PEFT is never instantiated here.
"""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest

from trainer.adapter_config import AdapterConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job(technique: str = "qlora", **kwargs) -> SimpleNamespace:
    """Minimal stand-in for a JobConfig with sensible defaults."""
    defaults = dict(
        technique=technique,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        use_all_linear=False,
        adalora_init_r=12,
        adalora_target_r=8,
        ia3_feedforward_modules=[],
        num_virtual_tokens=20,
        task_type="CAUSAL_LM",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# from_job_config round-trip — one per technique
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("technique", ["qlora", "lora", "adalora", "ia3", "prefix", "prompt"])
def test_from_job_config_method_field(technique):
    cfg = AdapterConfig.from_job_config(_job(technique=technique))
    assert cfg.method == technique


def test_qlora_defaults():
    cfg = AdapterConfig.from_job_config(_job("qlora"))
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.bias == "none"
    assert cfg.use_all_linear is False
    assert cfg.target_modules is None


def test_lora_custom_rank():
    cfg = AdapterConfig.from_job_config(_job("lora", lora_rank=64, lora_alpha=128))
    assert cfg.r == 64
    assert cfg.lora_alpha == 128


def test_adalora_specific_fields():
    cfg = AdapterConfig.from_job_config(_job("adalora", adalora_init_r=24, adalora_target_r=16))
    assert cfg.adalora_init_r == 24
    assert cfg.adalora_target_r == 16


def test_ia3_feedforward_modules():
    mods = ["fc1", "fc2"]
    cfg = AdapterConfig.from_job_config(_job("ia3", ia3_feedforward_modules=mods))
    assert cfg.ia3_feedforward_modules == mods


def test_prefix_num_virtual_tokens():
    cfg = AdapterConfig.from_job_config(_job("prefix", num_virtual_tokens=50))
    assert cfg.num_virtual_tokens == 50


def test_prompt_num_virtual_tokens():
    cfg = AdapterConfig.from_job_config(_job("prompt", num_virtual_tokens=10))
    assert cfg.num_virtual_tokens == 10


def test_use_all_linear_propagates():
    cfg = AdapterConfig.from_job_config(_job(use_all_linear=True))
    assert cfg.use_all_linear is True


# ---------------------------------------------------------------------------
# to_lora_cfg_dict — keys match what LoraConfig expects
# ---------------------------------------------------------------------------

_LORA_CFG_KEYS = {"r", "lora_alpha", "lora_dropout", "bias", "task_type",
                  "target_modules", "use_all_linear", "init_lora_weights"}

def test_to_lora_cfg_dict_keys():
    cfg = AdapterConfig.from_job_config(_job())
    assert set(cfg.to_lora_cfg_dict().keys()) == _LORA_CFG_KEYS


def test_to_lora_cfg_dict_values_match_fields():
    cfg = AdapterConfig.from_job_config(_job(lora_rank=32, lora_alpha=64))
    d = cfg.to_lora_cfg_dict()
    assert d["r"] == 32
    assert d["lora_alpha"] == 64
    assert d["task_type"] == "CAUSAL_LM"


# ---------------------------------------------------------------------------
# to_dict — full round-trip
# ---------------------------------------------------------------------------

def test_to_dict_contains_all_fields():
    cfg = AdapterConfig()
    d = cfg.to_dict()
    for f in fields(AdapterConfig):
        assert f.name in d, f"Missing field: {f.name}"


def test_to_dict_round_trip():
    cfg = AdapterConfig(method="adalora", r=8, adalora_init_r=24, adalora_target_r=12)
    d = cfg.to_dict()
    restored = AdapterConfig(**d)
    assert restored == cfg


# ---------------------------------------------------------------------------
# Missing attributes on caller object — graceful fallback
# ---------------------------------------------------------------------------

def test_from_job_config_missing_attrs():
    """from_job_config must not raise when optional fields are absent."""
    minimal = SimpleNamespace(technique="lora", lora_rank=8, lora_alpha=16, lora_dropout=0.1)
    cfg = AdapterConfig.from_job_config(minimal)
    assert cfg.method == "lora"
    assert cfg.r == 8
    # Defaults kick in for everything else
    assert cfg.num_virtual_tokens == 20
