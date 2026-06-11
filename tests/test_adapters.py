"""Unit tests for trainer/adapters.py — strategy registry."""

from __future__ import annotations

import pytest

from trainer.adapters import ADAPTER_META, REGISTRY, get_strategy

# ── Registry completeness ──────────────────────────────────────────────────────

EXPECTED_TECHNIQUES = {"lora", "qlora", "adalora", "ia3", "prefix", "prompt"}


def test_all_expected_techniques_registered():
    assert EXPECTED_TECHNIQUES <= set(REGISTRY), (
        f"Missing from registry: {EXPECTED_TECHNIQUES - set(REGISTRY)}"
    )


def test_adapter_meta_keys_match_registry():
    """Every entry in REGISTRY should have a corresponding ADAPTER_META entry."""
    assert set(REGISTRY) == set(ADAPTER_META), (
        f"Mismatch: registry={set(REGISTRY)}, meta={set(ADAPTER_META)}"
    )


def test_adapter_meta_has_required_fields():
    required = {"label", "short", "params"}
    for name, meta in ADAPTER_META.items():
        missing = required - set(meta)
        assert not missing, f"ADAPTER_META[{name!r}] missing fields: {missing}"


# ── get_strategy ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("technique", sorted(EXPECTED_TECHNIQUES))
def test_get_strategy_returns_object(technique):
    strategy = get_strategy(technique)
    assert strategy is not None
    assert hasattr(strategy, "prepare"), f"{technique!r} strategy has no prepare() method"


def test_get_strategy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown adapter technique"):
        get_strategy("nonexistent_technique")


def test_get_strategy_unknown_message_lists_supported():
    with pytest.raises(ValueError) as exc_info:
        get_strategy("magic_lora")
    msg = str(exc_info.value)
    for technique in REGISTRY:
        assert technique in msg, f"Error message should list {technique!r} as supported"


def test_get_strategy_same_instance_each_call():
    """Registry returns the same singleton strategy object for repeated calls."""
    assert get_strategy("lora") is get_strategy("lora")
    assert get_strategy("qlora") is get_strategy("qlora")


# ── TrainingConfig technique field ─────────────────────────────────────────────


def test_training_config_default_technique():
    from trainer.config import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.technique == "qlora"


def test_training_config_accepts_all_techniques():
    from trainer.config import TrainingConfig

    for technique in REGISTRY:
        cfg = TrainingConfig(technique=technique)
        assert cfg.technique == technique


# ── Strategy prepare() is callable (duck-type only, no model load) ─────────────


def test_all_strategies_have_callable_prepare():
    for name, strategy in REGISTRY.items():
        assert callable(getattr(strategy, "prepare", None)), (
            f"Strategy {name!r}: prepare() is not callable"
        )
