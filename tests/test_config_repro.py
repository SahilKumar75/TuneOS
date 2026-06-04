"""Reproducibility/seeding and torch_compile config defaults.

Pure-dataclass assertions — no torch/datasets imports, so these are safe to run
alongside the torch-mocking unit tests. The end-to-end wiring (seed determinism,
TrainingArguments threading) lives in tests/test_trainer_integration.py.
"""

from __future__ import annotations

from trainer.config import TrainingConfig


def test_training_config_has_seed_default():
    assert TrainingConfig().seed == 42


def test_training_config_torch_compile_off_by_default():
    assert TrainingConfig().use_torch_compile is False


def test_seed_is_overridable():
    assert TrainingConfig(seed=123).seed == 123
