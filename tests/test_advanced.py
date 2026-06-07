"""Tests for the advanced P4-F additions — sweep grid, distill/FSDP config."""

from __future__ import annotations

from trainer.config import DistillConfig, TrainingConfig
from trainer.sweep import expand_grid


def test_expand_grid_cartesian_product():
    out = expand_grid({"epochs": 3}, {"lr": [1, 2], "bs": [4, 8]})
    assert len(out) == 4
    assert all(d["epochs"] == 3 for d in out)
    assert {"epochs": 3, "lr": 1, "bs": 4} in out
    assert {"epochs": 3, "lr": 2, "bs": 8} in out


def test_expand_grid_empty_returns_base_copy():
    base = {"a": 1}
    out = expand_grid(base, {})
    assert out == [{"a": 1}]
    out[0]["a"] = 2  # must be a copy, not the original
    assert base["a"] == 1


def test_distill_config_defaults():
    c = DistillConfig()
    assert c.temperature == 2.0
    assert 0.0 <= c.alpha <= 1.0
    assert c.teacher_model == ""


def test_training_config_fsdp_defaults_off():
    cfg = TrainingConfig()
    assert cfg.fsdp == ""
    assert cfg.fsdp_config is None


def test_advanced_modules_importable():
    import trainer.benchmark_eval  # noqa: F401
    import trainer.kd  # noqa: F401
    import trainer.quantize  # noqa: F401
