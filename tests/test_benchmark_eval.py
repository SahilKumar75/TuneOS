"""Interface tests for the optional lm-eval wrapper (no lm-eval required)."""

from __future__ import annotations

import pytest

from trainer.benchmark_eval import DEFAULT_TASKS, lm_eval_available, run_benchmarks


def test_available_returns_bool():
    assert isinstance(lm_eval_available(), bool)


def test_default_tasks_nonempty():
    assert "hellaswag" in DEFAULT_TASKS


def test_run_benchmarks_raises_when_unavailable():
    if not lm_eval_available():
        with pytest.raises(RuntimeError, match="lm-eval"):
            run_benchmarks("/tmp/does-not-matter")
