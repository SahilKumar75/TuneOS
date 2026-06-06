"""Tests for the DPO data path and config — no torch/trl required.

Tests that construct a real HF ``Dataset`` are gated behind
``TUNEOS_INTEGRATION_TESTS`` because the unit suite mocks torch/transformers in
``sys.modules`` (for the worker tests), which breaks ``datasets`` fingerprinting.
Column-validation and pure-config tests run unconditionally.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from trainer.config import DPOConfig
from trainer.dataset import detect_dataset_type, load_preference_pairs

_needs_real_datasets = pytest.mark.skipif(
    not os.getenv("TUNEOS_INTEGRATION_TESTS"),
    reason="constructs a real HF Dataset; set TUNEOS_INTEGRATION_TESTS=1 to run",
)


def test_detect_dataset_type():
    assert detect_dataset_type(["prompt", "chosen", "rejected"]) == "preference"
    assert detect_dataset_type(["Prompt", "Chosen", "Rejected", "extra"]) == "preference"
    assert detect_dataset_type(["instruction", "output"]) == "instruction"
    assert detect_dataset_type(["prompt", "chosen"]) == "instruction"


@_needs_real_datasets
def test_load_preference_pairs_csv(tmp_path):
    csv = tmp_path / "prefs.csv"
    pd.DataFrame(
        {
            "prompt": ["q1", "q2"],
            "chosen": ["good1", "good2"],
            "rejected": ["bad1", "bad2"],
            "extra": ["x", "y"],
        }
    ).to_csv(csv, index=False)

    ds = load_preference_pairs(str(csv))
    assert sorted(ds.column_names) == ["chosen", "prompt", "rejected"]  # extra dropped
    assert ds[0]["prompt"] == "q1"
    assert ds[0]["chosen"] == "good1"
    assert ds[0]["rejected"] == "bad1"


@_needs_real_datasets
def test_load_preference_pairs_custom_columns(tmp_path):
    csv = tmp_path / "prefs.csv"
    pd.DataFrame(
        {"q": ["a"], "win": ["w"], "lose": ["l"]},
    ).to_csv(csv, index=False)

    ds = load_preference_pairs(str(csv), prompt_col="q", chosen_col="win", rejected_col="lose")
    assert sorted(ds.column_names) == ["chosen", "prompt", "rejected"]
    assert ds[0]["prompt"] == "a"


def test_load_preference_pairs_missing_column(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"prompt": ["a"], "chosen": ["b"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing column"):
        load_preference_pairs(str(csv))


def test_dpo_config_defaults():
    cfg = DPOConfig()
    assert cfg.beta == 0.1
    assert cfg.max_length == 1024
    assert cfg.max_prompt_length == 512
    assert cfg.num_train_epochs == 1
