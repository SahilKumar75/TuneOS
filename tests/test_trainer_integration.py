"""End-to-end trainer integration test on a tiny model.

This actually loads a model, runs a training step, and asserts the loss is
finite and an adapter is written. It is skipped by default because it pulls a
(small) model from the Hub; enable with:

    TUNEOS_INTEGRATION_TESTS=1 poetry run pytest tests/test_trainer_integration.py

Uses `sshleifer/tiny-gpt2` (~2 MB) so it runs in seconds on CPU — no GPU/4-bit.
"""

from __future__ import annotations

import json
import math
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("TUNEOS_INTEGRATION_TESTS") != "1",
    reason="integration test — set TUNEOS_INTEGRATION_TESTS=1 to run",
)

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture
def tiny_dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [{"instruction": f"Question {i}", "output": f"Answer {i}"} for i in range(8)]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def _load_tiny():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    return model, tok


def test_dataset_tokenizes_to_expected_shape(tiny_dataset):
    from trainer.dataset import load_and_tokenize

    _, tok = _load_tiny()
    ds = load_and_tokenize(tiny_dataset, tok, max_seq_length=32)
    assert len(ds) == 8
    assert "input_ids" in ds.column_names
    assert "labels" in ds.column_names
    assert len(ds[0]["input_ids"]) == 32


def test_perplexity_is_finite(tiny_dataset):
    from trainer.dataset import load_and_tokenize
    from trainer.metrics import compute_perplexity

    model, tok = _load_tiny()
    ds = load_and_tokenize(tiny_dataset, tok, max_seq_length=32)
    ppl = compute_perplexity(model, tok, ds)
    assert ppl is not None
    assert math.isfinite(ppl)
    assert ppl > 1.0


def test_train_step_runs_and_writes_adapter(tiny_dataset, tmp_path):
    """Train one short run via SFTTrainer directly and confirm an adapter is saved."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model
    from transformers import TrainingArguments
    from trl import SFTTrainer

    from trainer.dataset import load_and_tokenize
    from trainer.lora import save_adapter

    model, tok = _load_tiny()
    model = get_peft_model(
        model,
        PeftLoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"),
    )
    ds = load_and_tokenize(tiny_dataset, tok, max_seq_length=32)
    out = tmp_path / "run"
    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        max_steps=2,
        report_to="none",
        logging_steps=1,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=args,
        tokenizer=tok,
        dataset_text_field="text",
        max_seq_length=32,
    )
    result = trainer.train()
    assert math.isfinite(result.training_loss)

    save_adapter(model, str(out))
    assert (out / "adapter_config.json").exists()
