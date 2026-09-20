"""Tests for trainer/dataset.py — prompt formatting and column validation.

Column validation happens before the tokenizer is touched, so these tests pass
a dummy tokenizer that would error if it were ever called.
"""

import csv
import json

import pytest

from trainer.dataset import _summarize_truncation, format_prompt, load_and_tokenize


def test_format_prompt_uses_instruction_and_output():
    text = format_prompt({"instruction": "Say hi", "output": "Hi!"})
    assert "Say hi" in text
    assert "Hi!" in text
    assert "### Instruction:" in text
    assert "### Response:" in text


def test_format_prompt_custom_columns():
    text = format_prompt(
        {"prompt": "Q", "answer": "A"}, instruction_col="prompt", output_col="answer"
    )
    assert "Q" in text
    assert "A" in text


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_load_and_tokenize_raises_on_missing_instruction_column(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, ["question", "output"], [["q1", "a1"]])

    def _boom(*_a, **_k):  # tokenizer must never be reached
        raise AssertionError("tokenizer should not be called when columns are invalid")

    with pytest.raises(ValueError, match="Instruction column"):
        load_and_tokenize(str(csv_path), _boom, instruction_col="instruction")


def test_load_and_tokenize_raises_on_missing_output_column(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, ["instruction", "result"], [["q1", "a1"]])

    def _boom(*_a, **_k):
        raise AssertionError("tokenizer should not be called when columns are invalid")

    with pytest.raises(ValueError, match="Output column"):
        load_and_tokenize(str(csv_path), _boom, output_col="output")


# ── truncation warning (#181) ───────────────────────────────────────


def test_summarize_truncation_returns_none_when_nothing_truncated():
    assert _summarize_truncation([10, 20, 30], max_seq_length=50) is None


def test_summarize_truncation_reports_percentage_and_average_overflow():
    # two samples fit, two exceed max_seq_length by 10 and 20 tokens
    msg = _summarize_truncation([50, 60, 70, 50], max_seq_length=50)
    assert msg is not None
    assert "50.0%" in msg
    assert "2/4" in msg
    assert "max_seq_length=50" in msg
    assert "avg 15.0 tokens" in msg


class _FakeTokenizer:
    """Minimal stand-in for a HF tokenizer: token count = word count (+BOS)."""

    bos_token_id = 0

    def __call__(
        self, texts, truncation=True, max_length=None, padding=None, add_special_tokens=True
    ):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            ids = ([self.bos_token_id] if add_special_tokens else []) + list(
                range(1, len(t.split()) + 1)
            )
            if truncation and max_length is not None:
                ids = ids[:max_length]
            if padding == "max_length" and max_length is not None:
                ids = ids + [0] * (max_length - len(ids))
            out.append(ids)
        return {"input_ids": out}


def test_load_and_tokenize_warns_when_a_sample_is_truncated(caplog):
    from datasets import Dataset

    preloaded = Dataset.from_dict(
        {
            "instruction": ["short one", "this instruction has quite a lot more words in it"],
            "output": ["ok", "also a fairly long response with several words"],
        }
    )

    with caplog.at_level("WARNING", logger="trainer.dataset"):
        load_and_tokenize(
            "unused.json",
            _FakeTokenizer(),
            max_seq_length=6,
            preloaded=preloaded,
        )

    assert any("exceeded max_seq_length=6" in r.message for r in caplog.records)


def test_load_and_tokenize_warns_when_tokenization_is_cached(tmp_path, caplog):
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "instruction": "this instruction contains enough words to overflow",
                "output": "and so does this response",
            }
        )
        + "\n"
    )
    tokenizer = _FakeTokenizer()

    load_and_tokenize(str(jsonl_path), tokenizer, max_seq_length=6)
    caplog.clear()
    with caplog.at_level("WARNING", logger="trainer.dataset"):
        load_and_tokenize(str(jsonl_path), tokenizer, max_seq_length=6)

    assert any("exceeded max_seq_length=6" in r.message for r in caplog.records)
