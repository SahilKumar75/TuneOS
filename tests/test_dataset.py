"""Tests for trainer/dataset.py — prompt formatting and column validation.

Column validation happens before the tokenizer is touched, so these tests pass
a dummy tokenizer that would error if it were ever called.
"""

import csv

import pytest

from trainer.dataset import format_prompt, load_and_tokenize


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
