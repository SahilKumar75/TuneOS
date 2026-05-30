"""
Tests for trainer/evaluate.py.
torch, transformers, and the evaluate library are mocked.
"""

import sys
from unittest.mock import MagicMock

import pytest

# Mock heavy ML deps before importing the module
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("evaluate", MagicMock())

# Pre-import so patch() can resolve dotted paths
import trainer.evaluate as _evaluate_mod  # noqa: E402


class TestEvaluateModel:
    """Tests for trainer.evaluate.evaluate_model."""

    def _make_mocks(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()
        return mock_model, mock_tokenizer, mock_dataset

    def test_returns_dict_with_perplexity_key(self):
        mock_model, mock_tokenizer, mock_dataset = self._make_mocks()
        result = _evaluate_mod.evaluate_model(mock_model, mock_tokenizer, mock_dataset)
        assert "perplexity" in result

    def test_returns_dict_with_bleu_key(self):
        mock_model, mock_tokenizer, mock_dataset = self._make_mocks()
        result = _evaluate_mod.evaluate_model(mock_model, mock_tokenizer, mock_dataset)
        assert "bleu" in result

    def test_sets_model_to_eval_mode(self):
        mock_model, mock_tokenizer, mock_dataset = self._make_mocks()
        _evaluate_mod.evaluate_model(mock_model, mock_tokenizer, mock_dataset)
        mock_model.eval.assert_called_once()

    def test_does_not_raise(self):
        mock_model, mock_tokenizer, mock_dataset = self._make_mocks()
        try:
            _evaluate_mod.evaluate_model(mock_model, mock_tokenizer, mock_dataset)
        except Exception as exc:
            pytest.fail(f"evaluate_model raised unexpectedly: {exc}")
