"""
Tests for trainer/evaluate.py.
torch, transformers, and the evaluate library are mocked.
"""

import sys
from unittest.mock import MagicMock

import pytest

# Mock heavy ML deps before importing the module.
# torch sub-modules must be registered individually so Python's import
# machinery doesn't try to traverse the MagicMock as a real package.
_torch_mock = MagicMock()
for _mod in ["torch", "torch.utils", "torch.utils.data", "torch.nn", "torch.cuda"]:
    sys.modules.setdefault(_mod, _torch_mock)
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


class TestEvaluateReferences:
    """Reference metrics (ROUGE-1/BLEU) are pure-Python and need no mocking."""

    def test_returns_rouge1_and_bleu_floats(self):
        predictions = ["the cat sat on the mat", "hello world"]
        references = ["the cat sat on the mat", "hello there world"]
        result = _evaluate_mod.evaluate_references(predictions, references)
        assert "rouge1" in result and "bleu" in result
        assert isinstance(result["rouge1"], float)
        assert isinstance(result["bleu"], float)

    def test_perfect_match_scores_high(self):
        pairs = ["the quick brown fox"]
        result = _evaluate_mod.evaluate_references(pairs, pairs)
        assert result["rouge1"] == 1.0
