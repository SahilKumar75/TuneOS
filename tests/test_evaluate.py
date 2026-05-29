"""
Tests for trainer/evaluate.py.
ML dependencies (torch, transformers, evaluate) are mocked.
"""
from unittest.mock import MagicMock, patch


class TestEvaluateModel:
    """Tests for trainer.evaluate.evaluate_model."""

    def _make_mocks(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()
        return mock_model, mock_tokenizer, mock_dataset

    def test_returns_dict_with_perplexity_key(self):
        mock_model, mock_tokenizer, mock_dataset = self._make_mocks()
        with patch("trainer.evaluate.evaluate.load", return_value=MagicMock()):
            from trainer.evaluate import evaluate_model
            result = evaluate_model(mock_model, mock_tokenizer, mock_dataset)
        assert "perplexity" in result

    def test_returns_dict_with_bleu_key(self):
        mock_model, mock_tokenizer, mock_dataset = self._make_mocks()
        with patch("trainer.evaluate.evaluate.load", return_value=MagicMock()):
            from trainer.evaluate import evaluate_model
            result = evaluate_model(mock_model, mock_tokenizer, mock_dataset)
        assert "bleu" in result

    def test_sets_model_to_eval_mode(self):
        mock_model, mock_tokenizer, mock_dataset = self._make_mocks()
        with patch("trainer.evaluate.evaluate.load", return_value=MagicMock()):
            from trainer.evaluate import evaluate_model
            evaluate_model(mock_model, mock_tokenizer, mock_dataset)
        mock_model.eval.assert_called_once()

    def test_does_not_raise(self):
        mock_model, mock_tokenizer, mock_dataset = self._make_mocks()
        with patch("trainer.evaluate.evaluate.load", return_value=MagicMock()):
            from trainer.evaluate import evaluate_model
            try:
                evaluate_model(mock_model, mock_tokenizer, mock_dataset)
            except Exception as exc:
                assert False, f"evaluate_model raised unexpectedly: {exc}"
