"""
Tests for workers/status.py and workers/train_task.py.
All Redis calls are mocked — no live Redis needed.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock all heavy/unavailable deps before importing workers modules
for _mod in [
    "redis",
    "celery",
    "celery.app",
    "celery.signals",
    "peft",
    "torch",
    "torch.nn",
    "torch.cuda",
    "transformers",
    "transformers.trainer_utils",
    "datasets",
    "accelerate",
    "trl",
    "bitsandbytes",
    "trainer.lora",
    "trainer.qlora",
    "trainer.dataset",
    "trainer.finetune",
    "trainer.callbacks",
    "trainer.config",
    "trainer.loader",
]:
    sys.modules.setdefault(_mod, MagicMock())

# Sibling test modules (collected earlier, alphabetically) may have inserted
# MagicMock stand-ins for the workers package into sys.modules. Evict them so
# the real implementations under test are imported here, with their heavy
# dependencies mocked above.
for _stale in ["workers", "workers.status", "workers.train_task", "workers.celery_app"]:
    sys.modules.pop(_stale, None)

# Pre-import so patch() can resolve the dotted paths
import workers.status  # noqa: E402
import workers.train_task  # noqa: E402

# ── workers/status.py ───────────────────────────────────────────


class TestGetJobStatus:
    """Tests for workers.status.get_job_status."""

    def _mock_redis(self, return_value):
        """Return a context manager that patches redis.from_url."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = return_value
        return patch("workers.status.redis.from_url", return_value=mock_redis)

    def test_returns_not_found_when_key_missing(self):
        with self._mock_redis(None):
            from workers.status import get_job_status

            result = get_job_status("missing-id")
        assert result["status"] == "not_found"
        assert result["job_id"] == "missing-id"

    def test_returns_parsed_status_when_key_exists(self):
        payload = json.dumps({"status": "running", "job_id": "abc-123"})
        with self._mock_redis(payload.encode()):
            from workers.status import get_job_status

            result = get_job_status("abc-123")
        assert result["status"] == "running"
        assert result["job_id"] == "abc-123"

    def test_returns_done_status(self):
        payload = json.dumps(
            {
                "status": "done",
                "job_id": "xyz",
                "output_path": "/outputs/xyz",
            }
        )
        with self._mock_redis(payload.encode()):
            from workers.status import get_job_status

            result = get_job_status("xyz")
        assert result["status"] == "done"
        assert result["output_path"] == "/outputs/xyz"

    def test_returns_failed_status(self):
        payload = json.dumps(
            {
                "status": "failed",
                "job_id": "bad-job",
                "error": "OOM",
            }
        )
        with self._mock_redis(payload.encode()):
            from workers.status import get_job_status

            result = get_job_status("bad-job")
        assert result["status"] == "failed"
        assert "error" in result

    def test_uses_correct_redis_key(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        with patch("workers.status.redis.from_url", return_value=mock_redis):
            from workers.status import get_job_status

            get_job_status("my-job")
        mock_redis.get.assert_called_once_with("job:my-job:status")


# ── workers/train_task.py ────────────────────────────────────────


class TestRunFinetuneTask:
    """Tests for workers.train_task.run_finetune (Celery task)."""

    def _base_configs(self):
        model_cfg = {
            "model_name": "google/gemma-2b",
            "use_4bit": True,
            "use_8bit": False,
            "trust_remote_code": False,
        }
        lora_cfg = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        }
        train_cfg = {
            "output_dir": "/tmp/test-output",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "learning_rate": 2e-4,
        }
        return model_cfg, lora_cfg, train_cfg

    def test_sets_running_status_on_start(self):
        model_cfg, lora_cfg, train_cfg = self._base_configs()
        mock_redis = MagicMock()
        mock_redis.getdel.return_value = None
        mock_finetune = MagicMock(return_value=("/outputs/job1", MagicMock(), MagicMock()))

        with (
            patch("workers.train_task.redis.from_url", return_value=mock_redis),
            patch("workers.train_task.finetune", mock_finetune),
        ):
            workers.train_task._run_finetune_impl(
                MagicMock(), "job1", model_cfg, lora_cfg, train_cfg, "/tmp/data.jsonl"
            )

        first_set_call = mock_redis.set.call_args_list[0]
        first_status = json.loads(first_set_call[0][1])
        assert first_status["status"] == "running"

    def test_sets_done_status_on_success(self):
        model_cfg, lora_cfg, train_cfg = self._base_configs()
        mock_redis = MagicMock()
        mock_redis.getdel.return_value = None
        mock_finetune = MagicMock(return_value=("/outputs/job2", MagicMock(), MagicMock()))

        with (
            patch("workers.train_task.redis.from_url", return_value=mock_redis),
            patch("workers.train_task.finetune", mock_finetune),
        ):
            result = workers.train_task._run_finetune_impl(
                MagicMock(), "job2", model_cfg, lora_cfg, train_cfg, "/tmp/data.jsonl"
            )

        assert result == "/outputs/job2"
        last_set_call = mock_redis.set.call_args_list[-1]
        last_status = json.loads(last_set_call[0][1])
        assert last_status["status"] == "done"
        assert last_status["output_path"] == "/outputs/job2"

    def test_sets_failed_status_on_exception(self):
        model_cfg, lora_cfg, train_cfg = self._base_configs()
        mock_redis = MagicMock()
        mock_redis.getdel.return_value = None

        with (
            patch("workers.train_task.redis.from_url", return_value=mock_redis),
            patch("workers.train_task.finetune", side_effect=RuntimeError("OOM")),
        ):
            with pytest.raises(RuntimeError):
                workers.train_task._run_finetune_impl(
                    MagicMock(), "job3", model_cfg, lora_cfg, train_cfg, "/tmp/data.jsonl"
                )

        last_set_call = mock_redis.set.call_args_list[-1]
        last_status = json.loads(last_set_call[0][1])
        assert last_status["status"] == "failed"
        assert "OOM" in last_status["error"]
