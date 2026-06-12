import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure heavy deps are mocked before any test imports them. setdefault is
# idempotent so tests that import this module multiple times stay safe.
_mock_task = MagicMock()
_mock_task.apply_async = MagicMock(return_value=MagicMock(id="test-task-id"))
_mock_celery = MagicMock()
_mock_celery.task = MagicMock(return_value=lambda f: _mock_task)

sys.modules.setdefault("celery", MagicMock())
sys.modules.setdefault("redis", MagicMock())
sys.modules.setdefault("workers.celery_app", MagicMock(celery_app=_mock_celery))
sys.modules.setdefault("workers.train_task", MagicMock(run_finetune=_mock_task))
sys.modules.setdefault(
    "workers.status",
    MagicMock(get_job_status=MagicMock(return_value={"status": "running", "job_id": "x"})),
)


@pytest.fixture(scope="session")
def test_client():
    from fastapi.testclient import TestClient

    from app.api import app_api

    return TestClient(app_api)


@pytest.fixture
def tmp_dataset_dir(tmp_path):
    """A temporary directory pre-populated with a minimal JSONL dataset."""
    ds = tmp_path / "dataset.jsonl"
    ds.write_text('{"instruction": "hello", "output": "world"}\n')
    return tmp_path


@pytest.fixture
def mock_redis(monkeypatch):
    """A MagicMock Redis client wired into the Redis URL env var."""
    client = MagicMock()
    client.get.return_value = None
    client.getdel.return_value = None
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    import redis

    monkeypatch.setattr(redis, "from_url", MagicMock(return_value=client))
    return client
