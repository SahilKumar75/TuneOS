"""
Tests for app/api.py REST endpoints.
Uses FastAPI TestClient — no running server needed.
"""

import sys
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

# Mock Celery and Redis before importing the API
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

from app.api import app_api

client = TestClient(app_api)


# ── /health ──────────────────────────────────────────────────────


def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_payload():
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "version" in data


# ── /gpu ─────────────────────────────────────────────────────────


def test_gpu_returns_200():
    resp = client.get("/gpu")
    assert resp.status_code == 200


def test_gpu_payload_shape():
    data = client.get("/gpu").json()
    assert "available" in data
    assert "backend" in data
    assert "name" in data
    assert isinstance(data["available"], bool)


# ── /models ──────────────────────────────────────────────────────


def test_models_returns_200():
    resp = client.get("/models")
    assert resp.status_code == 200


def test_models_returns_list():
    data = client.get("/models").json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_models_each_has_required_fields():
    data = client.get("/models").json()
    for model in data:
        assert "name" in model
        assert "hf_id" in model


def test_models_contains_mistral():
    data = client.get("/models").json()
    hf_ids = [m["hf_id"] for m in data]
    assert "mistralai/Mistral-7B-v0.1" in hf_ids


# ── /jobs ────────────────────────────────────────────────────────


def test_list_jobs_returns_empty_list():
    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_job_returns_201():
    payload = {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "dataset_path": "/tmp/test.jsonl",
    }
    resp = client.post("/jobs", json=payload)
    assert resp.status_code == 201


def test_create_job_returns_job_id():
    payload = {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "dataset_path": "/tmp/test.jsonl",
    }
    data = client.post("/jobs", json=payload).json()
    assert "job_id" in data
    assert len(data["job_id"]) == 36  # UUID format


def test_create_job_defaults():
    payload = {
        "model_id": "google/gemma-2b",
        "dataset_path": "/tmp/data.csv",
    }
    resp = client.post("/jobs", json=payload)
    assert resp.status_code == 201


def test_create_job_missing_model_id_fails():
    resp = client.post("/jobs", json={"dataset_path": "/tmp/data.csv"})
    assert resp.status_code == 422


def test_create_job_without_dataset_path_succeeds():
    # dataset_path is optional — hub dataset jobs omit it
    resp = client.post("/jobs", json={"model_id": "google/gemma-2b"})
    assert resp.status_code == 201


def test_get_job_status_returns_200():
    resp = client.get("/jobs/some-job-id")
    assert resp.status_code == 200


def test_get_job_status_payload_shape():
    data = client.get("/jobs/test-id-123").json()
    assert "job_id" in data
    assert "status" in data


def test_cancel_job_returns_200():
    resp = client.delete("/jobs/some-job-id")
    assert resp.status_code == 200


def test_cancel_job_payload_shape():
    data = client.delete("/jobs/some-job-id").json()
    assert data["job_id"] == "some-job-id"
    assert "status" in data
