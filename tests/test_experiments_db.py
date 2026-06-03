"""Tests for app/state/experiments_db — SQLite backend (default)."""

from __future__ import annotations

import os
import sys

import pytest

# Stub heavy deps before importing anything from the app package.
for _mod in [
    "reflex",
    "torch",
    "transformers",
    "peft",
    "datasets",
    "accelerate",
    "trl",
    "bitsandbytes",
    "celery",
    "redis",
    "httpx",
]:
    sys.modules.setdefault(_mod, type(sys)(_mod))

import importlib

# ── Helpers ────────────────────────────────────────────────────────


def _reload_db_module(tmp_path):
    """Re-import experiments_db pointing at a fresh temp DB file."""
    db_file = str(tmp_path / "test_experiments.db")
    os.environ["EXPERIMENT_DB"] = db_file
    os.environ.pop("EXPERIMENTS_DB_URL", None)

    import app.state.experiments_db as _old

    importlib.reload(_old)
    import app.state.experiments_db as db

    importlib.reload(db)
    db._init_db()
    return db


# ── _adapt_sql ─────────────────────────────────────────────────────


def test_adapt_sql_sqlite_passthrough():
    import app.state.experiments_db as db

    sql = "SELECT * FROM runs WHERE id = ?"
    assert db._adapt_sql(sql) == sql


def test_adapt_sql_postgres_replaces():
    """Verify placeholder replacement when _USE_POSTGRES is toggled."""
    import app.state.experiments_db as db

    original = db._USE_POSTGRES
    try:
        db._USE_POSTGRES = True
        assert db._adapt_sql("INSERT INTO t VALUES (?, ?)") == "INSERT INTO t VALUES (%s, %s)"
    finally:
        db._USE_POSTGRES = original


# ── _get_conn / _init_db ───────────────────────────────────────────


def test_init_db_creates_tables(tmp_path):
    db = _reload_db_module(tmp_path)
    with db._get_conn() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"runs", "run_metrics", "run_params", "registered_models"}.issubset(tables)


# ── save_run_metrics ───────────────────────────────────────────────


def test_save_run_metrics_roundtrip(tmp_path):
    db = _reload_db_module(tmp_path)
    history = [
        {"step": 0, "loss": 2.5, "epoch": 1.0, "learning_rate": 2e-4},
        {"step": 1, "loss": 2.1, "epoch": 1.0, "learning_rate": 2e-4},
    ]
    db.save_run_metrics("run-abc", history)
    result = db.get_run_metrics(["run-abc"], metric_key="loss")
    assert "run-abc" in result
    assert len(result["run-abc"]) == 2
    values = [pt["value"] for pt in result["run-abc"]]
    assert pytest.approx(values[0], abs=1e-6) == 2.5
    assert pytest.approx(values[1], abs=1e-6) == 2.1


def test_save_run_metrics_idempotent(tmp_path):
    db = _reload_db_module(tmp_path)
    history = [{"step": 0, "loss": 1.9}]
    db.save_run_metrics("run-idem", history)
    db.save_run_metrics("run-idem", history)  # second write must not duplicate
    result = db.get_run_metrics(["run-idem"], metric_key="loss")
    assert len(result["run-idem"]) == 1


# ── save_run_params ────────────────────────────────────────────────


def test_save_run_params_roundtrip(tmp_path):
    db = _reload_db_module(tmp_path)
    db.save_run_params("run-p1", {"lr": "2e-4", "epochs": 3, "lora_r": 16})
    with db._get_conn() as conn:
        rows = conn.execute(
            "SELECT key, value FROM run_params WHERE run_id = ?", ("run-p1",)
        ).fetchall()
    params = {r["key"]: r["value"] for r in rows}
    assert params["lr"] == "2e-4"
    assert params["epochs"] == "3"
    assert params["lora_r"] == "16"


# ── save_experiment_run ────────────────────────────────────────────


def test_save_experiment_run_upsert(tmp_path):
    db = _reload_db_module(tmp_path)
    run = {
        "id": "run-exp1",
        "name": "my-run",
        "model_id": "mistralai/Mistral-7B-v0.1",
        "model_source": "hub",
        "technique": "qlora",
        "epochs": 3,
        "learning_rate": "2e-4",
        "lora_r": 16,
        "batch_size": 4,
        "dataset_name": "alpaca",
        "user_intent": "chatbot",
        "final_loss": 1.23,
        "perplexity": 5.6,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T01:00:00Z",
        "status": "done",
        "output_path": "/tmp/out",
        "loss_history": [],
    }
    db.save_experiment_run(run)
    with db._get_conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", ("run-exp1",)).fetchone()
    assert row["name"] == "my-run"
    assert pytest.approx(row["final_loss"], abs=1e-6) == 1.23

    # Upsert: update perplexity
    run["perplexity"] = 4.0
    db.save_experiment_run(run)
    with db._get_conn() as conn:
        row2 = conn.execute("SELECT * FROM runs WHERE id = ?", ("run-exp1",)).fetchone()
    assert pytest.approx(row2["perplexity"], abs=1e-6) == 4.0


# ── write_job_status ───────────────────────────────────────────────


def test_write_job_status_insert_and_update(tmp_path):
    db = _reload_db_module(tmp_path)
    db.write_job_status("job-1", "running", name="test-job", model_id="m1", started_at="t0")
    with db._get_conn() as conn:
        row = conn.execute("SELECT status FROM runs WHERE id = ?", ("job-1",)).fetchone()
    assert row["status"] == "running"

    db.write_job_status("job-1", "done", finished_at="t1")
    with db._get_conn() as conn:
        row2 = conn.execute("SELECT status FROM runs WHERE id = ?", ("job-1",)).fetchone()
    assert row2["status"] == "done"


# ── register_model / list_registered_models ────────────────────────


def test_register_model_roundtrip(tmp_path):
    db = _reload_db_module(tmp_path)
    db.register_model("prod-v1", "run-abc", alias="latest", metric_snapshot={"perplexity": 3.2})
    models = db.list_registered_models()
    assert len(models) == 1
    assert models[0]["name"] == "prod-v1"
    assert models[0]["run_id"] == "run-abc"
    assert models[0]["metric_snapshot"]["perplexity"] == pytest.approx(3.2, abs=1e-6)


def test_register_model_upsert(tmp_path):
    db = _reload_db_module(tmp_path)
    db.register_model("prod-v1", "run-old", alias="latest")
    db.register_model("prod-v1", "run-new", alias="champion")
    models = db.list_registered_models()
    assert len(models) == 1
    assert models[0]["run_id"] == "run-new"
    assert models[0]["alias"] == "champion"


# ── get_run_metrics — multiple runs ────────────────────────────────


def test_get_run_metrics_multiple_runs(tmp_path):
    db = _reload_db_module(tmp_path)
    db.save_run_metrics("r1", [{"step": 0, "loss": 2.0}, {"step": 1, "loss": 1.8}])
    db.save_run_metrics("r2", [{"step": 0, "loss": 2.5}])
    result = db.get_run_metrics(["r1", "r2"], metric_key="loss")
    assert len(result["r1"]) == 2
    assert len(result["r2"]) == 1


def test_get_run_metrics_empty_ids(tmp_path):
    db = _reload_db_module(tmp_path)
    assert db.get_run_metrics([]) == {}


# ── list_runs ──────────────────────────────────────────────────────


def test_list_runs(tmp_path):
    db = _reload_db_module(tmp_path)
    db.write_job_status("j1", "done")
    db.write_job_status("j2", "running")
    runs = db.list_runs()
    ids = [r["job_id"] for r in runs]
    assert "j1" in ids
    assert "j2" in ids
