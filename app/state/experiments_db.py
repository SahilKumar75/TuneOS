"""Pure SQLite persistence for experiment tracking.

This module is intentionally free of any Reflex (UI) dependency so that the
Celery worker and trainer — which run in a headless backend process — can
persist run metrics and job status without importing the UI framework.

The Reflex-facing layer (``experiment_state.py``) re-exports these helpers
and adds the ``rx.State`` class on top.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.getenv("EXPERIMENT_DB", os.path.join(_PROJECT_ROOT, "storage", "experiments.db"))


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                name TEXT,
                model_id TEXT,
                model_source TEXT,
                technique TEXT,
                epochs INTEGER,
                learning_rate TEXT,
                lora_r INTEGER,
                batch_size INTEGER,
                dataset_name TEXT,
                user_intent TEXT,
                final_loss REAL,
                perplexity REAL,
                started_at TEXT,
                finished_at TEXT,
                status TEXT,
                output_path TEXT,
                loss_history TEXT
            )
        """)
        # Step-level metrics table for queryable run history
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_metrics (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL,
                step INTEGER NOT NULL DEFAULT 0,
                timestamp REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (run_id, key, step)
            )
        """)
        # Immutable hyperparameter snapshot per run
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_params (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (run_id, key)
            )
        """)


def save_run_metrics(run_id: str, loss_history: list[dict[str, Any]]) -> None:
    """Unpack a loss_history list into per-step rows in run_metrics.

    Each entry must have at least {step, loss}; optional keys: epoch,
    learning_rate, eval_loss.  Safe to call multiple times — uses
    INSERT OR REPLACE so re-runs overwrite stale rows.
    """
    if not loss_history:
        return
    ts = time.time()
    rows = []
    for pt in loss_history:
        step = int(pt.get("step", 0))
        for key in ("loss", "eval_loss", "learning_rate", "epoch"):
            val = pt.get(key)
            if val is not None:
                rows.append((run_id, key, float(val), step, ts))
    if not rows:
        return
    try:
        _init_db()
        with _get_conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO run_metrics (run_id, key, value, step, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
    except Exception:
        pass


def save_run_params(run_id: str, params: dict[str, Any]) -> None:
    """Persist hyperparameter key/value pairs for a run (immutable snapshot)."""
    if not params:
        return
    rows = [(run_id, str(k), str(v)) for k, v in params.items()]
    try:
        _init_db()
        with _get_conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO run_params (run_id, key, value) VALUES (?, ?, ?)",
                rows,
            )
    except Exception:
        pass


def write_job_status(
    run_id: str,
    status: str,
    *,
    name: str = "",
    model_id: str = "",
    started_at: str = "",
    finished_at: str = "",
    output_path: str = "",
    error: str = "",
) -> None:
    """Durable (SQLite) job lifecycle record — complementary to ephemeral Redis status.

    Called by the worker at job start, completion, and failure so the API
    can serve job state even when Redis is unavailable.
    """
    try:
        _init_db()
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO runs (id, name, model_id, status, started_at, finished_at, output_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    finished_at=COALESCE(NULLIF(excluded.finished_at,''), finished_at),
                    output_path=COALESCE(NULLIF(excluded.output_path,''), output_path)
                """,
                (run_id, name, model_id, status, started_at, finished_at, output_path),
            )
    except Exception:
        pass


def save_experiment_run(run_data: dict[str, Any]):
    """Insert/replace a full run record. Called from the UI after training."""
    try:
        _init_db()
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (id, name, model_id, model_source, technique, epochs, learning_rate,
                 lora_r, batch_size, dataset_name, user_intent, final_loss, perplexity,
                 started_at, finished_at, status, output_path, loss_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_data.get("id", ""),
                    run_data.get("name", ""),
                    run_data.get("model_id", ""),
                    run_data.get("model_source", "hub"),
                    run_data.get("technique", "qlora"),
                    run_data.get("epochs", 3),
                    run_data.get("learning_rate", "2e-4"),
                    run_data.get("lora_r", 16),
                    run_data.get("batch_size", 4),
                    run_data.get("dataset_name", ""),
                    run_data.get("user_intent", ""),
                    run_data.get("final_loss", 0.0),
                    run_data.get("perplexity", 0.0),
                    run_data.get("started_at", ""),
                    run_data.get("finished_at", ""),
                    run_data.get("status", "unknown"),
                    run_data.get("output_path", ""),
                    json.dumps(run_data.get("loss_history", [])),
                ),
            )
    except Exception:
        pass
