"""Pure SQLite/PostgreSQL persistence for experiment tracking.

Top-level package so workers can import without pulling in app/.
The app-side shim at app/state/experiments_db.py re-exports everything here.

Backend selection
-----------------
Default: local SQLite at ``storage/experiments.db``.
Set ``EXPERIMENTS_DB_URL`` to a PostgreSQL DSN to switch to Postgres.
``psycopg2-binary`` must be installed for the Postgres backend.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

_logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.getenv("EXPERIMENT_DB", os.path.join(_PROJECT_ROOT, "storage", "experiments.db"))

_POSTGRES_URL: str = os.getenv("EXPERIMENTS_DB_URL", "")
_USE_POSTGRES: bool = _POSTGRES_URL.startswith(("postgres://", "postgresql://"))


def _adapt_sql(sql: str) -> str:
    """Convert ``?`` parameter markers to ``%s`` for the PostgreSQL driver."""
    if _USE_POSTGRES:
        return sql.replace("?", "%s")
    return sql


class _PgConnAdapter:
    """Wraps a psycopg2 connection + RealDictCursor to look like sqlite3."""

    def __init__(self, conn, cur) -> None:  # type: ignore[type-arg]
        self._conn = conn
        self._cur = cur

    def execute(self, sql: str, params: tuple = ()):
        self._cur.execute(_adapt_sql(sql), params)
        return self._cur

    def executemany(self, sql: str, params_seq):
        self._cur.executemany(_adapt_sql(sql), params_seq)
        return self._cur


@contextmanager
def _get_conn() -> Generator[Any, None, None]:
    """Context manager that yields a DB connection normalised to the sqlite3 API."""
    if _USE_POSTGRES:
        try:
            import psycopg2  # type: ignore[import]
            import psycopg2.extras  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "psycopg2-binary is required for the PostgreSQL backend. "
                "Run: pip install psycopg2-binary"
            ) from exc
        conn = psycopg2.connect(_POSTGRES_URL)
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            yield _PgConnAdapter(conn, cur)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()


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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_params (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (run_id, key)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS registered_models (
                name TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                alias TEXT NOT NULL DEFAULT 'latest',
                metric_snapshot TEXT NOT NULL DEFAULT '{}',
                registered_at TEXT NOT NULL
            )
        """)
        # #18 — index for ORDER BY created_at DESC in list_runs
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(started_at DESC)")


def save_run_metrics(run_id: str, loss_history: list[dict[str, Any]]) -> None:
    if not loss_history:
        return
    now = time.time()
    rows = []
    for pt in loss_history:
        step = int(pt.get("step", 0))
        ts = float(pt.get("timestamp") or pt.get("ts") or now)
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
                "INSERT INTO run_metrics (run_id, key, value, step, timestamp) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (run_id, key, step) DO UPDATE SET "
                "value=EXCLUDED.value, timestamp=EXCLUDED.timestamp",
                rows,
            )
    except Exception:
        _logger.exception("Failed to save run_metrics for run_id=%s", run_id)
        raise


def save_final_metrics(run_id: str, metrics: dict[str, Any]) -> None:
    rows = [
        (run_id, key, float(val), -1, time.time())
        for key, val in metrics.items()
        if val is not None
    ]
    if not rows:
        return
    try:
        _init_db()
        with _get_conn() as conn:
            conn.executemany(
                "INSERT INTO run_metrics (run_id, key, value, step, timestamp) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (run_id, key, step) DO UPDATE SET "
                "value=EXCLUDED.value, timestamp=EXCLUDED.timestamp",
                rows,
            )
    except Exception:
        _logger.exception("Failed to save final metrics for run_id=%s", run_id)
        raise


def save_run_params(run_id: str, params: dict[str, Any]) -> None:
    if not params:
        return
    rows = [(run_id, str(k), str(v)) for k, v in params.items()]
    try:
        _init_db()
        with _get_conn() as conn:
            conn.executemany(
                "INSERT INTO run_params (run_id, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT (run_id, key) DO UPDATE SET value=EXCLUDED.value",
                rows,
            )
    except Exception:
        _logger.exception("Failed to save run_params for run_id=%s", run_id)


def write_job_status(
    run_id: str,
    status: str,
    *,
    name: str = "",
    model_id: str = "",
    started_at: str = "",
    finished_at: str = "",
    output_path: str = "",
) -> None:
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
        _logger.exception("Failed to write job status for run_id=%s", run_id)


def list_runs(limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    try:
        _init_db()
        sql = "SELECT id, status, output_path FROM runs ORDER BY started_at DESC"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = (int(limit), int(offset))
        with _get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "job_id": r["id"],
                "status": r["status"] or "unknown",
                "output_path": r["output_path"] or "",
            }
            for r in rows
        ]
    except Exception:
        _logger.exception("Failed to list runs")
        return []


def list_stale_running_jobs(max_age_seconds: int = 7200) -> list[str]:
    """Return run IDs stuck in 'running' longer than max_age_seconds.

    Used by the startup sweep to recover from worker crashes.
    """
    try:
        _init_db()
        cutoff = datetime.fromtimestamp(time.time() - max_age_seconds, tz=timezone.utc).isoformat()
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT id FROM runs WHERE status = ? AND started_at < ?",
                ("running", cutoff),
            ).fetchall()
        return [r["id"] for r in rows]
    except Exception:
        _logger.exception("Failed to list stale running jobs")
        return []


def get_final_metrics(run_id: str) -> dict[str, float]:
    try:
        _init_db()
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT key, value FROM run_metrics WHERE run_id = ? AND step = -1",
                (run_id,),
            ).fetchall()
        return {r["key"]: r["value"] for r in rows}
    except Exception:
        _logger.exception("Failed to read final metrics for run_id=%s", run_id)
        return {}


def get_run_metrics(
    run_ids: list[str], metric_key: str = "loss"
) -> dict[str, list[dict[str, Any]]]:
    if not run_ids:
        return {}
    try:
        _init_db()
        placeholders = ",".join("?" * len(run_ids))
        with _get_conn() as conn:
            rows = conn.execute(
                f"SELECT run_id, step, value FROM run_metrics "
                f"WHERE run_id IN ({placeholders}) AND key = ? "
                f"ORDER BY run_id, step",
                (*run_ids, metric_key),
            ).fetchall()
        result: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            result.setdefault(r["run_id"], []).append({"step": r["step"], "value": r["value"]})
        return result
    except Exception:
        _logger.exception("Failed to get run_metrics for runs=%s key=%s", run_ids, metric_key)
        return {}


def register_model(
    name: str,
    run_id: str,
    *,
    alias: str = "latest",
    metric_snapshot: dict[str, Any] | None = None,
) -> None:
    try:
        _init_db()
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO registered_models (name, run_id, alias, metric_snapshot, registered_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    run_id=excluded.run_id,
                    alias=excluded.alias,
                    metric_snapshot=excluded.metric_snapshot,
                    registered_at=excluded.registered_at
                """,
                (
                    name,
                    run_id,
                    alias,
                    json.dumps(metric_snapshot or {}),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except Exception:
        _logger.exception("Failed to register model name=%s run_id=%s", name, run_id)


def list_registered_models() -> list[dict[str, Any]]:
    try:
        _init_db()
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT name, run_id, alias, metric_snapshot, registered_at "
                "FROM registered_models ORDER BY registered_at DESC"
            ).fetchall()
        return [
            {
                "name": r["name"],
                "run_id": r["run_id"],
                "alias": r["alias"],
                "metric_snapshot": json.loads(r["metric_snapshot"] or "{}"),
                "registered_at": r["registered_at"],
            }
            for r in rows
        ]
    except Exception:
        _logger.exception("Failed to list registered models")
        return []


def save_experiment_run(run_data: dict[str, Any]):
    try:
        _init_db()
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO runs
                (id, name, model_id, model_source, technique, epochs, learning_rate,
                 lora_r, batch_size, dataset_name, user_intent, final_loss, perplexity,
                 started_at, finished_at, status, output_path, loss_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=EXCLUDED.name,
                    model_id=EXCLUDED.model_id,
                    model_source=EXCLUDED.model_source,
                    technique=EXCLUDED.technique,
                    epochs=EXCLUDED.epochs,
                    learning_rate=EXCLUDED.learning_rate,
                    lora_r=EXCLUDED.lora_r,
                    batch_size=EXCLUDED.batch_size,
                    dataset_name=EXCLUDED.dataset_name,
                    user_intent=EXCLUDED.user_intent,
                    final_loss=EXCLUDED.final_loss,
                    perplexity=EXCLUDED.perplexity,
                    started_at=EXCLUDED.started_at,
                    finished_at=EXCLUDED.finished_at,
                    status=EXCLUDED.status,
                    output_path=EXCLUDED.output_path,
                    loss_history=EXCLUDED.loss_history
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
        _logger.exception("Failed to save experiment run %s", run_data.get("id", ""))
