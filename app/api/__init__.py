"""TuneOS REST API package.

Exports ``app_api`` — a FastAPI instance with all routers mounted.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from app.logging_config import TraceIDMiddleware, configure_logging

configure_logging()
_logger = logging.getLogger(__name__)

_STALE_RUNNING_SECONDS = 7200  # jobs running longer than this on startup are zombie


def _sweep_stale_jobs() -> None:
    """Mark any SQLite jobs stuck in 'running' for > 2 h as failed.

    Called once at API startup — handles the case where a Celery worker died
    mid-job and never wrote a terminal status.
    """
    try:
        import sqlite3

        from app.state.experiments_db import DB_PATH, write_job_status

        now = datetime.now(timezone.utc)
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, started_at FROM runs WHERE status = 'running'"
            ).fetchall()

        for row in rows:
            started = row["started_at"]
            if not started:
                continue
            try:
                started_dt = datetime.fromisoformat(started)
                if started_dt.tzinfo is None:
                    started_dt = started_dt.replace(tzinfo=timezone.utc)
                age = (now - started_dt).total_seconds()
                if age > _STALE_RUNNING_SECONDS:
                    write_job_status(row["id"], "failed", finished_at=now.isoformat())
                    _logger.warning("Marked stale job %s as failed (age=%.0fs)", row["id"], age)
            except Exception:
                pass
    except Exception:
        pass  # never crash the API startup


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _sweep_stale_jobs()
    yield


from app.api import (  # noqa: E402
    datasets_routes,
    experiments_routes,
    jobs_routes,
    models_routes,
    system,
)

app_api = FastAPI(title="TuneOS API", lifespan=_lifespan)
if TraceIDMiddleware is not None:
    app_api.add_middleware(TraceIDMiddleware)

app_api.include_router(system.router)
app_api.include_router(models_routes.router)
app_api.include_router(datasets_routes.router)
app_api.include_router(jobs_routes.router)
app_api.include_router(experiments_routes.router)

__all__ = ["app_api"]
