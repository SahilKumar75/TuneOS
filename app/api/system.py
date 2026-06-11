"""System-level API routes: /health, /health/workers, and /gpu."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import _detect_gpu
from app.api.schemas import GpuInfo, HealthResponse

router = APIRouter()


@router.get("/health", response_model=None)
async def health():
    """#13 — include Celery worker liveness so callers know if jobs can run."""
    try:
        from workers.celery_app import celery_app

        inspector = celery_app.control.inspect(timeout=1.0)
        active = inspector.active()
        celery_alive = bool(active)
    except Exception:
        celery_alive = False

    if not celery_alive:
        import json

        from fastapi import Response

        body = json.dumps({"status": "degraded", "celery_alive": False})
        return Response(content=body, status_code=503, media_type="application/json")

    return HealthResponse()


@router.get("/health/workers")
async def workers_health():
    """Return whether at least one Celery worker is alive and ready."""
    try:
        from workers.celery_app import celery_app

        inspector = celery_app.control.inspect(timeout=2.0)
        active = inspector.active()
        alive = bool(active)
        worker_names = list(active.keys()) if active else []
    except Exception as exc:
        return {"workers_alive": False, "workers": [], "error": str(exc)}

    return {"workers_alive": alive, "workers": worker_names}


@router.get("/gpu", response_model=GpuInfo)
async def gpu_info():
    return _detect_gpu()


def recover_stale_jobs() -> None:
    """#11 — mark running jobs older than 2h as failed on app startup.

    Covers the case where a Celery worker was OOM-killed or SIGKILL'd mid-job,
    leaving SQLite rows stuck at status='running' with no worker to update them.
    """
    import json
    import logging
    import os

    import redis as _redis

    from db.experiments_db import list_stale_running_jobs, write_job_status

    _log = logging.getLogger(__name__)
    try:
        stale = list_stale_running_jobs(max_age_seconds=7200)
        if not stale:
            return
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        for job_id in stale:
            write_job_status(job_id, "failed")
            r.set(
                f"job:{job_id}:status",
                json.dumps(
                    {
                        "status": "failed",
                        "job_id": job_id,
                        "error": "Stale — worker likely restarted",
                    }
                ),
            )
            r.expire(f"job:{job_id}:status", 172800)
            _log.warning("Marked stale job %s as failed on startup", job_id)
    except Exception:
        _log.exception("Startup stale-job recovery failed")
