"""System-level API routes: /health, /health/workers, and /gpu."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.deps import _detect_gpu
from app.api.schemas import GpuInfo, HealthResponse

router = APIRouter()


def _celery_alive() -> bool:
    """Return True if at least one Celery worker responds within 1 s."""
    try:
        from workers.celery_app import celery_app
        return bool(celery_app.control.inspect(timeout=1.0).active())
    except Exception:
        return False


@router.get("/health")
async def health():
    """Return 200 ok when healthy, 503 degraded when no Celery worker is up."""
    alive = _celery_alive()
    body = {"status": "ok" if alive else "degraded", "celery_alive": alive}
    return JSONResponse(content=body, status_code=200 if alive else 503)


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
