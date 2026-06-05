"""System-level API routes: /health, /health/workers, and /gpu."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import _detect_gpu
from app.api.schemas import GpuInfo, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
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
