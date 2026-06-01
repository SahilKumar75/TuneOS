"""System-level API routes: /health and /gpu."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import _detect_gpu
from app.api.schemas import GpuInfo, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@router.get("/gpu", response_model=GpuInfo)
async def gpu_info():
    return _detect_gpu()
