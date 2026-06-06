"""Shared constants and helper functions used across all API routers."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

from app.api.schemas import GpuInfo

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
DATASET_DIR = os.getenv("DATASET_DIR", "./storage/datasets")


def artifact_path(job_id: str, artifact: str) -> Path:
    """Return the canonical path for a job artifact (adapter, merged, gguf, …).

    All workers and API routes should use this instead of constructing paths
    ad-hoc. The resolved path is validated to stay within the job's own
    directory so that absolute values or ``..`` segments in ``job_id`` /
    ``artifact`` cannot escape ``OUTPUT_DIR``.

    Raises:
        ValueError: if the resolved path escapes the job directory.
    """
    base = Path(OUTPUT_DIR).resolve()
    job_root = (base / job_id).resolve()
    candidate = (job_root / artifact).resolve()
    try:
        # job_root must stay under base (guards `..` in job_id), and the final
        # candidate must stay under job_root (guards `..`/absolute artifact).
        job_root.relative_to(base)
        candidate.relative_to(job_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe artifact path for job {job_id!r}: {artifact!r}") from exc
    return candidate


_SUPPORTED_MODELS: list[dict] = [
    {
        "name": "Mistral 7B",
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "notes": "Primary target, well-tested with QLoRA",
    },
    {"name": "Llama 3 8B", "hf_id": "meta-llama/Meta-Llama-3-8B", "notes": "Requires HF token"},
    {
        "name": "Phi-3 Mini",
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "notes": "Fast, runs on smaller GPUs",
    },
    {"name": "Gemma 2B", "hf_id": "google/gemma-2b", "notes": "Good for low-VRAM environments"},
]


def _redis_sync():
    import redis

    return redis.from_url(REDIS_URL)


def _get_job_status_from_redis(job_id: str) -> dict:
    try:
        from workers.status import get_job_status

        return get_job_status(job_id)
    except Exception:
        return {"status": "unknown", "job_id": job_id}


def _cuda_version() -> str:
    """CUDA version from torch if it's already imported — never force-import it."""
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return ""
    return getattr(getattr(torch, "version", None), "cuda", "") or ""


def _detect_gpu() -> GpuInfo:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
            first = [c.strip() for c in lines[0].split(",")]
            total_gb = round(float(first[1]) / 1024, 1) if len(first) > 1 else 0.0
            free_gb = round(float(first[2]) / 1024, 1) if len(first) > 2 else 0.0
            return GpuInfo(
                available=True,
                backend="cuda",
                name=first[0],
                detail=result.stdout.strip(),
                device_count=len(lines),
                vram_total_gb=total_gb,
                vram_free_gb=free_gb,
                cuda_version=_cuda_version(),
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "Apple" in result.stdout:
                return GpuInfo(
                    available=True,
                    backend="mps",
                    name=result.stdout.strip(),
                    detail="Apple Metal Performance Shaders",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return GpuInfo(available=False, backend="cpu", name="CPU only")
