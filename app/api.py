"""
TuneOS — REST API endpoints (FastAPI Router).

Mounted under ``/api`` by the Reflex application.  Provides health
checks, GPU detection, model listing, and CRUD placeholders for
fine-tuning jobs.
"""
from __future__ import annotations

import platform
import subprocess
import uuid
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Router ───────────────────────────────────────────────────────
app_api = FastAPI(title="TuneOS API")

# ── Constants ────────────────────────────────────────────────────
_VERSION = "0.1.0"

_SUPPORTED_MODELS: list[dict[str, str]] = [
    {
        "name": "Mistral 7B",
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "notes": "Primary target, well-tested with QLoRA",
    },
    {
        "name": "Llama 3 8B",
        "hf_id": "meta-llama/Meta-Llama-3-8B",
        "notes": "Requires HF token",
    },
    {
        "name": "Phi-3 Mini",
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "notes": "Fast, runs on smaller GPUs",
    },
    {
        "name": "Gemma 2B",
        "hf_id": "google/gemma-2b",
        "notes": "Good for low-VRAM environments",
    },
]


# ── Pydantic Schemas ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Health-check response."""
    status: str = "ok"
    version: str = _VERSION


class GpuInfo(BaseModel):
    """GPU detection result."""
    available: bool
    backend: str
    name: str
    detail: str = ""


class ModelInfo(BaseModel):
    """A supported base model."""
    name: str
    hf_id: str
    notes: str = ""


class JobConfig(BaseModel):
    """Request body for creating a new fine-tuning job."""
    model_id: str = Field(..., description="Hugging Face model ID")
    dataset_path: str = Field(..., description="Path to the uploaded dataset")
    lora_rank: int = Field(default=8, ge=1, le=256)
    lora_alpha: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1)


class JobStatus(BaseModel):
    """Response schema for job status."""
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""


class JobCreated(BaseModel):
    """Response returned when a job is successfully queued."""
    job_id: str
    status: str = "queued"


# ── GPU Detection ────────────────────────────────────────────────
def _detect_gpu() -> GpuInfo:
    """Detect the available GPU (NVIDIA via nvidia-smi, Apple MPS via sysctl)."""
    # NVIDIA check
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0]
            return GpuInfo(
                available=True,
                backend="cuda",
                name=gpu_name,
                detail=result.stdout.strip(),
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Apple Silicon MPS check
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                cpu = result.stdout.strip()
                if "Apple" in cpu:
                    return GpuInfo(
                        available=True,
                        backend="mps",
                        name=cpu,
                        detail="Apple Metal Performance Shaders",
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return GpuInfo(available=False, backend="cpu", name="CPU only")


# ── Endpoints ────────────────────────────────────────────────────
@app_api.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Basic liveness / readiness check."""
    return HealthResponse()


@app_api.get("/gpu", response_model=GpuInfo)
async def gpu_info() -> GpuInfo:
    """Detect and return GPU information."""
    return _detect_gpu()


@app_api.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """Return the list of supported base models."""
    return [ModelInfo(**m) for m in _SUPPORTED_MODELS]


# ── Celery + Redis wiring ────────────────────────────────────────
def _get_celery():
    """Import Celery app lazily so the API starts even if Redis is down."""
    from workers.celery_app import celery_app
    return celery_app


def _get_job_status_from_redis(job_id: str) -> dict:
    """Read job status from Redis. Returns dict with at least 'status' key."""
    try:
        from workers.status import get_job_status
        return get_job_status(job_id)
    except Exception:
        return {"status": "unknown", "job_id": job_id}


# ── Job CRUD ─────────────────────────────────────────────────────
@app_api.get("/jobs", response_model=list[JobStatus])
async def list_jobs() -> list[JobStatus]:
    """List all fine-tuning jobs (placeholder — returns empty list)."""
    return []


@app_api.post("/jobs", response_model=JobCreated, status_code=201)
async def create_job(config: JobConfig) -> JobCreated:
    """Create and enqueue a new fine-tuning job."""
    job_id = str(uuid.uuid4())

    model_cfg = {
        "model_name": config.model_id,
        "use_4bit": True,
        "use_8bit": False,
        "trust_remote_code": False,
    }
    lora_cfg = {
        "r": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    }
    train_cfg = {
        "output_dir": f"./outputs/{job_id}",
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    }

    try:
        from workers.train_task import run_finetune
        run_finetune.apply_async(
            args=[job_id, model_cfg, lora_cfg, train_cfg, config.dataset_path],
            task_id=job_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not enqueue job: {exc}")

    return JobCreated(job_id=job_id)


@app_api.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    """Get real-time status of a job from Redis."""
    state = _get_job_status_from_redis(job_id)
    return JobStatus(
        job_id=job_id,
        status=state.get("status", "unknown"),
        progress=state.get("progress", 0.0),
        message=state.get("error", ""),
    )


@app_api.delete("/jobs/{job_id}", response_model=JobStatus)
async def cancel_job(job_id: str) -> JobStatus:
    """Revoke a queued or running Celery task."""
    try:
        celery_app = _get_celery()
        celery_app.control.revoke(job_id, terminate=True, signal="SIGTERM")
    except Exception:
        pass  # Best-effort cancellation
    return JobStatus(job_id=job_id, status="cancelled")
