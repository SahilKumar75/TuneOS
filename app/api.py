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


# ── Job CRUD Placeholders ────────────────────────────────────────
@app_api.get("/jobs", response_model=list[JobStatus])
async def list_jobs() -> list[JobStatus]:
    """List all fine-tuning jobs (placeholder — returns empty list)."""
    return []


@app_api.post("/jobs", response_model=JobCreated, status_code=201)
async def create_job(config: JobConfig) -> JobCreated:
    """Create a new fine-tuning job (placeholder — returns mock ID)."""
    job_id = str(uuid.uuid4())
    # TODO: Enqueue a Celery task with ``config`` and persist job metadata.
    return JobCreated(job_id=job_id)


@app_api.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    """Get status of a specific job (placeholder)."""
    # TODO: Look up real job state from Redis / DB.
    return JobStatus(
        job_id=job_id,
        status="unknown",
        message="Job tracking is not yet implemented.",
    )


@app_api.delete("/jobs/{job_id}", response_model=JobStatus)
async def cancel_job(job_id: str) -> JobStatus:
    """Cancel a running job (placeholder)."""
    # TODO: Revoke the Celery task and update state.
    return JobStatus(
        job_id=job_id,
        status="cancelled",
        message="Job cancellation is not yet implemented.",
    )
