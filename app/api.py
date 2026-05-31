"""
TuneOS — REST API endpoints (FastAPI Router).

Mounted under ``/api`` by the Reflex application.  Provides health
checks, GPU detection, model listing, and CRUD placeholders for
fine-tuning jobs.
"""

from __future__ import annotations

import io
import json
import os
import platform
import subprocess
import uuid
import zipfile

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── Router ───────────────────────────────────────────────────────
app_api = FastAPI(title="TuneOS API")

# ── Constants ────────────────────────────────────────────────────
_VERSION = "0.1.0"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")

# Process-level inference cache: job_id -> (model, tokenizer)
# Max 1 entry — evicted when a different job_id is requested.
_INFER_CACHE: dict = {}

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
        raise HTTPException(status_code=503, detail=f"Could not enqueue job: {exc}") from exc

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


# ── Post-training endpoints ───────────────────────────────────────

@app_api.get("/jobs/{job_id}/download")
async def download_adapter(job_id: str):
    """Zip and stream the adapter weights directory."""
    adapter_dir = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(adapter_dir):
            full = os.path.join(adapter_dir, fname)
            if os.path.isfile(full):
                zf.write(full, fname)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=adapter_{job_id[:8]}.zip"},
    )


class PushHubRequest(BaseModel):
    repo_name: str
    hf_token: str = ""


@app_api.post("/jobs/{job_id}/push_hub")
async def push_to_hub(job_id: str, req: PushHubRequest):
    """Push the adapter to Hugging Face Hub."""
    token = req.hf_token or os.getenv("HF_TOKEN", "")
    if not token:
        raise HTTPException(status_code=400, detail="HF token required")

    adapter_dir = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter not found")

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=req.repo_name, repo_type="model", exist_ok=True, private=True)
        api.upload_folder(folder_path=adapter_dir, repo_id=req.repo_name, repo_type="model")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "pushed", "repo_url": f"https://huggingface.co/{req.repo_name}"}


@app_api.get("/jobs/{job_id}/eval")
async def get_eval(job_id: str):
    """Read evaluation metrics written by the training worker."""
    try:
        import redis as _redis

        r = _redis.from_url(REDIS_URL)
        raw = r.get(f"job:{job_id}:eval")
        if not raw:
            return {"status": "not_ready", "perplexity": None, "bleu": None}
        data = json.loads(raw)
        return {"status": "done", **data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class InferRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7


@app_api.post("/jobs/{job_id}/infer")
async def infer(job_id: str, req: InferRequest):
    """Run inference using the fine-tuned adapter (loaded lazily, cached in-process)."""
    import torch

    state = _get_job_status_from_redis(job_id)
    if state.get("status") != "done":
        raise HTTPException(status_code=400, detail="Job not complete")

    adapter_dir = state.get("output_path") or os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter directory not found")

    global _INFER_CACHE

    if job_id not in _INFER_CACHE:
        # Evict any previously cached model to free VRAM
        _INFER_CACHE.clear()

        config_path = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="adapter_config.json not found")

        with open(config_path) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path", "")
        if not base_model_name:
            raise HTTPException(status_code=500, detail="base_model_name_or_path missing in adapter_config.json")

        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = PeftModel.from_pretrained(base_model, adapter_dir)
            model.eval()
            _INFER_CACHE[job_id] = (model, tokenizer)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    model, tokenizer = _INFER_CACHE[job_id]

    try:
        inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {"response": response}
