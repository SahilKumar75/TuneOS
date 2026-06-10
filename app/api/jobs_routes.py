"""Job CRUD and per-job action routes."""

from __future__ import annotations

import io
import json
import os
import threading
import uuid
import zipfile

from cachetools import LRUCache
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import OUTPUT_DIR, REDIS_URL, _get_job_status_from_redis
from app.api.schemas import (
    CommentaryRequest,
    DistillJobConfig,
    DPOJobConfig,
    GgufRequest,
    GitHubPushRequest,
    InferRequest,
    JobConfig,
    JobCreated,
    JobStatus,
    MergeRequest,
    PushHubRequest,
    SweepRequest,
)

router = APIRouter()

_T5_FAMILIES = ("t5", "mt5")


def _detect_task_type(model_id: str) -> str:
    """Return PEFT TaskType string based on model ID. T5/mT5 need SEQ_2_SEQ_LM."""
    name = model_id.lower()
    if any(f in name for f in _T5_FAMILIES):
        return "SEQ_2_SEQ_LM"
    return "CAUSAL_LM"


# Bounded LRU of loaded inference models — each is GBs, so cap how many stay
# resident; the least-recently-used is evicted when full. A single lock
# serializes loads and evictions across requests.
_INFER_CACHE: LRUCache = LRUCache(maxsize=3)
_INFER_CACHE_LOCK = threading.Lock()


def _resolve_job_dir(job_id: str) -> str:
    """Resolve job_id to an absolute path inside OUTPUT_DIR, rejecting traversal."""
    base = os.path.abspath(OUTPUT_DIR)
    candidate = os.path.abspath(os.path.join(base, job_id))
    if os.path.commonpath([base, candidate]) != base:
        raise HTTPException(status_code=400, detail="Invalid job id")
    return candidate


@router.get("/jobs", response_model=list[JobStatus])
async def list_jobs(limit: int = 50, offset: int = 0):
    """Return runs from the durable SQLite store, most-recent first (paginated)."""
    from app.state.experiments_db import list_runs

    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    return [JobStatus(**row) for row in list_runs(limit=limit, offset=offset)]


def _build_finetune_kwargs(config: JobConfig) -> dict:
    """Translate a JobConfig into the run_finetune task kwargs."""
    job_id = config.experiment_id or str(uuid.uuid4())
    model_cfg = {
        "model_name": config.model_id,
        "use_4bit": config.use_4bit,
        "use_8bit": False,
        "trust_remote_code": False,
        "max_seq_length": config.max_seq_length,
        "hf_token": config.hf_token,
        "local_model_path": config.local_model_path,
        "model_source": config.model_source,
    }
    lora_cfg = {
        "r": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "bias": "none",
        "task_type": _detect_task_type(config.model_id),
        "target_modules": None,  # auto-detected from model architecture in trainer/lora.py
    }
    train_cfg = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "fp16": not config.bf16,
        "bf16": config.bf16,
        "logging_steps": 1,
        "save_steps": 100,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "optim": "paged_adamw_32bit",
        "max_grad_norm": 0.3,
        "eval_split_ratio": config.eval_split_ratio,
        "early_stopping_patience": config.early_stopping_patience,
        "eval_steps": config.eval_steps,
        "resume_from_checkpoint": config.resume_from_checkpoint or None,
        "seed": config.seed,
        "use_torch_compile": config.use_torch_compile,
        "compute_backend": config.compute_backend,
        "prompt_template": config.prompt_template,
        "packing": config.packing,
        "fsdp": config.fsdp,
    }
    return {
        "job_id": job_id,
        "model_cfg": model_cfg,
        "lora_cfg": lora_cfg,
        "train_cfg": train_cfg,
        "dataset_path": config.dataset_path,
        "hub_dataset_id": config.hub_dataset_id,
        "hub_split": config.hub_dataset_split,
        "instruction_col": config.instruction_col,
        "output_col": config.output_col,
    }


def _enqueue_finetune(config: JobConfig) -> str:
    """Enqueue one SFT job (worker-alive guarded); returns its job_id."""
    kwargs = _build_finetune_kwargs(config)
    _ensure_worker_alive()
    try:
        from workers.train_task import run_finetune

        run_finetune.apply_async(kwargs=kwargs, task_id=kwargs["job_id"])
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not enqueue job: {exc}") from exc
    return kwargs["job_id"]


@router.post("/jobs", response_model=JobCreated, status_code=201)
async def create_job(config: JobConfig):
    """Create and enqueue a new fine-tuning job."""
    return JobCreated(job_id=_enqueue_finetune(config))


def _ensure_worker_alive() -> None:
    """Raise 503 if no Celery worker is listening — avoids silently queuing a
    job that would sit forever. Inspect failures (e.g. Redis down) are tolerated;
    the enqueue below will surface those."""
    try:
        from workers.celery_app import celery_app as _celery

        if not _celery.control.inspect(timeout=2.0).active():
            raise HTTPException(
                status_code=503,
                detail=(
                    "No training workers are running. Start the desktop app, or run: "
                    "celery -A workers.celery_app worker --loglevel=info"
                ),
            )
    except HTTPException:
        raise
    except Exception:
        pass


@router.post("/jobs/dpo", response_model=JobCreated, status_code=201)
async def create_dpo_job(config: DPOJobConfig):
    """Create and enqueue a DPO (preference) fine-tuning job."""
    job_id = config.experiment_id or str(uuid.uuid4())

    model_cfg = {
        "model_name": config.model_id,
        "use_4bit": config.use_4bit,
        "use_8bit": False,
        "trust_remote_code": False,
        "max_seq_length": config.max_length,
        "hf_token": config.hf_token,
        "local_model_path": config.local_model_path,
        "model_source": config.model_source,
    }
    lora_cfg = {
        "r": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "bias": "none",
        "task_type": _detect_task_type(config.model_id),
        "target_modules": None,
    }
    dpo_cfg = {
        "output_dir": OUTPUT_DIR,
        "beta": config.beta,
        "max_length": config.max_length,
        "max_prompt_length": config.max_prompt_length,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "fp16": not config.bf16,
        "bf16": config.bf16,
        "seed": config.seed,
    }

    _ensure_worker_alive()

    try:
        from workers.dpo_task import run_dpo

        run_dpo.apply_async(
            kwargs={
                "job_id": job_id,
                "model_cfg": model_cfg,
                "lora_cfg": lora_cfg,
                "dpo_cfg": dpo_cfg,
                "dataset_path": config.dataset_path,
                "hub_dataset_id": config.hub_dataset_id,
                "hub_split": config.hub_dataset_split,
                "prompt_col": config.prompt_col,
                "chosen_col": config.chosen_col,
                "rejected_col": config.rejected_col,
            },
            task_id=job_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not enqueue DPO job: {exc}") from exc

    return JobCreated(job_id=job_id)


@router.post("/jobs/distill", response_model=JobCreated, status_code=201)
async def create_distill_job(config: DistillJobConfig):
    """Create and enqueue a knowledge-distillation job (teacher → LoRA student)."""
    job_id = config.experiment_id or str(uuid.uuid4())

    model_cfg = {
        "model_name": config.model_id,
        "use_4bit": config.use_4bit,
        "use_8bit": False,
        "trust_remote_code": False,
        "max_seq_length": config.max_seq_length,
        "hf_token": config.hf_token,
        "local_model_path": config.local_model_path,
        "model_source": config.model_source,
    }
    lora_cfg = {
        "r": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "bias": "none",
        "task_type": _detect_task_type(config.model_id),
        "target_modules": None,
    }
    distill_cfg = {
        "output_dir": OUTPUT_DIR,
        "teacher_model": config.teacher_model,
        "temperature": config.temperature,
        "alpha": config.alpha,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "max_seq_length": config.max_seq_length,
        "fp16": not config.bf16,
        "bf16": config.bf16,
        "seed": config.seed,
        "prompt_template": config.prompt_template,
    }

    _ensure_worker_alive()
    try:
        from workers.kd_task import run_distill

        run_distill.apply_async(
            kwargs={
                "job_id": job_id,
                "model_cfg": model_cfg,
                "lora_cfg": lora_cfg,
                "distill_cfg": distill_cfg,
                "dataset_path": config.dataset_path,
                "hub_dataset_id": config.hub_dataset_id,
                "hub_split": config.hub_dataset_split,
                "instruction_col": config.instruction_col,
                "output_col": config.output_col,
            },
            task_id=job_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Could not enqueue distillation job: {exc}"
        ) from exc

    return JobCreated(job_id=job_id)


@router.post("/jobs/sweep")
async def create_sweep(req: SweepRequest):
    """Expand a hyperparameter grid over the base config and enqueue one SFT job
    per combination. Returns the list of job ids (visualize them on /compare)."""
    from trainer.sweep import expand_grid

    base = req.base.model_dump()
    configs = expand_grid(base, req.grid)
    job_ids: list[str] = []
    for cfg in configs:
        cfg["experiment_id"] = ""  # each combination gets a fresh id
        job_ids.append(_enqueue_finetune(JobConfig(**cfg)))
    return {"count": len(job_ids), "job_ids": job_ids}


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    state = _get_job_status_from_redis(job_id)
    return JobStatus(
        job_id=job_id,
        status=state.get("status", "unknown"),
        progress=state.get("progress", 0.0),
        message=state.get("error", ""),
        output_path=state.get("output_path", ""),
        error=state.get("error", ""),
    )


@router.delete("/jobs/{job_id}", response_model=JobStatus)
async def cancel_job(job_id: str):
    try:
        from workers.celery_app import celery_app

        celery_app.control.revoke(job_id, terminate=True, signal="SIGTERM")
    except Exception:
        pass
    return JobStatus(job_id=job_id, status="cancelled")


@router.get("/jobs/{job_id}/download")
async def download_adapter(job_id: str):
    adapter_dir = _resolve_job_dir(job_id)
    if not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(adapter_dir):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, adapter_dir)
                zf.write(full, arcname)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=adapter_{job_id[:8]}.zip"},
    )


@router.post("/jobs/{job_id}/push_hub")
async def push_to_hub(job_id: str, req: PushHubRequest):
    token = req.hf_token or os.getenv("HF_TOKEN", "")
    if not token:
        raise HTTPException(status_code=400, detail="HF token required")

    adapter_dir = _resolve_job_dir(job_id)
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


@router.post("/jobs/{job_id}/merge", status_code=202)
async def merge_adapter(job_id: str, req: MergeRequest):
    adapter_dir = _resolve_job_dir(job_id)
    if not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter not found")

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="adapter_config.json not found")

    with open(config_path) as f:
        adapter_cfg = json.load(f)
    base_model_id = adapter_cfg.get("base_model_name_or_path", "")
    if not base_model_id:
        raise HTTPException(status_code=500, detail="base_model_name_or_path missing")

    try:
        from workers.merge_task import merge_adapter_task

        merge_adapter_task.apply_async(
            kwargs={
                "job_id": job_id,
                "base_model_id": base_model_id,
                "adapter_path": adapter_dir,
                "hf_token": req.hf_token or os.getenv("HF_TOKEN", ""),
            },
            task_id=f"{job_id}-merge",
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not enqueue merge: {exc}") from exc

    return {"status": "merging", "job_id": job_id}


@router.get("/jobs/{job_id}/download-merged")
async def download_merged(job_id: str):
    merged_dir = os.path.join(_resolve_job_dir(job_id), "merged")
    if not os.path.isdir(merged_dir):
        raise HTTPException(status_code=404, detail="Merged model not found — run merge first")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(merged_dir):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, merged_dir)
                zf.write(full, arcname)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=merged_{job_id[:8]}.zip"},
    )


@router.post("/jobs/{job_id}/export-gguf", status_code=202)
async def export_gguf(job_id: str, req: GgufRequest):
    merged_dir = os.path.join(_resolve_job_dir(job_id), "merged")
    if not os.path.isdir(merged_dir):
        raise HTTPException(status_code=400, detail="Merge the model first before exporting GGUF")

    try:
        from workers.merge_task import export_gguf_task

        export_gguf_task.apply_async(
            kwargs={
                "job_id": job_id,
                "merged_model_path": merged_dir,
                "quant_type": req.quant_type,
            },
            task_id=f"{job_id}-gguf",
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"status": "exporting", "quant_type": req.quant_type}


@router.post("/jobs/{job_id}/push-github")
async def push_github(job_id: str, req: GitHubPushRequest):
    adapter_dir = _resolve_job_dir(job_id)
    if not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter not found")

    try:
        from workers.merge_task import push_github_task

        push_github_task.apply_async(
            kwargs={
                "job_id": job_id,
                "adapter_path": adapter_dir,
                "repo_url": req.repo_url,
                "github_token": req.github_token,
            },
            task_id=f"{job_id}-github",
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"status": "pushing"}


@router.post("/jobs/{job_id}/commentary")
async def get_commentary(job_id: str, req: CommentaryRequest):
    """Return a plain-English sentence describing training progress."""
    epoch_frac = req.epoch / max(req.total_epochs, 1)
    drop = req.loss_drop_pct

    quality = (
        "great" if drop > 40 else ("healthy" if drop > 20 else ("slow" if drop > 5 else "stalled"))
    )
    phase = "early" if epoch_frac < 0.33 else ("middle" if epoch_frac < 0.67 else "final")
    intent_frag = f" for your {req.intent}" if req.intent else ""
    loss_verb = {
        "great": "dropped significantly",
        "healthy": "is decreasing steadily",
        "slow": "is decreasing slowly",
        "stalled": "has barely moved",
    }[quality]

    templates = {
        (
            "great",
            "early",
        ): f"Strong start! Loss {loss_verb} — the model is picking up patterns{intent_frag} quickly.",
        (
            "great",
            "middle",
        ): f"Training is going well. Loss {loss_verb} and the model is solidifying its skills{intent_frag}.",
        (
            "great",
            "final",
        ): f"Excellent run! Loss {loss_verb}. Your model looks ready{intent_frag}.",
        (
            "healthy",
            "early",
        ): f"Good progress. Loss {loss_verb} — on track for a solid result{intent_frag}.",
        ("healthy", "middle"): f"Training looks healthy. Loss {loss_verb}. Keep it running.",
        ("healthy", "final"): f"Looking good in the final stretch. Loss {loss_verb}.",
        ("slow", "early"): f"Loss {loss_verb} — a slow start is normal. Give it a few more epochs.",
        ("slow", "middle"): f"Loss {loss_verb}. Consider a higher learning rate if this continues.",
        (
            "slow",
            "final",
        ): f"Loss {loss_verb}. The model may need more data or more epochs next time.",
        (
            "stalled",
            "early",
        ): f"Loss {loss_verb} yet. Try a higher learning rate or check your dataset.",
        (
            "stalled",
            "middle",
        ): f"Loss {loss_verb}. Training may be stuck — check the learning rate.",
        (
            "stalled",
            "final",
        ): f"Loss {loss_verb} much. Try more epochs or a larger learning rate next run.",
    }

    commentary = templates.get(
        (quality, phase), f"Training in progress. Current loss: {req.current_loss:.4f}."
    )
    return {"commentary": commentary}


@router.get("/jobs/{job_id}/eval")
async def get_eval(job_id: str):
    import redis.asyncio as aioredis

    r = aioredis.from_url(REDIS_URL)
    try:
        raw = await r.get(f"job:{job_id}:eval")
        if raw:
            return {"status": "done", **json.loads(raw)}
    except Exception:
        # Redis unavailable — fall through to the durable SQLite store.
        pass
    finally:
        await r.aclose()

    # Fallback: metrics persisted to SQLite survive a Redis restart / TTL expiry.
    from app.state.experiments_db import get_final_metrics

    persisted = get_final_metrics(job_id)
    if persisted:
        return {"status": "done", **persisted}
    return {"status": "not_ready", "perplexity": None, "bleu": None}


@router.post("/jobs/{job_id}/infer")
async def infer(job_id: str, req: InferRequest):
    import torch

    state = _get_job_status_from_redis(job_id)
    if state.get("status") != "done":
        raise HTTPException(status_code=400, detail="Job not complete")

    adapter_dir = state.get("output_path") or _resolve_job_dir(job_id)
    if not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter directory not found")

    with _INFER_CACHE_LOCK:
        if job_id not in _INFER_CACHE:
            config_path = os.path.join(adapter_dir, "adapter_config.json")
            if not os.path.exists(config_path):
                raise HTTPException(status_code=404, detail="adapter_config.json not found")
            with open(config_path) as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path", "")
            if not base_model_name:
                raise HTTPException(status_code=500, detail="base_model_name_or_path missing")

            try:
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name, device_map="auto", torch_dtype=torch.float16
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
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {"response": response}
