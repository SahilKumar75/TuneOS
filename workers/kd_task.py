"""Celery task for knowledge-distillation fine-tuning."""

import json
import os
import traceback
from datetime import datetime, timezone

import redis

from trainer.config import DistillConfig, LoraConfig, ModelConfig
from trainer.kd import distill
from workers.celery_app import celery_app

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    import spaces

    _gpu_decorator = spaces.GPU
except ImportError:
    _gpu_decorator = lambda fn: fn  # noqa: E731


@_gpu_decorator
def _run_distill_impl(
    task_self,
    job_id: str,
    model_cfg: dict,
    lora_cfg: dict,
    distill_cfg: dict,
    dataset_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
):
    from app.state.experiments_db import save_run_metrics, write_job_status

    r = redis.from_url(REDIS_URL)
    status_key = f"job:{job_id}:status"
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        write_job_status(job_id, "running", started_at=started_at)
        r.set(status_key, json.dumps({"status": "running", "job_id": job_id}))

        output_path, _, _ = distill(
            ModelConfig(**model_cfg),
            LoraConfig(**lora_cfg),
            DistillConfig(**distill_cfg),
            dataset_path,
            job_id,
            hub_dataset_id=hub_dataset_id,
            hub_split=hub_split,
            instruction_col=instruction_col,
            output_col=output_col,
        )

        finished_at = datetime.now(timezone.utc).isoformat()
        write_job_status(job_id, "done", finished_at=finished_at, output_path=output_path)
        r.set(
            status_key,
            json.dumps({"status": "done", "job_id": job_id, "output_path": output_path}),
        )
        return output_path

    except Exception as e:
        finished_at = datetime.now(timezone.utc).isoformat()
        write_job_status(job_id, "failed", finished_at=finished_at)
        r.set(
            status_key,
            json.dumps(
                {
                    "status": "failed",
                    "job_id": job_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
        )
        raise

    finally:
        try:
            key = f"job:{job_id}:loss_history"
            raw_entries = r.lrange(key, 0, -1)
            if raw_entries:
                save_run_metrics(job_id, [json.loads(e) for e in raw_entries])
                r.delete(key)
        except Exception:
            pass


@celery_app.task(bind=True, name="workers.kd_task.run_distill", time_limit=7200)
def run_distill(
    self,
    job_id: str,
    model_cfg: dict,
    lora_cfg: dict,
    distill_cfg: dict,
    dataset_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
):
    return _run_distill_impl(
        self,
        job_id,
        model_cfg,
        lora_cfg,
        distill_cfg,
        dataset_path,
        hub_dataset_id=hub_dataset_id,
        hub_split=hub_split,
        instruction_col=instruction_col,
        output_col=output_col,
    )
