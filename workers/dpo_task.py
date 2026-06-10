"""Celery task for DPO (preference) fine-tuning.

Mirrors workers/train_task.py: writes durable job status to SQLite + Redis at
start/done/failure, and persists step metrics on completion. DPO has no
post-training perplexity eval, so the eval step is omitted.
"""

import json
import os
import traceback
from datetime import datetime, timezone

import redis

from trainer.config import DPOConfig, LoraConfig, ModelConfig
from trainer.dpo import train_dpo
from workers.celery_app import celery_app

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    import spaces

    _gpu_decorator = spaces.GPU
except ImportError:
    _gpu_decorator = lambda fn: fn  # noqa: E731


@_gpu_decorator
def _run_dpo_impl(
    task_self,
    job_id: str,
    model_cfg: dict,
    lora_cfg: dict,
    dpo_cfg: dict,
    dataset_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    prompt_col: str = "prompt",
    chosen_col: str = "chosen",
    rejected_col: str = "rejected",
):
    """Core logic, separated so it can be unit-tested without a live broker."""
    from app.state.experiments_db import save_run_metrics, write_job_status

    r = redis.from_url(REDIS_URL)
    status_key = f"job:{job_id}:status"
    started_at = datetime.now(timezone.utc).isoformat()

    _stored_token = r.getdel(f"job:{job_id}:hf_token")
    if _stored_token:
        os.environ["HF_TOKEN"] = _stored_token.decode() if isinstance(_stored_token, bytes) else _stored_token

    try:
        write_job_status(job_id, "running", started_at=started_at)
        r.set(status_key, json.dumps({"status": "running", "job_id": job_id}))

        output_path, _, _ = train_dpo(
            ModelConfig(**model_cfg),
            LoraConfig(**lora_cfg),
            DPOConfig(**dpo_cfg),
            dataset_path,
            job_id,
            hub_dataset_id=hub_dataset_id,
            hub_split=hub_split,
            prompt_col=prompt_col,
            chosen_col=chosen_col,
            rejected_col=rejected_col,
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


@celery_app.task(bind=True, name="workers.dpo_task.run_dpo", time_limit=7200)
def run_dpo(
    self,
    job_id: str,
    model_cfg: dict,
    lora_cfg: dict,
    dpo_cfg: dict,
    dataset_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    prompt_col: str = "prompt",
    chosen_col: str = "chosen",
    rejected_col: str = "rejected",
):
    return _run_dpo_impl(
        self,
        job_id,
        model_cfg,
        lora_cfg,
        dpo_cfg,
        dataset_path,
        hub_dataset_id=hub_dataset_id,
        hub_split=hub_split,
        prompt_col=prompt_col,
        chosen_col=chosen_col,
        rejected_col=rejected_col,
    )
