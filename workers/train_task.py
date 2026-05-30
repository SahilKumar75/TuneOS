import json
import os
import traceback

import redis

from trainer.config import LoraConfig, ModelConfig, TrainingConfig
from trainer.finetune import finetune
from workers.celery_app import celery_app

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _run_finetune_impl(
    task_self, job_id: str, model_cfg: dict, lora_cfg: dict, train_cfg: dict, dataset_path: str
):
    """Core logic, separated so it can be unit-tested without a live Celery broker."""
    r = redis.from_url(REDIS_URL)
    status_key = f"job:{job_id}:status"

    try:
        r.set(status_key, json.dumps({"status": "running", "job_id": job_id}))

        output_path = finetune(
            model_cfg=ModelConfig(**model_cfg),
            lora_cfg=LoraConfig(**lora_cfg),
            train_cfg=TrainingConfig(**train_cfg),
            dataset_path=dataset_path,
            job_id=job_id,
        )

        r.set(
            status_key,
            json.dumps(
                {
                    "status": "done",
                    "job_id": job_id,
                    "output_path": output_path,
                }
            ),
        )
        return output_path

    except Exception as e:
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


@celery_app.task(bind=True, name="workers.train_task.run_finetune")
def run_finetune(
    self, job_id: str, model_cfg: dict, lora_cfg: dict, train_cfg: dict, dataset_path: str
):
    return _run_finetune_impl(self, job_id, model_cfg, lora_cfg, train_cfg, dataset_path)
