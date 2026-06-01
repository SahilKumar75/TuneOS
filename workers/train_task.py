import json
import os
import traceback

import redis

from trainer.config import LoraConfig, ModelConfig, TrainingConfig
from trainer.finetune import finetune
from workers.celery_app import celery_app

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    import spaces
    _gpu_decorator = spaces.GPU
except ImportError:
    _gpu_decorator = lambda fn: fn  # noqa: E731


@_gpu_decorator
def _run_finetune_impl(
    task_self, job_id: str, model_cfg: dict, lora_cfg: dict, train_cfg: dict, dataset_path: str,
    hub_dataset_id: str = "", hub_split: str = "train",
    instruction_col: str = "instruction", output_col: str = "output",
):
    """Core logic, separated so it can be unit-tested without a live Celery broker."""
    r = redis.from_url(REDIS_URL)
    status_key = f"job:{job_id}:status"

    try:
        r.set(status_key, json.dumps({"status": "running", "job_id": job_id}))

        cfg = ModelConfig(**model_cfg)
        output_path, model, tokenizer = finetune(
            model_cfg=cfg,
            lora_cfg=LoraConfig(**lora_cfg),
            train_cfg=TrainingConfig(**train_cfg),
            dataset_path=dataset_path,
            job_id=job_id,
            hub_dataset_id=hub_dataset_id,
            hub_split=hub_split,
            instruction_col=instruction_col,
            output_col=output_col,
        )

        # Evaluate on a 20% random sample of the training data
        try:
            from trainer.dataset import load_and_tokenize
            from trainer.evaluate import evaluate_model

            full_dataset = load_and_tokenize(
                dataset_path, tokenizer, cfg.max_seq_length,
                hub_dataset_id=hub_dataset_id, hub_split=hub_split,
                instruction_col=instruction_col, output_col=output_col,
            )
            n_eval = max(1, int(0.2 * len(full_dataset)))
            eval_sample = full_dataset.shuffle(seed=42).select(range(n_eval))
            eval_results = evaluate_model(model, tokenizer, eval_sample)
            r.set(f"job:{job_id}:eval", json.dumps(eval_results))
        except Exception:
            # Eval failure must not fail the whole job
            r.set(f"job:{job_id}:eval", json.dumps({"perplexity": None, "bleu": None}))

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


@celery_app.task(bind=True, name="workers.train_task.run_finetune", time_limit=7200)
def run_finetune(
    self, job_id: str, model_cfg: dict, lora_cfg: dict, train_cfg: dict, dataset_path: str,
    hub_dataset_id: str = "", hub_split: str = "train",
    instruction_col: str = "instruction", output_col: str = "output",
):
    return _run_finetune_impl(
        self, job_id, model_cfg, lora_cfg, train_cfg, dataset_path,
        hub_dataset_id=hub_dataset_id, hub_split=hub_split,
        instruction_col=instruction_col, output_col=output_col,
    )
