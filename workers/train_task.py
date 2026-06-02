import json
import os
import traceback
from datetime import datetime, timezone

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
    task_self,
    job_id: str,
    model_cfg: dict,
    lora_cfg: dict,
    train_cfg: dict,
    dataset_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
):
    """Core logic, separated so it can be unit-tested without a live Celery broker."""
    from app.state.experiment_state import save_run_metrics, write_job_status

    r = redis.from_url(REDIS_URL)
    status_key = f"job:{job_id}:status"
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        # Durable record so GET /jobs works even if Redis is unavailable
        write_job_status(job_id, "running", started_at=started_at)
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
                dataset_path,
                tokenizer,
                cfg.max_seq_length,
                hub_dataset_id=hub_dataset_id,
                hub_split=hub_split,
                instruction_col=instruction_col,
                output_col=output_col,
            )
            n_eval = max(1, int(0.2 * len(full_dataset)))
            eval_sample = full_dataset.shuffle(seed=42).select(range(n_eval))
            eval_results = evaluate_model(model, tokenizer, eval_sample)
            r.set(f"job:{job_id}:eval", json.dumps(eval_results))
        except Exception:
            # Eval failure must not fail the whole job
            r.set(f"job:{job_id}:eval", json.dumps({"perplexity": None, "bleu": None}))

        # Persist step-level metrics to the queryable run_metrics table.
        # The callback appends JSON payloads to a Redis list; read them all.
        try:
            raw_entries = r.lrange(f"job:{job_id}:loss_history", 0, -1)
            if raw_entries:
                loss_history = [json.loads(e) for e in raw_entries]
                save_run_metrics(job_id, loss_history)
        except Exception:
            pass

        finished_at = datetime.now(timezone.utc).isoformat()
        # Durable SQLite record — survives Redis restart
        write_job_status(job_id, "done", finished_at=finished_at, output_path=output_path)
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


@celery_app.task(bind=True, name="workers.train_task.run_finetune", time_limit=7200)
def run_finetune(
    self,
    job_id: str,
    model_cfg: dict,
    lora_cfg: dict,
    train_cfg: dict,
    dataset_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
):
    return _run_finetune_impl(
        self,
        job_id,
        model_cfg,
        lora_cfg,
        train_cfg,
        dataset_path,
        hub_dataset_id=hub_dataset_id,
        hub_split=hub_split,
        instruction_col=instruction_col,
        output_col=output_col,
    )
