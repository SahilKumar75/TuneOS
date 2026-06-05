import json
import os
import traceback
from datetime import datetime, timezone

import redis

from trainer.config import LoraConfig, ModelConfig, TrainingConfig
from trainer.finetune import finetune
from workers.celery_app import celery_app

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Models that reliably require an HF token to download (accept-on-Hub gating).
# Only include families where a missing token causes a cryptic silent failure.
_GATED_MODEL_PREFIXES = (
    "meta-llama/",
    "meta-llama/Meta-Llama",
)

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
    from app.state.experiments_db import save_run_metrics, write_job_status

    r = redis.from_url(REDIS_URL)
    status_key = f"job:{job_id}:status"
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        # Early-fail for gated models with no token — avoids a silent hang during download.
        _model_name = model_cfg.get("model_name", "")
        _hf_token = model_cfg.get("hf_token", "") or os.getenv("HF_TOKEN", "")
        if not _hf_token and any(
            _model_name.startswith(p) for p in _GATED_MODEL_PREFIXES
        ):
            _msg = (
                f"Model '{_model_name}' requires a Hugging Face token. "
                "Add HF_TOKEN to your .env file or paste it in the model settings."
            )
            write_job_status(job_id, "failed", started_at=started_at, finished_at=started_at)
            r.set(status_key, json.dumps({"status": "failed", "job_id": job_id, "error": _msg}))
            raise RuntimeError(_msg)

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
            from trainer.dataset import load_and_tokenize, load_instruction_pairs
            from trainer.evaluate import (
                evaluate_model,
                evaluate_references,
                generate_predictions,
            )

            seed = TrainingConfig(**train_cfg).seed
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
            eval_sample = full_dataset.shuffle(seed=seed).select(range(n_eval))
            eval_results = evaluate_model(model, tokenizer, eval_sample)

            # Reference metrics (ROUGE-1/BLEU) need generated predictions vs. the
            # raw reference text. Sample the same indices from the raw pairs.
            try:
                raw_pairs = load_instruction_pairs(
                    dataset_path,
                    hub_dataset_id=hub_dataset_id,
                    hub_split=hub_split,
                    instruction_col=instruction_col,
                    output_col=output_col,
                )
                n_ref = min(len(raw_pairs), 32)
                ref_sample = raw_pairs.shuffle(seed=seed).select(range(n_ref))
                instructions = [row["instruction"] for row in ref_sample]
                references = [row["output"] for row in ref_sample]
                predictions = generate_predictions(model, tokenizer, instructions)
                eval_results.update(evaluate_references(predictions, references))
            except Exception:
                # Reference metrics are best-effort; never let them break eval.
                eval_results.setdefault("rouge1", None)
                eval_results.setdefault("bleu", None)

            r.set(f"job:{job_id}:eval", json.dumps(eval_results))
        except Exception:
            # Eval failure must not fail the whole job
            r.set(
                f"job:{job_id}:eval",
                json.dumps({"perplexity": None, "rouge1": None, "bleu": None}),
            )

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
        # Surface a remediation hint for OOM so the UI can guide the user.
        suggestion = getattr(e, "suggestion", "")
        payload = {
            "status": "failed",
            "job_id": job_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        if suggestion:
            payload["suggestion"] = suggestion
        r.set(status_key, json.dumps(payload))
        raise

    finally:
        # Persist accumulated step-level metrics even if training failed, then
        # drop the Redis list so it doesn't grow unboundedly across jobs.
        try:
            key = f"job:{job_id}:loss_history"
            raw_entries = r.lrange(key, 0, -1)
            if raw_entries:
                loss_history = [json.loads(e) for e in raw_entries]
                save_run_metrics(job_id, loss_history)
                r.delete(key)
        except Exception:
            pass


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
