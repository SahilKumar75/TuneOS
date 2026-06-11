import json
import logging
import os
import traceback
from datetime import datetime, timezone

import redis

_logger = logging.getLogger(__name__)

from trainer.config import LoraConfig, ModelConfig, TrainingConfig
from trainer.evaluate import evaluate_run
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


# Shared, framework-light eval lives in trainer.evaluate so the Modal runner can
# reuse it without importing celery/redis. Kept as a module alias for callers.
_compute_eval = evaluate_run


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
    compose_adapters: bool = False,
    overlay_technique: str = "lora",
):
    """Core logic, separated so it can be unit-tested without a live Celery broker."""
    from db.experiments_db import save_run_metrics, write_job_status

    r = redis.from_url(REDIS_URL)
    status_key = f"job:{job_id}:status"
    started_at = datetime.now(timezone.utc).isoformat()

    # Retrieve the short-TTL token stored by the API before enqueue; never travels in task kwargs.
    _stored_token = r.getdel(f"job:{job_id}:hf_token")
    if _stored_token:
        os.environ["HF_TOKEN"] = (
            _stored_token.decode() if isinstance(_stored_token, bytes) else _stored_token
        )

    try:
        # Early-fail for gated models with no token — avoids a silent hang during download.
        _model_name = model_cfg.get("model_name", "")
        _hf_token = os.getenv("HF_TOKEN", "")
        if not _hf_token and any(_model_name.startswith(p) for p in _GATED_MODEL_PREFIXES):
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
        r.expire(status_key, 21600)  # 6h — covers Modal max job time

        backend = train_cfg.get("compute_backend", "local")
        output_path = os.path.join(train_cfg["output_dir"], job_id)

        # Fail fast on an unknown backend rather than silently running locally.
        # "hf_spaces" intentionally uses the local path — the Space supplies the
        # GPU via the @spaces.GPU decorator on this function.
        if backend not in ("local", "modal", "hf_spaces"):
            raise ValueError(
                f"Unknown compute_backend '{backend}' for job {job_id}. "
                "Expected one of: local, modal, hf_spaces."
            )

        if backend == "modal":
            # Train on a Modal T4 GPU. The remote run produces the adapter and
            # eval metrics; we write the adapter to local disk and persist
            # metrics through the same path as a local run.
            from workers.modal_runner import run_on_modal

            # #16 — signal GPU provisioning so UI shows a banner instead of spinner
            r.set(
                status_key,
                json.dumps(
                    {
                        "status": "provisioning",
                        "job_id": job_id,
                        "message": "Provisioning GPU on Modal...",
                    }
                ),
            )
            r.expire(status_key, 21600)

            result = run_on_modal(
                job_id=job_id,
                model_cfg=model_cfg,
                lora_cfg=lora_cfg,
                train_cfg=train_cfg,
                dataset_path=dataset_path,
                output_path=output_path,
                hub_dataset_id=hub_dataset_id,
                hub_split=hub_split,
                instruction_col=instruction_col,
                output_col=output_col,
            )
            eval_results = result.get("eval") or {
                "perplexity": None,
                "rouge1": None,
                "bleu": None,
            }
            # Replay any returned step metrics into Redis so the `finally` block
            # persists them uniformly with the local path.
            for entry in result.get("loss_history", []):
                r.rpush(f"job:{job_id}:loss_history", json.dumps(entry))
            r.set(f"job:{job_id}:eval", json.dumps(eval_results))
        else:
            output_path, model, tokenizer = finetune(
                model_cfg=ModelConfig(**model_cfg),
                lora_cfg=LoraConfig(**lora_cfg),
                train_cfg=TrainingConfig(**train_cfg),
                dataset_path=dataset_path,
                job_id=job_id,
                hub_dataset_id=hub_dataset_id,
                hub_split=hub_split,
                instruction_col=instruction_col,
                output_col=output_col,
                redis_client=r,
            )

            if compose_adapters and overlay_technique:
                from trainer.adapters import stack_adapter

                mixed = stack_adapter(
                    model,
                    technique=overlay_technique,
                    r=lora_cfg.get("r", 8),
                    lora_alpha=lora_cfg.get("lora_alpha", 16),
                    lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                )
                mixed.save_pretrained(output_path)

            if train_cfg.get("eval_split_ratio", 0.1) == 0.0:
                eval_results = {"perplexity": None, "rouge1": None, "bleu": None}
            else:
                eval_results = _compute_eval(
                    model,
                    tokenizer,
                    model_cfg,
                    train_cfg,
                    dataset_path,
                    hub_dataset_id,
                    hub_split,
                    instruction_col,
                    output_col,
                )
            r.set(f"job:{job_id}:eval", json.dumps(eval_results))

        # Persist eval to SQLite as well, so GET /jobs/{id}/eval survives a Redis
        # restart / TTL expiry (durable fallback alongside the Redis copy).
        try:
            from db.experiments_db import save_final_metrics

            save_final_metrics(
                job_id,
                {k: v for k, v in eval_results.items() if isinstance(v, int | float)},
            )
        except Exception:
            _logger.warning("Failed to persist final eval metrics for job %s", job_id, exc_info=True)

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
        r.expire(status_key, 172800)  # 48h
        r.expire(f"job:{job_id}:eval", 172800)  # 48h
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
        r.expire(status_key, 172800)  # 48h
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
            _logger.warning("Failed to persist loss history for job %s", job_id, exc_info=True)


@celery_app.task(bind=True, name="workers.train_task.run_finetune", time_limit=7200, queue="sft")
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
    compose_adapters: bool = False,
    overlay_technique: str = "lora",
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
        compose_adapters=compose_adapters,
        overlay_technique=overlay_technique,
    )
