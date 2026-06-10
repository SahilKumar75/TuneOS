import os

from celery import Celery
from celery.signals import worker_process_init
from kombu import Queue

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "finetune_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    # Ensure the worker registers all task modules at startup.
    include=[
        "workers.train_task",
        "workers.dpo_task",
        "workers.kd_task",
        "workers.merge_task",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=7200,  # 2 hour hard limit per job
    # Named queues — lets workers be pinned to specific workload types.
    # e.g. `celery worker -Q sft` for GPU-heavy SFT, `-Q dpo` for DPO, etc.
    task_queues=(
        Queue("sft"),
        Queue("dpo"),
        Queue("kd"),
        Queue("default"),
    ),
    task_default_queue="default",
    task_routes={
        "workers.train_task.run_finetune": {"queue": "sft"},
        "workers.dpo_task.run_dpo": {"queue": "dpo"},
        "workers.kd_task.run_distill": {"queue": "kd"},
    },
)


@worker_process_init.connect
def _configure_worker_logging(**kwargs):
    """Enable structured JSON logging in every Celery worker process."""
    try:
        from app.logging_config import configure_logging

        configure_logging()
    except Exception:
        pass  # never crash the worker process
