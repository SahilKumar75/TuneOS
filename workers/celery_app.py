import os

from celery import Celery

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
)
