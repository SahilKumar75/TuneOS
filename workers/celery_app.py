import logging
import os

from celery import Celery
from celery.signals import setup_logging

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "finetune_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
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
    task_time_limit=7200,
    # #20 — dedicated queues prevent KD OOM from killing SFT jobs
    task_default_queue="sft",
    task_queues={
        "sft": {"exchange": "sft", "routing_key": "sft"},
        "dpo": {"exchange": "dpo", "routing_key": "dpo"},
        "kd": {"exchange": "kd", "routing_key": "kd"},
    },
)


# #22 — structured JSON logging for all worker processes
@setup_logging.connect
def configure_worker_logging(**kwargs):
    try:
        from pythonjsonlogger import jsonlogger  # type: ignore[import]

        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(logging.INFO)
    except ImportError:
        logging.basicConfig(level=logging.INFO)
