import json
import os
import time

import redis
import torch
from transformers import TrainerCallback

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class RedisLossCallback(TrainerCallback):
    """
    Publishes training progress to Redis channel 'job:<job_id>:progress'
    after every logging step so the frontend can stream live metrics.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.redis = redis.from_url(REDIS_URL)
        self.channel = f"job:{job_id}:progress"
        self._start_time: float = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return

        gpu_mem = 0.0
        try:
            if torch.cuda.is_available():
                gpu_mem = round(torch.cuda.memory_allocated() / 1e9, 2)
        except Exception:
            pass

        payload = {
            "step": state.global_step,
            "loss": round(logs["loss"], 4),
            "epoch": round(state.epoch, 2) if state.epoch else 0,
            "learning_rate": logs.get("learning_rate", 0),
            "eval_loss": logs.get("eval_loss"),
            "total_steps": state.max_steps or 0,
            "elapsed_seconds": int(time.time() - self._start_time),
            "gpu_memory_used_gb": gpu_mem,
            "status": "running",
        }
        self.redis.publish(self.channel, json.dumps(payload))
        # Also accumulate in a list key so the worker can persist to run_metrics
        self.redis.rpush(f"job:{self.job_id}:loss_history", json.dumps(payload))

    def on_train_end(self, args, state, control, **kwargs):
        payload = {
            "status": "done",
            "step": state.global_step,
            "elapsed_seconds": int(time.time() - self._start_time),
        }
        self.redis.publish(self.channel, json.dumps(payload))
