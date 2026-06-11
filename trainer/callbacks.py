import json
import os
import time

import redis
import torch
from transformers import TrainerCallback

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_FLUSH_EVERY = 10  # #12 — batch rpush every N steps to cut Redis round-trips


class RedisLossCallback(TrainerCallback):
    """
    Publishes training progress to Redis channel 'job:<job_id>:progress'
    after every logging step so the frontend can stream live metrics.

    Step payloads are batched into a local buffer and flushed to the
    loss_history list key every _FLUSH_EVERY steps (and on train end) to
    reduce Redis I/O for long runs. Per-step publish() is kept fire-and-forget
    for the real-time UI channel — it does not block training.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.redis = redis.from_url(REDIS_URL)
        self.channel = f"job:{job_id}:progress"
        self._start_time: float = 0.0
        self._buffer: list[str] = []

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()

    def _flush_buffer(self):
        if self._buffer:
            self.redis.rpush(f"job:{self.job_id}:loss_history", *self._buffer)
            self._buffer = []

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
            "grad_norm": (round(float(logs["grad_norm"]), 4) if "grad_norm" in logs else None),
            "total_steps": state.max_steps or 0,
            "elapsed_seconds": int(time.time() - self._start_time),
            "gpu_memory_used_gb": gpu_mem,
            "status": "running",
        }
        encoded = json.dumps(payload)
        # Fire-and-forget publish for real-time UI — does not block training
        self.redis.publish(self.channel, encoded)
        # Buffer for batched persistence
        self._buffer.append(encoded)
        if state.global_step % _FLUSH_EVERY == 0:
            self._flush_buffer()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        payload = {
            "step": state.global_step,
            "epoch": round(state.epoch, 2) if state.epoch else 0,
            "eval_loss": round(float(metrics["eval_loss"]), 4),
            "elapsed_seconds": int(time.time() - self._start_time),
            "status": "running",
        }
        encoded = json.dumps(payload)
        self.redis.publish(self.channel, encoded)
        self._buffer.append(encoded)

    def on_train_end(self, args, state, control, **kwargs):
        self._flush_buffer()
        payload = {
            "status": "done",
            "step": state.global_step,
            "elapsed_seconds": int(time.time() - self._start_time),
        }
        self.redis.publish(self.channel, json.dumps(payload))
