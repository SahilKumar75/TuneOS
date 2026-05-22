import json
import redis
import os
from transformers import TrainerCallback

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class RedisLossCallback(TrainerCallback):
    """
    Publishes training loss to Redis channel 'job:<job_id>:loss'
    after every logging step so the frontend can stream it live.
    """
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.redis = redis.from_url(REDIS_URL)
        self.channel = f"job:{job_id}:loss"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            payload = {
                "step": state.global_step,
                "loss": round(logs["loss"], 4),
                "epoch": round(state.epoch, 2) if state.epoch else 0,
                "learning_rate": logs.get("learning_rate", 0),
            }
            self.redis.publish(self.channel, json.dumps(payload))
