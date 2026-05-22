import reflex as rx
import redis, json, os, asyncio
from typing import List

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class LossPoint(rx.Base):
    step: int
    loss: float
    epoch: float

class JobState(rx.State):
    job_id: str = ""
    status: str = "idle"          # idle | running | done | failed
    loss_history: List[LossPoint] = []
    output_path: str = ""
    error_msg: str = ""

    @rx.background
    async def poll_job(self, job_id: str):
        """
        Subscribe to Redis pub/sub channel for live loss updates.
        Updates state after every logged step.
        """
        async with self:
            self.job_id = job_id
            self.status = "running"
            self.loss_history = []

        r = redis.from_url(REDIS_URL)
        pubsub = r.pubsub()
        pubsub.subscribe(f"job:{job_id}:loss")
        status_key = f"job:{job_id}:status"

        while True:
            msg = pubsub.get_message(ignore_subscribe_messages=True)
            if msg:
                data = json.loads(msg["data"])
                async with self:
                    self.loss_history.append(
                        LossPoint(
                            step=data["step"],
                            loss=data["loss"],
                            epoch=data["epoch"],
                        )
                    )
            # Check if job finished
            status_raw = r.get(status_key)
            if status_raw:
                status_data = json.loads(status_raw)
                if status_data["status"] in ("done", "failed"):
                    async with self:
                        self.status = status_data["status"]
                        self.output_path = status_data.get("output_path", "")
                        self.error_msg = status_data.get("error", "")
                    break

            await asyncio.sleep(1)
