import json
import os
from typing import Any

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def get_job_status(job_id: str) -> dict[str, Any]:
    r = redis.from_url(REDIS_URL)
    status_key = f"job:{job_id}:status"
    status_raw = r.get(status_key)

    if not status_raw:
        return {"status": "not_found", "job_id": job_id}

    return json.loads(status_raw)
