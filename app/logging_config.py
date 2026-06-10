"""Structured JSON logging with per-request/per-task trace IDs.

Usage
-----
API startup::

    from app.logging_config import configure_logging
    configure_logging()

FastAPI middleware (already wired in app/api/__init__.py)::

    from app.logging_config import TraceIDMiddleware

Celery worker startup (wired via ``worker_process_init`` signal in
workers/celery_app.py)::

    from app.logging_config import configure_logging
    configure_logging()

Within any module, get/set the current trace ID::

    from app.logging_config import get_trace_id, set_trace_id
    logger.info("job started", extra={"job_id": job_id})
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime, timezone

# Per-coroutine / per-thread trace ID — set by middleware or task preamble.
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    return _trace_id_var.get()


def set_trace_id(tid: str) -> None:
    _trace_id_var.set(tid)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    _SKIP = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        record.message = record.getMessage()
        obj: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }
        tid = get_trace_id()
        if tid:
            obj["trace_id"] = tid

        # Attach any extra fields the caller passed via extra={...}
        for key, val in record.__dict__.items():
            if key not in self._SKIP and not key.startswith("_"):
                obj[key] = val

        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)

        return json.dumps(obj, default=str)


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger with the JSON formatter.

    Safe to call multiple times — idempotent.
    """
    lvl = getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO)
    root = logging.getLogger()
    if any(
        isinstance(h, logging.StreamHandler) and isinstance(h.formatter, _JsonFormatter)
        for h in root.handlers
    ):
        return  # already configured

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root.setLevel(lvl)
    root.addHandler(handler)
    # Silence noisy third-party loggers
    for noisy in ("transformers", "datasets", "peft", "torch", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# FastAPI middleware
# ---------------------------------------------------------------------------

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    class TraceIDMiddleware(BaseHTTPMiddleware):
        """Inject a trace_id into every request context and response header."""

        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            tid = request.headers.get("X-Trace-ID") or new_trace_id()
            set_trace_id(tid)
            response = await call_next(request)
            response.headers["X-Trace-ID"] = tid
            return response

except ImportError:
    # Starlette not available (e.g. in unit tests that import this module standalone)
    TraceIDMiddleware = None  # type: ignore[assignment,misc]
