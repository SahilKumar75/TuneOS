"""
Deprecated — all job state now lives in FinetuneState.
This stub exists only to prevent import errors in legacy pages.
"""

import reflex as rx


class JobState(rx.State):
    job_id: str = ""
    status: str = "idle"
    loss_history: list = []
    output_path: str = ""
    error_msg: str = ""
