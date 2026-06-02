"""Experiment tracking — Reflex state layer over the SQLite persistence in experiments_db."""

from __future__ import annotations

import reflex as rx
from pydantic import BaseModel

from app.state.experiments_db import (
    DB_PATH,
    _get_conn,
    _init_db,
    save_experiment_run,
    save_run_metrics,
    save_run_params,
    write_job_status,
)

# Re-exported for backward compatibility with existing imports.
__all__ = [
    "DB_PATH",
    "ExperimentRun",
    "ExperimentState",
    "save_experiment_run",
    "save_run_metrics",
    "save_run_params",
    "write_job_status",
]

_init_db()


class ExperimentRun(BaseModel):
    id: str = ""
    name: str = ""
    model_id: str = ""
    model_source: str = "hub"
    technique: str = "qlora"
    epochs: int = 3
    learning_rate: str = "2e-4"
    lora_r: int = 16
    batch_size: int = 4
    dataset_name: str = ""
    user_intent: str = ""
    final_loss: float = 0.0
    perplexity: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    status: str = "unknown"
    output_path: str = ""


class ExperimentState(rx.State):
    runs: list[ExperimentRun] = []
    selected_run_ids: list[str] = []
    is_loading: bool = False

    @rx.var
    def selected_runs(self) -> list[ExperimentRun]:
        ids = set(self.selected_run_ids)
        return [r for r in self.runs if r.id in ids]

    @rx.var
    def completed_runs(self) -> list[ExperimentRun]:
        return [r for r in self.runs if r.status == "done"]

    @rx.event
    def load_runs(self):
        try:
            with _get_conn() as conn:
                rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
            self.runs = [
                ExperimentRun(
                    id=r["id"],
                    name=r["name"] or "",
                    model_id=r["model_id"] or "",
                    model_source=r["model_source"] or "hub",
                    technique=r["technique"] or "qlora",
                    epochs=r["epochs"] or 3,
                    learning_rate=r["learning_rate"] or "2e-4",
                    lora_r=r["lora_r"] or 16,
                    batch_size=r["batch_size"] or 4,
                    dataset_name=r["dataset_name"] or "",
                    user_intent=r["user_intent"] or "",
                    final_loss=r["final_loss"] or 0.0,
                    perplexity=r["perplexity"] or 0.0,
                    started_at=r["started_at"] or "",
                    finished_at=r["finished_at"] or "",
                    status=r["status"] or "unknown",
                    output_path=r["output_path"] or "",
                )
                for r in rows
            ]
        except Exception:
            self.runs = []

    @rx.event
    def toggle_run_selection(self, run_id: str):
        if run_id in self.selected_run_ids:
            self.selected_run_ids = [i for i in self.selected_run_ids if i != run_id]
        else:
            self.selected_run_ids = [*self.selected_run_ids, run_id]

    @rx.event
    def delete_run(self, run_id: str):
        try:
            with _get_conn() as conn:
                conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            self.runs = [r for r in self.runs if r.id != run_id]
            self.selected_run_ids = [i for i in self.selected_run_ids if i != run_id]
        except Exception:
            pass
