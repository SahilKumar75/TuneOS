"""Experiment tracking — Reflex state layer over the SQLite persistence in experiments_db."""

from __future__ import annotations

import reflex as rx
from pydantic import BaseModel

from app.state.experiments_db import (
    DB_PATH,
    _get_conn,
    _init_db,
    get_run_metrics,
    list_registered_models,
    register_model,
    save_experiment_run,
    save_final_metrics,
    save_run_metrics,
    save_run_params,
    write_job_status,
)

# Re-exported for backward compatibility with existing imports.
__all__ = [
    "DB_PATH",
    "ExperimentRun",
    "ExperimentState",
    "RegisteredModel",
    "get_run_metrics",
    "list_registered_models",
    "register_model",
    "save_experiment_run",
    "save_final_metrics",
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
    # ── Comparison view (overlaid loss curves) ────────────────────
    compare_metric: str = "loss"  # loss | eval_loss | learning_rate
    compare_data: list[dict] = []  # pivoted rows: {"step", "run0", "run1", ...}
    compare_labels: list[str] = []

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
    def set_compare_metric(self, metric: str):
        self.compare_metric = metric

    @rx.event
    def load_comparison(self):
        """Fetch step-level metrics for the selected runs and pivot them into
        the wide shape the overlaid chart expects ({step, run0, run1, ...})."""
        ids = self.selected_run_ids[:6]  # chart palette caps at 6 series
        series = get_run_metrics(ids, self.compare_metric)
        ordered = [rid for rid in ids if rid in series]
        step_map: dict[int, dict] = {}
        for i, rid in enumerate(ordered):
            for pt in series[rid]:
                row = step_map.setdefault(int(pt["step"]), {"step": int(pt["step"])})
                row[f"run{i}"] = pt["value"]
        self.compare_data = [step_map[s] for s in sorted(step_map)]
        name_by_id = {r.id: (r.name or r.id[:8]) for r in self.runs}
        self.compare_labels = [name_by_id.get(rid, rid[:8]) for rid in ordered]

    @rx.event
    def delete_run(self, run_id: str):
        try:
            with _get_conn() as conn:
                conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            self.runs = [r for r in self.runs if r.id != run_id]
            self.selected_run_ids = [i for i in self.selected_run_ids if i != run_id]
        except Exception:
            pass


# ── Typed model for the chart overlay ────────────────────────────


class ComparePoint(BaseModel):
    """One step-level data point for the comparison chart."""

    step: int = 0
    value: float = 0.0
    run_id: str = ""


class RegisteredModel(BaseModel):
    """A named model entry in the registry."""

    name: str = ""
    run_id: str = ""
    alias: str = "latest"
    registered_at: str = ""
    perplexity: float = 0.0
    final_loss: float = 0.0


class ModelRegistryState(rx.State):
    """Manages the model registry table and run comparison state."""

    models: list[RegisteredModel] = []
    is_registering: bool = False
    register_error: str = ""
    register_name: str = ""
    # Tracks the run_id that was most recently registered successfully.
    # Empty string means no registration has occurred in this session.
    registered_run_id: str = ""

    @rx.event
    def load_models(self):
        raw = list_registered_models()
        self.models = [
            RegisteredModel(
                name=m["name"],
                run_id=m["run_id"],
                alias=m["alias"],
                registered_at=m["registered_at"],
                perplexity=float(m["metric_snapshot"].get("perplexity") or 0.0),
                final_loss=float(m["metric_snapshot"].get("final_loss") or 0.0),
            )
            for m in raw
        ]

    @rx.event
    def register_current_run(self, run_id: str, name: str, metrics: dict):
        """Register a completed run under a user-provided name."""
        if not name.strip():
            self.register_error = "Model name cannot be empty"
            return
        self.is_registering = True
        self.register_error = ""
        try:
            register_model(name.strip(), run_id, alias="latest", metric_snapshot=metrics)
        except Exception as exc:
            self.register_error = str(exc)
        finally:
            self.is_registering = False
        return ModelRegistryState.load_models

    @rx.event
    def set_register_name(self, value: str):
        self.register_name = value
        self.registered_run_id = ""
        self.register_error = ""

    @rx.event
    def clear_registration(self):
        """Reset registration state when a new experiment is started."""
        self.register_name = ""
        self.registered_run_id = ""
        self.register_error = ""

    @rx.event
    def do_register(self, run_id: str, perplexity: float, final_loss: float):
        """Register the current run using ``self.register_name``.

        Only snapshots metrics when they are passed in at click-time (not during
        render), so perplexity will reflect the actual evaluated value.
        """
        if not self.register_name.strip():
            self.register_error = "Model name cannot be empty"
            return
        self.is_registering = True
        self.register_error = ""
        self.registered_run_id = ""
        try:
            register_model(
                self.register_name.strip(),
                run_id,
                alias="latest",
                metric_snapshot={"perplexity": perplexity, "final_loss": final_loss},
            )
            self.registered_run_id = run_id
        except Exception as exc:
            self.register_error = str(exc)
        finally:
            self.is_registering = False
        return ModelRegistryState.load_models

    @rx.event
    def delete_model(self, name: str):
        try:
            with _get_conn() as conn:
                conn.execute("DELETE FROM registered_models WHERE name = ?", (name,))
            self.models = [m for m in self.models if m.name != name]
        except Exception:
            pass
