"""Experiment tracking — persists all fine-tuning runs across sessions in SQLite."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

import reflex as rx
from pydantic import BaseModel

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.getenv("EXPERIMENT_DB", os.path.join(_PROJECT_ROOT, "storage", "experiments.db"))


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                name TEXT,
                model_id TEXT,
                model_source TEXT,
                technique TEXT,
                epochs INTEGER,
                learning_rate TEXT,
                lora_r INTEGER,
                batch_size INTEGER,
                dataset_name TEXT,
                user_intent TEXT,
                final_loss REAL,
                perplexity REAL,
                started_at TEXT,
                finished_at TEXT,
                status TEXT,
                output_path TEXT,
                loss_history TEXT
            )
        """)


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


def save_experiment_run(run_data: dict[str, Any]):
    """Called from FinetuneState._save_experiment_record() — writes to SQLite."""
    try:
        _init_db()
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (id, name, model_id, model_source, technique, epochs, learning_rate,
                 lora_r, batch_size, dataset_name, user_intent, final_loss, perplexity,
                 started_at, finished_at, status, output_path, loss_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_data.get("id", ""),
                    run_data.get("name", ""),
                    run_data.get("model_id", ""),
                    run_data.get("model_source", "hub"),
                    run_data.get("technique", "qlora"),
                    run_data.get("epochs", 3),
                    run_data.get("learning_rate", "2e-4"),
                    run_data.get("lora_r", 16),
                    run_data.get("batch_size", 4),
                    run_data.get("dataset_name", ""),
                    run_data.get("user_intent", ""),
                    run_data.get("final_loss", 0.0),
                    run_data.get("perplexity", 0.0),
                    run_data.get("started_at", ""),
                    run_data.get("finished_at", ""),
                    run_data.get("status", "unknown"),
                    run_data.get("output_path", ""),
                    json.dumps(run_data.get("loss_history", [])),
                ),
            )
    except Exception:
        pass
