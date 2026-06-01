"""Experiment tracking API routes."""

from __future__ import annotations

import os
import sqlite3

from fastapi import APIRouter, HTTPException

router = APIRouter()


def _get_db_path() -> str:
    return os.getenv("EXPERIMENT_DB", "./storage/experiments.db")


@router.get("/experiments")
async def list_experiments():
    try:
        db_path = _get_db_path()
        if not os.path.exists(db_path):
            return {"runs": []}
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
        conn.close()
        return {"runs": [dict(r) for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    try:
        db_path = _get_db_path()
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM runs WHERE id = ?", (experiment_id,))
        conn.commit()
        conn.close()
        return {"status": "deleted"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
