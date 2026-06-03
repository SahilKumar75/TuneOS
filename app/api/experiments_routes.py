"""Experiment tracking API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.state.experiments_db import (
    _get_conn,
    _init_db,
    get_run_metrics,
    list_registered_models,
    register_model,
)

router = APIRouter()


# ── Runs ──────────────────────────────────────────────────────────


@router.get("/experiments")
async def list_experiments():
    """List all recorded runs, most-recent first."""
    try:
        _init_db()
        with _get_conn() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
        return {"runs": [dict(r) for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete a run record by id."""
    try:
        _init_db()
        with _get_conn() as conn:
            conn.execute("DELETE FROM runs WHERE id = ?", (experiment_id,))
        return {"status": "deleted"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/experiments/compare")
async def compare_runs(ids: str, metric: str = "loss"):
    """Return step-level metric values for multiple runs overlaid.

    Query params:
      ids    — comma-separated run IDs (e.g. ``?ids=run1,run2``)
      metric — metric key to compare (default: ``loss``); also accepts
               ``eval_loss``, ``learning_rate``, ``epoch``

    Response:
      {
        "metric": "loss",
        "runs": {
          "run_id_1": [{"step": 0, "value": 1.23}, ...],
          "run_id_2": [...]
        }
      }
    """
    run_ids = [r.strip() for r in ids.split(",") if r.strip()]
    if not run_ids:
        raise HTTPException(status_code=422, detail="ids must be a non-empty comma-separated list")
    if len(run_ids) > 10:
        raise HTTPException(status_code=422, detail="At most 10 runs can be compared at once")
    data = get_run_metrics(run_ids, metric_key=metric)
    return {"metric": metric, "runs": data}


# ── Model registry ────────────────────────────────────────────────


class RegisterModelRequest(BaseModel):
    name: str
    run_id: str
    alias: str = "latest"
    metric_snapshot: dict = {}


@router.get("/experiments/models")
async def list_models():
    """Return all entries in the model registry."""
    return {"models": list_registered_models()}


@router.post("/experiments/models", status_code=201)
async def register_model_endpoint(req: RegisterModelRequest):
    """Register (or update) a named model pointing to a training run.

    This is the "Register" action on the Results step: it records which
    run produced the canonical version of a given model name.
    """
    if not req.name.strip() or not req.run_id.strip():
        raise HTTPException(status_code=422, detail="name and run_id are required")
    register_model(
        req.name.strip(),
        req.run_id.strip(),
        alias=req.alias or "latest",
        metric_snapshot=req.metric_snapshot,
    )
    return {"status": "registered", "name": req.name, "run_id": req.run_id}


@router.delete("/experiments/models/{name}")
async def delete_registered_model(name: str):
    """Remove a named model from the registry."""
    try:
        _init_db()
        with _get_conn() as conn:
            conn.execute("DELETE FROM registered_models WHERE name = ?", (name,))
        return {"status": "deleted"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
