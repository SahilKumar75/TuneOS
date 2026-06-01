"""TuneOS REST API package.

Exports ``app_api`` — a FastAPI instance with all routers mounted.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api import (
    datasets_routes,
    experiments_routes,
    jobs_routes,
    models_routes,
    system,
)

app_api = FastAPI(title="TuneOS API")

app_api.include_router(system.router)
app_api.include_router(models_routes.router)
app_api.include_router(datasets_routes.router)
app_api.include_router(jobs_routes.router)
app_api.include_router(experiments_routes.router)

__all__ = ["app_api"]
