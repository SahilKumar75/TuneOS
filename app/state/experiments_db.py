"""Backward-compat shim — real implementation lives in db.experiments_db.

Workers import directly from db.experiments_db to avoid pulling in app/.
App-side code and existing tests continue to use this path unchanged.

Reloading this module (e.g. in tests via importlib.reload) also reloads the
underlying db module so env-var overrides (EXPERIMENT_DB) take effect.
"""

from __future__ import annotations

import importlib
import sys

import db.experiments_db as _db_module

# Re-run the underlying module so env-var overrides applied by test fixtures
# (monkeypatch.setenv) take effect when THIS shim is reloaded.
importlib.reload(_db_module)

from db.experiments_db import (  # noqa: E402, F401
    _USE_POSTGRES,
    DB_PATH,
    _get_conn,
    _init_db,
    _PgConnAdapter,
    get_final_metrics,
    get_run_metrics,
    list_registered_models,
    list_runs,
    list_stale_running_jobs,
    register_model,
    save_experiment_run,
    save_final_metrics,
    save_run_metrics,
    save_run_params,
    write_job_status,
)


def _adapt_sql(sql: str) -> str:
    # Reads _USE_POSTGRES from this shim's own namespace so test fixtures that
    # set `app.state.experiments_db._USE_POSTGRES = True` are visible here.
    if sys.modules[__name__]._USE_POSTGRES:
        return sql.replace("?", "%s")
    return sql
