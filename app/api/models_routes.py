"""Model listing and validation API routes."""

from __future__ import annotations

import os

from fastapi import APIRouter

from app.api.deps import _SUPPORTED_MODELS
from app.api.schemas import ModelInfo, ModelValidateRequest

router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
async def list_models():
    return [ModelInfo(**m) for m in _SUPPORTED_MODELS]


@router.post("/models/validate")
async def validate_model(req: ModelValidateRequest):
    """Validate that a model ID is loadable (HF Hub or local path)."""
    import asyncio
    token = req.hf_token or os.getenv("HF_TOKEN") or None

    def _check():
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(req.model_id, token=token, trust_remote_code=False)
        return cfg.model_type, getattr(cfg, "num_parameters", lambda: None)()

    try:
        model_type, num_params = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _check), timeout=20.0
        )
        param_str = f"{num_params / 1e9:.1f}B" if num_params else "unknown size"
        return {"valid": True, "model_type": model_type, "num_params": param_str, "error": ""}
    except Exception as exc:
        return {"valid": False, "model_type": "", "num_params": "", "error": str(exc)}
