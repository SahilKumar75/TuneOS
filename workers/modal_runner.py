"""Modal.com training backend.

Runs the exact same `trainer.finetune` pipeline on a Modal-managed T4 GPU so a
user with no local GPU can still fine-tune. The local Celery worker stays the
orchestrator: it serializes the dataset, calls `run_finetune_modal.remote(...)`,
and writes the returned adapter + metrics back to local disk. This keeps the
storage / status / Redis layer identical to the local path.

Import is lazy and guarded — `modal` is an optional dependency. `modal_available()`
lets callers (and tests) check without importing the SDK.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

try:
    import modal

    _MODAL_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only without modal installed
    modal = None  # type: ignore[assignment]
    _MODAL_IMPORT_ERROR = exc


def modal_available() -> bool:
    """True when the SDK is installed and Modal credentials are configured.

    Accepts either env-var tokens (MODAL_TOKEN_ID/SECRET — best for deploys) or
    a local profile written by `modal token set` (~/.modal.toml — best for dev).
    """
    if modal is None:
        return False
    if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
        return True
    return (Path.home() / ".modal.toml").exists()


# The remote app + function are only defined when `modal` is importable, so that
# importing this module never fails on a machine without the SDK.
if modal is not None:
    modal_image = modal.Image.debian_slim(python_version="3.10").pip_install(
        "torch",
        "transformers>=4.44.0",
        "peft>=0.11.0",
        "trl>=0.8.0",
        "bitsandbytes>=0.43.0",
        "datasets>=2.19.0",
        "accelerate>=0.30.0",
    )
    modal_app = modal.App("tuneos-trainer")

    @modal_app.function(gpu="t4", timeout=7200, image=modal_image)
    def run_finetune_modal(
        job_id: str,
        model_cfg: dict,
        lora_cfg: dict,
        train_cfg: dict,
        dataset_bytes: bytes,
        hub_dataset_id: str = "",
        hub_split: str = "train",
        instruction_col: str = "instruction",
        output_col: str = "output",
    ) -> dict:
        """Run one fine-tune on a Modal T4 and return adapter + eval as a dict.

        Returns ``{"adapter_zip": bytes, "eval": {...}, "loss_history": [...]}``.
        The caller unpacks the zip into the local OUTPUT_DIR and reuses the
        existing status / metrics persistence.
        """
        import tempfile

        from trainer.config import LoraConfig, ModelConfig, TrainingConfig
        from trainer.finetune import finetune

        # Materialize the dataset the local worker streamed up to us. An empty
        # payload means "use the HF Hub dataset id instead".
        dataset_path = ""
        if dataset_bytes:
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                f.write(dataset_bytes)
                dataset_path = f.name

        # Force the remote run into a known, writable output directory.
        train_cfg = {**train_cfg, "output_dir": "/tmp/tuneos_out"}

        output_path, model, tokenizer = finetune(
            model_cfg=ModelConfig(**model_cfg),
            lora_cfg=LoraConfig(**lora_cfg),
            train_cfg=TrainingConfig(**train_cfg),
            dataset_path=dataset_path,
            job_id=job_id,
            hub_dataset_id=hub_dataset_id,
            hub_split=hub_split,
            instruction_col=instruction_col,
            output_col=output_col,
        )

        # Evaluate remotely while the model is still in GPU memory — the local
        # worker has no model object to score against.
        from workers.train_task import _compute_eval

        eval_results = _compute_eval(
            model,
            tokenizer,
            model_cfg,
            train_cfg,
            dataset_path,
            hub_dataset_id,
            hub_split,
            instruction_col,
            output_col,
        )

        return {
            "adapter_zip": _zip_dir(output_path),
            "eval": eval_results,
            "loss_history": [],  # live streaming is Phase E
        }


def _zip_dir(path: str) -> bytes:
    """Zip a directory tree into an in-memory bytes blob."""
    root = Path(path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in root.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(root))
    return buf.getvalue()


def _unzip_to(blob: bytes, dest: str) -> str:
    """Extract a zip blob produced by ``_zip_dir`` into ``dest``. Returns ``dest``."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(dest)
    return dest


def run_on_modal(
    job_id: str,
    model_cfg: dict,
    lora_cfg: dict,
    train_cfg: dict,
    dataset_path: str,
    output_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
) -> dict:
    """Orchestrate a Modal run from the local worker.

    Reads the local dataset, invokes the remote function, writes the returned
    adapter to ``output_path``, and returns the eval/loss_history dict for the
    caller to persist. Raises ``RuntimeError`` if Modal is unavailable.
    """
    if not modal_available():
        raise RuntimeError(
            "Modal backend selected but unavailable. Install `modal` and set "
            "MODAL_TOKEN_ID + MODAL_TOKEN_SECRET in your .env."
        )

    dataset_bytes = b""
    if dataset_path and Path(dataset_path).exists():
        dataset_bytes = Path(dataset_path).read_bytes()

    with modal_app.run():
        result = run_finetune_modal.remote(
            job_id=job_id,
            model_cfg=model_cfg,
            lora_cfg=lora_cfg,
            train_cfg=train_cfg,
            dataset_bytes=dataset_bytes,
            hub_dataset_id=hub_dataset_id,
            hub_split=hub_split,
            instruction_col=instruction_col,
            output_col=output_col,
        )

    _unzip_to(result["adapter_zip"], output_path)
    return {"eval": result.get("eval", {}), "loss_history": result.get("loss_history", [])}
