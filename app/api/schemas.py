"""Pydantic schemas / response models for the TuneOS API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

_VERSION = "0.2.0"


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = _VERSION


class GpuInfo(BaseModel):
    available: bool
    backend: str
    name: str
    detail: str = ""
    device_count: int = 0
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    cuda_version: str = ""


class ModelInfo(BaseModel):
    name: str
    hf_id: str
    notes: str = ""


class JobConfig(BaseModel):
    model_id: str
    model_source: str = "hub"
    local_model_path: str = ""
    hf_token: str = ""
    dataset_path: str = ""
    hub_dataset_id: str = ""
    hub_dataset_split: str = "train"
    instruction_col: str = "instruction"
    output_col: str = "output"
    technique: str = "qlora"
    use_4bit: bool = True
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    learning_rate: float = Field(default=2e-4, gt=0)
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1)
    max_seq_length: int = Field(default=512, ge=64)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    lr_scheduler_type: str = "cosine"
    bf16: bool = False
    # ── Phase 2: validation + resumption ──────────────────────────
    eval_split_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    early_stopping_patience: int = Field(default=0, ge=0, le=20)
    eval_steps: int = Field(default=0, ge=0)
    resume_from_checkpoint: str = ""
    # Seeds every source of randomness so a run is reproducible.
    seed: int = Field(default=42, ge=0)
    # Opt-in PyTorch 2.0 compilation for faster training on supported GPUs.
    use_torch_compile: bool = False
    # Compute backend for training execution.
    compute_backend: Literal["local", "modal", "hf_spaces"] = "local"
    # Prompt formatting + sample packing.
    prompt_template: Literal["alpaca", "chatml", "llama3", "phi3", "zephyr", "gemma", "mistral"] = (
        "alpaca"
    )
    packing: bool = False
    # Multi-GPU FSDP option string (empty = off), e.g. "full_shard auto_wrap".
    fsdp: str = ""
    optim: str = "paged_adamw_32bit"
    use_all_linear: bool = False
    user_intent: str = ""
    experiment_name: str = ""
    experiment_id: str = ""
    compose_adapters: bool = False
    overlay_technique: str = "lora"


class DPOJobConfig(BaseModel):
    """Config for a DPO (preference) fine-tuning job — POST /api/jobs/dpo."""

    model_id: str
    model_source: str = "hub"
    local_model_path: str = ""
    hf_token: str = ""
    dataset_path: str = ""
    hub_dataset_id: str = ""
    hub_dataset_split: str = "train"
    prompt_col: str = "prompt"
    chosen_col: str = "chosen"
    rejected_col: str = "rejected"
    use_4bit: bool = True
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    beta: float = Field(default=0.1, gt=0.0, le=1.0)
    learning_rate: float = Field(default=5e-5, gt=0)
    epochs: int = Field(default=1, ge=1, le=100)
    batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    max_length: int = Field(default=1024, ge=64)
    max_prompt_length: int = Field(default=512, ge=16)
    bf16: bool = False
    seed: int = Field(default=42, ge=0)
    experiment_id: str = ""


class DistillJobConfig(BaseModel):
    """Config for a knowledge-distillation job — POST /api/jobs/distill."""

    model_id: str  # student model
    model_source: str = "hub"
    local_model_path: str = ""
    hf_token: str = ""
    teacher_model: str  # required — the (larger) teacher, same tokenizer family
    dataset_path: str = ""
    hub_dataset_id: str = ""
    hub_dataset_split: str = "train"
    instruction_col: str = "instruction"
    output_col: str = "output"
    use_4bit: bool = True
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    temperature: float = Field(default=2.0, gt=0.0, le=10.0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    learning_rate: float = Field(default=2e-4, gt=0)
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    max_seq_length: int = Field(default=512, ge=64)
    bf16: bool = False
    seed: int = Field(default=42, ge=0)
    prompt_template: Literal["alpaca", "chatml", "llama3", "phi3", "zephyr", "gemma", "mistral"] = (
        "alpaca"
    )
    experiment_id: str = ""


class VisionJobConfig(BaseModel):
    """Config for a VLM (vision-language model) fine-tuning job — POST /api/jobs/vision."""

    model_id: str
    model_source: str = "hub"
    local_model_path: str = ""
    hf_token: str = ""
    dataset_path: str = ""
    hub_dataset_id: str = ""
    hub_dataset_split: str = "train"
    instruction_col: str = "instruction"
    output_col: str = "output"
    image_col: str = "image"
    use_4bit: bool = True
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    learning_rate: float = Field(default=2e-4, gt=0)
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=2, ge=1)
    max_seq_length: int = Field(default=512, ge=64)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    bf16: bool = False
    seed: int = Field(default=42, ge=0)
    experiment_id: str = ""


class SweepRequest(BaseModel):
    """Fan out a hyperparameter grid into multiple fine-tune jobs."""

    base: JobConfig
    # field name -> list of values to try; cartesian product is enqueued.
    grid: dict[str, list] = {}


class SweepResponse(BaseModel):
    count: int
    job_ids: list[str]


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    output_path: str = ""
    error: str = ""


class JobCreated(BaseModel):
    job_id: str
    status: str = "queued"


class ModelValidateRequest(BaseModel):
    model_id: str
    hf_token: str = ""


class DatasetGenRequest(BaseModel):
    user_intent: str
    method: str = "self_instruct"
    n_samples: int = Field(default=50, ge=5, le=500)
    seed_examples: list[dict] = []
    hf_token: str = ""
    quality_threshold: float = Field(default=0.0, ge=0.0, le=5.0)
    personas: list[str] = []
    export_format: str = "jsonl"  # "jsonl" | "alpaca_json" | "sharegpt_json"


class CommentaryRequest(BaseModel):
    epoch: float
    total_epochs: int
    loss_drop_pct: float
    current_loss: float
    intent: str = ""


class PushHubRequest(BaseModel):
    repo_name: str
    hf_token: str = ""


class InferRequest(BaseModel):
    prompt: str = Field(..., max_length=8192)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class MergeRequest(BaseModel):
    hf_token: str = ""


class GgufRequest(BaseModel):
    quant_type: str = "Q4_K_M"


class GitHubPushRequest(BaseModel):
    repo_url: str
    github_token: str
