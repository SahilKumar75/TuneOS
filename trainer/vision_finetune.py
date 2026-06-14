"""VLM fine-tuning — LoRA adapter training for vision-language models.

Uses AutoProcessor + AutoModelForVision2Seq (covers LLaVA-1.5, Qwen2-VL, InstructBLIP, etc.).
Mirrors the structure of trainer/finetune.py so workers/vision_task.py can call it uniformly.
"""

from __future__ import annotations

import os

import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from trainer.callbacks import RedisLossCallback
from trainer.config import LoraConfig, ModelConfig, TrainingConfig
from trainer.dataset import load_multimodal
from trainer.lora import inject_lora


def finetune_vision(
    model_cfg: ModelConfig,
    lora_cfg: LoraConfig,
    train_cfg: TrainingConfig,
    dataset_path: str,
    job_id: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
    image_col: str = "image",
) -> str:
    """Fine-tune a VLM with a LoRA adapter. Returns the output directory path."""
    processor = AutoProcessor.from_pretrained(
        model_cfg.model_name,
        token=model_cfg.hf_token or None,
        trust_remote_code=False,
    )

    _bnb_cfg = None
    if model_cfg.use_4bit:
        _bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if train_cfg.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForVision2Seq.from_pretrained(
        model_cfg.model_name,
        token=model_cfg.hf_token or None,
        trust_remote_code=False,
        quantization_config=_bnb_cfg,
        device_map="auto",
    )

    model = inject_lora(model, lora_cfg)

    dataset = load_multimodal(
        dataset_path,
        processor,
        max_seq_length=model_cfg.max_seq_length,
        hub_dataset_id=hub_dataset_id,
        hub_split=hub_split,
        instruction_col=instruction_col,
        output_col=output_col,
        image_col=image_col,
    )

    output_dir = os.path.join(train_cfg.output_dir, job_id)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        optim=train_cfg.optim,
        seed=train_cfg.seed,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[RedisLossCallback(job_id)],
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
