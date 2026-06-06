import os

from transformers import EarlyStoppingCallback, TrainingArguments, set_seed
from trl import SFTTrainer

from trainer.callbacks import RedisLossCallback
from trainer.config import LoraConfig, ModelConfig, TrainingConfig
from trainer.dataset import load_and_tokenize
from trainer.lora import save_adapter
from trainer.qlora import prepare_qlora_model


class OutOfMemoryError(RuntimeError):
    """Raised when training fails due to GPU OOM, carrying a remediation hint."""

    def __init__(self, original: BaseException):
        self.suggestion = (
            "CUDA out of memory — reduce batch_size, max_seq_length, or "
            "gradient_accumulation_steps, or pick a smaller model."
        )
        super().__init__(f"{self.suggestion} (original error: {original})")


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg or "cuda oom" in msg or exc.__class__.__name__ == "OutOfMemoryError"
    )


def finetune(
    model_cfg: ModelConfig,
    lora_cfg: LoraConfig,
    train_cfg: TrainingConfig,
    dataset_path: str,
    job_id: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
):
    """
    Full fine-tuning pipeline:
      1. Load model (any source: HF Hub, local, custom string)
      2. Load + tokenize dataset, optionally splitting off a validation set
      3. Train with SFTTrainer (with optional early stopping + checkpoint resume)
      4. Save adapter weights
    Returns (output_path, model, tokenizer).

    Raises OutOfMemoryError with a remediation hint if training hits GPU OOM.
    """
    # Seed every source of randomness up front so the run is reproducible.
    set_seed(train_cfg.seed)

    # 1. Prepare model
    model, tokenizer = prepare_qlora_model(model_cfg, lora_cfg)

    # 2. Load dataset (optionally splitting off a validation set below). With
    # packing the trainer tokenizes raw text itself; otherwise we pre-tokenize.
    # Both honor the selected prompt template.
    if train_cfg.packing:
        from trainer.dataset import load_raw_text

        dataset = load_raw_text(
            dataset_path,
            hub_dataset_id=hub_dataset_id,
            hub_split=hub_split,
            instruction_col=instruction_col,
            output_col=output_col,
            template=train_cfg.prompt_template,
        )
    else:
        dataset = load_and_tokenize(
            dataset_path,
            tokenizer,
            model_cfg.max_seq_length,
            hub_dataset_id=hub_dataset_id,
            hub_split=hub_split,
            instruction_col=instruction_col,
            output_col=output_col,
            template=train_cfg.prompt_template,
        )

    eval_dataset = None
    ratio = train_cfg.eval_split_ratio
    # Only split if a meaningful validation set (>=1 example) can be carved out.
    if ratio and 0.0 < ratio < 1.0 and len(dataset) >= 2 and int(len(dataset) * ratio) >= 1:
        split = dataset.train_test_split(test_size=ratio, seed=train_cfg.seed)
        dataset, eval_dataset = split["train"], split["test"]

    use_early_stopping = bool(train_cfg.early_stopping_patience) and eval_dataset is not None
    eval_strategy = "epoch" if eval_dataset is not None else "no"
    # load_best_model_at_end requires matching eval/save strategies.
    save_strategy = "epoch" if use_early_stopping else "steps"

    # 3. Training arguments
    output_path = os.path.join(train_cfg.output_dir, job_id)
    training_args = TrainingArguments(
        output_dir=output_path,
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
        max_grad_norm=train_cfg.max_grad_norm,
        report_to=train_cfg.report_to,  # external tracker(s); "none" by default
        gradient_checkpointing=True,
        seed=train_cfg.seed,
        data_seed=train_cfg.seed,
        torch_compile=train_cfg.use_torch_compile,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss" if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
    )

    callbacks = [RedisLossCallback(job_id=job_id)]
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=train_cfg.early_stopping_patience)
        )

    # 4. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=model_cfg.max_seq_length,
        packing=train_cfg.packing,
        callbacks=callbacks,
    )

    try:
        trainer.train(resume_from_checkpoint=train_cfg.resume_from_checkpoint or None)
    except Exception as exc:  # noqa: BLE001 — re-raised below
        if _is_oom(exc):
            raise OutOfMemoryError(exc) from exc
        raise

    save_adapter(model, output_path)
    return output_path, model, tokenizer
