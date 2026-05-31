import os

from transformers import TrainingArguments
from trl import SFTTrainer

from trainer.callbacks import RedisLossCallback
from trainer.config import LoraConfig, ModelConfig, TrainingConfig
from trainer.dataset import load_and_tokenize
from trainer.lora import save_adapter
from trainer.qlora import prepare_qlora_model


def finetune(
    model_cfg: ModelConfig,
    lora_cfg: LoraConfig,
    train_cfg: TrainingConfig,
    dataset_path: str,
    job_id: str,
) -> str:
    """
    Full fine-tuning pipeline:
      1. Load QLoRA model
      2. Load + tokenize dataset
      3. Train with SFTTrainer
      4. Save adapter weights
    Returns path to saved adapter.
    """
    # 1. Prepare QLoRA model
    model, tokenizer = prepare_qlora_model(model_cfg, lora_cfg)

    # 2. Load dataset
    dataset = load_and_tokenize(dataset_path, tokenizer, model_cfg.max_seq_length)

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
        report_to="none",  # disable wandb/mlflow by default
        gradient_checkpointing=True,
    )

    # 4. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=model_cfg.max_seq_length,
        callbacks=[RedisLossCallback(job_id=job_id)],
    )

    trainer.train()
    save_adapter(model, output_path)

    return output_path, model, tokenizer
