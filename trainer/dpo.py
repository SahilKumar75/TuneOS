"""Direct Preference Optimization (DPO) training.

Trains a LoRA adapter on (prompt, chosen, rejected) preference triples using
``trl.DPOTrainer``. Reuses the same model loader and LoRA injection as SFT, so a
DPO run produces an adapter in the identical output layout.
"""

import os

from transformers import set_seed

from trainer.callbacks import RedisLossCallback
from trainer.config import DPOConfig, LoraConfig, ModelConfig
from trainer.dataset import load_preference_pairs
from trainer.loader import load_model_and_tokenizer
from trainer.lora import inject_lora, save_adapter


def train_dpo(
    model_cfg: ModelConfig,
    lora_cfg: LoraConfig,
    dpo_cfg: DPOConfig,
    dataset_path: str,
    job_id: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    prompt_col: str = "prompt",
    chosen_col: str = "chosen",
    rejected_col: str = "rejected",
):
    """Run a DPO fine-tune and save the LoRA adapter.

    Returns ``(output_path, model, tokenizer)``. With a PEFT model and no
    explicit ``ref_model``, DPOTrainer uses the adapter-disabled base model as
    the implicit reference, so no second copy of the weights is needed.
    """
    # trl re-exports its own DPOConfig (a TrainingArguments subclass); import it
    # lazily and alias to avoid clashing with our trainer.config.DPOConfig.
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    set_seed(dpo_cfg.seed)

    model, tokenizer = load_model_and_tokenizer(model_cfg)
    model = inject_lora(model, lora_cfg)

    dataset = load_preference_pairs(
        dataset_path,
        hub_dataset_id=hub_dataset_id,
        hub_split=hub_split,
        prompt_col=prompt_col,
        chosen_col=chosen_col,
        rejected_col=rejected_col,
    )

    output_path = os.path.join(dpo_cfg.output_dir, job_id)
    args = TRLDPOConfig(
        output_dir=output_path,
        beta=dpo_cfg.beta,
        max_length=dpo_cfg.max_length,
        max_prompt_length=dpo_cfg.max_prompt_length,
        num_train_epochs=dpo_cfg.num_train_epochs,
        per_device_train_batch_size=dpo_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=dpo_cfg.gradient_accumulation_steps,
        learning_rate=dpo_cfg.learning_rate,
        warmup_ratio=dpo_cfg.warmup_ratio,
        lr_scheduler_type=dpo_cfg.lr_scheduler_type,
        fp16=dpo_cfg.fp16,
        bf16=dpo_cfg.bf16,
        seed=dpo_cfg.seed,
        data_seed=dpo_cfg.seed,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[RedisLossCallback(job_id=job_id)],
    )
    trainer.train()

    save_adapter(model, output_path)
    return output_path, model, tokenizer
