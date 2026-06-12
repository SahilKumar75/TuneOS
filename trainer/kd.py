"""Knowledge distillation.

Trains a (LoRA) student model to mimic a larger frozen teacher: the loss blends
the hard-label cross-entropy with the KL divergence between the temperature-
softened student and teacher next-token distributions.

Teacher and student must share the same tokenizer/vocabulary so their logits are
comparable (e.g. distil a 7B into a 1–3B of the same family). Heavy imports are
done lazily so importing this module stays cheap.
"""

import os

from trainer.config import DistillConfig, LoraConfig, ModelConfig
from trainer.dataset import load_and_tokenize
from trainer.loader import load_model_and_tokenizer
from trainer.lora import save_adapter
from trainer.qlora import prepare_qlora_model


def distill(
    model_cfg: ModelConfig,
    lora_cfg: LoraConfig,
    distill_cfg: DistillConfig,
    dataset_path: str,
    job_id: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
):
    """Distil ``distill_cfg.teacher_model`` into the LoRA student. Returns
    ``(output_path, student, tokenizer)``."""
    import torch
    import torch.nn.functional as F
    from transformers import Trainer, TrainingArguments, set_seed

    from trainer.callbacks import RedisLossCallback

    if not distill_cfg.teacher_model:
        raise ValueError("DistillConfig.teacher_model must be set for distillation.")

    set_seed(distill_cfg.seed)

    # Student gets the LoRA adapter; teacher is loaded frozen (4-bit to save VRAM).
    student, tokenizer = prepare_qlora_model(model_cfg, lora_cfg)
    teacher, _ = load_model_and_tokenizer(
        ModelConfig(
            model_name=distill_cfg.teacher_model,
            use_4bit=True,
            max_seq_length=distill_cfg.max_seq_length,
            hf_token=model_cfg.hf_token,
        )
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Put teacher on a secondary GPU when available to avoid competing with the
    # student for VRAM on the primary device. Falls back to CPU so training still
    # works on single-GPU setups (at the cost of host↔device transfer overhead).
    import torch as _torch

    if _torch.cuda.is_available() and _torch.cuda.device_count() > 1:
        teacher = teacher.to("cuda:1")
        _teacher_device = "cuda:1"
    else:
        teacher = teacher.to("cpu")
        _teacher_device = "cpu"

    dataset = load_and_tokenize(
        dataset_path,
        tokenizer,
        distill_cfg.max_seq_length,
        hub_dataset_id=hub_dataset_id,
        hub_split=hub_split,
        instruction_col=instruction_col,
        output_col=output_col,
        template=distill_cfg.prompt_template,
    )

    temperature = distill_cfg.temperature
    alpha = distill_cfg.alpha

    class _DistillTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            ce_loss = outputs.loss  # hard-label cross-entropy
            with torch.no_grad():
                teacher_inputs = {
                    k: v.to(_teacher_device)
                    for k, v in inputs.items()
                    if k in ("input_ids", "attention_mask")
                }
                teacher_logits = teacher(**teacher_inputs).logits.to(outputs.logits.device)
            t = temperature
            kl = F.kl_div(
                F.log_softmax(outputs.logits / t, dim=-1),
                F.softmax(teacher_logits / t, dim=-1),
                reduction="batchmean",
            ) * (t * t)
            loss = alpha * kl + (1 - alpha) * ce_loss
            return (loss, outputs) if return_outputs else loss

    output_path = os.path.join(distill_cfg.output_dir, job_id)
    args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=distill_cfg.num_train_epochs,
        per_device_train_batch_size=distill_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=distill_cfg.gradient_accumulation_steps,
        learning_rate=distill_cfg.learning_rate,
        warmup_ratio=distill_cfg.warmup_ratio,
        lr_scheduler_type=distill_cfg.lr_scheduler_type,
        fp16=distill_cfg.fp16,
        bf16=distill_cfg.bf16,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        seed=distill_cfg.seed,
        data_seed=distill_cfg.seed,
    )

    trainer = _DistillTrainer(
        model=student,
        args=args,
        train_dataset=dataset,
        callbacks=[RedisLossCallback(job_id=job_id)],
    )
    trainer.train()

    save_adapter(student, output_path)
    return output_path, student, tokenizer
