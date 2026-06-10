from peft import (
    LoraConfig as PeftLoraConfig,
)
from peft import (
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from trainer.config import LoraConfig, get_target_modules


def inject_lora(model, lora_cfg: LoraConfig):
    """Prepare model for k-bit training then inject LoRA adapters.

    If lora_cfg.target_modules is None, the correct modules are
    auto-detected from model.config.model_type so Gemma, Phi-3,
    Falcon etc. don't silently receive wrong projection names.
    """
    model = prepare_model_for_kbit_training(model)

    if lora_cfg.target_modules is None:
        # Coerce to str — custom configs may carry a non-string model_type.
        model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "")
        target_modules = get_target_modules(model_type)
    else:
        target_modules = lora_cfg.target_modules

    peft_config = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=getattr(TaskType, lora_cfg.task_type),
        target_modules=target_modules,
        init_lora_weights=lora_cfg.init_lora_weights,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # log how many params are trainable
    return model


def save_adapter(model, output_dir: str):
    """Save only the LoRA adapter weights (small — a few MB)."""
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Adapter saved to {output_dir}")


def merge_and_save(model, tokenizer, output_dir: str):
    """
    Merge LoRA adapter back into the base model weights and save the
    full merged model. Use when the client wants a standalone model.
    """
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")
