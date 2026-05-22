from peft import (
    get_peft_model,
    LoraConfig as PeftLoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)
from trainer.config import LoraConfig
import torch

def inject_lora(model, lora_cfg: LoraConfig):
    """
    Prepare model for k-bit training then inject LoRA adapters.
    """
    model = prepare_model_for_kbit_training(model)

    peft_config = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()   # log how many params are trainable
    return model


def save_adapter(model, output_dir: str):
    """Save only the LoRA adapter weights (small — a few MB)."""
    model.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")


def merge_and_save(model, tokenizer, output_dir: str):
    """
    Merge LoRA adapter back into the base model weights and save the
    full merged model. Use when the client wants a standalone model.
    """
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")
