from trainer.config import ModelConfig, LoraConfig, TrainingConfig
from trainer.loader import load_model_and_tokenizer
from trainer.lora import inject_lora

def prepare_qlora_model(model_cfg: ModelConfig, lora_cfg: LoraConfig):
    """
    Full QLoRA setup:
      1. Load model with 4-bit NF4 quantization
      2. Prepare for k-bit training (cast layer norms to fp32)
      3. Inject LoRA adapters
    Returns (model, tokenizer) ready for training.
    """
    model_cfg.use_4bit = True
    model, tokenizer = load_model_and_tokenizer(model_cfg)
    model = inject_lora(model, lora_cfg)
    return model, tokenizer
