from trainer.config import LoraConfig, ModelConfig
from trainer.loader import load_model_and_tokenizer
from trainer.lora import inject_lora


def prepare_qlora_model(model_cfg: ModelConfig, lora_cfg: LoraConfig):
    """
    Full QLoRA setup:
      1. Load model with 4-bit NF4 quantization
      2. Prepare for k-bit training (cast layer norms to fp32)
      3. Inject LoRA adapters
    Returns (model, tokenizer) ready for training.

    Honors ``model_cfg.use_4bit`` instead of forcing it on — callers select
    QLoRA (4-bit) vs plain LoRA (full/8-bit) via the config.
    """
    model, tokenizer = load_model_and_tokenizer(model_cfg)
    model = inject_lora(model, lora_cfg)
    return model, tokenizer
