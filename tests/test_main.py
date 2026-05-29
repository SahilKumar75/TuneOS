import pytest
from trainer.config import ModelConfig, LoraConfig, TrainingConfig


# --- ModelConfig ---

def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.model_name == "mistralai/Mistral-7B-v0.1"
    assert cfg.use_4bit is True
    assert cfg.use_8bit is False
    assert cfg.max_seq_length == 512

def test_model_config_custom():
    cfg = ModelConfig(model_name="google/gemma-2b", use_4bit=False, max_seq_length=1024)
    assert cfg.model_name == "google/gemma-2b"
    assert cfg.use_4bit is False
    assert cfg.max_seq_length == 1024

def test_model_config_4bit_and_8bit_independent():
    cfg = ModelConfig(use_4bit=False, use_8bit=True)
    assert cfg.use_4bit is False
    assert cfg.use_8bit is True


# --- LoraConfig ---

def test_lora_config_defaults():
    cfg = LoraConfig()
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.bias == "none"
    assert cfg.task_type == "CAUSAL_LM"
    assert "q_proj" in cfg.target_modules
    assert "v_proj" in cfg.target_modules

def test_lora_config_custom_rank():
    cfg = LoraConfig(r=64, lora_alpha=128)
    assert cfg.r == 64
    assert cfg.lora_alpha == 128

def test_lora_config_target_modules_mutable():
    cfg1 = LoraConfig()
    cfg2 = LoraConfig()
    cfg1.target_modules.append("k_proj")
    assert "k_proj" not in cfg2.target_modules  # default_factory ensures no shared state

def test_lora_config_dropout_range():
    cfg = LoraConfig(lora_dropout=0.1)
    assert 0.0 <= cfg.lora_dropout <= 1.0


# --- TrainingConfig ---

def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.num_train_epochs == 3
    assert cfg.learning_rate == pytest.approx(2e-4)
    assert cfg.fp16 is True
    assert cfg.bf16 is False
    assert cfg.lr_scheduler_type == "cosine"
    assert cfg.optim == "paged_adamw_32bit"

def test_training_config_custom():
    cfg = TrainingConfig(num_train_epochs=5, learning_rate=1e-4, fp16=False, bf16=True)
    assert cfg.num_train_epochs == 5
    assert cfg.learning_rate == pytest.approx(1e-4)
    assert cfg.fp16 is False
    assert cfg.bf16 is True

def test_training_config_fp16_bf16_exclusive():
    # Configs allow both flags — verify they don't silently override each other
    cfg = TrainingConfig(fp16=True, bf16=False)
    assert cfg.fp16 is True
    assert cfg.bf16 is False

def test_training_config_output_dir():
    cfg = TrainingConfig(output_dir="/tmp/test_run")
    assert cfg.output_dir == "/tmp/test_run"
