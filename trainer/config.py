from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    use_4bit: bool = True
    use_8bit: bool = False
    trust_remote_code: bool = False
    max_seq_length: int = 512
    hf_token: str = ""
    local_model_path: str = ""
    model_source: str = "hub"  # "hub" | "local" | "custom_string"


@dataclass
class LoraConfig:
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # scaling factor
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 10
    save_steps: int = 100
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 0.3
