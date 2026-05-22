import reflex as rx
import uuid
import os
from workers.train_task import run_finetune
from app.state.job_state import JobState

class ModelState(rx.State):
    model_name: str = "mistralai/Mistral-7B-v0.1"
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    
    # Training parameters
    epochs: int = 3
    learning_rate: str = "2e-4"
    dataset_path: str = ""

    def set_dataset_path(self, path: str):
        self.dataset_path = path
        
    def set_model_name(self, name: str):
        self.model_name = name

    def start_training(self):
        if not self.dataset_path:
            # Need to handle no dataset selected
            return rx.window_alert("Please upload and select a dataset first.")
            
        job_id = str(uuid.uuid4())
        
        # Prepare configs to match our backend structure
        model_cfg = {
            "model_name": self.model_name,
            "use_4bit": True,
            "use_8bit": False,
            "trust_remote_code": False,
            "max_seq_length": 512
        }
        
        lora_cfg = {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "v_proj"]
        }
        
        train_cfg = {
            "output_dir": os.getenv("OUTPUT_DIR", "./outputs"),
            "num_train_epochs": self.epochs,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": float(self.learning_rate),
            "fp16": True,
            "bf16": False,
            "logging_steps": 1, # Set lower for demonstration of UI streaming
            "save_steps": 100,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "optim": "paged_adamw_32bit",
            "max_grad_norm": 0.3
        }

        # Kick off Celery task
        run_finetune.delay(
            job_id=job_id,
            model_cfg=model_cfg,
            lora_cfg=lora_cfg,
            train_cfg=train_cfg,
            dataset_path=self.dataset_path
        )
        
        return [
            JobState.poll_job(job_id),
            rx.redirect("/training")
        ]
