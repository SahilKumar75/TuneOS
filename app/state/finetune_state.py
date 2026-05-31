"""Wizard state for the /finetune dedicated flow."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import reflex as rx

from app.state.job_state import JobState

DATASET_DIR = os.getenv("DATASET_DIR", "./storage/datasets")


class FinetuneState(rx.State):
    # ── Step tracking ─────────────────────────────────────────────
    current_step: int = 1  # 1–5

    # ── Step 1: Model + Technique ─────────────────────────────────
    selected_model_id: str = ""
    selected_model_name: str = ""
    selected_technique: str = "qlora"  # "qlora" | "lora"

    # ── Step 2: Dataset ───────────────────────────────────────────
    dataset_path: str = ""
    dataset_filename: str = ""
    dataset_preview: list[dict[str, Any]] = []
    dataset_error: str = ""
    is_uploading: bool = False
    existing_datasets: list[str] = []

    # ── Step 3: Hyperparameters ───────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    epochs: int = 3
    learning_rate: str = "2e-4"
    batch_size: int = 4
    max_seq_length: int = 512

    # ── Step 4: Training ──────────────────────────────────────────
    job_id: str = ""
    is_starting: bool = False
    start_error: str = ""
    training_start_time: str = ""

    # ── Step 5: Results ───────────────────────────────────────────
    hf_token_input: str = ""
    hf_repo_name: str = ""
    push_status: str = "idle"  # idle | pushing | done | error
    push_error: str = ""
    push_repo_url: str = ""
    eval_perplexity: float = 0.0
    eval_status: str = "idle"  # idle | running | done | error | not_ready
    chat_input: str = ""
    chat_response: str = ""
    chat_loading: bool = False
    chat_error: str = ""

    # ── Computed vars ─────────────────────────────────────────────
    @rx.var
    def can_go_to_dataset(self) -> bool:
        return bool(self.selected_model_id)

    @rx.var
    def can_go_to_configure(self) -> bool:
        return bool(self.dataset_path) and not bool(self.dataset_error)

    @rx.var
    def can_start_training(self) -> bool:
        return self.can_go_to_configure and bool(self.selected_model_id)

    @rx.var
    def technique_label(self) -> str:
        return "QLoRA" if self.selected_technique == "qlora" else "LoRA"

    @rx.var
    def technique_description(self) -> str:
        if self.selected_technique == "qlora":
            return "Trains a small adapter in compressed mode. Works on 12 GB+ GPU. Recommended."
        return "Trains a small adapter in float16. Needs ~16 GB GPU for 7B models."

    # ── Step 1 events ─────────────────────────────────────────────
    @rx.event
    def select_model(self, model_id: str, model_name: str):
        self.selected_model_id = model_id
        self.selected_model_name = model_name

    @rx.event
    def select_technique(self, technique: str):
        self.selected_technique = technique

    # ── Step 2 events ─────────────────────────────────────────────
    @rx.event
    def load_existing_datasets(self):
        if not os.path.exists(DATASET_DIR):
            self.existing_datasets = []
            return
        self.existing_datasets = [
            f
            for f in os.listdir(DATASET_DIR)
            if os.path.isfile(os.path.join(DATASET_DIR, f))
        ]

    @rx.event
    def select_existing_dataset(self, filename: str):
        path = os.path.join(DATASET_DIR, filename)
        self.dataset_path = path
        self.dataset_filename = filename
        self._validate_dataset_at(path)

    async def handle_dataset_upload(self, files: list[rx.UploadFile]):
        self.is_uploading = True
        self.dataset_error = ""
        self.dataset_preview = []
        yield

        if not files:
            self.is_uploading = False
            return

        file = files[0]
        data = await file.read()

        os.makedirs(DATASET_DIR, exist_ok=True)
        out_path = os.path.join(DATASET_DIR, file.filename)
        with open(out_path, "wb") as f:
            f.write(data)

        self.dataset_path = out_path
        self.dataset_filename = file.filename
        self.is_uploading = False

        # Refresh existing list
        yield FinetuneState.load_existing_datasets()
        self._validate_dataset_at(out_path)

    def _validate_dataset_at(self, path: str):
        """Read up to 10 rows, check for required columns, populate preview."""
        import pandas as pd

        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, nrows=10)
            else:
                rows = []
                with open(path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rows.append(json.loads(line))
                        if len(rows) == 10:
                            break
                df = pd.DataFrame(rows)

            required = {"instruction", "output"}
            missing = required - set(df.columns)
            if missing:
                self.dataset_error = (
                    f"Missing columns: {', '.join(sorted(missing))}. "
                    "File must contain 'instruction' and 'output' fields."
                )
                self.dataset_preview = []
            else:
                self.dataset_error = ""
                self.dataset_preview = (
                    df[["instruction", "output"]]
                    .head(5)
                    .fillna("")
                    .to_dict("records")
                )
        except Exception as exc:
            self.dataset_error = f"Could not read file: {exc}"
            self.dataset_preview = []

    # ── Navigation ────────────────────────────────────────────────
    @rx.event
    def go_to_step(self, step: int):
        self.current_step = step

    @rx.event
    def next_step(self):
        self.current_step += 1

    @rx.event
    def prev_step(self):
        if self.current_step > 1:
            self.current_step -= 1

    # ── Step 3 setters ────────────────────────────────────────────
    @rx.event
    def set_lora_r(self, value: int):
        self.lora_r = int(value)

    @rx.event
    def set_lora_alpha(self, value: int):
        self.lora_alpha = int(value)

    @rx.event
    def set_epochs(self, value: str):
        try:
            self.epochs = max(1, min(20, int(value)))
        except ValueError:
            pass

    @rx.event
    def set_learning_rate(self, value: str):
        self.learning_rate = value

    @rx.event
    def set_batch_size(self, value: str):
        self.batch_size = int(value)

    @rx.event
    def set_max_seq_length(self, value: str):
        self.max_seq_length = int(value)

    # ── Step 4: Start training ────────────────────────────────────
    @rx.event
    def start_training(self):
        if not self.can_start_training:
            return

        import uuid
        from datetime import datetime

        job_id = str(uuid.uuid4())
        self.job_id = job_id
        self.is_starting = True
        self.start_error = ""
        self.training_start_time = datetime.utcnow().isoformat()

        use_4bit = self.selected_technique == "qlora"
        model_cfg = {
            "model_name": self.selected_model_id,
            "use_4bit": use_4bit,
            "use_8bit": False,
            "trust_remote_code": False,
            "max_seq_length": self.max_seq_length,
        }
        lora_cfg = {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "v_proj"],
        }
        train_cfg = {
            "output_dir": os.getenv("OUTPUT_DIR", "./outputs"),
            "num_train_epochs": self.epochs,
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": 4,
            "learning_rate": float(self.learning_rate),
            "fp16": True,
            "bf16": False,
            "logging_steps": 1,
            "save_steps": 100,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "optim": "paged_adamw_32bit",
            "max_grad_norm": 0.3,
        }

        try:
            from workers.train_task import run_finetune

            run_finetune.delay(
                job_id=job_id,
                model_cfg=model_cfg,
                lora_cfg=lora_cfg,
                train_cfg=train_cfg,
                dataset_path=self.dataset_path,
            )
        except Exception as exc:
            self.start_error = str(exc)
            self.is_starting = False
            return

        self.is_starting = False
        self.current_step = 4
        return JobState.poll_job(job_id)

    # ── Step 5: Post-training actions ─────────────────────────────
    @rx.event
    def download_adapter(self):
        return rx.redirect(f"/api/jobs/{self.job_id}/download")

    @rx.event(background=True)
    async def push_to_hub(self):
        async with self:
            self.push_status = "pushing"
            self.push_error = ""

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"http://localhost:8000/api/jobs/{self.job_id}/push_hub",
                    json={
                        "repo_name": self.hf_repo_name,
                        "hf_token": self.hf_token_input,
                    },
                )
            if resp.status_code == 200:
                data = resp.json()
                async with self:
                    self.push_status = "done"
                    self.push_repo_url = data.get("repo_url", "")
            else:
                async with self:
                    self.push_status = "error"
                    self.push_error = resp.json().get("detail", "Push failed")
        except Exception as exc:
            async with self:
                self.push_status = "error"
                self.push_error = str(exc)

    @rx.event(background=True)
    async def run_eval(self):
        async with self:
            self.eval_status = "running"

        import asyncio

        for _ in range(60):  # Poll for up to 60 seconds
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"http://localhost:8000/api/jobs/{self.job_id}/eval"
                    )
                data = resp.json()
                if data.get("status") == "done":
                    ppl = data.get("perplexity")
                    async with self:
                        self.eval_status = "done"
                        self.eval_perplexity = float(ppl) if ppl is not None else 0.0
                    return
                elif data.get("status") == "not_ready":
                    await asyncio.sleep(2)
                else:
                    break
            except Exception:
                await asyncio.sleep(2)

        async with self:
            self.eval_status = "not_ready"

    @rx.event(background=True)
    async def send_test_chat(self):
        prompt = self.chat_input
        if not prompt.strip():
            return

        async with self:
            self.chat_loading = True
            self.chat_response = ""
            self.chat_error = ""

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    f"http://localhost:8000/api/jobs/{self.job_id}/infer",
                    json={"prompt": prompt, "max_new_tokens": 200, "temperature": 0.7},
                )
            if resp.status_code == 200:
                async with self:
                    self.chat_response = resp.json().get("response", "")
                    self.chat_loading = False
            else:
                async with self:
                    self.chat_error = resp.json().get("detail", "Inference failed")
                    self.chat_loading = False
        except Exception as exc:
            async with self:
                self.chat_error = str(exc)
                self.chat_loading = False

    @rx.event
    def set_chat_input(self, value: str):
        self.chat_input = value

    @rx.event
    def set_hf_repo_name(self, value: str):
        self.hf_repo_name = value

    @rx.event
    def set_hf_token_input(self, value: str):
        self.hf_token_input = value
