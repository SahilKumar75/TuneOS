"""Training runtime state — polling loop, live metrics, eval, and test-chat.

Inherits from FinetuneState (wizard config + navigation) so all wizard fields
(lora_r, epochs, lr, effective_model_id, etc.) are readable via ``self``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import reflex as rx

from app.state.experiment_state import (
    ExperimentState,
    save_experiment_run,
    save_final_metrics,
)
from app.state.finetune_state import (
    API_BASE,
    REDIS_URL,
    ChatMessage,
    EpochLogEntry,
    FinetuneState,
    LossPoint,
)


class TrainingPollerState(FinetuneState):
    """Training runtime — added on top of FinetuneState wizard config."""

    # ── Step 5: Training dashboard ────────────────────────────────
    job_id: str = ""
    is_starting: bool = False
    start_error: str = ""
    training_start_time: str = ""
    training_status: str = "idle"  # idle | running | done | failed
    current_epoch: float = 0.0
    total_steps: int = 0
    elapsed_seconds: int = 0
    gpu_memory_used_gb: float = 0.0
    ai_commentary: str = ""
    output_path: str = ""
    error_msg: str = ""
    loss_history: list[LossPoint] = []
    epoch_log: list[EpochLogEntry] = []  # one entry per completed epoch
    show_grad_norm: bool = False  # toggle grad-norm series on the live chart

    # Experiment tracking
    experiment_id: str = ""

    # ── Step 6: Results ───────────────────────────────────────────
    eval_perplexity: float = 0.0
    eval_bleu: float = 0.0
    eval_rouge1: float = 0.0
    eval_rouge2: float = 0.0
    eval_rougeL: float = 0.0
    eval_meteor: float = 0.0
    eval_status: str = "idle"  # idle | running | done | error | not_ready
    test_chat_history: list[ChatMessage] = []
    chat_input: str = ""
    chat_loading: bool = False
    chat_error: str = ""

    # ── Computed vars ─────────────────────────────────────────────
    @rx.var
    def elapsed_time_display(self) -> str:
        s = self.elapsed_seconds
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m"
        if m:
            return f"{m}m {sec}s"
        return f"{sec}s"

    @rx.var
    def current_total_steps_display(self) -> str:
        current = len(self.loss_history)
        total = self.total_steps
        if total:
            return f"{current} / {total}"
        return str(current)

    @rx.var
    def gpu_memory_display(self) -> str:
        if self.gpu_memory_used_gb == 0.0:
            return "—"
        return f"{self.gpu_memory_used_gb:.1f} GB"

    @rx.var
    def epoch_progress_pct(self) -> int:
        if self.epochs == 0:
            return 0
        return min(100, int(round((self.current_epoch / self.epochs) * 100, 0)))

    @rx.var
    def loss_history_chart_data(self) -> list[dict[str, Any]]:
        return [pt.model_dump() for pt in self.loss_history]

    @rx.var
    def current_epoch_display(self) -> str:
        return f"{min(int(self.current_epoch) + 1, self.epochs)} / {self.epochs}"

    @rx.var
    def last_train_loss(self) -> float:
        if self.loss_history:
            return self.loss_history[-1].loss
        return 0.0

    # ── Step 5: Start training ────────────────────────────────────
    @rx.event(background=True)
    async def start_training(self):
        if not self.can_start_training:
            return

        exp_id = str(__import__("uuid").uuid4())
        exp_name = (
            self.experiment_name
            or f"{self.effective_model_name}-{datetime.now().strftime('%m%d-%H%M')}"
        )

        async with self:
            self.is_starting = True
            self.start_error = ""
            self.experiment_id = exp_id
            self.experiment_name = exp_name
            self.training_start_time = datetime.now(timezone.utc).isoformat()
            self.loss_history = []
            self.epoch_log = []
            self.ai_commentary = ""
            self.training_status = "idle"

        _ds_path = self.dataset_path if self.data_source not in ("hub_dataset", "skip") else ""
        _hub_id = self.hub_dataset_id if self.data_source == "hub_dataset" else ""

        if self.is_dpo:
            endpoint = f"{API_BASE}/api/jobs/dpo"
            payload = {
                "model_id": self.effective_model_id,
                "model_source": self.model_source,
                "local_model_path": self.local_model_path,
                "hf_token": self.hf_token,
                "dataset_path": _ds_path,
                "hub_dataset_id": _hub_id,
                "hub_dataset_split": self.hub_dataset_split,
                "prompt_col": self.dpo_prompt_col,
                "chosen_col": self.dpo_chosen_col,
                "rejected_col": self.dpo_rejected_col,
                "use_4bit": True,
                "lora_rank": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "beta": self.dpo_beta,
                "learning_rate": float(self.learning_rate),
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_length": self.dpo_max_length,
                "max_prompt_length": self.dpo_max_prompt_length,
                "bf16": self.bf16,
                "seed": 42,
                "experiment_id": exp_id,
            }
        elif self.is_kd:
            endpoint = f"{API_BASE}/api/jobs/distill"
            payload = {
                "model_id": self.effective_model_id,
                "model_source": self.model_source,
                "local_model_path": self.local_model_path,
                "hf_token": self.hf_token,
                "teacher_model": self.kd_teacher_model,
                "dataset_path": _ds_path,
                "hub_dataset_id": _hub_id,
                "hub_dataset_split": self.hub_dataset_split,
                "instruction_col": self.hub_dataset_instruction_col,
                "output_col": self.hub_dataset_output_col,
                "use_4bit": True,
                "lora_rank": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "temperature": self.kd_temperature,
                "alpha": self.kd_alpha,
                "learning_rate": float(self.learning_rate),
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "seed": 42,
                "experiment_id": exp_id,
            }
        else:
            endpoint = f"{API_BASE}/api/jobs"
            payload = {
                "model_id": self.effective_model_id,
                "model_source": self.model_source,
                "local_model_path": self.local_model_path,
                "hf_token": self.hf_token,
                "dataset_path": _ds_path,
                "hub_dataset_id": _hub_id,
                "hub_dataset_split": self.hub_dataset_split,
                "instruction_col": self.hub_dataset_instruction_col,
                "output_col": self.hub_dataset_output_col,
                "technique": self.selected_technique,
                "use_4bit": self.selected_technique == "qlora",
                "lora_rank": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "learning_rate": float(self.learning_rate),
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "max_seq_length": self.max_seq_length,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "warmup_ratio": self.warmup_ratio,
                "lr_scheduler_type": self.lr_scheduler,
                "bf16": self.bf16,
                "user_intent": self.user_intent,
                "experiment_name": exp_name,
                "experiment_id": exp_id,
                "compute_backend": self.compute_backend,
                "prompt_template": self.prompt_template,
                "packing": self.packing,
            }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(endpoint, json=payload)
            if resp.status_code in (200, 201):
                job_id = resp.json()["job_id"]
                async with self:
                    self.job_id = job_id
                    self.is_starting = False
                    self.current_step = 5
                    self.training_status = "running"
                await self._poll_job_loop(job_id)
            else:
                async with self:
                    self.start_error = resp.json().get("detail", "Failed to start training job")
                    self.is_starting = False
        except Exception as exc:
            async with self:
                self.start_error = str(exc)
                self.is_starting = False

    async def _poll_job_loop(self, job_id: str):
        import redis.asyncio as aioredis

        r = aioredis.from_url(REDIS_URL)
        pubsub = r.pubsub()
        await pubsub.subscribe(f"job:{job_id}:progress")

        prev_epoch = 0.0
        epoch_start_loss: float | None = None

        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = json.loads(message["data"])

            current_loss = data.get("loss", 0)
            current_epoch = data.get("epoch", 0)

            async with self:
                self.loss_history.append(
                    LossPoint(
                        step=data.get("step", 0),
                        loss=current_loss,
                        epoch=current_epoch,
                        learning_rate=data.get("learning_rate", 0),
                        eval_loss=data.get("eval_loss"),
                    )
                )
                self.current_epoch = current_epoch
                self.total_steps = data.get("total_steps", 0)
                self.elapsed_seconds = data.get("elapsed_seconds", 0)
                self.gpu_memory_used_gb = data.get("gpu_memory_used_gb", 0.0)

            # Detect epoch boundary and log a summary
            if int(current_epoch) > int(prev_epoch):
                if epoch_start_loss is not None and self.loss_history:
                    drop_pct = round(
                        (epoch_start_loss - current_loss) / max(epoch_start_loss, 1e-9) * 100, 1
                    )
                    async with self:
                        self.epoch_log.append(
                            EpochLogEntry(
                                epoch=int(prev_epoch) + 1,
                                loss_start=round(epoch_start_loss, 4),
                                loss_end=round(current_loss, 4),
                                drop_pct=drop_pct,
                                elapsed_seconds=self.elapsed_seconds,
                            )
                        )
                    await self._refresh_commentary(current_loss, drop_pct, int(current_epoch))
                epoch_start_loss = current_loss
            elif epoch_start_loss is None:
                epoch_start_loss = current_loss

            prev_epoch = current_epoch

            if data.get("status") in ("done", "failed"):
                async with self:
                    self.training_status = data.get("status", "done")
                break

        # After loop: fetch final status + output_path from the REST API
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{API_BASE}/api/jobs/{job_id}")
            if resp.status_code == 200:
                sdata = resp.json()
                async with self:
                    self.output_path = sdata.get("output_path", "")
                    self.error_msg = sdata.get("error", "")
                    self.training_status = sdata.get("status", self.training_status)
        except Exception:
            pass

        await pubsub.unsubscribe()
        await r.aclose()

        await self._save_experiment_record()

        if self.training_status == "done":
            async with self:
                self.current_step = 6
            await self._auto_eval()

    async def _refresh_commentary(self, current_loss: float, drop_pct: float, epoch: int):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{API_BASE}/api/jobs/{self.job_id}/commentary",
                    json={
                        "epoch": epoch,
                        "total_epochs": self.epochs,
                        "loss_drop_pct": drop_pct,
                        "current_loss": current_loss,
                        "intent": self.user_intent,
                    },
                )
            if resp.status_code == 200:
                async with self:
                    self.ai_commentary = resp.json().get("commentary", "")
        except Exception:
            pass

    async def _auto_eval(self):
        for _ in range(30):
            import asyncio

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(f"{API_BASE}/api/jobs/{self.job_id}/eval")
                data = resp.json()
                if data.get("status") == "done":
                    async with self:
                        self.eval_status = "done"
                        ppl = data.get("perplexity")
                        self.eval_perplexity = float(ppl) if ppl is not None else 0.0
                        rouge1 = data.get("rouge1")
                        self.eval_rouge1 = float(rouge1) if rouge1 is not None else 0.0
                        bleu = data.get("bleu")
                        self.eval_bleu = float(bleu) if bleu is not None else 0.0
                        rouge2 = data.get("rouge2")
                        self.eval_rouge2 = float(rouge2) if rouge2 is not None else 0.0
                        rougel = data.get("rougeL")
                        self.eval_rougeL = float(rougel) if rougel is not None else 0.0
                        meteor = data.get("meteor")
                        self.eval_meteor = float(meteor) if meteor is not None else 0.0
                    return
            except Exception:
                pass
            await asyncio.sleep(3)

        async with self:
            self.eval_status = "not_ready"

    async def _save_experiment_record(self):
        final_loss = self.loss_history[-1].loss if self.loss_history else 0.0
        save_experiment_run(
            {
                "id": self.experiment_id,
                "name": self.experiment_name,
                "model_id": self.effective_model_id,
                "model_source": self.model_source,
                "technique": self.selected_technique,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "lora_r": self.lora_r,
                "batch_size": self.batch_size,
                "dataset_name": self.dataset_name,
                "user_intent": self.user_intent,
                "final_loss": final_loss,
                "perplexity": self.eval_perplexity,
                "started_at": self.training_start_time,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": self.training_status,
                "output_path": self.output_path,
                "loss_history": [pt.model_dump() for pt in self.loss_history],
            }
        )
        save_final_metrics(
            self.experiment_id,
            {"rouge1": self.eval_rouge1, "bleu": self.eval_bleu},
        )
        async with self:
            pass
        return ExperimentState.load_runs()

    # ── Step 6: Eval & test-chat ───────────────────────────────────
    @rx.event(background=True)
    async def run_eval(self):
        async with self:
            self.eval_status = "running"
        await self._auto_eval()

    @rx.event
    def set_chat_input(self, value: str):
        self.chat_input = value

    @rx.event
    def handle_chat_key(self, key: str):
        if key == "Enter":
            return TrainingPollerState.send_test_chat

    @rx.event(background=True)
    async def send_test_chat(self):
        prompt = self.chat_input.strip()
        if not prompt:
            return

        system = self.user_intent or ""
        full_prompt = f"[System: {system}]\n\n{prompt}" if system else prompt

        async with self:
            self.chat_loading = True
            self.chat_error = ""
            self.test_chat_history = [
                *self.test_chat_history,
                ChatMessage(role="user", content=prompt),
            ]

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    f"{API_BASE}/api/jobs/{self.job_id}/infer",
                    json={"prompt": full_prompt, "max_new_tokens": 300, "temperature": 0.7},
                )
            if resp.status_code == 200:
                response = resp.json().get("response", "")
                async with self:
                    self.test_chat_history = [
                        *self.test_chat_history,
                        ChatMessage(role="assistant", content=response),
                    ]
                    self.chat_input = ""
                    self.chat_loading = False
            else:
                async with self:
                    self.chat_error = resp.json().get("detail", "Inference failed")
                    self.chat_loading = False
        except Exception as exc:
            async with self:
                self.chat_error = str(exc)
                self.chat_loading = False

    # ── Split-brain recovery on page load ─────────────────────────
    @rx.event(background=True)
    async def rehydrate_from_api(self):
        """On /finetune page load: if a job is already running, reconnect to it."""
        async with self:
            if self.job_id:
                return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{API_BASE}/api/jobs?limit=5")
            jobs = resp.json()
        except Exception:
            return
        if not jobs:
            return
        latest = jobs[0]
        status = latest.get("status", "")
        job_id = latest.get("id") or latest.get("job_id", "")
        if status in ("running", "provisioning"):
            async with self:
                self.job_id = job_id
                self.training_status = "running"
                self.current_step = 5
            await self._poll_job_loop(job_id)
        elif status == "done":
            async with self:
                self.job_id = job_id
                self.training_status = "done"
                self.output_path = latest.get("output_path", "")
                self.current_step = 6
