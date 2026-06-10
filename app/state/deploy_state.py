"""Deploy state — Step 7 export/push operations.

Inherits from TrainingPollerState so it can guard against deploying before
training is complete (reads ``self.training_status`` and ``self.job_id``).
"""

from __future__ import annotations

import httpx
import reflex as rx

from app.state.finetune_state import API_BASE
from app.state.training_poller_state import TrainingPollerState


class DeployState(TrainingPollerState):
    """Deploy runtime — added on top of TrainingPollerState."""

    # ── Step 7: Deploy ────────────────────────────────────────────
    deploy_adapter: bool = True
    deploy_merged: bool = False
    deploy_hub: bool = False
    deploy_gguf: bool = False
    deploy_github: bool = False
    hf_token_input: str = ""
    hf_repo_name: str = ""
    push_status: str = "idle"
    push_error: str = ""
    push_repo_url: str = ""
    gguf_quantization: str = "Q4_K_M"
    github_repo_url: str = ""
    github_token: str = ""
    merge_status: str = "idle"
    deploy_log: str = ""
    gguf_status: str = "idle"
    github_push_status: str = "idle"

    # ── Deploy event handlers ─────────────────────────────────────
    @rx.event
    def toggle_deploy_target(self, target: str):
        targets = {
            "adapter": "deploy_adapter",
            "merged": "deploy_merged",
            "hub": "deploy_hub",
            "gguf": "deploy_gguf",
            "github": "deploy_github",
        }
        if target in targets:
            attr = targets[target]
            setattr(self, attr, not getattr(self, attr))

    @rx.event
    def set_hf_repo_name(self, value: str):
        self.hf_repo_name = value

    @rx.event
    def set_hf_token_input(self, value: str):
        self.hf_token_input = value

    @rx.event
    def set_gguf_quantization(self, value: str):
        self.gguf_quantization = value

    @rx.event
    def set_github_repo_url(self, value: str):
        self.github_repo_url = value

    @rx.event
    def set_github_token(self, value: str):
        self.github_token = value

    @rx.event
    def download_adapter(self):
        return rx.redirect(f"{API_BASE}/api/jobs/{self.job_id}/download")

    @rx.event(background=True)
    async def push_to_hub(self):
        async with self:
            self.push_status = "pushing"
            self.push_error = ""

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    f"{API_BASE}/api/jobs/{self.job_id}/push_hub",
                    json={"repo_name": self.hf_repo_name, "hf_token": self.hf_token_input},
                )
            if resp.status_code == 200:
                async with self:
                    self.push_status = "done"
                    self.push_repo_url = resp.json().get("repo_url", "")
            else:
                async with self:
                    self.push_status = "error"
                    self.push_error = resp.json().get("detail", "Push failed")
        except Exception as exc:
            async with self:
                self.push_status = "error"
                self.push_error = str(exc)

    @rx.event(background=True)
    async def start_merge(self):
        async with self:
            self.merge_status = "merging"
            self.deploy_log = "Starting model merge..."

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{API_BASE}/api/jobs/{self.job_id}/merge",
                    json={"hf_token": self.hf_token_input},
                )
            if resp.status_code in (200, 202):
                async with self:
                    self.deploy_log += "\nMerge job submitted. This may take 5–15 minutes."
            else:
                async with self:
                    self.merge_status = "error"
                    self.deploy_log += f"\nMerge failed: {resp.json().get('detail', 'Unknown')}"
        except Exception as exc:
            async with self:
                self.merge_status = "error"
                self.deploy_log += f"\nMerge error: {exc}"

    @rx.event(background=True)
    async def start_gguf_export(self):
        async with self:
            self.gguf_status = "exporting"
            self.deploy_log += "\nStarting GGUF export..."

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{API_BASE}/api/jobs/{self.job_id}/export-gguf",
                    json={"quant_type": self.gguf_quantization},
                )
            if resp.status_code in (200, 202):
                async with self:
                    self.deploy_log += "\nGGUF export job submitted."
            else:
                async with self:
                    self.gguf_status = "error"
                    self.deploy_log += f"\nGGUF export failed: {resp.json().get('detail', '')}"
        except Exception as exc:
            async with self:
                self.gguf_status = "error"
                self.deploy_log += f"\nGGUF export error: {exc}"

    @rx.event(background=True)
    async def push_to_github(self):
        async with self:
            self.github_push_status = "pushing"
            self.deploy_log += "\nPushing adapter to GitHub..."

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{API_BASE}/api/jobs/{self.job_id}/push-github",
                    json={"repo_url": self.github_repo_url, "github_token": self.github_token},
                )
            if resp.status_code == 200:
                async with self:
                    self.github_push_status = "done"
                    self.deploy_log += f"\nPushed to {self.github_repo_url}"
            else:
                async with self:
                    self.github_push_status = "error"
                    self.deploy_log += f"\nGitHub push failed: {resp.json().get('detail', '')}"
        except Exception as exc:
            async with self:
                self.github_push_status = "error"
                self.deploy_log += f"\nGitHub push error: {exc}"
