"""
TuneOS — Application state for the landing page UI.
Manages sidebar projects, input forms, and theme.
"""
import re

import httpx
import reflex as rx
from typing import List
from pydantic import BaseModel


class ProjectItem(BaseModel):
    """A single project entry in the sidebar history."""
    id: str
    name: str
    base_model: str
    status: str   # 'training' | 'completed' | 'failed' | 'queued'
    created_at: str


class AppState(rx.State):
    """Main application state for the landing page."""

    # ── Active input tab ──────────────────────────────────────────
    active_tab: str = "huggingface"

    # ── Form values ───────────────────────────────────────────────
    local_model_path: str = ""
    local_model_format: str = "safetensors"
    github_url: str = ""
    github_branch: str = "main"
    huggingface_model_id: str = ""
    huggingface_revision: str = "main"

    # ── Sidebar controls ──────────────────────────────────────────
    search_query: str = ""
    show_model_selector: bool = False
    show_permission_selector: bool = False
    permission_mode: str = "training"
    sidebar_collapsed: bool = False
    tree_n: str = "3"

    # ── Link preview / workspace state ───────────────────────────
    preview_loading: bool = False
    preview_ready: bool = False
    preview_error: str = ""
    workspace_active: bool = False
    preview_kind: str = ""
    preview_url: str = ""
    preview_title: str = ""
    preview_owner: str = ""
    preview_summary: str = ""
    preview_meta: str = ""
    chat_input: str = ""

    # ── Mock project history ──────────────────────────────────────
    projects: List[ProjectItem] = [
        ProjectItem(
            id="1", name="Mistral-7B Customer Support",
            base_model="mistralai/Mistral-7B-v0.1",
            status="completed", created_at="2 hours ago",
        ),
        ProjectItem(
            id="2", name="Llama-3 Code Assistant",
            base_model="meta-llama/Meta-Llama-3-8B",
            status="training", created_at="5 hours ago",
        ),
        ProjectItem(
            id="3", name="Phi-3 Summarizer",
            base_model="microsoft/Phi-3-mini-4k-instruct",
            status="completed", created_at="1 day ago",
        ),
        ProjectItem(
            id="4", name="Gemma-2B Chatbot",
            base_model="google/gemma-2b",
            status="failed", created_at="2 days ago",
        ),
        ProjectItem(
            id="5", name="Mistral-7B Legal QA",
            base_model="mistralai/Mistral-7B-v0.1",
            status="completed", created_at="3 days ago",
        ),
        ProjectItem(
            id="6", name="Llama-3 Medical Notes",
            base_model="meta-llama/Meta-Llama-3-8B",
            status="queued", created_at="4 days ago",
        ),
        ProjectItem(
            id="7", name="Phi-3 Email Drafter",
            base_model="microsoft/Phi-3-mini-4k-instruct",
            status="completed", created_at="1 week ago",
        ),
        ProjectItem(
            id="8", name="Gemma-2B Translator",
            base_model="google/gemma-2b",
            status="completed", created_at="1 week ago",
        ),
    ]

    # ── Computed vars ─────────────────────────────────────────────
    @rx.var
    def filtered_projects(self) -> List[ProjectItem]:
        if not self.search_query:
            return self.projects
        q = self.search_query.lower()
        return [
            p for p in self.projects
            if q in p.name.lower() or q in p.base_model.lower()
        ]

    @rx.var
    def active_projects(self) -> List[ProjectItem]:
        return [
            p for p in self.filtered_projects
            if p.status in ("training", "queued")
        ]

    @rx.var
    def recent_projects(self) -> List[ProjectItem]:
        return [
            p for p in self.filtered_projects
            if p.status not in ("training", "queued")
        ]

    @rx.var
    def input_placeholder(self) -> str:
        placeholders = {
            "huggingface": "Ask TuneOS anything. @ to mention files or datasets",
            "github": "Paste a GitHub repository or describe the training task",
            "local": "Enter a local model path or describe the training task",
        }
        return placeholders.get(self.active_tab, "")

    @rx.var
    def current_input_value(self) -> str:
        if self.active_tab == "huggingface":
            return self.huggingface_model_id
        elif self.active_tab == "github":
            return self.github_url
        return self.local_model_path

    @rx.var
    def preview_source_label(self) -> str:
        labels = {
            "huggingface": "Hugging Face model",
            "github": "GitHub repository",
            "local": "Local model",
        }
        return labels.get(self.preview_kind, "Model")

    @rx.var
    def tab_label(self) -> str:
        labels = {
            "huggingface": "Hugging Face",
            "github": "GitHub",
            "local": "Local",
        }
        return labels.get(self.active_tab, "Hugging Face")

    @rx.var
    def permission_label(self) -> str:
        labels = {
            "analytics": "Analytics",
            "training": "Training",
            "finetuning": "Fine-tuning",
        }
        return labels.get(self.permission_mode, "Training")

    @rx.var
    def balanced_tree_count(self) -> str:
        try:
            n = int(self.tree_n)
            if n < 0:
                return "0"
            if n == 0 or n == 1:
                return "1"
            mod = 1000000007
            dp = [0] * (n + 1)
            dp[0] = 1
            dp[1] = 1
            for i in range(2, n + 1):
                prev1 = dp[i - 1]
                prev2 = dp[i - 2]
                bothMinusOne = (prev1 * prev1) % mod
                oneMinusTwo = (2 * prev1 * prev2) % mod
                dp[i] = (bothMinusOne + oneMinusTwo) % mod
            return str(dp[n])
        except Exception:
            return "0"

    # ── Event handlers ────────────────────────────────────────────
    @rx.event
    def set_search_query(self, query: str):
        self.search_query = query

    @rx.event
    def set_tree_n(self, value: str):
        self.tree_n = value

    @rx.event
    def set_active_tab(self, tab: str):
        self.active_tab = tab
        self.show_model_selector = False
        self.show_permission_selector = False

    @rx.event
    def handle_input_change(self, value: str):
        self.preview_ready = False
        self.preview_error = ""
        self.workspace_active = False
        if self.active_tab == "huggingface":
            self.huggingface_model_id = value
        elif self.active_tab == "github":
            self.github_url = value
        else:
            self.local_model_path = value

    @rx.event
    def toggle_model_selector(self):
        self.show_model_selector = not self.show_model_selector
        self.show_permission_selector = False

    @rx.event
    def toggle_permission_selector(self):
        self.show_permission_selector = not self.show_permission_selector
        self.show_model_selector = False

    @rx.event
    def close_menus(self):
        self.show_model_selector = False
        self.show_permission_selector = False

    @rx.event
    def select_tab_from_menu(self, tab: str):
        self.active_tab = tab
        self.show_model_selector = False
        self.show_permission_selector = False

    @rx.event
    def select_permission_mode(self, mode: str):
        self.permission_mode = mode
        self.show_permission_selector = False

    @rx.event
    def new_project(self):
        self.local_model_path = ""
        self.github_url = ""
        self.huggingface_model_id = ""
        self.active_tab = "huggingface"
        self.show_model_selector = False
        self.show_permission_selector = False
        self.preview_loading = False
        self.preview_ready = False
        self.preview_error = ""
        self.workspace_active = False
        self.preview_kind = ""
        self.preview_url = ""
        self.preview_title = ""
        self.preview_owner = ""
        self.preview_summary = ""
        self.preview_meta = ""
        self.chat_input = ""

    @rx.event
    def toggle_sidebar(self):
        self.sidebar_collapsed = not self.sidebar_collapsed

    @rx.event
    def set_hf_model(self, model_id: str):
        self.huggingface_model_id = model_id
        self.active_tab = "huggingface"

    def _input_text(self) -> str:
        if self.active_tab == "huggingface":
            return self.huggingface_model_id.strip()
        if self.active_tab == "github":
            return self.github_url.strip()
        return self.local_model_path.strip()

    def _extract_hf_repo(self, text: str) -> tuple[str, str]:
        cleaned = text.strip()
        match = re.search(r"huggingface\.co/(?:models/)?([^/\s]+/[^/\s?#]+)", cleaned)
        if match:
            repo_id = match.group(1).rstrip("/")
            return repo_id, f"https://huggingface.co/{repo_id}"
        if re.fullmatch(r"[\w.-]+/[\w.-]+", cleaned):
            return cleaned, f"https://huggingface.co/{cleaned}"
        return "", cleaned

    def _extract_github_repo(self, text: str) -> tuple[str, str]:
        cleaned = text.strip()
        match = re.search(r"github\.com[:/]([^/\s]+)/([^/\s?#.]+)", cleaned)
        if not match:
            return "", cleaned
        repo = f"{match.group(1)}/{match.group(2).replace('.git', '')}".rstrip("/")
        return repo, f"https://github.com/{repo}"

    async def _fetch_hf_preview(self, text: str) -> dict[str, str]:
        repo_id, url = self._extract_hf_repo(text)
        if not repo_id:
            raise ValueError("Paste a Hugging Face model link or model id like owner/model.")

        data: dict = {}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"https://huggingface.co/api/models/{repo_id}")
                if response.status_code == 200:
                    data = response.json()
        except Exception:
            data = {}

        tags = data.get("tags") or []
        pipeline = data.get("pipeline_tag") or "model"
        downloads = data.get("downloads")
        likes = data.get("likes")
        meta_parts = [pipeline]
        if downloads is not None:
            meta_parts.append(f"{downloads:,} downloads")
        if likes is not None:
            meta_parts.append(f"{likes:,} likes")

        summary_bits = []
        if tags:
            summary_bits.append("Tags: " + ", ".join(tags[:6]))
        card_data = data.get("cardData") or {}
        if card_data.get("license"):
            summary_bits.append(f"License: {card_data['license']}")

        return {
            "kind": "huggingface",
            "url": url,
            "title": data.get("modelId") or repo_id,
            "owner": repo_id.split("/")[0],
            "summary": ". ".join(summary_bits) or "Hugging Face model repository ready for fine-tuning setup.",
            "meta": " • ".join(meta_parts),
        }

    async def _fetch_github_preview(self, text: str) -> dict[str, str]:
        repo, url = self._extract_github_repo(text)
        if not repo:
            raise ValueError("Paste a GitHub repository link like https://github.com/owner/repo.")

        data: dict = {}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"https://api.github.com/repos/{repo}",
                    headers={"Accept": "application/vnd.github+json"},
                )
                if response.status_code == 200:
                    data = response.json()
        except Exception:
            data = {}

        stars = data.get("stargazers_count")
        forks = data.get("forks_count")
        language = data.get("language") or "repository"
        branch = data.get("default_branch") or "main"
        meta_parts = [language, f"default branch {branch}"]
        if stars is not None:
            meta_parts.append(f"{stars:,} stars")
        if forks is not None:
            meta_parts.append(f"{forks:,} forks")

        return {
            "kind": "github",
            "url": url,
            "title": data.get("full_name") or repo,
            "owner": repo.split("/")[0],
            "summary": data.get("description") or "GitHub repository ready for import and training setup.",
            "meta": " • ".join(meta_parts),
        }

    def _set_preview(self, preview: dict[str, str]):
        self.preview_kind = preview["kind"]
        self.preview_url = preview["url"]
        self.preview_title = preview["title"]
        self.preview_owner = preview["owner"]
        self.preview_summary = preview["summary"]
        self.preview_meta = preview["meta"]
        self.preview_ready = True

    @rx.event
    async def start_project(self):
        text = self._input_text()
        if not text:
            self.preview_error = "Paste a Hugging Face or GitHub link first."
            return

        self.preview_loading = True
        self.preview_ready = False
        self.preview_error = ""
        self.workspace_active = False
        self.show_model_selector = False
        self.show_permission_selector = False
        yield

        try:
            if "github.com" in text or self.active_tab == "github":
                preview = await self._fetch_github_preview(text)
            else:
                preview = await self._fetch_hf_preview(text)
            self._set_preview(preview)
        except ValueError as exc:
            self.preview_error = str(exc)
        finally:
            self.preview_loading = False

    @rx.event
    def confirm_preview(self):
        if not self.preview_ready:
            return
        self.workspace_active = True
        self.sidebar_collapsed = True

    @rx.event
    def cancel_preview(self):
        self.preview_loading = False
        self.preview_ready = False
        self.preview_error = ""
        self.workspace_active = False

    @rx.event
    def set_chat_input(self, value: str):
        self.chat_input = value
