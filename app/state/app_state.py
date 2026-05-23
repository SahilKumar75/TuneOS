"""
TuneOS — Application state for the landing page UI.
Manages sidebar projects, input forms, and theme.
"""
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
    show_action_menu: bool = False
    show_model_selector: bool = False
    tree_n: str = "3"

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
    def tab_label(self) -> str:
        labels = {
            "huggingface": "Hugging Face",
            "github": "GitHub",
            "local": "Local",
        }
        return labels.get(self.active_tab, "Hugging Face")

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
        self.show_action_menu = False

    @rx.event
    def handle_input_change(self, value: str):
        if self.active_tab == "huggingface":
            self.huggingface_model_id = value
        elif self.active_tab == "github":
            self.github_url = value
        else:
            self.local_model_path = value

    @rx.event
    def toggle_action_menu(self):
        self.show_action_menu = not self.show_action_menu
        self.show_model_selector = False

    @rx.event
    def toggle_model_selector(self):
        self.show_model_selector = not self.show_model_selector
        self.show_action_menu = False

    @rx.event
    def close_menus(self):
        self.show_action_menu = False
        self.show_model_selector = False

    @rx.event
    def select_tab_from_menu(self, tab: str):
        self.active_tab = tab
        self.show_action_menu = False
        self.show_model_selector = False

    @rx.event
    def new_project(self):
        self.local_model_path = ""
        self.github_url = ""
        self.huggingface_model_id = ""
        self.active_tab = "huggingface"
        self.show_action_menu = False
        self.show_model_selector = False

    @rx.event
    def set_hf_model(self, model_id: str):
        self.huggingface_model_id = model_id
        self.active_tab = "huggingface"

    @rx.event
    def start_project(self):
        # Will be wired to the actual training pipeline later
        pass
