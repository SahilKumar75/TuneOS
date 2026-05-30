"""
TuneOS — Application state for the landing page UI.
Manages sidebar projects, input forms, and theme.
"""
import os
import re

import httpx
import reflex as rx
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=True)


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
    preview_tags: list[str] = []
    preview_license: str = ""
    preview_pipeline: str = ""
    preview_downloads: str = ""
    preview_likes: str = ""
    preview_model_id: str = ""
    preview_architecture: str = ""
    preview_params: str = ""
    preview_formats: str = ""
    preview_library: str = ""
    preview_total_files: str = ""
    preview_created: str = ""
    preview_updated: str = ""
    preview_readme: str = ""
    chat_input: str = ""
    chat_messages: list[dict[str, str]] = []
    is_chat_loading: bool = False
    chat_model: str = "auto"
    last_used_model: str = ""

    # ── Project history (real, built from user actions) ───────────
    projects: List[ProjectItem] = []
    _next_project_id: int = 1

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
        return self.huggingface_model_id

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

    def _extract_bench_bounds(self) -> tuple[int, int]:
        """Return (start, end) line indices of the benchmark section, or (-1,-1)."""
        if not self.preview_readme:
            return (-1, -1)
        lines = self.preview_readme.splitlines()
        bench_kws = {"benchmark", "evaluation", "results", "performance", "leaderboard"}
        start = -1
        depth = 0
        for i, line in enumerate(lines):
            stripped = line.lstrip("#")
            level = len(line) - len(stripped)
            if level == 0:
                continue
            title = stripped.strip().lower()
            if start == -1:
                if any(k in title for k in bench_kws):
                    start = i
                    depth = level
            else:
                if level <= depth and i > start:
                    return (start, i)
        if start != -1:
            return (start, len(lines))
        return (-1, -1)

    @rx.var
    def preview_benchmark(self) -> str:
        """Extract the benchmark/evaluation section from the README."""
        s, e = self._extract_bench_bounds()
        if s == -1:
            return ""
        lines = self.preview_readme.splitlines()
        return "\n".join(lines[s:e]).strip()

    @rx.var
    def preview_readme_no_bench(self) -> str:
        """README with the benchmark section removed (to avoid duplication)."""
        s, e = self._extract_bench_bounds()
        if s == -1:
            return self.preview_readme
        lines = self.preview_readme.splitlines()
        return "\n".join(lines[:s] + lines[e:]).strip()

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
        self.huggingface_model_id = value
        self.github_url = value
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
        self.chat_messages = []
        self.preview_tags = []
        self.preview_license = ""
        self.preview_pipeline = ""
        self.preview_downloads = ""
        self.preview_likes = ""
        self.preview_model_id = ""
        self.preview_architecture = ""
        self.preview_params = ""
        self.preview_formats = ""
        self.preview_library = ""
        self.preview_total_files = ""
        self.preview_created = ""
        self.preview_updated = ""
        self.preview_readme = ""

    @rx.event
    def toggle_sidebar(self):
        self.sidebar_collapsed = not self.sidebar_collapsed

    @rx.event
    def set_hf_model(self, model_id: str):
        self.huggingface_model_id = model_id
        self.active_tab = "huggingface"

    def _input_text(self) -> str:
        return self.huggingface_model_id.strip()

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

        import os
        data: dict = {}
        readme_content: str = ""
        hf_token = os.getenv("HF_TOKEN", "")
        headers: dict = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://huggingface.co/api/models/{repo_id}",
                    headers=headers,
                )
                if response.status_code == 200:
                    data = response.json()
                # Fetch README / model card (try resolve endpoint with auth)
                for readme_url in [
                    f"https://huggingface.co/{repo_id}/resolve/main/README.md",
                    f"https://huggingface.co/{repo_id}/raw/main/README.md",
                ]:
                    readme_resp = await client.get(readme_url, headers=headers, follow_redirects=True)
                    if readme_resp.status_code == 200:
                        raw = readme_resp.text
                        # Strip YAML front matter
                        if raw.startswith("---"):
                            end = raw.find("---", 3)
                            if end != -1:
                                raw = raw[end + 3:].strip()
                        readme_content = raw[:12000]
                        break
        except Exception:
            data = data or {}

        tags = data.get("tags") or []
        pipeline = data.get("pipeline_tag") or "model"
        downloads = data.get("downloads")
        likes = data.get("likes")
        meta_parts = [pipeline]
        if downloads is not None:
            meta_parts.append(f"{downloads:,} downloads")
        if likes is not None:
            meta_parts.append(f"{likes:,} likes")

        card_data = data.get("cardData") or {}
        license_str = card_data.get("license") or ""

        # Build a proper summary from model card
        description = ""
        if card_data.get("model_name"):
            description = f"{card_data['model_name']} is a {pipeline} model."
        if not description:
            description = data.get("description") or f"A {pipeline} model hosted on Hugging Face."

        # Siblings info for file formats and size
        siblings = data.get("siblings") or []
        file_types = set()
        total_files = len(siblings)
        for s in siblings:
            fname = s.get("rfilename", "")
            if fname.endswith(".safetensors"):
                file_types.add("SafeTensors")
            elif fname.endswith(".gguf"):
                file_types.add("GGUF")
            elif fname.endswith(".bin"):
                file_types.add("PyTorch")

        # Architecture info
        config = data.get("config") or {}
        model_type = config.get("model_type") or ""
        arch_list = config.get("architectures") or []
        architecture = arch_list[0] if arch_list else model_type

        # Parameter count
        safetensors_info = data.get("safetensors") or {}
        params_total = safetensors_info.get("total") or 0
        if params_total > 1_000_000_000:
            params_str = f"{params_total / 1_000_000_000:.1f}B"
        elif params_total > 1_000_000:
            params_str = f"{params_total / 1_000_000:.0f}M"
        else:
            params_str = ""

        # Library
        library = data.get("library_name") or ""

        # Created/updated dates
        created = (data.get("createdAt") or "")[:10]
        updated = (data.get("lastModified") or "")[:10]

        return {
            "kind": "huggingface",
            "url": url,
            "title": data.get("modelId") or repo_id,
            "owner": repo_id.split("/")[0],
            "summary": description,
            "meta": " • ".join(meta_parts),
            "tags": tags[:8],
            "license": license_str,
            "pipeline": pipeline,
            "downloads": f"{downloads:,}" if downloads is not None else "",
            "likes": f"{likes:,}" if likes is not None else "",
            "model_id": data.get("modelId") or repo_id,
            "architecture": architecture,
            "params": params_str,
            "formats": ", ".join(sorted(file_types)) if file_types else "",
            "library": library,
            "total_files": str(total_files),
            "created": created,
            "updated": updated,
            "readme": readme_content,
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

    def _handle_local_path(self, path: str) -> dict:
        import os
        name = os.path.basename(path.rstrip("/")) or path
        return {
            "kind": "local",
            "url": path,
            "title": name,
            "owner": "local",
            "summary": f"Local model or dataset at {path}",
            "meta": "Local • No network required",
        }

    def _set_preview(self, preview: dict):
        self.preview_kind = preview["kind"]
        self.preview_url = preview["url"]
        self.preview_title = preview["title"]
        self.preview_owner = preview["owner"]
        self.preview_summary = preview["summary"]
        self.preview_meta = preview["meta"]
        self.preview_tags = preview.get("tags") or []
        self.preview_license = preview.get("license") or ""
        self.preview_pipeline = preview.get("pipeline") or ""
        self.preview_downloads = preview.get("downloads") or ""
        self.preview_likes = preview.get("likes") or ""
        self.preview_model_id = preview.get("model_id") or ""
        self.preview_architecture = preview.get("architecture") or ""
        self.preview_params = preview.get("params") or ""
        self.preview_formats = preview.get("formats") or ""
        self.preview_library = preview.get("library") or ""
        self.preview_total_files = preview.get("total_files") or ""
        self.preview_created = preview.get("created") or ""
        self.preview_updated = preview.get("updated") or ""
        self.preview_readme = preview.get("readme") or ""
        self.preview_ready = True

    @rx.event
    async def start_project(self):
        text = self._input_text()
        if not text:
            self.preview_error = "Paste a Hugging Face link, GitHub URL, or local path first."
            return

        self.preview_loading = True
        self.preview_ready = False
        self.preview_error = ""
        self.workspace_active = False
        self.show_model_selector = False
        self.show_permission_selector = False
        yield

        try:
            if text.startswith("/") or text.startswith("~/") or text.startswith("./"):
                preview = self._handle_local_path(text)
            elif "github.com" in text or self.active_tab == "github":
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
        # Add to project history
        already = any(p.base_model == self.preview_model_id for p in self.projects)
        if not already and self.preview_title:
            new_proj = ProjectItem(
                id=str(self._next_project_id),
                name=self.preview_title,
                base_model=self.preview_model_id or self.preview_title,
                status="active",
                created_at="Just now",
            )
            self.projects = [new_proj, *self.projects]
            self._next_project_id += 1

    @rx.event
    def cancel_preview(self):
        self.preview_loading = False
        self.preview_ready = False
        self.preview_error = ""
        self.workspace_active = False

    @rx.event
    def set_chat_input(self, value: str):
        self.chat_input = value

    @rx.event
    def handle_chat_key(self, key: str):
        if key == "Enter":
            return AppState.send_chat_message

    # Models available for manual selection
    CHAT_MODELS: list[dict[str, str]] = [
        {"id": "auto",                                       "label": "Auto (smart route)"},
        {"id": "anthropic/claude-sonnet-4-5",               "label": "Claude Sonnet 4.5"},
        {"id": "openai/gpt-oss-120b:free",                  "label": "GPT OSS 120B (free)"},
        {"id": "deepseek/deepseek-v4-flash:free",           "label": "DeepSeek V4 Flash (free)"},
        {"id": "qwen/qwen3-coder:free",                     "label": "Qwen3 Coder (free)"},
        {"id": "meta-llama/llama-3.3-70b-instruct:free",    "label": "Llama 3.3 70B (free)"},
        {"id": "nvidia/nemotron-3-super-120b-a12b:free",    "label": "Nemotron 120B (free)"},
    ]

    @rx.event
    def set_chat_model(self, model_id: str):
        self.chat_model = model_id

    # Fallback chain tried in order when a model is rate-limited
    # Each entry: (model_id, base_url, env_key)
    # base_url=None means OpenRouter
    _FREE_FALLBACKS: list[tuple[str, str, str]] = [
        ("openai/gpt-oss-120b:free",                 "openrouter", "OPENROUTER_API_KEY"),
        ("deepseek/deepseek-v4-flash:free",          "openrouter", "OPENROUTER_API_KEY"),
        ("qwen/qwen3-coder:free",                    "openrouter", "OPENROUTER_API_KEY"),
        ("meta-llama/llama-3.3-70b-instruct:free",   "openrouter", "OPENROUTER_API_KEY"),
        ("llama-3.3-70b-versatile",                  "groq",       "GROQ_API_KEY"),
        ("llama3-70b-8192",                          "groq",       "GROQ_API_KEY"),
        ("mixtral-8x7b-32768",                       "groq",       "GROQ_API_KEY"),
        ("gemma2-9b-it",                             "groq",       "GROQ_API_KEY"),
        ("nvidia/nemotron-3-super-120b-a12b:free",   "openrouter", "OPENROUTER_API_KEY"),
        ("moonshotai/kimi-k2.6:free",                "openrouter", "OPENROUTER_API_KEY"),
    ]

    def _route_model(self, text: str) -> tuple[str, str, str]:
        lower = text.lower()
        code_kws = {"config", "yaml", "json", "code", "script", "train", "lora", "qlora",
                    "peft", "batch", "epoch", "lr", "learning rate", "trl", "transformers",
                    "accelerate", "bitsandbytes", "quantiz", "merge", "export", "convert"}
        reasoning_kws = {"why", "explain", "compare", "difference", "tradeoff", "should i",
                         "best practice", "recommend", "analyse", "analyze", "pros", "cons"}
        if any(k in lower for k in code_kws):
            return ("qwen/qwen3-coder:free", "openrouter", "OPENROUTER_API_KEY")
        if any(k in lower for k in reasoning_kws):
            return ("deepseek/deepseek-v4-flash:free", "openrouter", "OPENROUTER_API_KEY")
        return ("llama-3.3-70b-versatile", "groq", "GROQ_API_KEY")

    def _build_system_prompt(self) -> str:
        parts = [
            "You are TuneOS Assistant — an expert in LLM fine-tuning, LoRA, QLoRA, and Hugging Face tooling.",
            "The user is working with the following model. Use this context to give accurate, specific advice.",
            "",
        ]
        if self.preview_title:
            parts.append(f"Model: {self.preview_title}")
        if self.preview_url:
            parts.append(f"URL: {self.preview_url}")
        if self.preview_summary:
            parts.append(f"Description: {self.preview_summary}")
        if self.preview_meta:
            parts.append(f"Stats: {self.preview_meta}")
        if self.preview_architecture:
            parts.append(f"Architecture: {self.preview_architecture}")
        if self.preview_params:
            parts.append(f"Parameters: {self.preview_params}")
        if self.preview_pipeline:
            parts.append(f"Pipeline: {self.preview_pipeline}")
        if self.preview_library:
            parts.append(f"Library: {self.preview_library}")
        if self.preview_license:
            parts.append(f"License: {self.preview_license}")
        if self.preview_tags:
            parts.append(f"Tags: {', '.join(self.preview_tags)}")
        if self.preview_readme:
            parts.append("")
            parts.append("Model Card (README):")
            parts.append(self.preview_readme[:6000])
        parts += [
            "",
            "Be concise and practical. For code/configs, use fenced code blocks.",
        ]
        return "\n".join(parts)

    @rx.event
    async def send_chat_message(self):
        text = self.chat_input.strip()
        if not text or self.is_chat_loading:
            return

        if self.chat_model == "auto":
            first = self._route_model(text)
        else:
            # manual pick — detect if it's a groq model by absence of "/"
            provider = "groq" if "/" not in self.chat_model else "openrouter"
            key_name = "GROQ_API_KEY" if provider == "groq" else "OPENROUTER_API_KEY"
            first = (self.chat_model, provider, key_name)

        self.chat_messages = [*self.chat_messages, {"role": "user", "text": text}]
        self.chat_input = ""
        self.is_chat_loading = True
        self.last_used_model = first[0]
        self.chat_messages = [*self.chat_messages, {"role": "assistant", "text": ""}]
        yield

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            self.chat_messages = [
                *self.chat_messages[:-1],
                {"role": "assistant", "text": "⚠️ No OPENROUTER_API_KEY found. Add it to your .env file."},
            ]
            self.is_chat_loading = False
            yield
            return

        system = self._build_system_prompt()
        history = [
            {"role": ("user" if m["role"] == "user" else "assistant"), "content": m["text"]}
            for m in self.chat_messages[:-1]
            if m["text"]
        ]

        import json
        import httpx as _httpx

        BASE_URLS = {
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
            "groq":       "https://api.groq.com/openai/v1/chat/completions",
        }

        # Build fallback queue: preferred model first, then rest
        fallbacks = [first] + [
            t for t in self._FREE_FALLBACKS if t[0] != first[0]
        ]

        full_text = ""
        last_error = ""

        try:
            async with _httpx.AsyncClient(timeout=30.0) as http:
                for model_id, provider, key_name in fallbacks:
                    key = os.environ.get(key_name, "")
                    if not key:
                        continue  # skip if no key configured

                    headers = {
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    }
                    if provider == "openrouter":
                        headers["X-Title"] = "TuneOS"

                    resp = await http.post(
                        BASE_URLS[provider],
                        headers=headers,
                        json={
                            "model": model_id,
                            "messages": [{"role": "system", "content": system}, *history],
                            "max_tokens": 1024,
                            "stream": True,
                        },
                    )
                    if resp.status_code in (429, 503):
                        last_error = f"{resp.status_code} on {model_id}"
                        continue
                    if resp.status_code != 200:
                        last_error = f"HTTP {resp.status_code} from {model_id}"
                        continue

                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content") or ""
                            full_text += delta
                            if full_text:
                                self.chat_messages = [
                                    *self.chat_messages[:-1],
                                    {"role": "assistant", "text": full_text},
                                ]
                                yield
                        except Exception:
                            continue

                    if full_text:
                        self.last_used_model = f"{model_id} ({provider})"
                        break  # success

            if not full_text:
                self.chat_messages = [
                    *self.chat_messages[:-1],
                    {"role": "assistant", "text": f"⚠️ All models unavailable. Last error: {last_error}"},
                ]
        except Exception as exc:
            self.chat_messages = [
                *self.chat_messages[:-1],
                {"role": "assistant", "text": f"⚠️ Error: {exc}"},
            ]
        finally:
            self.is_chat_loading = False
            yield
