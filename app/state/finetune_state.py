"""Wizard state for the /finetune dedicated flow — single source of truth."""

from __future__ import annotations

import json
import os
import uuid

import httpx
import reflex as rx
from pydantic import BaseModel

_PRESET_META: dict[str, dict[str, str]] = {
    "mistralai/Mistral-7B-v0.1": {
        "name": "Mistral 7B",
        "org": "mistralai",
        "size": "7B params",
        "notes": "Well-tested with QLoRA, great all-rounder",
        "license": "Apache 2.0",
        "arch": "Decoder-only",
    },
    "meta-llama/Meta-Llama-3-8B": {
        "name": "Llama 3 8B",
        "org": "meta-llama",
        "size": "8B params",
        "notes": "Strong general-purpose model",
        "license": "Llama 3 Community",
        "arch": "Decoder-only",
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "name": "Phi-3 Mini",
        "org": "microsoft",
        "size": "3.8B params",
        "notes": "Fast, runs on smaller GPUs",
        "license": "MIT",
        "arch": "Decoder-only",
    },
    "google/gemma-2b": {
        "name": "Gemma 2B",
        "org": "google",
        "size": "2B params",
        "notes": "Good for low-VRAM environments",
        "license": "Gemma License",
        "arch": "Decoder-only",
    },
    "EleutherAI/pythia-410m": {
        "name": "Pythia 410M",
        "org": "EleutherAI",
        "size": "410M params",
        "notes": "Tiny model — great for testing pipelines fast",
        "license": "Apache 2.0",
        "arch": "Decoder-only",
    },
    "bigcode/starcoder2-3b": {
        "name": "StarCoder2 3B",
        "org": "bigcode",
        "size": "3B params",
        "notes": "Excellent for code generation tasks",
        "license": "BigCode OpenRAIL-M",
        "arch": "Decoder-only",
    },
}

# Organisation meta — keyed lowercase HF org name.
# `logo` is the Clearbit CDN URL for a clean official logo.
# Clearbit provides 128px PNGs of official company marks — consistent, well-cropped.
_ORG_META: dict[str, dict[str, str]] = {
    "mistralai": {
        "initial": "Mi",
        "color": "#FF7000",
        "logo": "https://github.com/mistralai.png?size=128",
    },
    "meta-llama": {
        "initial": "M",
        "color": "#0668E1",
        "logo": "https://github.com/meta-llama.png?size=128",
    },
    "microsoft": {
        "initial": "Ms",
        "color": "#00A4EF",
        "logo": "https://github.com/microsoft.png?size=128",
    },
    "google": {
        "initial": "G",
        "color": "#4285F4",
        "logo": "https://github.com/google.png?size=128",
    },
    "eleutherai": {
        "initial": "EA",
        "color": "#6E40C9",
        "logo": "https://github.com/EleutherAI.png?size=128",
    },
    "bigcode": {
        "initial": "BC",
        "color": "#0EA5E9",
        "logo": "https://github.com/bigcode-project.png?size=128",
    },
    "huggingface": {
        "initial": "HF",
        "color": "#FF9D00",
        "logo": "https://github.com/huggingface.png?size=128",
    },
    "stabilityai": {
        "initial": "SA",
        "color": "#6366F1",
        "logo": "https://github.com/Stability-AI.png?size=128",
    },
    "tiiuae": {
        "initial": "FA",
        "color": "#059669",
        "logo": "https://github.com/tiiuae.png?size=128",
    },
    "qwen": {"initial": "Q", "color": "#7C3AED", "logo": "https://github.com/QwenLM.png?size=128"},
    "cohere": {
        "initial": "Co",
        "color": "#39594D",
        "logo": "https://github.com/cohere-ai.png?size=128",
    },
    "openai": {
        "initial": "Oa",
        "color": "#10A37F",
        "logo": "https://github.com/openai.png?size=128",
    },
    "nvidia": {
        "initial": "Nv",
        "color": "#76B900",
        "logo": "https://github.com/NVIDIA.png?size=128",
    },
    "apple": {"initial": "Ap", "color": "#555555", "logo": "https://github.com/apple.png?size=128"},
    "deepmind": {
        "initial": "DM",
        "color": "#4285F4",
        "logo": "https://github.com/google-deepmind.png?size=128",
    },
    "anthropic": {
        "initial": "An",
        "color": "#C97E45",
        "logo": "https://github.com/anthropics.png?size=128",
    },
}


class DatasetRow(BaseModel):
    instruction: str = ""
    output: str = ""


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class LossPoint(BaseModel):
    step: int = 0
    loss: float = 0.0
    epoch: float = 0.0
    learning_rate: float = 0.0
    eval_loss: float | None = None


class EpochLogEntry(BaseModel):
    epoch: int = 0
    loss_start: float = 0.0
    loss_end: float = 0.0
    drop_pct: float = 0.0
    elapsed_seconds: int = 0


class SeedExample(BaseModel):
    instruction: str = ""
    output: str = ""


DATASET_DIR = os.getenv("DATASET_DIR", "./storage/datasets")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
API_BASE = os.getenv("API_BASE", "http://localhost:8000")


class FinetuneState(rx.State):
    # ── Step tracking ─────────────────────────────────────────────
    current_step: int = 1  # 1–7

    # ── Step 1: Model source ──────────────────────────────────────
    model_source: str = "hub"  # "hub" | "local" | "custom_string"
    selected_model_id: str = ""
    selected_model_name: str = ""
    custom_model_str: str = ""
    local_model_path: str = ""
    model_url_error: str = ""
    is_validating_model: bool = False
    hf_token: str = ""  # for gated models
    selected_technique: str = "qlora"  # "qlora" | "lora" | "adalora" | "ia3" | "prefix" | "prompt"
    step1_show_picker: bool = False  # True = full grid picker (advanced)
    # Extended preview info fetched live from HF Hub API
    model_downloads: str = ""
    model_likes: str = ""
    model_pipeline: str = ""
    model_hf_tags: list[str] = []
    model_context_window: str = ""  # max_position_embeddings from config
    model_type_hf: str = ""  # model_type from config (gemma, llama, etc.)
    model_languages: str = ""  # comma-joined language codes
    model_last_updated: str = ""  # ISO date string from lastModified
    model_bio: str = ""  # first paragraph from model README
    model_fetch_error: str = ""  # debug: last error from fetch_model_info
    is_fetching_model_info: bool = False

    # ── Step 2: Intent ────────────────────────────────────────────
    user_intent: str = ""  # written by approve_intent() for API compat

    # Phase A – filter chips (all optional)
    intent_use_for: str = ""  # "personal" | "company" | "research" | "education" | ""
    intent_domain: str = ""  # "healthcare" | "finance" | "education" | "legal" | "creative" | "technology" | "ecommerce" | "customer_service" | ""
    intent_task_type: str = (
        ""  # "text" | "vision" | "audio" | "code" | "translation" | "summarization" | ""
    )

    # New input fields for Phase A
    intent_project_name: str = ""  # project name
    intent_description: str = ""  # project description
    intent_request_volume: str = ""  # expected request volume
    intent_accuracy_req: str = ""  # accuracy requirements

    # Phase progression
    intent_phase: int = 1  # 1 = filter chips, 2 = questions, 3 = preview

    # Phase B – questionnaire (now dynamically generated)
    intent_question_idx: int = 0
    intent_questions: list[dict] = []  # Dynamic questions generated by AI
    intent_answers: list[str] = []
    intent_custom_answers: list[str] = []
    intent_is_custom: list[bool] = []
    intent_is_generating_questions: bool = False
    intent_live_plan: str = ""  # Updates as user answers questions

    # Phase C – markdown preview
    intent_md: str = ""
    intent_approved: bool = False

    # ── Step 3: Data ──────────────────────────────────────────────
    data_source: str = "upload"  # "upload" | "hub_dataset" | "generate"
    dataset_path: str = ""
    dataset_filename: str = ""
    dataset_preview: list[DatasetRow] = []
    dataset_error: str = ""
    is_uploading: bool = False
    existing_datasets: list[str] = []
    # Hub dataset
    hub_dataset_id: str = ""
    hub_dataset_split: str = "train"
    hub_dataset_instruction_col: str = "instruction"
    hub_dataset_output_col: str = "output"
    hub_dataset_preview: list[DatasetRow] = []
    hub_dataset_columns: list[str] = []
    is_loading_hub_preview: bool = False
    hub_preview_error: str = ""
    # Synthetic generation
    is_generating: bool = False
    generation_method: str = "self_instruct"  # "self_instruct" | "few_shot" | "template"
    generation_n: int = 50
    generation_status: str = ""
    generated_samples: list[DatasetRow] = []
    generation_diversity_score: float = 0.0
    seed_examples: list[SeedExample] = []

    # ── Step 3: Data stats ────────────────────────────────────────
    dataset_row_count: int = 0
    dataset_avg_tokens: float = 0.0

    # ── Step 4: Configure ─────────────────────────────────────────
    ui_mode: str = "simple"  # "simple" | "advanced"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    epochs: int = 3
    learning_rate: str = "2e-4"
    batch_size: int = 4
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    bf16: bool = False
    experiment_name: str = ""
    eval_split_ratio: float = 0.1
    early_stopping_patience: int = 0
    compute_backend: str = "local"  # "local" | "modal" | "hf_spaces"
    prompt_template: str = "alpaca"  # alpaca | chatml | llama3 | phi3 | zephyr
    packing: bool = False

    # ── Computed vars ─────────────────────────────────────────────
    @rx.var
    def can_go_to_intent(self) -> bool:
        return bool(self.effective_model_id)

    @rx.var
    def can_go_to_data(self) -> bool:
        return bool(self.intent_approved)

    @rx.var
    def intent_all_answered(self) -> bool:
        return all(a != "" for a in self.intent_answers)

    @rx.var
    def can_go_to_configure(self) -> bool:
        has_data = (
            (
                self.data_source == "upload"
                and bool(self.dataset_path)
                and not bool(self.dataset_error)
            )
            or (self.data_source == "hub_dataset" and bool(self.hub_dataset_id))
            or (self.data_source == "generate" and bool(self.dataset_path))
            or self.data_source == "skip"
        )
        return has_data

    @rx.var
    def can_start_training(self) -> bool:
        return self.can_go_to_configure and bool(self.effective_model_id)

    @rx.var
    def step_can_advance(self) -> bool:
        """True when the wizard is allowed to move forward from current_step.

        Used by next_step() to enforce guards server-side so no client-side
        trick (e.g. a stray keyboard event) can skip a locked step.
        """
        if self.current_step == 1:
            return self.can_go_to_intent
        if self.current_step == 2:
            # Step 2 uses approve_intent which sets intent_approved; block raw
            # next_step() calls when intent hasn't been approved yet.
            return self.can_go_to_data
        if self.current_step == 3:
            return self.can_go_to_configure
        # Steps 4–7: always allow backward/forward within the workspace
        return True

    @rx.var
    def effective_model_id(self) -> str:
        if self.model_source == "hub" and self.selected_model_id:
            return self.selected_model_id
        if self.model_source == "local" and self.local_model_path:
            return self.local_model_path
        if self.model_source == "custom_string" and self.custom_model_str:
            return self.custom_model_str
        return ""

    @rx.var
    def effective_model_name(self) -> str:
        if self.model_source == "hub":
            return self.selected_model_name or self.selected_model_id
        if self.model_source == "local":
            return os.path.basename(self.local_model_path) if self.local_model_path else ""
        return self.custom_model_str

    @rx.var
    def technique_label(self) -> str:
        _labels = {
            "qlora": "QLoRA",
            "lora": "LoRA",
            "adalora": "AdaLoRA",
            "ia3": "IA³",
            "prefix": "Prefix Tuning",
            "prompt": "Prompt Tuning",
        }
        return _labels.get(self.selected_technique, self.selected_technique.upper())

    @rx.var
    def dataset_name(self) -> str:
        if self.data_source == "skip":
            return "None (skipped)"
        if self.data_source == "hub_dataset":
            return self.hub_dataset_id
        if self.dataset_filename:
            return self.dataset_filename
        return "Unknown dataset"

    @rx.var
    def config_summary_lr(self) -> str:
        lr_map = {"1e-4": "Slow & careful", "2e-4": "Balanced", "5e-4": "Fast learning"}
        return lr_map.get(self.learning_rate, self.learning_rate)

    @rx.var
    def selected_model_size(self) -> str:
        return _PRESET_META.get(self.selected_model_id, {}).get("size", "")

    @rx.var
    def selected_model_notes(self) -> str:
        return _PRESET_META.get(self.selected_model_id, {}).get("notes", "")

    @rx.var
    def selected_model_license(self) -> str:
        return _PRESET_META.get(self.selected_model_id, {}).get("license", "")

    @rx.var
    def selected_model_arch(self) -> str:
        return _PRESET_META.get(self.selected_model_id, {}).get("arch", "")

    @rx.var
    def selected_model_source_label(self) -> str:
        return {"hub": "Hub model", "local": "Local file", "custom_string": "Custom ID"}.get(
            self.model_source, ""
        )

    @rx.var
    def selected_model_org(self) -> str:
        if "/" in self.selected_model_id:
            return self.selected_model_id.split("/")[0]
        return ""

    @rx.var
    def selected_model_org_initial(self) -> str:
        org = self.selected_model_id.split("/")[0] if "/" in self.selected_model_id else ""
        meta = _ORG_META.get(org.lower(), {})
        return meta.get("initial", org[:2].upper() if org else "?")

    @rx.var
    def selected_model_org_color(self) -> str:
        org = self.selected_model_id.split("/")[0] if "/" in self.selected_model_id else ""
        return _ORG_META.get(org.lower(), {}).get("color", "#6366F1")

    @rx.var
    def selected_model_org_avatar(self) -> str:
        """Clearbit logo URL for known orgs; GitHub fallback for unknown ones."""
        org = self.selected_model_id.split("/")[0] if "/" in self.selected_model_id else ""
        if not org:
            return ""
        meta = _ORG_META.get(org.lower(), {})
        return meta.get("logo", f"https://github.com/{org}.png?size=128")

    @rx.var
    def suggested_technique(self) -> str:
        """Recommend a technique based on model size + user intent (Step 2)."""
        intent = (self.intent_domain + " " + " ".join(self.intent_answers)).lower()
        size = self.selected_model_size.lower()

        # Intent-based signal: preference / alignment → DPO
        if any(k in intent for k in ["preference", "alignment", "reward", "rlhf", "helpful"]):
            return "dpo"

        # Parse size to float billions
        size_b = 0.0
        try:
            if "b params" in size:
                size_b = float(size.replace("b params", "").strip())
            elif "m params" in size:
                size_b = float(size.replace("m params", "").strip()) / 1000.0
        except ValueError:
            pass

        # Small models (< 1.5B) can handle standard LoRA efficiently
        if 0 < size_b < 1.5:
            return "lora"
        # Everything ≥ 1.5B benefits from QLoRA's memory efficiency
        return "qlora"

    # ── Step 1 events ─────────────────────────────────────────────
    @rx.event
    def prefill_model(self, model_id: str, model_name: str):
        """Pre-populate model from an external context (e.g. landing page preview).

        Lands the wizard on the confirmation card so the user sees what model
        was chosen and can confirm or swap it — without being dumped back into
        the full picker.  No-op if the wizard is already in progress.
        """
        # Only skip pre-fill when a flow is genuinely in progress (past Step 1).
        # A stale selected_model_id from a previous closed tab should not block the
        # new pre-fill — the tab close should reset state, but if it didn't we
        # still want the new model to take over at Step 1.
        if self.current_step > 1:
            return
        if not model_id:
            return
        meta = _PRESET_META.get(model_id, {})
        self.selected_model_id = model_id
        self.selected_model_name = meta.get("name", model_name or model_id)
        self.model_source = "hub"
        self.custom_model_str = ""
        self.model_url_error = ""
        self.step1_show_picker = False
        self._clear_model_preview()
        return FinetuneState.fetch_model_info

    @rx.event
    def select_model(self, model_id: str, model_name: str):
        self.selected_model_id = model_id
        self.selected_model_name = model_name
        self.model_source = "hub"
        self.custom_model_str = ""
        self.model_url_error = ""
        self.step1_show_picker = False
        self._clear_model_preview()
        return FinetuneState.fetch_model_info

    def _clear_model_preview(self):
        self.model_downloads = ""
        self.model_likes = ""
        self.model_pipeline = ""
        self.model_hf_tags = []
        self.model_context_window = ""
        self.model_type_hf = ""
        self.model_languages = ""
        self.model_last_updated = ""
        self.model_bio = ""
        self.model_fetch_error = ""

    @rx.event
    def select_preset(self, model_id: str):
        """Select a model by ID from the confirmation-card dropdown."""
        meta = _PRESET_META.get(model_id, {})
        self.selected_model_id = model_id
        self.selected_model_name = meta.get("name", model_id)
        self.model_source = "hub"
        self.custom_model_str = ""
        self.model_url_error = ""
        self._clear_model_preview()
        return FinetuneState.fetch_model_info

    @rx.event
    def set_custom_confirm_input(self, value: str):
        """Update model from the text input; auto-strips HF/GitHub URLs to model IDs."""
        cleaned = value.strip()
        # Parse HuggingFace URLs → extract org/model slug
        for prefix in (
            "https://huggingface.co/",
            "https://hf.co/",
            "http://huggingface.co/",
            "huggingface.co/",
        ):
            if cleaned.startswith(prefix):
                slug = cleaned[len(prefix) :].rstrip("/")
                parts = slug.split("/")
                cleaned = "/".join(parts[:2]) if len(parts) >= 2 else slug
                break
        self.custom_model_str = cleaned
        self.selected_model_id = cleaned
        self.selected_model_name = cleaned
        self.model_source = "custom_string"
        self.model_url_error = ""
        self._clear_model_preview()

    @rx.event
    def show_model_picker(self):
        """Open the full grid picker."""
        self.step1_show_picker = True

    @rx.event
    def hide_model_picker(self):
        """Return to the confirmation / selection card."""
        self.step1_show_picker = False

    @rx.event
    def select_technique(self, technique: str):
        self.selected_technique = technique

    @rx.event
    def set_model_source(self, source: str):
        self.model_source = source
        self.model_url_error = ""

    @rx.event
    def set_custom_model_str(self, value: str):
        self.custom_model_str = value
        self.model_url_error = ""

    @rx.event
    def set_hf_token(self, value: str):
        self.hf_token = value

    @rx.event(background=True)
    async def validate_and_select_custom_model(self):
        model_str = self.custom_model_str.strip()
        if not model_str:
            return
        async with self:
            self.is_validating_model = True
            self.model_url_error = ""

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    f"{API_BASE}/api/models/validate",
                    json={"model_id": model_str, "hf_token": self.hf_token},
                )
            data = resp.json()
            if data.get("valid"):
                async with self:
                    self.selected_model_id = model_str
                    self.selected_model_name = data.get("model_type", model_str)
                    self.is_validating_model = False
                    self.step1_show_picker = False
                    self._clear_model_preview()
                return FinetuneState.fetch_model_info
            else:
                async with self:
                    self.model_url_error = data.get("error", "Model not found or inaccessible.")
                    self.is_validating_model = False
        except Exception as exc:
            async with self:
                self.model_url_error = f"Validation failed: {exc}"
                self.is_validating_model = False

    @rx.event(background=True)
    async def fetch_model_info(self):
        """Fetch live metadata from HF Hub API and populate extended preview fields."""
        async with self:
            model_id = self.selected_model_id  # snapshot; used to discard stale responses
            token = self.hf_token
        if not model_id or "/" not in model_id:
            return
        async with self:
            self.is_fetching_model_info = True
        try:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            _SKIP_STARTS = ("#", "!", "[", "<", "|", "*", "-", ">", "_", ":")

            def _is_prose(line: str) -> bool:
                s = line.strip()
                return (
                    bool(s)
                    and len(s) > 60
                    and not any(s.startswith(c) for c in _SKIP_STARTS)
                    and not s[0].isdigit()
                )

            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(
                    f"https://huggingface.co/api/models/{model_id}",
                    headers=headers,
                )
                if resp.status_code != 200:
                    return
                data = resp.json()

                # Bio: fetch README inside the same client session
                bio = ""
                try:
                    readme_resp = await client.get(
                        f"https://huggingface.co/{model_id}/resolve/main/README.md",
                        headers=headers,
                    )
                    if readme_resp.status_code == 200:
                        readme = readme_resp.text
                        if readme.startswith("---"):
                            end = readme.find("---", 3)
                            if end != -1:
                                readme = readme[end + 3 :].strip()
                        lines = readme.split("\n")

                        def _collect_paragraph(start_idx: int) -> str:
                            """Join consecutive non-empty prose lines into one paragraph."""
                            parts = []
                            for ln in lines[start_idx:]:
                                s = ln.strip()
                                if not s:
                                    if parts:
                                        break  # blank line ends the paragraph
                                    continue
                                if s.startswith("#"):
                                    break
                                parts.append(s)
                            return " ".join(parts)

                        # Pass 1: paragraph after a description/about/overview heading
                        in_desc = False
                        for idx, line in enumerate(lines):
                            stripped = line.strip()
                            if stripped.startswith("#") and any(
                                kw in stripped.lower()
                                for kw in (
                                    "description",
                                    "about",
                                    "overview",
                                    "introduction",
                                    "summary",
                                )
                            ):
                                in_desc = True
                                continue
                            if in_desc:
                                if stripped.startswith("#"):
                                    break
                                if _is_prose(stripped):
                                    para = _collect_paragraph(idx)
                                    bio = para[:900] + ("…" if len(para) > 900 else "")
                                    break
                        # Pass 2: fall back to first substantial prose paragraph anywhere
                        if not bio:
                            for idx, line in enumerate(lines):
                                if _is_prose(line.strip()):
                                    para = _collect_paragraph(idx)
                                    bio = para[:900] + ("…" if len(para) > 900 else "")
                                    break
                except Exception:
                    pass

            dl = data.get("downloads", 0) or 0
            likes = data.get("likes", 0) or 0
            pipeline = (data.get("pipeline_tag") or "").replace("-", " ").title()

            # Useful capability tags only
            keep = {
                "text-generation",
                "conversational",
                "code",
                "summarization",
                "translation",
                "question-answering",
                "fill-mask",
                "rlhf",
                "instruction-tuned",
                "chat",
                "fine-tuned",
                "causal-lm",
            }
            raw_tags = data.get("tags") or []
            tags = [t for t in raw_tags if t.lower() in keep][:4]

            # Architecture / config details
            cfg = data.get("config") or {}
            model_type_raw = cfg.get("model_type") or ""
            ctx = cfg.get("max_position_embeddings") or cfg.get("max_seq_len") or 0

            # Languages from card metadata
            card = data.get("cardData") or {}
            langs = card.get("language") or []
            lang_str = ", ".join(str(lg) for lg in langs[:4]) if langs else ""

            # Last-modified date (trim to YYYY-MM-DD)
            last_mod = (data.get("lastModified") or "")[:10]

            async with self:
                # Discard if the user selected a different model while we were fetching
                if self.selected_model_id != model_id:
                    return
                self.model_downloads = (
                    (
                        f"{dl / 1_000_000:.1f}M"
                        if dl >= 1_000_000
                        else f"{dl // 1_000}k"
                        if dl >= 1_000
                        else str(dl)
                    )
                    if dl
                    else ""
                )
                self.model_likes = str(likes) if likes else ""
                self.model_pipeline = pipeline
                self.model_hf_tags = tags
                self.model_type_hf = model_type_raw.title() if model_type_raw else ""
                self.model_context_window = f"{ctx:,} tokens" if ctx else ""
                self.model_languages = lang_str
                self.model_last_updated = last_mod
                self.model_bio = bio
                self.model_fetch_error = ""
                self.is_fetching_model_info = False
        except Exception as _exc:
            import traceback

            _tb = traceback.format_exc()
            print("fetch_model_info ERROR:", _tb)
            async with self:
                if self.selected_model_id == model_id:
                    self.model_fetch_error = str(_exc)
                    self.is_fetching_model_info = False

    async def handle_local_model_upload(self, files: list[rx.UploadFile]):
        self.is_validating_model = True
        self.model_url_error = ""
        yield

        if not files:
            self.is_validating_model = False
            return

        file = files[0]
        data = await file.read()
        dest_dir = os.path.join("./storage/models", uuid.uuid4().hex)
        os.makedirs(dest_dir, exist_ok=True)
        safe_name = os.path.basename(file.filename)
        dest_path = os.path.join(dest_dir, safe_name)
        with open(dest_path, "wb") as f:
            f.write(data)

        if safe_name.endswith(".zip"):
            import zipfile

            with zipfile.ZipFile(dest_path) as archive:
                archive.extractall(dest_dir)
            self.local_model_path = dest_dir
        else:
            self.local_model_path = dest_path
        self.model_source = "local"
        self.is_validating_model = False

    # ── Step 2 events ─────────────────────────────────────────────

    @rx.event
    def set_user_intent(self, value: str):
        """Legacy setter kept for test compat."""
        self.user_intent = value

    @rx.event
    def set_intent_use_for(self, v: str):
        self.intent_use_for = "" if self.intent_use_for == v else v

    @rx.event
    def set_intent_domain(self, v: str):
        self.intent_domain = "" if self.intent_domain == v else v

    @rx.event
    def set_intent_task_type(self, v: str):
        self.intent_task_type = "" if self.intent_task_type == v else v

    @rx.event
    def set_intent_project_name(self, v: str):
        self.intent_project_name = v

    @rx.event
    def set_intent_description(self, v: str):
        self.intent_description = v

    @rx.event
    def set_intent_request_volume(self, v: str):
        self.intent_request_volume = v

    @rx.event
    def set_intent_accuracy_req(self, v: str):
        self.intent_accuracy_req = v

    @rx.event
    async def intent_next_phase(self):
        if self.intent_phase == 1:
            # Moving from Phase A to Phase B - generate personalized questions
            await self._generate_personalized_questions()
        elif self.intent_phase == 2:
            self._generate_intent_md()
        if self.intent_phase < 3:
            self.intent_phase += 1
            yield

    @rx.event
    def intent_prev_phase(self):
        if self.intent_phase > 1:
            self.intent_phase -= 1
            if self.intent_phase == 2:
                self.intent_question_idx = len(self.intent_answers) - 1

    @rx.event
    def intent_next_question(self):
        if self.intent_question_idx < len(self.intent_answers) - 1:
            self.intent_question_idx += 1
        else:
            self._generate_intent_md()
            self.intent_phase = 3

    @rx.event
    def intent_prev_question(self):
        if self.intent_question_idx > 0:
            self.intent_question_idx -= 1
        else:
            self.intent_phase = 1

    @rx.event
    def intent_edit_question(self, idx: int):
        self.intent_question_idx = idx
        yield rx.call_script(
            f"setTimeout(()=>{{const e=document.getElementById('intent-q-{idx}');if(e)e.scrollIntoView({{behavior:'smooth',block:'start'}});}},50)"
        )

    @rx.event
    async def set_intent_answer(self, idx: int, value: str):
        answers = list(self.intent_answers)
        answers[idx] = value
        self.intent_answers = answers
        is_custom = list(self.intent_is_custom)
        is_custom[idx] = False
        self.intent_is_custom = is_custom

        # Update live plan immediately
        await self._update_live_plan()

        # Auto-advance focus and scroll to next question
        if self.intent_question_idx == idx and idx < len(self.intent_answers) - 1:
            self.intent_question_idx = idx + 1
            next_id = f"intent-q-{idx + 1}"
            yield rx.call_script(
                f"setTimeout(()=>{{const e=document.getElementById('{next_id}');if(e)e.scrollIntoView({{behavior:'smooth',block:'start'}});}},120)"
            )

    @rx.event
    def set_intent_custom_answer(self, idx: int, value: str):
        custom = list(self.intent_custom_answers)
        custom[idx] = value
        self.intent_custom_answers = custom
        answers = list(self.intent_answers)
        answers[idx] = value
        self.intent_answers = answers

    @rx.event
    def toggle_intent_custom(self, idx: int):
        is_custom = list(self.intent_is_custom)
        is_custom[idx] = not is_custom[idx]
        self.intent_is_custom = is_custom
        if is_custom[idx]:
            answers = list(self.intent_answers)
            answers[idx] = self.intent_custom_answers[idx]
            self.intent_answers = answers
        else:
            answers = list(self.intent_answers)
            answers[idx] = ""
            self.intent_answers = answers

    def _generate_intent_md(self):
        use_for_label = {"personal": "Personal", "company": "Company product"}.get(
            self.intent_use_for, "Not specified"
        )
        domain_label = {
            "healthcare": "Healthcare",
            "finance": "Finance",
            "education": "Education",
            "legal": "Legal",
            "creative": "Creative",
        }.get(self.intent_domain, "Not specified")
        task_label = {
            "text": "Text generation",
            "vision": "Image / Vision",
            "audio": "Audio / Speech",
            "code": "Code",
        }.get(self.intent_task_type, "Not specified")

        # Use live plan if available
        if self.intent_live_plan:
            summary = self.intent_live_plan
        else:
            filled = [a for a in self.intent_answers if a]
            if filled:
                summary = f"A {task_label.lower()} model for {domain_label.lower()}."
            else:
                summary = "Intent not fully specified — all fields can be updated before training."

        q_lines = "\n".join(
            f"{i + 1}. **{self.intent_questions[i]['heading'] if i < len(self.intent_questions) else f'Question {i + 1}'}:** {self.intent_answers[i] or 'Not specified'}"
            for i in range(len(self.intent_answers))
        )

        self.intent_md = f"""# Fine-Tuning Intent Profile

## Summary
{summary}

## Use Case Context
- **Project Name:** {self.intent_project_name or "Not specified"}
- **Description:** {self.intent_description or "Not specified"}
- **Use for:** {use_for_label}
- **Domain:** {domain_label}
- **Task type:** {task_label}
- **Expected Volume:** {self.intent_request_volume or "Not specified"}
- **Accuracy Requirements:** {self.intent_accuracy_req or "Not specified"}

## Questionnaire
{q_lines}

## Machine Context
```
intent_use_for: {self.intent_use_for or "not_set"}
intent_domain: {self.intent_domain or "not_set"}
intent_task_type: {self.intent_task_type or "not_set"}
intent_answers: {self.intent_answers}
```
"""

    async def _generate_personalized_questions(self):
        """Generate personalized questions using OpenRouter API based on Phase A inputs."""
        import json
        import os

        import httpx

        self.intent_is_generating_questions = True
        yield

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            # Fallback to default questions
            self.intent_questions = [
                {
                    "heading": "What is the primary goal of this model?",
                    "options": [
                        "Answer questions / provide information",
                        "Generate or transform content",
                        "Classify, analyze, or extract data",
                    ],
                },
                {
                    "heading": "Who is the target audience?",
                    "options": [
                        "General public / consumers",
                        "Domain professionals",
                        "Internal team / developers",
                    ],
                },
                {
                    "heading": "What is the primary input format?",
                    "options": [
                        "Free-form text / conversations",
                        "Structured data or documents",
                        "Mixed / varies",
                    ],
                },
                {
                    "heading": "What tone and style should the model use?",
                    "options": [
                        "Formal and precise",
                        "Friendly and conversational",
                        "Concise and direct",
                    ],
                },
                {
                    "heading": "How will you measure success?",
                    "options": [
                        "Accuracy / factual correctness",
                        "User satisfaction / engagement",
                        "Task completion / automation rate",
                    ],
                },
            ]
            self.intent_answers = [""] * len(self.intent_questions)
            self.intent_custom_answers = [""] * len(self.intent_questions)
            self.intent_is_custom = [False] * len(self.intent_questions)
            self.intent_is_generating_questions = False
            yield
            return

        # Build context from Phase A
        context = f"""Generate 5 personalized questions for a fine-tuning project with these details:
- Project Name: {self.intent_project_name or "Not specified"}
- Description: {self.intent_description or "General purpose"}
- Use Case: {self.intent_use_for or "Not specified"}
- Domain: {self.intent_domain or "General"}
- Task Type: {self.intent_task_type or "Text"}
- Expected Volume: {self.intent_request_volume or "Not specified"}
- Accuracy Requirements: {self.intent_accuracy_req or "Standard"}

Generate 5 highly relevant questions that will help refine this fine-tuning project. Each question should have 3-4 specific answer options.

Return ONLY valid JSON in this exact format, no other text:
{{
  "questions": [
    {{
      "heading": "Your question here?",
      "options": ["Option 1", "Option 2", "Option 3"]
    }}
  ]
}}"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "X-Title": "TuneOS Intent Questions",
                    },
                    json={
                        "model": "deepseek/deepseek-v4-flash:free",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful AI that generates JSON only. Never include explanations, just pure JSON.",
                            },
                            {"role": "user", "content": context},
                        ],
                        "max_tokens": 1500,
                        "temperature": 0.7,
                    },
                )

                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]

                    # Extract JSON from response (in case model adds extra text)
                    import re

                    json_match = re.search(r"\{.*\}", content, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        self.intent_questions = parsed.get("questions", [])[:5]

                        # Initialize answer arrays
                        self.intent_answers = [""] * len(self.intent_questions)
                        self.intent_custom_answers = [""] * len(self.intent_questions)
                        self.intent_is_custom = [False] * len(self.intent_questions)
                    else:
                        raise ValueError("No JSON found in response")
                else:
                    raise Exception(f"API returned {resp.status_code}")

        except Exception as e:
            # Fallback to default questions on error
            print(f"Error generating questions: {e}")
            self.intent_questions = [
                {
                    "heading": "What is the primary goal of this model?",
                    "options": [
                        "Answer questions / provide information",
                        "Generate or transform content",
                        "Classify, analyze, or extract data",
                    ],
                },
                {
                    "heading": "Who is the target audience?",
                    "options": [
                        "General public / consumers",
                        "Domain professionals",
                        "Internal team / developers",
                    ],
                },
                {
                    "heading": "What is the primary input format?",
                    "options": [
                        "Free-form text / conversations",
                        "Structured data or documents",
                        "Mixed / varies",
                    ],
                },
            ]
            self.intent_answers = [""] * len(self.intent_questions)
            self.intent_custom_answers = [""] * len(self.intent_questions)
            self.intent_is_custom = [False] * len(self.intent_questions)

        self.intent_is_generating_questions = False
        yield

    async def _update_live_plan(self):
        """Update the live plan based on current answers using OpenRouter API."""
        import os

        import httpx

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return

        # Build context
        answered = []
        for i, answer in enumerate(self.intent_answers):
            if answer and i < len(self.intent_questions):
                answered.append(f"Q: {self.intent_questions[i]['heading']}\nA: {answer}")

        if not answered:
            return

        context = f"""Based on these project details and answers, write a concise 2-3 sentence summary of what this fine-tuned model will do:

Project Context:
- Name: {self.intent_project_name or "Not specified"}
- Description: {self.intent_description or "General purpose"}
- Domain: {self.intent_domain or "General"}
- Task: {self.intent_task_type or "Text"}

Answered Questions:
{chr(10).join(answered)}

Write ONLY the summary, no other text."""

        try:
            async with httpx.AsyncClient(timeout=15.0) as http:
                resp = await http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "X-Title": "TuneOS Plan Update",
                    },
                    json={
                        "model": "deepseek/deepseek-v4-flash:free",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You write concise, clear summaries. Never use markdown or formatting.",
                            },
                            {"role": "user", "content": context},
                        ],
                        "max_tokens": 200,
                        "temperature": 0.5,
                    },
                )

                if resp.status_code == 200:
                    data = resp.json()
                    summary = data["choices"][0]["message"]["content"].strip()
                    self.intent_live_plan = summary
                    yield

        except Exception as e:
            print(f"Error updating live plan: {e}")
            # Silent fail - plan update is optional

    @rx.event
    def approve_intent(self):
        self.intent_approved = True
        self.user_intent = self.intent_md
        return FinetuneState.next_step()

    # ── Step 3 events ─────────────────────────────────────────────
    @rx.event
    def set_data_source(self, source: str):
        self.data_source = source
        if source == "skip":
            self.dataset_path = ""
            self.dataset_filename = ""
            self.dataset_error = ""

    @rx.event
    def set_hub_dataset_id(self, dataset_id: str):
        self.hub_dataset_id = dataset_id
        self.data_source = "hub_dataset"

    @rx.event
    def set_hub_instruction_col(self, value: str):
        self.hub_dataset_instruction_col = value

    @rx.event
    def set_hub_output_col(self, value: str):
        self.hub_dataset_output_col = value

    @rx.event(background=True)
    async def load_hub_dataset_preview(self):
        if not self.hub_dataset_id:
            return
        async with self:
            self.is_loading_hub_preview = True
            self.hub_preview_error = ""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{API_BASE}/api/datasets/{self.hub_dataset_id}/preview")
            if resp.status_code == 200:
                data = resp.json()
                async with self:
                    self.hub_dataset_columns = data.get("columns", [])
                    raw_rows = data.get("rows", [])
                    self.hub_dataset_preview = [
                        DatasetRow(instruction=r.get("instruction", ""), output=r.get("output", ""))
                        for r in raw_rows
                    ]
                    self.is_loading_hub_preview = False
                    # Auto-detect instruction/output columns
                    cols = data.get("columns", [])
                    if "instruction" in cols:
                        self.hub_dataset_instruction_col = "instruction"
                    if "output" in cols:
                        self.hub_dataset_output_col = "output"
            else:
                async with self:
                    self.hub_preview_error = resp.json().get("detail", "Failed to load preview")
                    self.is_loading_hub_preview = False
        except Exception as exc:
            async with self:
                self.hub_preview_error = str(exc)
                self.is_loading_hub_preview = False

    @rx.event
    def load_existing_datasets(self):
        if not os.path.exists(DATASET_DIR):
            self.existing_datasets = []
            return
        self.existing_datasets = [
            f for f in os.listdir(DATASET_DIR) if os.path.isfile(os.path.join(DATASET_DIR, f))
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
        out_path = os.path.join(DATASET_DIR, os.path.basename(file.filename))
        with open(out_path, "wb") as f:
            f.write(data)

        self.dataset_path = out_path
        self.dataset_filename = file.filename
        self.is_uploading = False

        yield FinetuneState.load_existing_datasets()
        self._validate_dataset_at(out_path)

    def _validate_dataset_at(self, path: str):
        import pandas as pd

        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, nrows=10)
            elif path.endswith(".json") and not path.endswith(".jsonl"):
                import json as _json

                with open(path) as fh:
                    raw = _json.load(fh)
                df = pd.DataFrame(raw if isinstance(raw, list) else [raw])
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

            # Accept any two columns — user can remap, but prefer instruction/output
            if len(df.columns) < 2:
                self.dataset_error = "Dataset must have at least 2 columns."
                self.dataset_preview = []
                return

            # Use instruction/output if present, else first two columns
            inst_col = "instruction" if "instruction" in df.columns else df.columns[0]
            out_col = "output" if "output" in df.columns else df.columns[1]

            self.dataset_error = ""
            records = (
                df[[inst_col, out_col]]
                .head(5)
                .fillna("")
                .rename(columns={inst_col: "instruction", out_col: "output"})
                .to_dict("records")
            )
            self.dataset_preview = [
                DatasetRow(instruction=r.get("instruction", ""), output=r.get("output", ""))
                for r in records
            ]
        except Exception as exc:
            self.dataset_error = f"Could not read file: {exc}"
            self.dataset_preview = []

    @rx.event
    def set_generation_method(self, method: str):
        self.generation_method = method

    @rx.event
    def set_generation_n(self, value: str):
        try:
            self.generation_n = int(value)
        except ValueError:
            pass

    @rx.event(background=True)
    async def generate_starter_dataset(self):
        async with self:
            self.is_generating = True
            self.generation_status = "Generating data..."
            self.generated_samples = []

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{API_BASE}/api/datasets/generate",
                    json={
                        "user_intent": self.user_intent,
                        "method": self.generation_method,
                        "n_samples": self.generation_n,
                        "seed_examples": self.seed_examples,
                        "hf_token": self.hf_token,
                    },
                )
            if resp.status_code == 200:
                data = resp.json()
                samples = data.get("samples", [])
                stats = data.get("stats", {})
                async with self:
                    self.dataset_path = data.get("dataset_path", "")
                    self.dataset_filename = os.path.basename(self.dataset_path)
                    preview_rows = [
                        DatasetRow(instruction=s.get("instruction", ""), output=s.get("output", ""))
                        for s in samples[:5]
                    ]
                    self.generated_samples = preview_rows
                    self.dataset_preview = preview_rows
                    n = stats.get("final_count", len(samples))
                    div = stats.get("diversity_score", 0)
                    self.generation_diversity_score = div
                    self.generation_status = f"Generated {n} examples" + (
                        f" · diversity {div:.2f}" if div else ""
                    )
                    self.is_generating = False
                    self.data_source = "generate"
            else:
                async with self:
                    self.generation_status = (
                        f"Generation failed: {resp.json().get('detail', 'Unknown error')}"
                    )
                    self.is_generating = False
        except Exception as exc:
            async with self:
                self.generation_status = f"Generation failed: {exc}"
                self.is_generating = False

    # ── Navigation ────────────────────────────────────────────────
    @rx.event
    def go_to_step(self, step: int):
        self.current_step = step

    @rx.event
    def next_step(self):
        if not self.step_can_advance:
            return
        self.current_step = min(7, self.current_step + 1)

    @rx.event
    def prev_step(self):
        if self.current_step > 1:
            self.current_step -= 1

    # ── Step 4 setters ────────────────────────────────────────────
    @rx.event
    def set_ui_mode(self, mode: str):
        self.ui_mode = mode

    @rx.event
    def toggle_ui_mode(self, advanced: bool):
        self.ui_mode = "advanced" if advanced else "simple"

    @rx.event
    def set_lora_r(self, value: list[float]):
        self.lora_r = int(value[0])

    @rx.event
    def set_lora_alpha(self, value: str):
        try:
            self.lora_alpha = max(1, int(value))
        except ValueError:
            pass

    @rx.event
    def set_lora_dropout(self, value: list[float]):
        self.lora_dropout = round(value[0], 2)

    @rx.event
    def set_epochs(self, value: str):
        try:
            self.epochs = max(1, min(50, int(value)))
        except ValueError:
            pass

    @rx.event
    def set_compute_backend(self, value: str):
        if value in ("local", "modal", "hf_spaces"):
            self.compute_backend = value

    @rx.event
    def set_prompt_template(self, value: str):
        if value in ("alpaca", "chatml", "llama3", "phi3", "zephyr"):
            self.prompt_template = value

    @rx.event
    def set_packing(self, value: bool):
        self.packing = value

    @rx.event
    def set_learning_rate(self, value: str):
        self.learning_rate = value

    @rx.event
    def set_batch_size(self, value: str):
        try:
            self.batch_size = int(value)
        except ValueError:
            pass

    @rx.event
    def set_max_seq_length(self, value: str):
        try:
            self.max_seq_length = int(value)
        except ValueError:
            pass

    @rx.event
    def set_gradient_accumulation_steps(self, value: str):
        try:
            self.gradient_accumulation_steps = int(value)
        except ValueError:
            pass

    @rx.event
    def set_warmup_ratio(self, value: list[float]):
        self.warmup_ratio = round(value[0], 2)

    @rx.event
    def set_lr_scheduler(self, value: str):
        self.lr_scheduler = value

    @rx.event
    def set_bf16(self, value: bool):
        self.bf16 = value

    @rx.event
    def set_experiment_name(self, value: str):
        self.experiment_name = value

    @rx.event
    def set_eval_split_ratio(self, value: list[float]):
        self.eval_split_ratio = round(
            float(value[0]) if isinstance(value, list) else float(value), 2
        )

    @rx.event
    def set_early_stopping_patience(self, value: str):
        try:
            self.early_stopping_patience = max(0, int(value))
        except (ValueError, TypeError):
            pass
