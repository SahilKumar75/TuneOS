"""Fine-tune wizard — Step 1: Model selection."""

from __future__ import annotations

import reflex as rx

from app.components.finetune.shared import _card, _label, _nav_buttons, _section_heading
from app.state.finetune_state import FinetuneState
from app.styles import c

_QUICK_MODELS = [
    ("mistralai/Mistral-7B-v0.1", "Mistral 7B", False),
    ("meta-llama/Meta-Llama-3-8B", "Llama 3 8B", True),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3 Mini", False),
    ("google/gemma-2b", "Gemma 2B", False),
    ("EleutherAI/pythia-410m", "Pythia 410M", False),
    ("bigcode/starcoder2-3b", "StarCoder2 3B", False),
]

_VLM_QUICK = [
    ("llava-hf/llava-1.5-7b-hf", "LLaVA-1.5 7B", False),
    ("Qwen/Qwen2-VL-2B-Instruct", "Qwen2-VL 2B", False),
]

_HF_LOGO = "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"


def _hf_source_badge() -> rx.Component:
    return rx.hstack(
        rx.image(src=_HF_LOGO, width="14px", height="14px"),
        rx.text("Hugging Face", font_size="0.72rem", color="#FF9D00", font_weight="600"),
        spacing="1",
        align="center",
    )


def _selected_model_panel() -> rx.Component:
    """Full info card shown as soon as a model is selected."""
    return rx.cond(
        FinetuneState.selected_model_id != "",
        rx.vstack(
            # Label row
            rx.hstack(
                rx.text(
                    "Current model selected",
                    font_size="0.72rem",
                    font_weight="700",
                    color=c("text_muted"),
                    text_transform="uppercase",
                    letter_spacing="0.06em",
                ),
                rx.spacer(),
                rx.cond(
                    FinetuneState.model_source == "hub",
                    _hf_source_badge(),
                    rx.fragment(),
                ),
                width="100%",
                align="center",
            ),
            # Card
            rx.box(
                rx.vstack(
                    # Org avatar + model name + spinner
                    rx.hstack(
                        rx.cond(
                            FinetuneState.selected_model_org_avatar != "",
                            rx.avatar(
                                src=FinetuneState.selected_model_org_avatar,
                                fallback=FinetuneState.selected_model_org_initial,
                                size="4",
                                radius="full",
                            ),
                            rx.avatar(
                                fallback=FinetuneState.selected_model_org_initial,
                                size="4",
                                radius="full",
                                color_scheme="indigo",
                            ),
                        ),
                        rx.vstack(
                            rx.text(
                                FinetuneState.selected_model_id,
                                font_size="1rem",
                                font_weight="700",
                                color=c("text_primary"),
                            ),
                            rx.text(
                                FinetuneState.selected_model_org,
                                font_size="0.75rem",
                                color=c("text_muted"),
                            ),
                            spacing="0",
                            align_items="flex-start",
                        ),
                        rx.spacer(),
                        rx.cond(
                            FinetuneState.is_fetching_model_info,
                            rx.spinner(size="2"),
                            rx.fragment(),
                        ),
                        spacing="3",
                        align="center",
                        width="100%",
                    ),
                    # Stats row
                    rx.cond(
                        ~FinetuneState.is_fetching_model_info,
                        rx.hstack(
                            rx.cond(
                                FinetuneState.model_downloads != "",
                                rx.hstack(
                                    rx.icon("download", size=12, color=c("text_muted")),
                                    rx.text(
                                        FinetuneState.model_downloads,
                                        font_size="0.76rem",
                                        color=c("text_secondary"),
                                    ),
                                    spacing="1",
                                    align="center",
                                ),
                                rx.fragment(),
                            ),
                            rx.cond(
                                FinetuneState.model_likes != "",
                                rx.hstack(
                                    rx.icon("heart", size=12, color=c("text_muted")),
                                    rx.text(
                                        FinetuneState.model_likes,
                                        font_size="0.76rem",
                                        color=c("text_secondary"),
                                    ),
                                    spacing="1",
                                    align="center",
                                ),
                                rx.fragment(),
                            ),
                            rx.cond(
                                FinetuneState.model_context_window != "",
                                rx.hstack(
                                    rx.icon("layers", size=12, color=c("text_muted")),
                                    rx.text(
                                        FinetuneState.model_context_window,
                                        font_size="0.76rem",
                                        color=c("text_secondary"),
                                    ),
                                    spacing="1",
                                    align="center",
                                ),
                                rx.fragment(),
                            ),
                            rx.cond(
                                FinetuneState.model_pipeline != "",
                                rx.badge(
                                    FinetuneState.model_pipeline,
                                    color_scheme="blue",
                                    size="1",
                                ),
                                rx.fragment(),
                            ),
                            spacing="3",
                            wrap="wrap",
                        ),
                        rx.fragment(),
                    ),
                    # Tags
                    rx.cond(
                        FinetuneState.model_hf_tags.length() > 0,
                        rx.hstack(
                            rx.foreach(
                                FinetuneState.model_hf_tags,
                                lambda tag: rx.badge(
                                    tag, color_scheme="gray", size="1", variant="soft"
                                ),
                            ),
                            wrap="wrap",
                            spacing="1",
                        ),
                        rx.fragment(),
                    ),
                    # Bio
                    rx.cond(
                        FinetuneState.model_bio != "",
                        rx.text(
                            FinetuneState.model_bio,
                            font_size="0.8rem",
                            color=c("text_secondary"),
                            line_height="1.6",
                        ),
                        rx.fragment(),
                    ),
                    spacing="3",
                    align_items="flex-start",
                    width="100%",
                ),
                background=c("bg_card"),
                border="2px solid",
                border_color=c("accent"),
                border_radius="12px",
                padding="18px",
                width="100%",
            ),
            spacing="2",
            width="100%",
            margin_bottom="20px",
        ),
        rx.fragment(),
    )


def _quick_chip(model_id: str, name: str, token_required: bool) -> rx.Component:
    is_sel = FinetuneState.selected_model_id == model_id
    return rx.button(
        rx.hstack(
            rx.text(name, font_size="0.8rem"),
            rx.cond(
                token_required,
                rx.badge("HF Token", color_scheme="orange", size="1"),
                rx.fragment(),
            ),
            spacing="2",
            align="center",
        ),
        on_click=FinetuneState.select_model(model_id, name),
        variant=rx.cond(is_sel, "solid", "soft"),
        color_scheme=rx.cond(is_sel, "blue", "gray"),
        size="2",
        radius="full",
    )


def _source_tab(source: str, label: str, icon: str) -> rx.Component:
    is_active = FinetuneState.model_source == source
    return rx.button(
        rx.hstack(rx.icon(icon, size=14), rx.text(label), spacing="2", align="center"),
        on_click=FinetuneState.set_model_source(source),
        variant=rx.cond(is_active, "solid", "soft"),
        color_scheme="blue",
        size="2",
    )


def _step1() -> rx.Component:
    return rx.vstack(
        _section_heading("Choose your model"),
        rx.text(
            "Pick a preset, paste any Hugging Face ID, or load a local file. "
            "We'll recommend the best training technique once we know your goal.",
            font_size="0.86rem",
            color=c("text_secondary"),
            margin_bottom="20px",
        ),
        # Current model selected — shown immediately on pick
        _selected_model_panel(),
        # Source tabs
        rx.hstack(
            _source_tab("hub", "HF Hub", "globe"),
            _source_tab("custom_string", "Any Model ID", "terminal"),
            _source_tab("local", "Local File", "folder-open"),
            spacing="2",
            margin_bottom="16px",
        ),
        # ── HF Hub tab ────────────────────────────────────────────
        rx.cond(
            FinetuneState.model_source == "hub",
            rx.vstack(
                rx.text(
                    rx.cond(
                        FinetuneState.intent_task_type == "vision",
                        "Vision-Language Models",
                        "Popular models",
                    ),
                    font_size="0.78rem",
                    font_weight="600",
                    color=c("text_muted"),
                    margin_bottom="8px",
                ),
                rx.cond(
                    FinetuneState.intent_task_type == "vision",
                    rx.flex(
                        *[_quick_chip(mid, n, t) for mid, n, t in _VLM_QUICK],
                        wrap="wrap",
                        gap="8px",
                    ),
                    rx.flex(
                        *[_quick_chip(mid, n, t) for mid, n, t in _QUICK_MODELS],
                        wrap="wrap",
                        gap="8px",
                    ),
                ),
                rx.box(height="12px"),
                _card(
                    rx.vstack(
                        _label("HF Token (required for gated models like Llama)"),
                        rx.input(
                            placeholder="hf_xxxxxxxxxxxxx",
                            type="password",
                            value=FinetuneState.hf_token,
                            on_change=FinetuneState.set_hf_token,
                            width="100%",
                        ),
                        spacing="1",
                    )
                ),
                width="100%",
                spacing="0",
            ),
            rx.fragment(),
        ),
        # ── Any Model ID tab ──────────────────────────────────────
        rx.cond(
            FinetuneState.model_source == "custom_string",
            _card(
                rx.vstack(
                    _label(
                        "Model ID or path (any string AutoModelForCausalLM.from_pretrained() accepts)"
                    ),
                    rx.hstack(
                        rx.input(
                            placeholder='e.g. "EleutherAI/gpt-j-6b" or "/local/path/to/model"',
                            value=FinetuneState.custom_model_str,
                            on_change=FinetuneState.set_custom_model_str,
                            flex="1",
                        ),
                        rx.button(
                            rx.cond(
                                FinetuneState.is_validating_model,
                                rx.hstack(
                                    rx.spinner(size="1"),
                                    rx.text("Checking..."),
                                    spacing="2",
                                ),
                                rx.text("Validate"),
                            ),
                            on_click=FinetuneState.validate_and_select_custom_model,
                            disabled=FinetuneState.is_validating_model,
                            color_scheme="blue",
                            size="2",
                        ),
                        spacing="2",
                    ),
                    rx.cond(
                        FinetuneState.model_url_error != "",
                        rx.callout(FinetuneState.model_url_error, color_scheme="red", size="1"),
                        rx.fragment(),
                    ),
                    _label("HF Token (for gated or private models)"),
                    rx.input(
                        placeholder="hf_xxxxxxxxxxxxx",
                        type="password",
                        value=FinetuneState.hf_token,
                        on_change=FinetuneState.set_hf_token,
                        width="100%",
                    ),
                    rx.text(
                        "Note: If you skip validation, any errors will appear when training starts.",
                        font_size="0.75rem",
                        color=c("text_muted"),
                    ),
                    spacing="2",
                )
            ),
            rx.fragment(),
        ),
        # ── Local File tab ────────────────────────────────────────
        rx.cond(
            FinetuneState.model_source == "local",
            _card(
                rx.vstack(
                    _label(
                        "Upload your model (.safetensors, .bin, .gguf, or .zip of model directory)"
                    ),
                    rx.upload(
                        rx.vstack(
                            rx.icon("upload", size=28, color=c("text_muted")),
                            rx.text("Drag & drop or click to upload", color=c("text_secondary")),
                            rx.text(
                                "Supports: .safetensors, .bin, .gguf, .zip",
                                font_size="0.75rem",
                                color=c("text_muted"),
                            ),
                            spacing="2",
                            align="center",
                        ),
                        id="model_upload",
                        border=f"2px dashed {c('border')}",
                        border_radius="10px",
                        padding="32px",
                        width="100%",
                        cursor="pointer",
                        on_drop=FinetuneState.handle_local_model_upload(
                            rx.upload_files(upload_id="model_upload")
                        ),
                    ),
                    rx.cond(
                        FinetuneState.local_model_path != "",
                        rx.callout(
                            rx.text(f"Loaded: {FinetuneState.local_model_path}"),
                            color_scheme="green",
                            size="1",
                        ),
                        rx.fragment(),
                    ),
                    spacing="2",
                )
            ),
            rx.fragment(),
        ),
        _nav_buttons(
            next_label="Next: Intent →",
            next_disabled=~FinetuneState.can_go_to_intent,
            show_back=False,
        ),
        spacing="0",
        width="100%",
        align_items="flex-start",
    )
