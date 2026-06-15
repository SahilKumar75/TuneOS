"""Fine-tune wizard — Step 1: Model selection."""

from __future__ import annotations

import reflex as rx

from app.components.finetune.shared import _card, _label, _nav_buttons, _section_heading
from app.state.finetune_state import FinetuneState
from app.styles import c

_QUICK_MODELS = [
    ("mistralai/Mistral-7B-v0.1", "Mistral 7B", False, "https://github.com/mistralai.png?size=32"),
    ("meta-llama/Meta-Llama-3-8B", "Llama 3 8B", True, "https://github.com/meta-llama.png?size=32"),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3 Mini", False, "https://github.com/microsoft.png?size=32"),
    ("google/gemma-2b", "Gemma 2B", False, "https://github.com/google.png?size=32"),
    ("EleutherAI/pythia-410m", "Pythia 410M", False, "https://github.com/EleutherAI.png?size=32"),
    ("bigcode/starcoder2-3b", "StarCoder2 3B", False, "https://github.com/bigcode-project.png?size=32"),
]

_VLM_QUICK = [
    ("llava-hf/llava-1.5-7b-hf", "LLaVA-1.5 7B", False, "https://github.com/haotian-liu.png?size=32"),
    ("Qwen/Qwen2-VL-2B-Instruct", "Qwen2-VL 2B", False, "https://github.com/QwenLM.png?size=32"),
]

_HF_LOGO = "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
_GH_LOGO = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"


def _stat_item(icon: str, value: rx.Component, icon_color: str = "") -> rx.Component:
    return rx.hstack(
        rx.icon(icon, size=13, color=icon_color or c("text_muted")),
        rx.text(value, font_size="0.78rem", color=c("text_secondary")),
        spacing="1",
        align="center",
    )


def _selected_model_panel() -> rx.Component:
    """Professional info card shown once a model is selected."""
    return rx.cond(
        FinetuneState.selected_model_id != "",
        rx.vstack(
            rx.hstack(
                rx.text(
                    "Current model",
                    font_size="0.72rem",
                    font_weight="700",
                    color=c("text_muted"),
                    text_transform="uppercase",
                    letter_spacing="0.06em",
                ),
                rx.spacer(),
                rx.cond(
                    FinetuneState.model_source == "hub",
                    rx.hstack(
                        rx.image(src=_HF_LOGO, width="13px", height="13px"),
                        rx.text("Hugging Face", font_size="0.7rem", color="#FF9D00", font_weight="600"),
                        spacing="1",
                        align="center",
                    ),
                    rx.fragment(),
                ),
                width="100%",
                align="center",
            ),
            rx.box(
                rx.vstack(
                    # ── Header: avatar + model id + org + spinner ──────────
                    rx.hstack(
                        rx.cond(
                            FinetuneState.selected_model_org_avatar != "",
                            rx.avatar(
                                src=FinetuneState.selected_model_org_avatar,
                                fallback=FinetuneState.selected_model_org_initial,
                                size="5",
                                radius="full",
                            ),
                            rx.avatar(
                                fallback=FinetuneState.selected_model_org_initial,
                                size="5",
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
                                word_break="break-all",
                            ),
                            rx.text(
                                FinetuneState.selected_model_org,
                                font_size="0.78rem",
                                color=c("text_muted"),
                                font_weight="500",
                            ),
                            spacing="0",
                            align_items="flex-start",
                        ),
                        rx.spacer(),
                        rx.cond(
                            FinetuneState.is_fetching_model_info,
                            rx.spinner(size="2", color="blue"),
                            rx.fragment(),
                        ),
                        spacing="3",
                        align="center",
                        width="100%",
                    ),
                    # ── Stats row: downloads · likes · context · pipeline ──
                    rx.cond(
                        ~FinetuneState.is_fetching_model_info,
                        rx.hstack(
                            rx.cond(
                                FinetuneState.model_downloads != "",
                                _stat_item("download", FinetuneState.model_downloads),
                                rx.fragment(),
                            ),
                            rx.cond(
                                FinetuneState.model_likes != "",
                                _stat_item("heart", FinetuneState.model_likes, "#E53E3E"),
                                rx.fragment(),
                            ),
                            rx.cond(
                                FinetuneState.model_context_window != "",
                                _stat_item("layers", FinetuneState.model_context_window),
                                rx.fragment(),
                            ),
                            rx.cond(
                                FinetuneState.model_pipeline != "",
                                rx.badge(FinetuneState.model_pipeline, color_scheme="blue", size="1"),
                                rx.fragment(),
                            ),
                            spacing="3",
                            wrap="wrap",
                            align="center",
                        ),
                        rx.fragment(),
                    ),
                    # ── Meta row: library · safetensors · arch · license ──
                    rx.cond(
                        ~FinetuneState.is_fetching_model_info,
                        rx.hstack(
                            rx.cond(
                                FinetuneState.model_library != "",
                                rx.badge(
                                    FinetuneState.model_library,
                                    color_scheme="purple",
                                    size="1",
                                    variant="soft",
                                ),
                                rx.fragment(),
                            ),
                            rx.cond(
                                FinetuneState.model_safetensors,
                                rx.badge("Safetensors", color_scheme="green", size="1", variant="soft"),
                                rx.fragment(),
                            ),
                            rx.cond(
                                FinetuneState.model_type_hf != "",
                                rx.badge(
                                    FinetuneState.model_type_hf,
                                    color_scheme="gray",
                                    size="1",
                                    variant="outline",
                                ),
                                rx.fragment(),
                            ),
                            rx.cond(
                                FinetuneState.model_license != "",
                                rx.text(
                                    FinetuneState.model_license,
                                    font_size="0.72rem",
                                    color=c("text_muted"),
                                ),
                                rx.fragment(),
                            ),
                            spacing="2",
                            wrap="wrap",
                            align="center",
                        ),
                        rx.fragment(),
                    ),
                    # ── Capability tags ───────────────────────────────────
                    rx.cond(
                        FinetuneState.model_hf_tags.length() > 0,
                        rx.hstack(
                            rx.foreach(
                                FinetuneState.model_hf_tags,
                                lambda tag: rx.badge(tag, color_scheme="gray", size="1", variant="soft"),
                            ),
                            wrap="wrap",
                            spacing="1",
                        ),
                        rx.fragment(),
                    ),
                    # ── Bio ───────────────────────────────────────────────
                    rx.cond(
                        FinetuneState.model_bio != "",
                        rx.text(
                            FinetuneState.model_bio,
                            font_size="0.8rem",
                            color=c("text_secondary"),
                            line_height="1.65",
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


def _quick_chip(model_id: str, name: str, token_required: bool, avatar_url: str) -> rx.Component:
    is_sel = FinetuneState.selected_model_id == model_id
    return rx.button(
        rx.hstack(
            rx.image(src=avatar_url, width="18px", height="18px", border_radius="50%"),
            rx.text(name, font_size="0.8rem"),
            rx.cond(
                token_required,
                rx.badge("Token", color_scheme="orange", size="1"),
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


def _search_result_row(result: dict) -> rx.Component:
    return rx.button(
        rx.hstack(
            rx.vstack(
                rx.text(
                    result["id"],
                    font_size="0.82rem",
                    font_weight="600",
                    color=c("text_primary"),
                    text_align="left",
                ),
                rx.text(result["pipeline"], font_size="0.72rem", color=c("text_muted"), text_align="left"),
                spacing="0",
                align_items="flex-start",
            ),
            rx.spacer(),
            rx.cond(
                result["downloads"] != "",
                rx.hstack(
                    rx.icon("download", size=11, color=c("text_muted")),
                    rx.text(result["downloads"], font_size="0.72rem", color=c("text_muted")),
                    spacing="1",
                    align="center",
                ),
                rx.fragment(),
            ),
            spacing="2",
            align="center",
            width="100%",
        ),
        on_click=FinetuneState.select_model(result["id"], result["id"]),
        variant="ghost",
        width="100%",
        padding_x="10px",
        padding_y="8px",
        justify="start",
    )


def _unified_card() -> rx.Component:
    """Single card: search box + popular chips + HF token."""
    return _card(
        rx.vstack(
            # ── Search row ──────────────────────────────────────────────
            rx.hstack(
                # Source segmented picker
                rx.hstack(
                    rx.button(
                        rx.hstack(
                            rx.image(src=_HF_LOGO, width="14px", height="14px"),
                            rx.text("HF Hub", font_size="0.8rem"),
                            spacing="1",
                            align="center",
                        ),
                        on_click=FinetuneState.set_model_search_source("hf"),
                        variant=rx.cond(
                            FinetuneState.model_search_source == "hf", "soft", "ghost"
                        ),
                        color_scheme=rx.cond(
                            FinetuneState.model_search_source == "hf", "orange", "gray"
                        ),
                        size="2",
                        border_right_radius="0",
                    ),
                    rx.button(
                        rx.hstack(
                            rx.image(
                                src=_GH_LOGO,
                                width="14px",
                                height="14px",
                                filter=rx.color_mode_cond(
                                    light="none",
                                    dark="brightness(0) invert(1)",
                                ),
                            ),
                            rx.text("GitHub", font_size="0.8rem"),
                            spacing="1",
                            align="center",
                        ),
                        on_click=FinetuneState.set_model_search_source("github"),
                        variant=rx.cond(
                            FinetuneState.model_search_source == "github", "soft", "ghost"
                        ),
                        color_scheme="gray",
                        size="2",
                        border_left_radius="0",
                    ),
                    spacing="0",
                    border="1px solid",
                    border_color=c("border"),
                    border_radius="8px",
                    overflow="hidden",
                ),
                # Live search input
                rx.input(
                    placeholder=rx.cond(
                        FinetuneState.model_search_source == "hf",
                        "Search Hugging Face models...",
                        "Search GitHub repositories...",
                    ),
                    value=FinetuneState.model_search_query,
                    on_change=FinetuneState.set_model_search_query,
                    flex="1",
                ),
                # Local import toggle
                rx.button(
                    rx.cond(
                        FinetuneState.show_local_upload,
                        rx.icon("x", size=14),
                        rx.icon("plus", size=14),
                    ),
                    on_click=FinetuneState.toggle_local_upload,
                    variant="soft",
                    color_scheme="gray",
                    size="2",
                    title="Import local model file",
                ),
                spacing="2",
                align="center",
                width="100%",
            ),
            # ── Search results ───────────────────────────────────────────
            rx.cond(
                FinetuneState.is_searching_models | (FinetuneState.model_search_results.length() > 0),
                rx.box(
                    rx.vstack(
                        rx.cond(
                            FinetuneState.is_searching_models,
                            rx.hstack(
                                rx.spinner(size="1"),
                                rx.text("Searching...", font_size="0.78rem", color=c("text_muted")),
                                spacing="2",
                                padding_x="10px",
                                padding_y="6px",
                            ),
                            rx.fragment(),
                        ),
                        rx.foreach(FinetuneState.model_search_results, _search_result_row),
                        spacing="0",
                        width="100%",
                    ),
                    border="1px solid",
                    border_color=c("border"),
                    border_radius="8px",
                    width="100%",
                    overflow="hidden",
                ),
                rx.fragment(),
            ),
            # ── Local upload area (toggle) ───────────────────────────────
            rx.cond(
                FinetuneState.show_local_upload,
                rx.vstack(
                    rx.upload(
                        rx.vstack(
                            rx.icon("upload", size=24, color=c("text_muted")),
                            rx.text(
                                "Drag & drop or click to upload",
                                font_size="0.82rem",
                                color=c("text_secondary"),
                            ),
                            rx.text(
                                ".safetensors · .bin · .gguf · .zip",
                                font_size="0.72rem",
                                color=c("text_muted"),
                            ),
                            spacing="2",
                            align="center",
                        ),
                        id="model_upload",
                        border=f"2px dashed {c('border')}",
                        border_radius="10px",
                        padding="24px",
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
                    width="100%",
                ),
                rx.fragment(),
            ),
            rx.divider(),
            # ── Popular model chips ──────────────────────────────────────
            rx.cond(
                FinetuneState.intent_task_type == "vision",
                _label("Vision-Language Models"),
                _label("Popular models"),
            ),
            rx.cond(
                FinetuneState.intent_task_type == "vision",
                rx.flex(
                    *[_quick_chip(mid, n, t, av) for mid, n, t, av in _VLM_QUICK],
                    wrap="wrap",
                    gap="8px",
                ),
                rx.flex(
                    *[_quick_chip(mid, n, t, av) for mid, n, t, av in _QUICK_MODELS],
                    wrap="wrap",
                    gap="8px",
                ),
            ),
            # ── HF Token (only for gated models) ────────────────────────
            rx.cond(
                FinetuneState.model_requires_token,
                rx.vstack(
                    rx.divider(),
                    _label("HF Token (this model is gated — token required)"),
                    rx.input(
                        placeholder="hf_xxxxxxxxxxxxx",
                        type="password",
                        value=FinetuneState.hf_token,
                        on_change=FinetuneState.set_hf_token,
                        width="100%",
                    ),
                    spacing="3",
                    width="100%",
                ),
                rx.fragment(),
            ),
            spacing="3",
            width="100%",
        )
    )


def _step1() -> rx.Component:
    return rx.vstack(
        _section_heading("Choose your model"),
        rx.text(
            "Search Hugging Face or GitHub, pick a preset, or upload a local file. "
            "We'll recommend the best training technique once we know your goal.",
            font_size="0.86rem",
            color=c("text_secondary"),
            margin_bottom="20px",
        ),
        _selected_model_panel(),
        _unified_card(),
        _nav_buttons(
            next_label="Next: Intent →",
            next_disabled=~FinetuneState.can_go_to_intent,
            show_back=False,
        ),
        spacing="0",
        width="100%",
        align_items="flex-start",
    )
