"""Fine-tune wizard — Step 1: Model selection and training technique."""

from __future__ import annotations

import reflex as rx

from app.state.finetune_state import FinetuneState
from app.styles import c
from app.components.finetune.shared import _card, _label, _section_heading, _nav_buttons

_MODELS = [
    {"id": "mistralai/Mistral-7B-v0.1", "name": "Mistral 7B", "size": "7B params",
     "notes": "Well-tested with QLoRA, great all-rounder", "token_required": False},
    {"id": "meta-llama/Meta-Llama-3-8B", "name": "Llama 3 8B", "size": "8B params",
     "notes": "Strong general-purpose model", "token_required": True},
    {"id": "microsoft/Phi-3-mini-4k-instruct", "name": "Phi-3 Mini", "size": "3.8B params",
     "notes": "Fast, runs on smaller GPUs", "token_required": False},
    {"id": "google/gemma-2b", "name": "Gemma 2B", "size": "2B params",
     "notes": "Good for low-VRAM environments", "token_required": False},
    {"id": "EleutherAI/pythia-410m", "name": "Pythia 410M", "size": "410M params",
     "notes": "Tiny model — great for testing pipelines fast", "token_required": False},
    {"id": "bigcode/starcoder2-3b", "name": "StarCoder2 3B", "size": "3B params",
     "notes": "Excellent for code generation tasks", "token_required": False},
]

_GGUF_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0", "F16"]


def _model_card(m: dict) -> rx.Component:
    is_selected = FinetuneState.selected_model_id == m["id"]
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(m["name"], font_size="0.92rem", font_weight="600", color=c("text_primary")),
                rx.cond(m["token_required"],
                        rx.badge("HF Token", color_scheme="orange", size="1"), rx.fragment()),
                justify="between", width="100%",
            ),
            rx.text(m["size"], font_size="0.78rem", color=c("text_secondary")),
            rx.text(m["notes"], font_size="0.78rem", color=c("text_muted")),
            spacing="1", align_items="flex-start", width="100%",
        ),
        background=rx.cond(is_selected, c("accent_soft"), c("bg_card")),
        border="2px solid",
        border_color=rx.cond(is_selected, c("accent"), c("border")),
        border_radius="10px", padding="14px", cursor="pointer", width="100%",
        on_click=FinetuneState.select_model(m["id"], m["name"]),
        _hover={"border_color": c("accent"), "background": c("accent_soft")},
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
        rx.text("Pick from common models, paste any Hugging Face ID, load a local file, "
                "or type any model string that Transformers accepts.",
                font_size="0.86rem", color=c("text_secondary"), margin_bottom="16px"),

        # Source switcher
        rx.hstack(
            _source_tab("hub", "HF Hub", "globe"),
            _source_tab("custom_string", "Any Model ID", "terminal"),
            _source_tab("local", "Local File", "folder-open"),
            spacing="2", margin_bottom="20px",
        ),

        # Hub tab
        rx.cond(
            FinetuneState.model_source == "hub",
            rx.vstack(
                rx.grid(*[_model_card(m) for m in _MODELS], columns="2", spacing="3", width="100%"),
                # HF token field for gated models
                rx.box(height="12px"),
                _card(
                    rx.vstack(
                        _label("HF Token (required for gated models like Llama)"),
                        rx.input(placeholder="hf_xxxxxxxxxxxxx", type="password",
                                 value=FinetuneState.hf_token,
                                 on_change=FinetuneState.set_hf_token,
                                 width="100%"),
                        spacing="1",
                    )
                ),
                width="100%", spacing="0",
            ),
            rx.fragment(),
        ),

        # Custom string tab
        rx.cond(
            FinetuneState.model_source == "custom_string",
            _card(
                rx.vstack(
                    _label("Model ID or path (any string AutoModelForCausalLM.from_pretrained() accepts)"),
                    rx.hstack(
                        rx.input(
                            placeholder='e.g. "EleutherAI/gpt-j-6b" or "/local/path/to/model"',
                            value=FinetuneState.custom_model_str,
                            on_change=FinetuneState.set_custom_model_str,
                            flex="1",
                        ),
                        rx.button(
                            rx.cond(FinetuneState.is_validating_model,
                                    rx.hstack(rx.spinner(size="1"), rx.text("Checking..."), spacing="2"),
                                    rx.text("Validate")),
                            on_click=FinetuneState.validate_and_select_custom_model,
                            disabled=FinetuneState.is_validating_model,
                            color_scheme="blue", size="2",
                        ),
                        spacing="2",
                    ),
                    rx.cond(
                        FinetuneState.model_url_error != "",
                        rx.callout(FinetuneState.model_url_error, color_scheme="red", size="1"),
                        rx.fragment(),
                    ),
                    rx.cond(
                        FinetuneState.selected_model_id != "",
                        rx.callout(
                            rx.hstack(rx.icon("check-circle", size=14),
                                      rx.text(f"Model ready: {FinetuneState.selected_model_id}"),
                                      spacing="2"),
                            color_scheme="green", size="1",
                        ),
                        rx.fragment(),
                    ),
                    _label("HF Token (for gated or private models)"),
                    rx.input(placeholder="hf_xxxxxxxxxxxxx", type="password",
                             value=FinetuneState.hf_token,
                             on_change=FinetuneState.set_hf_token, width="100%"),
                    rx.text("Note: If you skip validation, any errors will appear when training starts.",
                            font_size="0.75rem", color=c("text_muted")),
                    spacing="2",
                )
            ),
            rx.fragment(),
        ),

        # Local file tab
        rx.cond(
            FinetuneState.model_source == "local",
            _card(
                rx.vstack(
                    _label("Upload your model (.safetensors, .bin, .gguf, or .zip of model directory)"),
                    rx.upload(
                        rx.vstack(
                            rx.icon("upload", size=28, color=c("text_muted")),
                            rx.text("Drag & drop or click to upload", color=c("text_secondary")),
                            rx.text("Supports: .safetensors, .bin, .gguf, .zip",
                                    font_size="0.75rem", color=c("text_muted")),
                            spacing="2", align="center",
                        ),
                        id="model_upload",
                        border=f"2px dashed {c('border')}",
                        border_radius="10px", padding="32px",
                        width="100%", cursor="pointer",
                        on_drop=FinetuneState.handle_local_model_upload(rx.upload_files(upload_id="model_upload")),
                    ),
                    rx.cond(
                        FinetuneState.local_model_path != "",
                        rx.callout(
                            rx.text(f"Loaded: {FinetuneState.local_model_path}"),
                            color_scheme="green", size="1",
                        ),
                        rx.fragment(),
                    ),
                    spacing="2",
                )
            ),
            rx.fragment(),
        ),

        # Technique selector (always visible)
        rx.box(height="20px"),
        _section_heading("Training technique"),
        rx.flex(
            *[
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.text(label, font_size="0.88rem", font_weight="500",
                                    color=rx.cond(
                                        FinetuneState.selected_technique == tech,
                                        c("accent"), c("text_primary")
                                    )),
                            rx.cond(
                                FinetuneState.selected_technique == tech,
                                rx.icon("check-circle", size=14, color=c("accent")),
                                rx.fragment(),
                            ),
                            rx.cond(coming_soon, rx.badge("Soon", color_scheme="gray", size="1"),
                                    rx.fragment()),
                            spacing="2", align="center",
                        ),
                        rx.text(desc, font_size="0.76rem", color=c("text_muted")),
                        spacing="1", align_items="flex-start",
                    ),
                    background=rx.cond(
                        FinetuneState.selected_technique == tech,
                        c("accent_soft"), c("bg_input"),
                    ),
                    border="1px solid",
                    border_color=rx.cond(
                        FinetuneState.selected_technique == tech, c("accent"), c("border"),
                    ),
                    border_radius="8px", padding="12px 14px",
                    cursor=rx.cond(coming_soon, "not-allowed", "pointer"),
                    opacity=rx.cond(coming_soon, "0.5", "1"),
                    on_click=rx.cond(coming_soon, rx.prevent_default,
                                     FinetuneState.select_technique(tech)),
                    flex="1", min_width="140px",
                )
                for tech, label, desc, coming_soon in [
                    ("qlora", "QLoRA", "4-bit compressed. Runs on 12 GB+ GPU. Recommended.", False),
                    ("lora", "LoRA", "Float16. Needs ~16 GB GPU for 7B models.", False),
                    ("full", "Full Fine-tune", "All weights updated. Needs 80 GB+ GPU.", True),
                    ("dpo", "DPO", "Preference tuning for alignment.", True),
                ]
            ],
            wrap="wrap", gap="10px", width="100%",
        ),

        _nav_buttons(next_label="Next: Intent →",
                     next_disabled=~FinetuneState.can_go_to_intent,
                     show_back=False),
        spacing="0", width="100%", align_items="flex-start",
    )
