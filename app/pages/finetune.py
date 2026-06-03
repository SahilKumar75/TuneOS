"""TuneOS — Fine-tuning wizard (/finetune) — 7-step flow."""

from __future__ import annotations

import reflex as rx

from app.components.loss_chart import loss_chart
from app.state.experiment_state import ExperimentState, ModelRegistryState
from app.state.finetune_state import FinetuneState
from app.styles import c

# ── Preset models ─────────────────────────────────────────────────
_MODELS = [
    {
        "id": "mistralai/Mistral-7B-v0.1",
        "name": "Mistral 7B",
        "size": "7B params",
        "notes": "Well-tested with QLoRA, great all-rounder",
        "token_required": False,
    },
    {
        "id": "meta-llama/Meta-Llama-3-8B",
        "name": "Llama 3 8B",
        "size": "8B params",
        "notes": "Strong general-purpose model",
        "token_required": True,
    },
    {
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "name": "Phi-3 Mini",
        "size": "3.8B params",
        "notes": "Fast, runs on smaller GPUs",
        "token_required": False,
    },
    {
        "id": "google/gemma-2b",
        "name": "Gemma 2B",
        "size": "2B params",
        "notes": "Good for low-VRAM environments",
        "token_required": False,
    },
    {
        "id": "EleutherAI/pythia-410m",
        "name": "Pythia 410M",
        "size": "410M params",
        "notes": "Tiny model — great for testing pipelines fast",
        "token_required": False,
    },
    {
        "id": "bigcode/starcoder2-3b",
        "name": "StarCoder2 3B",
        "size": "3B params",
        "notes": "Excellent for code generation tasks",
        "token_required": False,
    },
]

_STEP_LABELS = ["Model", "Intent", "Data", "Configure", "Train", "Results", "Deploy"]

_INTENT_IDEAS = [
    "Health chatbot for diabetes patients",
    "Python code review assistant",
    "Customer support for SaaS products",
    "Legal document summarizer",
    "Recipe recommendation assistant",
    "Scientific paper Q&A bot",
    "SQL query generator",
    "Children's education tutor",
]

_LR_PRESETS = [
    ("1e-4", "Slow & careful"),
    ("2e-4", "Balanced (recommended)"),
    ("5e-4", "Fast learning"),
]

_GGUF_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0", "F16"]


# ── Shared helpers ────────────────────────────────────────────────
def _card(
    *children,
    padding: str = "20px",
    width: str = "100%",
    background: str | None = None,
    **props,
) -> rx.Component:
    return rx.box(
        *children,
        background=c("bg_card") if background is None else background,
        border="1px solid",
        border_color=c("border"),
        border_radius="12px",
        padding=padding,
        width=width,
        **props,
    )


def _label(text: str) -> rx.Component:
    return rx.text(
        text, font_size="0.8rem", font_weight="500", color=c("text_secondary"), margin_bottom="6px"
    )


def _section_heading(text: str) -> rx.Component:
    return rx.text(
        text, font_size="1.05rem", font_weight="600", color=c("text_primary"), margin_bottom="16px"
    )


def _nav_buttons(
    back_label: str = "← Back",
    next_label: str = "Next →",
    next_disabled: bool = False,
    next_event=None,
    show_back: bool = True,
) -> rx.Component:
    return rx.hstack(
        rx.button(
            back_label,
            on_click=FinetuneState.prev_step,
            variant="soft",
            color_scheme="gray",
            size="2",
        )
        if show_back
        else rx.fragment(),
        rx.spacer(),
        rx.button(
            next_label,
            on_click=next_event or FinetuneState.next_step,
            disabled=next_disabled,
            size="3",
            color_scheme="blue",
        ),
        width="100%",
        padding_top="16px",
    )


def _badge_status(status: str) -> rx.Component:
    color = rx.match(
        status,
        ("running", "blue"),
        ("done", "green"),
        ("failed", "red"),
        "gray",
    )
    return rx.badge(status.upper(), color_scheme=color, size="2")


# ── Progress bar ──────────────────────────────────────────────────
def _step_dot(index: int) -> rx.Component:
    step_num = index + 1
    is_done = FinetuneState.current_step > step_num
    is_active = FinetuneState.current_step == step_num
    return rx.vstack(
        rx.box(
            rx.cond(
                is_done,
                rx.icon("check", size=12, color="white"),
                rx.text(
                    str(step_num),
                    font_size="0.72rem",
                    font_weight="600",
                    color=rx.cond(is_active, "white", c("text_muted")),
                ),
            ),
            width="26px",
            height="26px",
            border_radius="50%",
            background=rx.cond(
                is_done, c("success"), rx.cond(is_active, c("accent"), c("bg_input"))
            ),
            border="2px solid",
            border_color=rx.cond(is_active | is_done, c("accent"), c("border")),
            display="flex",
            align_items="center",
            justify_content="center",
        ),
        rx.text(
            _STEP_LABELS[index],
            font_size="0.68rem",
            color=rx.cond(is_active, c("text_primary"), c("text_muted")),
            font_weight=rx.cond(is_active, "500", "400"),
        ),
        spacing="1",
        align="center",
    )


def _progress_bar() -> rx.Component:
    return rx.hstack(
        *[
            rx.hstack(
                _step_dot(i),
                rx.box(
                    height="2px",
                    flex="1",
                    background=rx.cond(
                        FinetuneState.current_step > i + 1, c("accent"), c("border")
                    ),
                    min_width="20px",
                )
                if i < len(_STEP_LABELS) - 1
                else rx.fragment(),
                spacing="0",
                align="center",
                flex="1" if i < len(_STEP_LABELS) - 1 else "0",
            )
            for i in range(len(_STEP_LABELS))
        ],
        width="100%",
        align="center",
        justify="center",
        margin_bottom="32px",
    )


# ── Step 1: Model Source ──────────────────────────────────────────
def _model_card(m: dict) -> rx.Component:
    is_selected = FinetuneState.selected_model_id == m["id"]
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(m["name"], font_size="0.92rem", font_weight="600", color=c("text_primary")),
                rx.cond(
                    m["token_required"],
                    rx.badge("HF Token", color_scheme="orange", size="1"),
                    rx.fragment(),
                ),
                justify="between",
                width="100%",
            ),
            rx.text(m["size"], font_size="0.78rem", color=c("text_secondary")),
            rx.text(m["notes"], font_size="0.78rem", color=c("text_muted")),
            spacing="1",
            align_items="flex-start",
            width="100%",
        ),
        background=rx.cond(is_selected, c("accent_soft"), c("bg_card")),
        border="2px solid",
        border_color=rx.cond(is_selected, c("accent"), c("border")),
        border_radius="10px",
        padding="14px",
        cursor="pointer",
        width="100%",
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


def _technique_selector() -> rx.Component:
    return rx.flex(
        *[
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.text(
                            label,
                            font_size="0.88rem",
                            font_weight="500",
                            color=rx.cond(
                                FinetuneState.selected_technique == tech,
                                c("accent"),
                                c("text_primary"),
                            ),
                        ),
                        rx.cond(
                            FinetuneState.selected_technique == tech,
                            rx.icon("circle-check", size=14, color=c("accent")),
                            rx.fragment(),
                        ),
                        rx.cond(
                            coming_soon,
                            rx.badge("Soon", color_scheme="gray", size="1"),
                            rx.fragment(),
                        ),
                        spacing="2",
                        align="center",
                    ),
                    rx.text(desc, font_size="0.76rem", color=c("text_muted")),
                    spacing="1",
                    align_items="flex-start",
                ),
                background=rx.cond(
                    FinetuneState.selected_technique == tech,
                    c("accent_soft"),
                    c("bg_input"),
                ),
                border="1px solid",
                border_color=rx.cond(
                    FinetuneState.selected_technique == tech,
                    c("accent"),
                    c("border"),
                ),
                border_radius="8px",
                padding="12px 14px",
                cursor=rx.cond(coming_soon, "not-allowed", "pointer"),
                opacity=rx.cond(coming_soon, "0.5", "1"),
                on_click=rx.cond(
                    coming_soon, rx.prevent_default, FinetuneState.select_technique(tech)
                ),
                flex="1",
                min_width="140px",
            )
            for tech, label, desc, coming_soon in [
                ("qlora", "QLoRA", "4-bit compressed. Runs on 12 GB+ GPU. Recommended.", False),
                ("lora", "LoRA", "Float16. Needs ~16 GB GPU for 7B models.", False),
                ("full", "Full Fine-tune", "All weights updated. Needs 80 GB+ GPU.", True),
                ("dpo", "DPO", "Preference tuning for alignment.", True),
            ]
        ],
        wrap="wrap",
        gap="10px",
        width="100%",
    )


def _step1_confirm() -> rx.Component:
    """Confirmation view — shown when a model has been pre-filled or selected."""
    return rx.vstack(
        # ── Model preview card ─────────────────────────────────────
        _card(
            rx.vstack(
                rx.hstack(
                    rx.box(
                        rx.icon("bot", size=26, color=c("accent")),
                        background=c("accent_soft"),
                        border_radius="10px",
                        padding="10px",
                        display="flex",
                        align_items="center",
                        justify_content="center",
                        flex_shrink="0",
                    ),
                    rx.vstack(
                        rx.hstack(
                            rx.text(
                                FinetuneState.effective_model_name,
                                font_size="1.05rem",
                                font_weight="700",
                                color=c("text_primary"),
                            ),
                            rx.badge(
                                FinetuneState.selected_model_source_label,
                                color_scheme="blue",
                                size="1",
                            ),
                            spacing="2",
                            align="center",
                            flex_wrap="wrap",
                        ),
                        rx.text(
                            FinetuneState.effective_model_id,
                            font_size="0.78rem",
                            color=c("text_muted"),
                            font_family="monospace",
                        ),
                        rx.cond(
                            FinetuneState.selected_model_size != "",
                            rx.hstack(
                                rx.text(
                                    FinetuneState.selected_model_size,
                                    font_size="0.78rem",
                                    color=c("text_secondary"),
                                ),
                                rx.text("·", color=c("text_muted"), font_size="0.78rem"),
                                rx.text(
                                    FinetuneState.selected_model_notes,
                                    font_size="0.78rem",
                                    color=c("text_secondary"),
                                ),
                                spacing="1",
                                flex_wrap="wrap",
                            ),
                            rx.fragment(),
                        ),
                        spacing="1",
                        align_items="flex-start",
                        flex="1",
                    ),
                    spacing="3",
                    align="start",
                    width="100%",
                ),
                spacing="0",
            )
        ),
        # ── Change model card ──────────────────────────────────────
        _card(
            rx.vstack(
                _label("Change model"),
                rx.select.root(
                    rx.select.trigger(
                        placeholder="Select a different preset…",
                        width="100%",
                    ),
                    rx.select.content(
                        *[
                            rx.select.item(m["name"], value=m["id"])
                            for m in _MODELS
                        ],
                    ),
                    value=rx.cond(
                        FinetuneState.model_source == "hub",
                        FinetuneState.selected_model_id,
                        "",
                    ),
                    on_change=FinetuneState.select_preset,
                    width="100%",
                ),
                rx.text(
                    "Or enter any model ID",
                    font_size="0.78rem",
                    font_weight="500",
                    color=c("text_secondary"),
                    margin_top="8px",
                ),
                rx.hstack(
                    rx.input(
                        placeholder='e.g. "EleutherAI/gpt-j-6b"',
                        value=rx.cond(
                            FinetuneState.model_source == "custom_string",
                            FinetuneState.custom_model_str,
                            "",
                        ),
                        on_change=FinetuneState.set_custom_confirm_input,
                        flex="1",
                    ),
                    rx.button(
                        rx.cond(
                            FinetuneState.is_validating_model,
                            rx.hstack(
                                rx.spinner(size="1"), rx.text("Checking…"), spacing="2"
                            ),
                            rx.text("Verify"),
                        ),
                        on_click=FinetuneState.validate_and_select_custom_model,
                        disabled=FinetuneState.is_validating_model
                        | (FinetuneState.custom_model_str == ""),
                        variant="soft",
                        color_scheme="gray",
                        size="2",
                    ),
                    spacing="2",
                ),
                rx.cond(
                    FinetuneState.model_url_error != "",
                    rx.callout(FinetuneState.model_url_error, color_scheme="red", size="1"),
                    rx.fragment(),
                ),
                spacing="3",
            )
        ),
        # ── Technique selector ─────────────────────────────────────
        rx.box(height="4px"),
        _section_heading("Training technique"),
        _technique_selector(),
        _nav_buttons(
            next_label="Confirm & continue →",
            next_disabled=~FinetuneState.can_go_to_intent,
            show_back=False,
        ),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


def _step1_picker() -> rx.Component:
    """Full model picker — shown when no model has been selected yet."""
    return rx.vstack(
        _section_heading("Choose your model"),
        rx.text(
            "Pick from common models, paste any Hugging Face ID, load a local file, "
            "or type any model string that Transformers accepts.",
            font_size="0.86rem",
            color=c("text_secondary"),
            margin_bottom="16px",
        ),
        # Source switcher
        rx.hstack(
            _source_tab("hub", "HF Hub", "globe"),
            _source_tab("custom_string", "Any Model ID", "terminal"),
            _source_tab("local", "Local File", "folder-open"),
            spacing="2",
            margin_bottom="20px",
        ),
        # Hub tab
        rx.cond(
            FinetuneState.model_source == "hub",
            rx.vstack(
                rx.grid(*[_model_card(m) for m in _MODELS], columns="2", spacing="3", width="100%"),
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
        # Custom string tab
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
                                    rx.spinner(size="1"), rx.text("Checking…"), spacing="2"
                                ),
                                rx.text("Verify"),
                            ),
                            on_click=FinetuneState.validate_and_select_custom_model,
                            disabled=FinetuneState.is_validating_model
                            | (FinetuneState.custom_model_str == ""),
                            variant="soft",
                            color_scheme="gray",
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
                        "Tip: click Verify to check the model is accessible before training.",
                        font_size="0.75rem",
                        color=c("text_muted"),
                    ),
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
        # Technique selector
        rx.box(height="20px"),
        _section_heading("Training technique"),
        _technique_selector(),
        _nav_buttons(
            next_label="Next: Intent →",
            next_disabled=~FinetuneState.can_go_to_intent,
            show_back=False,
        ),
        spacing="0",
        width="100%",
        align_items="flex-start",
    )


def _step1() -> rx.Component:
    return rx.cond(
        FinetuneState.step1_show_picker,
        _step1_picker(),
        _step1_confirm(),
    )


# ── Step 2: Intent ────────────────────────────────────────────────
def _step2() -> rx.Component:
    return rx.vstack(
        _section_heading("What are you building?"),
        rx.text(
            "Describe your use-case in plain English. TuneOS uses this to generate starter data, "
            "guide the training dashboard, and pre-fill the system prompt for testing.",
            font_size="0.86rem",
            color=c("text_secondary"),
            margin_bottom="16px",
        ),
        _card(
            rx.vstack(
                _label("Your goal (1–3 sentences)"),
                rx.text_area(
                    placeholder="e.g. A health chatbot that answers questions for people with Type 2 diabetes in simple language.",
                    value=FinetuneState.user_intent,
                    on_change=FinetuneState.set_user_intent,
                    rows="4",
                    width="100%",
                    resize="vertical",
                ),
                rx.text("Quick ideas:", font_size="0.76rem", color=c("text_muted")),
                rx.flex(
                    *[
                        rx.badge(
                            idea,
                            cursor="pointer",
                            on_click=FinetuneState.set_user_intent(idea),
                            color_scheme="blue",
                            variant="soft",
                            size="1",
                        )
                        for idea in _INTENT_IDEAS
                    ],
                    wrap="wrap",
                    gap="6px",
                ),
                spacing="3",
            )
        ),
        _nav_buttons(
            next_label="Next: Add Data →",
            next_disabled=FinetuneState.user_intent == "",
        ),
        spacing="0",
        width="100%",
        align_items="flex-start",
    )


# ── Step 3: Data ──────────────────────────────────────────────────
def _preview_table(rows: list, label: str = "Preview") -> rx.Component:
    return rx.vstack(
        rx.text(label, font_size="0.78rem", font_weight="500", color=c("text_muted")),
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Instruction"),
                    rx.table.column_header_cell("Output"),
                )
            ),
            rx.table.body(
                rx.foreach(
                    rows,
                    lambda row: rx.table.row(
                        rx.table.cell(
                            rx.text(
                                row.instruction,
                                font_size="0.78rem",
                                overflow="hidden",
                                text_overflow="ellipsis",
                                white_space="nowrap",
                                max_width="300px",
                            )
                        ),
                        rx.table.cell(
                            rx.text(
                                row.output,
                                font_size="0.78rem",
                                overflow="hidden",
                                text_overflow="ellipsis",
                                white_space="nowrap",
                                max_width="260px",
                            )
                        ),
                    ),
                )
            ),
            width="100%",
            variant="surface",
            size="1",
        ),
        width="100%",
        spacing="2",
    )


def _data_mode_btn(mode: str, label: str, icon: str) -> rx.Component:
    is_active = FinetuneState.data_source == mode
    return rx.button(
        rx.hstack(rx.icon(icon, size=14), rx.text(label), spacing="2", align="center"),
        on_click=FinetuneState.set_data_source(mode),
        variant=rx.cond(is_active, "solid", "soft"),
        color_scheme="blue",
        size="2",
    )


def _upload_panel() -> rx.Component:
    return _card(
        rx.vstack(
            _label("Upload CSV, JSONL, or JSON array — any two columns work, you can remap them"),
            rx.upload(
                rx.vstack(
                    rx.icon("upload", size=28, color=c("text_muted")),
                    rx.text("Drag & drop or click to select a file", color=c("text_secondary")),
                    rx.text(".csv · .jsonl · .json", font_size="0.75rem", color=c("text_muted")),
                    spacing="2",
                    align="center",
                ),
                id="dataset_upload",
                border=f"2px dashed {c('border')}",
                border_radius="10px",
                padding="24px",
                width="100%",
                cursor="pointer",
                on_drop=FinetuneState.handle_dataset_upload(
                    rx.upload_files(upload_id="dataset_upload")
                ),
            ),
            rx.cond(
                FinetuneState.is_uploading,
                rx.hstack(
                    rx.spinner(size="2"), rx.text("Uploading...", font_size="0.84rem"), spacing="2"
                ),
                rx.fragment(),
            ),
            rx.cond(
                FinetuneState.dataset_error != "",
                rx.callout(FinetuneState.dataset_error, color_scheme="red", size="1"),
                rx.fragment(),
            ),
            rx.cond(
                FinetuneState.dataset_preview.length() > 0,
                _preview_table(FinetuneState.dataset_preview, "File preview (first 5 rows)"),
                rx.fragment(),
            ),
            spacing="3",
        )
    )


def _hub_dataset_panel() -> rx.Component:
    return _card(
        rx.vstack(
            rx.cond(
                FinetuneState.hub_dataset_id != "",
                rx.vstack(
                    rx.hstack(
                        rx.icon("database", size=16, color=c("accent")),
                        rx.text(
                            FinetuneState.hub_dataset_id, font_weight="500", color=c("text_primary")
                        ),
                        spacing="2",
                        align="center",
                    ),
                    rx.hstack(
                        rx.vstack(
                            _label("Instruction column"),
                            rx.input(
                                value=FinetuneState.hub_dataset_instruction_col,
                                on_change=FinetuneState.set_hub_instruction_col,
                                size="2",
                                width="180px",
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Output column"),
                            rx.input(
                                value=FinetuneState.hub_dataset_output_col,
                                on_change=FinetuneState.set_hub_output_col,
                                size="2",
                                width="180px",
                            ),
                            spacing="1",
                        ),
                        rx.button(
                            "Load preview",
                            size="2",
                            color_scheme="blue",
                            variant="soft",
                            on_click=FinetuneState.load_hub_dataset_preview,
                            align_self="flex-end",
                        ),
                        spacing="4",
                        wrap="wrap",
                    ),
                    rx.cond(
                        FinetuneState.is_loading_hub_preview,
                        rx.hstack(
                            rx.spinner(size="2"),
                            rx.text("Loading...", font_size="0.84rem"),
                            spacing="2",
                        ),
                        rx.fragment(),
                    ),
                    rx.cond(
                        FinetuneState.hub_preview_error != "",
                        rx.callout(FinetuneState.hub_preview_error, color_scheme="red", size="1"),
                        rx.fragment(),
                    ),
                    rx.cond(
                        FinetuneState.hub_dataset_preview.length() > 0,
                        _preview_table(FinetuneState.hub_dataset_preview),
                        rx.fragment(),
                    ),
                    spacing="3",
                    width="100%",
                ),
                rx.vstack(
                    rx.text("No dataset selected yet.", color=c("text_muted"), font_size="0.86rem"),
                    rx.text(
                        'Go to the Datasets tab and click "Use in Fine-tune" on any dataset.',
                        color=c("text_muted"),
                        font_size="0.82rem",
                    ),
                    rx.button(
                        "Browse Datasets →",
                        on_click=rx.redirect("/datasets"),
                        color_scheme="blue",
                        variant="soft",
                        size="2",
                    ),
                    spacing="3",
                ),
            ),
            spacing="2",
        )
    )


def _generate_panel() -> rx.Component:
    return _card(
        rx.vstack(
            rx.hstack(
                rx.icon("sparkles", size=16, color=c("accent")),
                rx.text(
                    "Generate synthetic training data", font_weight="500", color=c("text_primary")
                ),
                spacing="2",
                align="center",
            ),
            rx.text(
                "TuneOS will create instruction/output pairs tailored to your stated goal using "
                "the Self-Instruct method (the same approach used to create Stanford Alpaca).",
                font_size="0.82rem",
                color=c("text_secondary"),
            ),
            rx.cond(
                FinetuneState.user_intent != "",
                rx.box(
                    rx.text(
                        f'Goal: "{FinetuneState.user_intent}"',
                        font_size="0.82rem",
                        color=c("text_muted"),
                        font_style="italic",
                    ),
                    background=c("bg_input"),
                    border_radius="6px",
                    padding="8px 12px",
                ),
                rx.fragment(),
            ),
            rx.hstack(
                rx.vstack(
                    _label("Method"),
                    rx.select.root(
                        rx.select.trigger(width="200px"),
                        rx.select.content(
                            rx.select.item("Self-Instruct (recommended)", value="self_instruct"),
                            rx.select.item("Few-Shot Expansion", value="few_shot"),
                            rx.select.item("Template-Based (offline)", value="template"),
                        ),
                        value=FinetuneState.generation_method,
                        on_change=FinetuneState.set_generation_method,
                    ),
                    spacing="1",
                ),
                rx.vstack(
                    _label("Number of examples"),
                    rx.select.root(
                        rx.select.trigger(width="120px"),
                        rx.select.content(
                            rx.select.item("50", value="50"),
                            rx.select.item("100", value="100"),
                            rx.select.item("250", value="250"),
                            rx.select.item("500", value="500"),
                        ),
                        value=FinetuneState.generation_n.to_string(),
                        on_change=FinetuneState.set_generation_n,
                    ),
                    spacing="1",
                ),
                spacing="4",
                wrap="wrap",
            ),
            rx.button(
                rx.cond(
                    FinetuneState.is_generating,
                    rx.hstack(rx.spinner(size="2"), rx.text("Generating..."), spacing="2"),
                    rx.hstack(
                        rx.icon("sparkles", size=14), rx.text("Generate examples"), spacing="2"
                    ),
                ),
                on_click=FinetuneState.generate_starter_dataset,
                disabled=FinetuneState.is_generating,
                color_scheme="blue",
                size="3",
            ),
            rx.cond(
                FinetuneState.generation_status != "",
                rx.text(
                    FinetuneState.generation_status, font_size="0.82rem", color=c("text_secondary")
                ),
                rx.fragment(),
            ),
            rx.cond(
                FinetuneState.generated_samples.length() > 0,
                _preview_table(FinetuneState.generated_samples, "Generated examples preview"),
                rx.fragment(),
            ),
            spacing="3",
        )
    )


def _step3() -> rx.Component:
    return rx.vstack(
        _section_heading("Add your training data"),
        rx.hstack(
            _data_mode_btn("upload", "Upload a file", "upload"),
            _data_mode_btn("hub_dataset", "HF Hub dataset", "database"),
            _data_mode_btn("generate", "Generate with AI", "sparkles"),
            spacing="2",
            margin_bottom="16px",
        ),
        rx.match(
            FinetuneState.data_source,
            ("upload", _upload_panel()),
            ("hub_dataset", _hub_dataset_panel()),
            ("generate", _generate_panel()),
            _upload_panel(),
        ),
        _nav_buttons(
            next_label="Next: Configure →",
            next_disabled=~FinetuneState.can_go_to_configure,
        ),
        spacing="0",
        width="100%",
        align_items="flex-start",
    )


# ── Step 4: Configure ─────────────────────────────────────────────
def _step4() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            _section_heading("Training configuration"),
            rx.spacer(),
            rx.hstack(
                rx.text("Simple", font_size="0.82rem", color=c("text_secondary")),
                rx.switch(
                    checked=FinetuneState.ui_mode == "advanced",
                    on_change=FinetuneState.toggle_ui_mode,
                    size="2",
                ),
                rx.text("Advanced", font_size="0.82rem", color=c("text_secondary")),
                spacing="2",
                align="center",
            ),
        ),
        # Simple mode
        _card(
            rx.vstack(
                rx.grid(
                    rx.vstack(
                        _label("Epochs"),
                        rx.input(
                            value=FinetuneState.epochs.to_string(),
                            on_change=FinetuneState.set_epochs,
                            type="number",
                            width="100%",
                        ),
                        rx.text(
                            "One full pass through your dataset",
                            font_size="0.72rem",
                            color=c("text_muted"),
                        ),
                        spacing="1",
                    ),
                    rx.vstack(
                        _label("Learning rate"),
                        rx.select.root(
                            rx.select.trigger(width="100%"),
                            rx.select.content(
                                *[
                                    rx.select.item(f"{lr} — {desc}", value=lr)
                                    for lr, desc in _LR_PRESETS
                                ],
                            ),
                            value=FinetuneState.learning_rate,
                            on_change=FinetuneState.set_learning_rate,
                        ),
                        spacing="1",
                    ),
                    rx.vstack(
                        _label("Technique"),
                        rx.text(
                            FinetuneState.technique_label,
                            font_size="0.88rem",
                            font_weight="500",
                            color=c("accent"),
                        ),
                        rx.text("Change in Step 1", font_size="0.72rem", color=c("text_muted")),
                        spacing="1",
                    ),
                    columns="3",
                    spacing="4",
                    width="100%",
                ),
                spacing="0",
            )
        ),
        # Advanced mode
        rx.cond(
            FinetuneState.ui_mode == "advanced",
            _card(
                rx.vstack(
                    rx.text(
                        "Advanced hyperparameters",
                        font_size="0.88rem",
                        font_weight="600",
                        color=c("text_primary"),
                        margin_bottom="12px",
                    ),
                    rx.grid(
                        rx.vstack(
                            _label("LoRA rank (r)"),
                            rx.slider(
                                min=4,
                                max=128,
                                step=4,
                                default_value=[FinetuneState.lora_r],
                                on_value_commit=FinetuneState.set_lora_r,
                            ),
                            rx.text(
                                FinetuneState.lora_r, font_size="0.82rem", color=c("text_secondary")
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("LoRA alpha"),
                            rx.input(
                                value=FinetuneState.lora_alpha.to_string(),
                                on_change=FinetuneState.set_lora_alpha,
                                type="number",
                                width="100%",
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("LoRA dropout"),
                            rx.slider(
                                min=0.0,
                                max=0.3,
                                step=0.01,
                                default_value=[FinetuneState.lora_dropout],
                                on_value_commit=FinetuneState.set_lora_dropout,
                            ),
                            rx.text(
                                FinetuneState.lora_dropout,
                                font_size="0.82rem",
                                color=c("text_secondary"),
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Batch size"),
                            rx.select.root(
                                rx.select.trigger(width="100%"),
                                rx.select.content(
                                    *[
                                        rx.select.item(str(v), value=str(v))
                                        for v in [1, 2, 4, 8, 16]
                                    ],
                                ),
                                value=FinetuneState.batch_size.to_string(),
                                on_change=FinetuneState.set_batch_size,
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Max sequence length"),
                            rx.select.root(
                                rx.select.trigger(width="100%"),
                                rx.select.content(
                                    *[
                                        rx.select.item(str(v), value=str(v))
                                        for v in [128, 256, 512, 1024, 2048]
                                    ],
                                ),
                                value=FinetuneState.max_seq_length.to_string(),
                                on_change=FinetuneState.set_max_seq_length,
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Gradient accumulation steps"),
                            rx.select.root(
                                rx.select.trigger(width="100%"),
                                rx.select.content(
                                    *[
                                        rx.select.item(str(v), value=str(v))
                                        for v in [1, 2, 4, 8, 16]
                                    ],
                                ),
                                value=FinetuneState.gradient_accumulation_steps.to_string(),
                                on_change=FinetuneState.set_gradient_accumulation_steps,
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("LR scheduler"),
                            rx.select.root(
                                rx.select.trigger(width="100%"),
                                rx.select.content(
                                    *[
                                        rx.select.item(v, value=v)
                                        for v in [
                                            "cosine",
                                            "linear",
                                            "constant",
                                            "cosine_with_restarts",
                                        ]
                                    ],
                                ),
                                value=FinetuneState.lr_scheduler,
                                on_change=FinetuneState.set_lr_scheduler,
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("BF16 mode (A100/H100 only)"),
                            rx.switch(
                                checked=FinetuneState.bf16,
                                on_change=FinetuneState.set_bf16,
                                size="2",
                            ),
                            rx.text(
                                "Better precision than FP16 on Ampere+ GPUs",
                                font_size="0.72rem",
                                color=c("text_muted"),
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Experiment name"),
                            rx.input(
                                placeholder="my-run-1",
                                value=FinetuneState.experiment_name,
                                on_change=FinetuneState.set_experiment_name,
                                width="100%",
                            ),
                            spacing="1",
                        ),
                        columns="3",
                        spacing="4",
                        width="100%",
                    ),
                    spacing="0",
                )
            ),
            rx.fragment(),
        ),
        # Run summary
        _card(
            rx.vstack(
                rx.text(
                    "Run summary",
                    font_size="0.82rem",
                    font_weight="600",
                    color=c("text_secondary"),
                    margin_bottom="8px",
                ),
                rx.grid(
                    rx.vstack(
                        rx.text("Model", font_size="0.72rem", color=c("text_muted")),
                        rx.text(
                            FinetuneState.effective_model_name,
                            font_size="0.84rem",
                            font_weight="500",
                            color=c("text_primary"),
                        ),
                        spacing="0",
                    ),
                    rx.vstack(
                        rx.text("Dataset", font_size="0.72rem", color=c("text_muted")),
                        rx.text(
                            FinetuneState.dataset_name,
                            font_size="0.84rem",
                            font_weight="500",
                            color=c("text_primary"),
                        ),
                        spacing="0",
                    ),
                    rx.vstack(
                        rx.text("Technique", font_size="0.72rem", color=c("text_muted")),
                        rx.text(
                            FinetuneState.technique_label,
                            font_size="0.84rem",
                            font_weight="500",
                            color=c("text_primary"),
                        ),
                        spacing="0",
                    ),
                    rx.vstack(
                        rx.text("Training", font_size="0.72rem", color=c("text_muted")),
                        rx.text(
                            FinetuneState.epochs.to_string()
                            + " epochs · lr="
                            + FinetuneState.learning_rate
                            + " · batch="
                            + FinetuneState.batch_size.to_string(),
                            font_size="0.82rem",
                            font_weight="500",
                            color=c("text_primary"),
                        ),
                        spacing="0",
                    ),
                    columns="2",
                    spacing="4",
                    width="100%",
                ),
                spacing="0",
            ),
            background=c("bg_input"),
        ),
        _nav_buttons(
            next_label="Start Training →",
            next_disabled=~FinetuneState.can_start_training,
            next_event=FinetuneState.start_training,
        ),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


# ── Step 5: Training Dashboard ────────────────────────────────────
def _metric_tile(label: str, value) -> rx.Component:
    return _card(
        rx.vstack(
            rx.text(
                label,
                font_size="0.68rem",
                font_weight="500",
                color=c("text_muted"),
                letter_spacing="0.05em",
            ),
            rx.text(value, font_size="1.5rem", font_weight="700", color=c("text_primary")),
            spacing="1",
        ),
        padding="14px",
    )


def _epoch_log_row(entry) -> rx.Component:
    return rx.hstack(
        rx.text(
            "Epoch " + entry.epoch.to_string(),
            font_size="0.78rem",
            font_weight="500",
            color=c("text_primary"),
            width="60px",
        ),
        rx.text(
            "loss " + entry.loss_start.to_string() + " → " + entry.loss_end.to_string(),
            font_size="0.78rem",
            color=c("text_secondary"),
            flex="1",
        ),
        rx.text(
            rx.cond(
                entry.drop_pct > 0,
                "↓" + entry.drop_pct.to_string() + "%",
                "Δ" + entry.drop_pct.to_string() + "%",
            ),
            font_size="0.78rem",
            color=rx.cond(entry.drop_pct > 10, c("success"), c("warning")),
            width="60px",
            text_align="right",
        ),
        width="100%",
    )


def _step5() -> rx.Component:
    return rx.vstack(
        # Header
        rx.hstack(
            rx.vstack(
                rx.text("Training", font_size="1.1rem", font_weight="700", color=c("text_primary")),
                rx.text(
                    FinetuneState.effective_model_name, font_size="0.82rem", color=c("text_muted")
                ),
                spacing="0",
            ),
            rx.spacer(),
            _badge_status(FinetuneState.training_status),
            align="center",
        ),
        # Start error
        rx.cond(
            FinetuneState.start_error != "",
            rx.callout(FinetuneState.start_error, color_scheme="red"),
            rx.fragment(),
        ),
        # Metric tiles
        rx.grid(
            _metric_tile("Epoch", FinetuneState.current_epoch_display),
            _metric_tile("Steps", FinetuneState.current_total_steps_display),
            _metric_tile("Elapsed", FinetuneState.elapsed_time_display),
            _metric_tile("GPU Memory", FinetuneState.gpu_memory_display),
            columns="4",
            spacing="3",
            width="100%",
        ),
        # Epoch progress bar
        rx.vstack(
            rx.hstack(
                rx.text("Epoch progress", font_size="0.76rem", color=c("text_muted")),
                rx.spacer(),
                rx.text(
                    FinetuneState.epoch_progress_pct.to_string() + "%",
                    font_size="0.76rem",
                    color=c("text_secondary"),
                ),
            ),
            rx.progress(
                value=FinetuneState.epoch_progress_pct, max=100, width="100%", color_scheme="blue"
            ),
            width="100%",
            spacing="1",
        ),
        # Loss + LR chart
        _card(loss_chart()),
        # AI Commentary
        rx.cond(
            FinetuneState.ai_commentary != "",
            _card(
                rx.hstack(
                    rx.icon("sparkles", size=16, color=c("accent")),
                    rx.text(
                        FinetuneState.ai_commentary, font_size="0.86rem", color=c("text_primary")
                    ),
                    spacing="2",
                    align="start",
                )
            ),
            rx.fragment(),
        ),
        # Epoch log
        rx.cond(
            FinetuneState.epoch_log.length() > 0,
            _card(
                rx.vstack(
                    rx.text(
                        "Epoch log",
                        font_size="0.78rem",
                        font_weight="600",
                        color=c("text_secondary"),
                        margin_bottom="8px",
                    ),
                    rx.foreach(FinetuneState.epoch_log, _epoch_log_row),
                    spacing="2",
                )
            ),
            rx.fragment(),
        ),
        # Status message after done
        rx.cond(
            FinetuneState.training_status == "done",
            rx.vstack(
                rx.callout(
                    rx.hstack(
                        rx.icon("circle-check", size=16),
                        rx.text("Training complete! Advancing to results..."),
                        spacing="2",
                    ),
                    color_scheme="green",
                ),
                rx.button(
                    "View Results →",
                    on_click=FinetuneState.go_to_step(6),
                    color_scheme="green",
                    size="3",
                ),
                spacing="3",
            ),
            rx.fragment(),
        ),
        rx.cond(
            FinetuneState.training_status == "failed",
            rx.callout(
                rx.vstack(
                    rx.text("Training failed", font_weight="600"),
                    rx.text(FinetuneState.error_msg, font_size="0.82rem"),
                ),
                color_scheme="red",
            ),
            rx.fragment(),
        ),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


# ── Step 6: Results ───────────────────────────────────────────────
def _step6() -> rx.Component:
    return rx.vstack(
        _section_heading("Results & Evaluation"),
        # Eval metrics
        _card(
            rx.vstack(
                rx.hstack(
                    rx.text(
                        "Evaluation metrics",
                        font_size="0.9rem",
                        font_weight="600",
                        color=c("text_primary"),
                    ),
                    rx.spacer(),
                    rx.cond(
                        FinetuneState.eval_status == "idle",
                        rx.button(
                            "Run evaluation",
                            on_click=FinetuneState.run_eval,
                            size="2",
                            color_scheme="blue",
                            variant="soft",
                        ),
                        rx.badge(FinetuneState.eval_status, color_scheme="blue", size="1"),
                    ),
                    align="center",
                ),
                rx.cond(
                    FinetuneState.eval_status == "done",
                    rx.grid(
                        rx.vstack(
                            rx.text("Perplexity", font_size="0.72rem", color=c("text_muted")),
                            rx.text(
                                FinetuneState.eval_perplexity.to_string(),
                                font_size="1.8rem",
                                font_weight="700",
                                color=c("accent"),
                            ),
                            rx.text("Lower is better", font_size="0.7rem", color=c("text_muted")),
                            spacing="0",
                        ),
                        rx.vstack(
                            rx.text("What it means", font_size="0.72rem", color=c("text_muted")),
                            rx.text(
                                rx.cond(
                                    FinetuneState.eval_perplexity < 10,
                                    "Excellent — model learned the domain well",
                                    rx.cond(
                                        FinetuneState.eval_perplexity < 30,
                                        "Good — decent task alignment",
                                        "Try more epochs or a larger dataset",
                                    ),
                                ),
                                font_size="0.84rem",
                                color=c("text_secondary"),
                            ),
                            spacing="1",
                        ),
                        columns="2",
                        spacing="4",
                    ),
                    rx.fragment(),
                ),
                spacing="3",
            )
        ),
        # Inference tester
        _card(
            rx.vstack(
                rx.text(
                    "Test your model",
                    font_size="0.9rem",
                    font_weight="600",
                    color=c("text_primary"),
                    margin_bottom="8px",
                ),
                rx.cond(
                    FinetuneState.user_intent != "",
                    rx.text(
                        f"System context: {FinetuneState.user_intent}",
                        font_size="0.76rem",
                        color=c("text_muted"),
                        font_style="italic",
                    ),
                    rx.fragment(),
                ),
                # Chat history
                rx.cond(
                    FinetuneState.test_chat_history.length() > 0,
                    rx.box(
                        rx.foreach(
                            FinetuneState.test_chat_history,
                            lambda msg: rx.box(
                                rx.text(
                                    msg.content,
                                    font_size="0.84rem",
                                    color=rx.cond(
                                        msg.role == "user",
                                        c("text_primary"),
                                        c("text_secondary"),
                                    ),
                                    padding="8px 12px",
                                    background=rx.cond(
                                        msg.role == "user",
                                        c("accent_soft"),
                                        c("bg_input"),
                                    ),
                                    border_radius="8px",
                                    align_self=rx.cond(
                                        msg.role == "user", "flex-end", "flex-start"
                                    ),
                                    max_width="80%",
                                ),
                                display="flex",
                                flex_direction=rx.cond(msg.role == "user", "row-reverse", "row"),
                                width="100%",
                                margin_bottom="6px",
                            ),
                        ),
                        width="100%",
                        max_height="300px",
                        overflow_y="auto",
                        padding="8px",
                        border="1px solid",
                        border_color=c("border"),
                        border_radius="8px",
                    ),
                    rx.fragment(),
                ),
                rx.hstack(
                    rx.input(
                        placeholder="Type a test message...",
                        value=FinetuneState.chat_input,
                        on_change=FinetuneState.set_chat_input,
                        on_key_down=FinetuneState.handle_chat_key,
                        flex="1",
                    ),
                    rx.button(
                        rx.cond(
                            FinetuneState.chat_loading,
                            rx.spinner(size="2"),
                            rx.icon("send", size=16),
                        ),
                        on_click=FinetuneState.send_test_chat,
                        disabled=FinetuneState.chat_loading,
                        color_scheme="blue",
                        size="2",
                    ),
                    spacing="2",
                ),
                rx.cond(
                    FinetuneState.chat_error != "",
                    rx.callout(FinetuneState.chat_error, color_scheme="red", size="1"),
                    rx.fragment(),
                ),
                spacing="3",
            )
        ),
        # Register to model registry
        rx.cond(
            (FinetuneState.training_status == "done") & (FinetuneState.experiment_id != ""),
            _card(
                rx.vstack(
                    rx.hstack(
                        rx.icon("bookmark", size=16, color=c("accent")),
                        rx.text(
                            "Register to model registry",
                            font_size="0.9rem",
                            font_weight="600",
                            color=c("text_primary"),
                        ),
                        spacing="2",
                        align="center",
                    ),
                    rx.text(
                        "Save this run under a name so you can promote it to production "
                        "or compare it against future runs.",
                        font_size="0.82rem",
                        color=c("text_secondary"),
                    ),
                    rx.hstack(
                        rx.input(
                            placeholder="my-chatbot-v1",
                            value=ModelRegistryState.register_name,
                            on_change=ModelRegistryState.set_register_name,
                            flex="1",
                        ),
                        rx.button(
                            rx.cond(
                                ModelRegistryState.is_registering,
                                rx.hstack(
                                    rx.spinner(size="2"), rx.text("Saving…"), spacing="2"
                                ),
                                rx.text("Register"),
                            ),
                            on_click=ModelRegistryState.do_register(
                                FinetuneState.experiment_id,
                                FinetuneState.eval_perplexity,
                                FinetuneState.last_train_loss,
                            ),
                            disabled=ModelRegistryState.is_registering,
                            color_scheme="blue",
                            size="2",
                        ),
                        spacing="2",
                        width="100%",
                    ),
                    rx.cond(
                        ModelRegistryState.register_error != "",
                        rx.callout(ModelRegistryState.register_error, color_scheme="red", size="1"),
                        rx.fragment(),
                    ),
                    rx.cond(
                        ModelRegistryState.register_success,
                        rx.callout(
                            rx.hstack(
                                rx.icon("circle-check", size=14),
                                rx.text(
                                    "Registered as "" + ModelRegistryState.register_name + """
                                ),
                                spacing="2",
                            ),
                            color_scheme="green",
                            size="1",
                        ),
                        rx.fragment(),
                    ),
                    spacing="3",
                )
            ),
            rx.fragment(),
        ),
        # Experiment comparison
        rx.cond(
            ExperimentState.completed_runs.length() > 1,
            _card(
                rx.vstack(
                    rx.text(
                        "Past runs with this setup",
                        font_size="0.9rem",
                        font_weight="600",
                        color=c("text_primary"),
                        margin_bottom="8px",
                    ),
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                rx.table.column_header_cell("Name"),
                                rx.table.column_header_cell("Model"),
                                rx.table.column_header_cell("Technique"),
                                rx.table.column_header_cell("LR"),
                                rx.table.column_header_cell("LoRA r"),
                                rx.table.column_header_cell("Batch"),
                                rx.table.column_header_cell("Epochs"),
                                rx.table.column_header_cell("Final Loss"),
                                rx.table.column_header_cell("Perplexity"),
                            )
                        ),
                        rx.table.body(
                            rx.foreach(
                                ExperimentState.completed_runs,
                                lambda r: rx.table.row(
                                    rx.table.cell(rx.text(r.name, font_size="0.8rem")),
                                    rx.table.cell(rx.text(r.model_id, font_size="0.8rem")),
                                    rx.table.cell(rx.text(r.technique, font_size="0.8rem")),
                                    rx.table.cell(
                                        rx.text(r.learning_rate, font_size="0.8rem")
                                    ),
                                    rx.table.cell(
                                        rx.text(r.lora_r.to_string(), font_size="0.8rem")
                                    ),
                                    rx.table.cell(
                                        rx.text(r.batch_size.to_string(), font_size="0.8rem")
                                    ),
                                    rx.table.cell(
                                        rx.text(r.epochs.to_string(), font_size="0.8rem")
                                    ),
                                    rx.table.cell(
                                        rx.text(r.final_loss.to_string(), font_size="0.8rem")
                                    ),
                                    rx.table.cell(
                                        rx.text(r.perplexity.to_string(), font_size="0.8rem")
                                    ),
                                ),
                            )
                        ),
                        variant="surface",
                        size="1",
                        width="100%",
                    ),
                    spacing="2",
                )
            ),
            rx.fragment(),
        ),
        _nav_buttons(next_label="Next: Deploy →"),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


# ── Step 7: Deploy ────────────────────────────────────────────────
def _deploy_target_row(
    target_key: str,
    label: str,
    description: str,
    icon: str,
    is_checked,
) -> rx.Component:
    return rx.hstack(
        rx.checkbox(
            checked=is_checked,
            on_change=lambda _: FinetuneState.toggle_deploy_target(target_key),
            size="2",
        ),
        rx.vstack(
            rx.text(label, font_weight="500", font_size="0.88rem", color=c("text_primary")),
            rx.text(description, font_size="0.76rem", color=c("text_muted")),
            spacing="0",
        ),
        spacing="3",
        align="start",
        padding="10px 0",
        border_bottom="1px solid",
        border_color=c("border"),
        width="100%",
    )


def _step7() -> rx.Component:
    return rx.vstack(
        _section_heading("Deploy your model"),
        rx.text(
            "Choose how you want to export or share your fine-tuned adapter.",
            font_size="0.86rem",
            color=c("text_secondary"),
            margin_bottom="16px",
        ),
        # Target selector
        _card(
            rx.vstack(
                _deploy_target_row(
                    "adapter",
                    "Download adapter",
                    "Zip the LoRA adapter files (~100 MB) — works with PEFT/Transformers",
                    "download",
                    FinetuneState.deploy_adapter,
                ),
                _deploy_target_row(
                    "merged",
                    "Download merged model",
                    "Merge adapter into base model → full standalone safetensors (~14 GB for 7B)",
                    "layers",
                    FinetuneState.deploy_merged,
                ),
                _deploy_target_row(
                    "hub",
                    "Push to Hugging Face Hub",
                    "Upload adapter to a public or private HF repository",
                    "globe",
                    FinetuneState.deploy_hub,
                ),
                _deploy_target_row(
                    "gguf",
                    "Export as GGUF",
                    "Convert to GGUF format for use with Ollama or llama.cpp (CPU inference)",
                    "cpu",
                    FinetuneState.deploy_gguf,
                ),
                _deploy_target_row(
                    "github",
                    "Push to GitHub",
                    "Commit adapter files to a GitHub repository using Git LFS",
                    "github",
                    FinetuneState.deploy_github,
                ),
                spacing="0",
            )
        ),
        # HF Hub fields
        rx.cond(
            FinetuneState.deploy_hub,
            _card(
                rx.vstack(
                    rx.text(
                        "Hugging Face Hub",
                        font_weight="600",
                        font_size="0.88rem",
                        color=c("text_primary"),
                    ),
                    rx.grid(
                        rx.vstack(
                            _label("HF Token"),
                            rx.input(
                                type="password",
                                placeholder="hf_xxxxxxxxxxxxx",
                                value=FinetuneState.hf_token_input,
                                on_change=FinetuneState.set_hf_token_input,
                                width="100%",
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Repository name (e.g. myuser/my-chatbot-lora)"),
                            rx.input(
                                placeholder="username/repo-name",
                                value=FinetuneState.hf_repo_name,
                                on_change=FinetuneState.set_hf_repo_name,
                                width="100%",
                            ),
                            spacing="1",
                        ),
                        columns="2",
                        spacing="3",
                        width="100%",
                    ),
                    rx.button(
                        rx.cond(
                            FinetuneState.push_status == "pushing",
                            rx.hstack(rx.spinner(size="2"), rx.text("Pushing..."), spacing="2"),
                            rx.text("Push to Hub"),
                        ),
                        on_click=FinetuneState.push_to_hub,
                        disabled=FinetuneState.push_status == "pushing",
                        color_scheme="blue",
                        size="2",
                    ),
                    rx.cond(
                        FinetuneState.push_status == "done",
                        rx.callout(
                            rx.hstack(
                                rx.icon("circle-check", size=14),
                                rx.text(f"Pushed to {FinetuneState.push_repo_url}"),
                                spacing="2",
                            ),
                            color_scheme="green",
                            size="1",
                        ),
                        rx.fragment(),
                    ),
                    rx.cond(
                        FinetuneState.push_error != "",
                        rx.callout(FinetuneState.push_error, color_scheme="red", size="1"),
                        rx.fragment(),
                    ),
                    spacing="3",
                )
            ),
            rx.fragment(),
        ),
        # GGUF fields
        rx.cond(
            FinetuneState.deploy_gguf,
            _card(
                rx.vstack(
                    rx.text(
                        "GGUF Export",
                        font_weight="600",
                        font_size="0.88rem",
                        color=c("text_primary"),
                    ),
                    rx.callout(
                        "GGUF export requires the model to be merged first. "
                        "Enable 'Download merged model' above to trigger the merge step.",
                        color_scheme="amber",
                        size="1",
                    ),
                    rx.hstack(
                        rx.vstack(
                            _label("Quantization"),
                            rx.select.root(
                                rx.select.trigger(width="160px"),
                                rx.select.content(
                                    *[rx.select.item(q, value=q) for q in _GGUF_QUANTS],
                                ),
                                value=FinetuneState.gguf_quantization,
                                on_change=FinetuneState.set_gguf_quantization,
                            ),
                            spacing="1",
                        ),
                        rx.button(
                            rx.cond(
                                FinetuneState.gguf_status == "exporting",
                                rx.hstack(
                                    rx.spinner(size="2"), rx.text("Exporting..."), spacing="2"
                                ),
                                rx.text("Export GGUF"),
                            ),
                            on_click=FinetuneState.start_gguf_export,
                            disabled=FinetuneState.gguf_status == "exporting",
                            color_scheme="blue",
                            size="2",
                            align_self="flex-end",
                        ),
                        spacing="3",
                        align="end",
                    ),
                    spacing="3",
                )
            ),
            rx.fragment(),
        ),
        # GitHub fields
        rx.cond(
            FinetuneState.deploy_github,
            _card(
                rx.vstack(
                    rx.text(
                        "GitHub Push",
                        font_weight="600",
                        font_size="0.88rem",
                        color=c("text_primary"),
                    ),
                    rx.grid(
                        rx.vstack(
                            _label("Repository URL (HTTPS)"),
                            rx.input(
                                placeholder="https://github.com/user/repo",
                                value=FinetuneState.github_repo_url,
                                on_change=FinetuneState.set_github_repo_url,
                                width="100%",
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("GitHub Token (needs repo scope)"),
                            rx.input(
                                type="password",
                                placeholder="ghp_xxxxxxxxxxxxx",
                                value=FinetuneState.github_token,
                                on_change=FinetuneState.set_github_token,
                                width="100%",
                            ),
                            spacing="1",
                        ),
                        columns="2",
                        spacing="3",
                        width="100%",
                    ),
                    rx.button(
                        rx.cond(
                            FinetuneState.github_push_status == "pushing",
                            rx.hstack(rx.spinner(size="2"), rx.text("Pushing..."), spacing="2"),
                            rx.text("Push to GitHub"),
                        ),
                        on_click=FinetuneState.push_to_github,
                        disabled=FinetuneState.github_push_status == "pushing",
                        color_scheme="blue",
                        size="2",
                    ),
                    rx.cond(
                        FinetuneState.github_push_status == "done",
                        rx.callout(
                            f"Pushed to {FinetuneState.github_repo_url}",
                            color_scheme="green",
                            size="1",
                        ),
                        rx.fragment(),
                    ),
                    spacing="3",
                )
            ),
            rx.fragment(),
        ),
        # Quick actions (always visible)
        rx.hstack(
            rx.button(
                rx.hstack(rx.icon("download", size=14), rx.text("Download adapter"), spacing="2"),
                on_click=FinetuneState.download_adapter,
                color_scheme="blue",
                variant="soft",
                size="2",
            ),
            rx.cond(
                FinetuneState.deploy_merged,
                rx.button(
                    rx.cond(
                        FinetuneState.merge_status == "merging",
                        rx.hstack(rx.spinner(size="2"), rx.text("Merging..."), spacing="2"),
                        rx.hstack(
                            rx.icon("layers", size=14), rx.text("Merge & download"), spacing="2"
                        ),
                    ),
                    on_click=FinetuneState.start_merge,
                    disabled=FinetuneState.merge_status == "merging",
                    color_scheme="blue",
                    variant="soft",
                    size="2",
                ),
                rx.fragment(),
            ),
            spacing="3",
            wrap="wrap",
        ),
        # Deploy log
        rx.cond(
            FinetuneState.deploy_log != "",
            _card(
                rx.vstack(
                    rx.text(
                        "Activity log",
                        font_size="0.78rem",
                        font_weight="600",
                        color=c("text_secondary"),
                    ),
                    rx.box(
                        rx.text(
                            FinetuneState.deploy_log,
                            font_size="0.76rem",
                            color=c("text_secondary"),
                            font_family="monospace",
                            white_space="pre-wrap",
                        ),
                        background=c("bg_input"),
                        border_radius="8px",
                        padding="12px",
                        max_height="200px",
                        overflow_y="auto",
                        width="100%",
                    ),
                    spacing="2",
                )
            ),
            rx.fragment(),
        ),
        # Start over
        rx.box(height="8px"),
        rx.button(
            "Start a new fine-tune →",
            on_click=rx.redirect("/finetune"),
            color_scheme="gray",
            variant="soft",
            size="2",
        ),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


# ── Page root ─────────────────────────────────────────────────────
def finetune_page() -> rx.Component:
    return rx.box(
        rx.vstack(
            # Header
            rx.hstack(
                rx.icon("flask-conical", size=20, color=c("accent")),
                rx.text(
                    "Fine-tune a model",
                    font_size="1.25rem",
                    font_weight="700",
                    color=c("text_primary"),
                ),
                spacing="2",
                align="center",
                margin_bottom="24px",
            ),
            # Progress bar
            _progress_bar(),
            # Step content
            rx.match(
                FinetuneState.current_step,
                (1, _step1()),
                (2, _step2()),
                (3, _step3()),
                (4, _step4()),
                (5, _step5()),
                (6, _step6()),
                (7, _step7()),
                rx.text("Invalid step", color=c("text_muted")),
            ),
            spacing="0",
            width="100%",
            align_items="flex-start",
            padding="32px 24px",
        ),
        width="100%",
        on_mount=[ExperimentState.load_runs, ModelRegistryState.load_models],
    )
