"""Fine-tune wizard — Step 3: Training data (upload / Hub / generate)."""

from __future__ import annotations

import reflex as rx

from app.components.finetune.shared import (
    _card,
    _label,
    _nav_buttons,
    _preview_table,
    _section_heading,
)
from app.state.finetune_state import FinetuneState
from app.styles import c


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
            rx.cond(
                FinetuneState.dataset_row_count > 0,
                rx.hstack(
                    rx.badge(
                        FinetuneState.dataset_row_count.to_string() + " rows",
                        color_scheme="blue",
                        size="1",
                    ),
                    rx.badge(
                        "~" + FinetuneState.dataset_avg_tokens.to_string() + " avg tokens",
                        color_scheme="gray",
                        size="1",
                    ),
                    rx.cond(
                        FinetuneState.dataset_row_count < 100,
                        rx.badge("⚠ Small dataset", color_scheme="orange", size="1"),
                        rx.fragment(),
                    ),
                    spacing="2",
                    margin_top="8px",
                ),
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
                        rx.icon("sparkles", size=14),
                        rx.text("Generate examples (~2 min)"),
                        spacing="2",
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


def _dpo_format_card() -> rx.Component:
    """Shown when the DPO technique is selected — preference data needs three
    columns (prompt / chosen / rejected); let the user remap them."""
    return _card(
        rx.vstack(
            rx.hstack(
                rx.icon("scale", size=15, color=c("accent")),
                rx.text("DPO preference data", font_weight="600", color=c("text_primary")),
                spacing="2",
                align="center",
            ),
            rx.text(
                "Your dataset needs three columns: the prompt, a chosen (preferred) "
                "response, and a rejected one. Remap the column names if they differ.",
                font_size="0.8rem",
                color=c("text_muted"),
            ),
            rx.grid(
                rx.vstack(
                    _label("Prompt column"),
                    rx.input(
                        value=FinetuneState.dpo_prompt_col,
                        on_change=FinetuneState.set_dpo_prompt_col,
                        width="100%",
                    ),
                    spacing="1",
                ),
                rx.vstack(
                    _label("Chosen column"),
                    rx.input(
                        value=FinetuneState.dpo_chosen_col,
                        on_change=FinetuneState.set_dpo_chosen_col,
                        width="100%",
                    ),
                    spacing="1",
                ),
                rx.vstack(
                    _label("Rejected column"),
                    rx.input(
                        value=FinetuneState.dpo_rejected_col,
                        on_change=FinetuneState.set_dpo_rejected_col,
                        width="100%",
                    ),
                    spacing="1",
                ),
                columns="3",
                spacing="3",
                width="100%",
            ),
            spacing="3",
            width="100%",
        ),
        margin_bottom="16px",
    )


def _step3() -> rx.Component:
    return rx.vstack(
        _section_heading("Add your training data"),
        rx.cond(FinetuneState.is_dpo, _dpo_format_card(), rx.fragment()),
        rx.hstack(
            _data_mode_btn("upload", "Upload a file ✓", "upload"),
            rx.badge("Recommended", color_scheme="green", size="1", variant="soft"),
            _data_mode_btn("hub_dataset", "HF Hub dataset", "database"),
            _data_mode_btn("generate", "Generate with AI ✨", "sparkles"),
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
