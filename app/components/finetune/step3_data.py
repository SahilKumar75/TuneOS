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


def _hf_icon(size: int = 20) -> rx.Component:
    """HuggingFace logo as inline SVG in brand orange."""
    s = str(size)
    svg = (
        f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
        '<ellipse cx="12" cy="12.5" rx="10.5" ry="10.5" fill="#FF9D00"/>'
        '<circle cx="8.8" cy="10.5" r="1.4" fill="#1a1a1a"/>'
        '<circle cx="15.2" cy="10.5" r="1.4" fill="#1a1a1a"/>'
        '<path d="M8.5 15.5 Q12 18.5 15.5 15.5" stroke="#1a1a1a" stroke-width="1.4"'
        ' fill="none" stroke-linecap="round"/>'
        '<line x1="8.8" y1="7" x2="8.8" y2="9" stroke="#1a1a1a" stroke-width="1.2"'
        ' stroke-linecap="round"/>'
        '<line x1="15.2" y1="7" x2="15.2" y2="9" stroke="#1a1a1a" stroke-width="1.2"'
        ' stroke-linecap="round"/>'
        "</svg>"
    )
    return rx.html(svg)


def _seg_btn(mode: str, *children, extra_event=None) -> rx.Component:
    """Uniform segmented-control pill — all tabs same size/shape."""
    is_active = FinetuneState.data_source == mode
    click = (
        [FinetuneState.set_data_source(mode), extra_event]
        if extra_event
        else FinetuneState.set_data_source(mode)
    )
    return rx.box(
        rx.hstack(*children, spacing="2", align="center"),
        on_click=click,
        cursor="pointer",
        padding="0 12px",
        height="30px",
        display="flex",
        align_items="center",
        border_radius="6px",
        background=rx.cond(is_active, "white", "transparent"),
        box_shadow=rx.cond(
            is_active,
            "0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.07)",
            "none",
        ),
        color=rx.cond(is_active, "var(--gray-12)", "var(--gray-11)"),
        style={
            "transition": "background 0.12s ease, box-shadow 0.12s ease",
            "white-space": "nowrap",
        },
    )


def _data_mode_btn(mode: str, label: str, icon: str) -> rx.Component:
    return _seg_btn(
        mode,
        rx.icon(icon, size=13),
        rx.text(label, font_size="0.82rem", font_weight="500"),
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
                        FinetuneState.hub_col_auto_detected,
                        rx.callout(
                            rx.text(
                                "Column names auto-detected (no exact 'instruction'/'output' match). "
                                "Available: ",
                                rx.text.span(
                                    FinetuneState.hub_columns_str,
                                    font_weight="500",
                                ),
                            ),
                            color_scheme="orange",
                            size="1",
                        ),
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
                    # Search bar
                    rx.hstack(
                        rx.icon("search", size=14, color="var(--gray-9)"),
                        rx.input(
                            placeholder="Search HuggingFace datasets...",
                            value=FinetuneState.hub_search_query,
                            on_change=FinetuneState.search_hub_datasets,
                            border="none",
                            outline="none",
                            background="transparent",
                            font_size="0.85rem",
                            flex="1",
                            _focus={"outline": "none", "box_shadow": "none"},
                        ),
                        rx.cond(
                            FinetuneState.hub_is_searching,
                            rx.spinner(size="1"),
                            rx.fragment(),
                        ),
                        padding="8px 12px",
                        border="1px solid var(--gray-5)",
                        border_radius="8px",
                        background="var(--gray-1)",
                        align="center",
                        width="100%",
                        spacing="2",
                    ),
                    # Results or recommended
                    rx.cond(
                        FinetuneState.hub_search_query != "",
                        # Search results
                        rx.cond(
                            FinetuneState.hub_search_results.length() > 0,
                            rx.vstack(
                                rx.foreach(
                                    FinetuneState.hub_search_results,
                                    lambda r: rx.box(
                                        rx.hstack(
                                            rx.vstack(
                                                rx.text(
                                                    r.id,
                                                    font_size="0.84rem",
                                                    font_weight="600",
                                                    color="var(--gray-12)",
                                                ),
                                                rx.text(
                                                    r.description,
                                                    font_size="0.75rem",
                                                    color="var(--gray-10)",
                                                    no_of_lines=1,
                                                ),
                                                rx.hstack(
                                                    rx.icon(
                                                        "download", size=11, color="var(--gray-9)"
                                                    ),
                                                    rx.text(
                                                        r.downloads.to_string(),
                                                        font_size="0.72rem",
                                                        color="var(--gray-9)",
                                                    ),
                                                    rx.icon(
                                                        "heart", size=11, color="var(--gray-9)"
                                                    ),
                                                    rx.text(
                                                        r.likes.to_string(),
                                                        font_size="0.72rem",
                                                        color="var(--gray-9)",
                                                    ),
                                                    spacing="1",
                                                    align="center",
                                                ),
                                                spacing="1",
                                                align_items="start",
                                            ),
                                            rx.button(
                                                "Use",
                                                size="1",
                                                variant="soft",
                                                color_scheme="blue",
                                                on_click=FinetuneState.set_hub_dataset_id(r.id),
                                            ),
                                            justify="between",
                                            align="center",
                                            width="100%",
                                        ),
                                        padding="10px 12px",
                                        border="1px solid var(--gray-4)",
                                        border_radius="8px",
                                        background="var(--gray-1)",
                                        width="100%",
                                        cursor="pointer",
                                        _hover={"background": "var(--gray-2)"},
                                    ),
                                ),
                                spacing="2",
                                width="100%",
                            ),
                            rx.text(
                                "No results found.",
                                font_size="0.83rem",
                                color="var(--gray-10)",
                                padding="8px 0",
                            ),
                        ),
                        # Intent-aware recommended datasets
                        rx.vstack(
                            rx.hstack(
                                rx.text(
                                    "Recommended for your intent",
                                    font_size="0.75rem",
                                    font_weight="600",
                                    color="var(--gray-9)",
                                    letter_spacing="0.04em",
                                ),
                                rx.cond(
                                    FinetuneState.hub_is_loading_recs,
                                    rx.spinner(size="1"),
                                    rx.fragment(),
                                ),
                                spacing="2",
                                align="center",
                            ),
                            rx.cond(
                                FinetuneState.hub_recommended.length() > 0,
                                rx.grid(
                                    rx.foreach(
                                        FinetuneState.hub_recommended,
                                        lambda r: rx.box(
                                            rx.vstack(
                                                # Header: big HF icon + full dataset id
                                                rx.hstack(
                                                    _hf_icon(18),
                                                    rx.vstack(
                                                        rx.text(
                                                            r.id,
                                                            font_size="0.8rem",
                                                            font_weight="700",
                                                            color="var(--gray-12)",
                                                            no_of_lines=1,
                                                        ),
                                                        rx.cond(
                                                            r.size_category != "",
                                                            rx.badge(
                                                                r.size_category,
                                                                size="1",
                                                                variant="soft",
                                                                color_scheme="gray",
                                                            ),
                                                            rx.fragment(),
                                                        ),
                                                        spacing="1",
                                                        align_items="start",
                                                    ),
                                                    spacing="2",
                                                    align="start",
                                                    width="100%",
                                                ),
                                                # Description
                                                rx.cond(
                                                    r.description != "",
                                                    rx.text(
                                                        r.description,
                                                        font_size="0.74rem",
                                                        color="var(--gray-10)",
                                                        no_of_lines=3,
                                                        line_height="1.45",
                                                    ),
                                                    rx.fragment(),
                                                ),
                                                # Task tags
                                                rx.cond(
                                                    r.tags.length() > 0,
                                                    rx.hstack(
                                                        rx.foreach(
                                                            r.tags,
                                                            lambda t: rx.badge(
                                                                t,
                                                                size="1",
                                                                variant="soft",
                                                                color_scheme="blue",
                                                            ),
                                                        ),
                                                        spacing="1",
                                                        wrap="wrap",
                                                    ),
                                                    rx.fragment(),
                                                ),
                                                # Footer: stats + Use button
                                                rx.hstack(
                                                    rx.hstack(
                                                        rx.icon(
                                                            "download",
                                                            size=11,
                                                            color="var(--gray-9)",
                                                        ),
                                                        rx.text(
                                                            r.downloads.to_string(),
                                                            font_size="0.72rem",
                                                            color="var(--gray-9)",
                                                        ),
                                                        rx.icon(
                                                            "heart",
                                                            size=11,
                                                            color="var(--gray-9)",
                                                        ),
                                                        rx.text(
                                                            r.likes.to_string(),
                                                            font_size="0.72rem",
                                                            color="var(--gray-9)",
                                                        ),
                                                        spacing="1",
                                                        align="center",
                                                    ),
                                                    rx.button(
                                                        "Use",
                                                        size="1",
                                                        variant="soft",
                                                        color_scheme="blue",
                                                        on_click=FinetuneState.set_hub_dataset_id(
                                                            r.id
                                                        ),
                                                    ),
                                                    justify="between",
                                                    width="100%",
                                                ),
                                                spacing="2",
                                                align_items="start",
                                            ),
                                            padding="14px",
                                            border="1px solid var(--gray-4)",
                                            border_radius="10px",
                                            background="var(--gray-1)",
                                            _hover={
                                                "background": "var(--gray-2)",
                                                "border-color": "var(--gray-5)",
                                            },
                                            style={"transition": "all 0.12s ease"},
                                        ),
                                    ),
                                    columns="2",
                                    spacing="3",
                                    width="100%",
                                ),
                                rx.cond(
                                    FinetuneState.hub_is_loading_recs,
                                    rx.fragment(),
                                    rx.text(
                                        "Click the HF Hub tab to load recommendations",
                                        font_size="0.83rem",
                                        color="var(--gray-9)",
                                        padding="8px 0",
                                    ),
                                ),
                            ),
                            spacing="2",
                            width="100%",
                        ),
                    ),
                    spacing="3",
                    width="100%",
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
                        rx.select.trigger(width="220px"),
                        rx.select.content(
                            rx.select.item("Self-Instruct (recommended)", value="self_instruct"),
                            rx.select.item("Evol-Instruct (WizardLM)", value="evol_instruct"),
                            rx.select.item("Persona-Based", value="persona"),
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
                rx.vstack(
                    _label("Export format"),
                    rx.select.root(
                        rx.select.trigger(width="160px"),
                        rx.select.content(
                            rx.select.item("JSONL (default)", value="jsonl"),
                            rx.select.item("Alpaca JSON", value="alpaca_json"),
                            rx.select.item("ShareGPT JSON", value="sharegpt_json"),
                        ),
                        value=FinetuneState.generation_export_format,
                        on_change=FinetuneState.set_export_format,
                    ),
                    spacing="1",
                ),
                spacing="4",
                wrap="wrap",
            ),
            # Quality filter slider
            rx.vstack(
                _label("Quality filter (0 = off, 3 = recommended)"),
                rx.hstack(
                    rx.slider(
                        min=0,
                        max=5,
                        step=0.5,
                        default_value=[FinetuneState.generation_quality_threshold],
                        on_value_commit=FinetuneState.set_quality_threshold,
                        width="200px",
                    ),
                    rx.text(
                        FinetuneState.generation_quality_threshold.to_string(),
                        font_size="0.82rem",
                        color=c("text_secondary"),
                    ),
                    spacing="3",
                    align="center",
                ),
                rx.text(
                    "Samples scoring below this threshold are discarded after generation.",
                    font_size="0.75rem",
                    color=c("text_muted"),
                ),
                spacing="1",
                width="100%",
            ),
            # Seed example editor
            rx.vstack(
                rx.hstack(
                    rx.text("Seed examples", font_weight="500", font_size="0.84rem"),
                    rx.badge(
                        FinetuneState.seed_examples.length().to_string(),
                        color_scheme="blue",
                        size="1",
                    ),
                    rx.button(
                        rx.cond(
                            FinetuneState.seed_editor_open,
                            "Hide editor",
                            "Add seeds",
                        ),
                        on_click=FinetuneState.toggle_seed_editor,
                        variant="ghost",
                        size="1",
                        color_scheme="blue",
                    ),
                    spacing="2",
                    align="center",
                ),
                rx.cond(
                    FinetuneState.seed_editor_open,
                    rx.vstack(
                        rx.text(
                            "Seed examples guide generation — the more specific, the better.",
                            font_size="0.76rem",
                            color=c("text_muted"),
                        ),
                        rx.hstack(
                            rx.vstack(
                                _label("Instruction"),
                                rx.text_area(
                                    value=FinetuneState.seed_editor_instruction,
                                    on_change=FinetuneState.set_seed_instruction,
                                    placeholder="e.g. What is insulin resistance?",
                                    rows="3",
                                    width="100%",
                                ),
                                spacing="1",
                                flex="1",
                            ),
                            rx.vstack(
                                _label("Output"),
                                rx.text_area(
                                    value=FinetuneState.seed_editor_output,
                                    on_change=FinetuneState.set_seed_output,
                                    placeholder="e.g. Insulin resistance is ...",
                                    rows="3",
                                    width="100%",
                                ),
                                spacing="1",
                                flex="1",
                            ),
                            spacing="3",
                            width="100%",
                        ),
                        rx.button(
                            rx.hstack(rx.icon("plus", size=12), rx.text("Add"), spacing="1"),
                            on_click=FinetuneState.add_seed_example,
                            size="1",
                            color_scheme="blue",
                            variant="soft",
                        ),
                        rx.cond(
                            FinetuneState.seed_examples.length() > 0,
                            rx.vstack(
                                rx.foreach(
                                    FinetuneState.seed_examples,
                                    lambda s: rx.hstack(
                                        rx.text(
                                            s.instruction,
                                            font_size="0.76rem",
                                            flex="1",
                                            no_of_lines=1,
                                        ),
                                        rx.icon_button(
                                            rx.icon("x", size=12),
                                            size="1",
                                            variant="ghost",
                                            color_scheme="red",
                                            on_click=FinetuneState.remove_seed_example(s.id),
                                        ),
                                        border="1px solid",
                                        border_color=c("border"),
                                        border_radius="6px",
                                        padding="6px 10px",
                                        width="100%",
                                        align="center",
                                    ),
                                ),
                                spacing="1",
                                width="100%",
                            ),
                            rx.fragment(),
                        ),
                        spacing="2",
                        width="100%",
                    ),
                    rx.fragment(),
                ),
                spacing="1",
                width="100%",
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
            rx.cond(
                FinetuneState.generation_alt_download_url != "",
                rx.hstack(
                    rx.icon("download", size=14, color=c("accent")),
                    rx.link(
                        "Download alternate format",
                        href=FinetuneState.generation_alt_download_url,
                        font_size="0.82rem",
                        color=c("accent"),
                    ),
                    spacing="2",
                    margin_top="4px",
                ),
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
            rx.cond(
                FinetuneState.dpo_column_error != "",
                rx.callout(
                    FinetuneState.dpo_column_error,
                    color_scheme="red",
                    size="1",
                ),
                rx.fragment(),
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
            _data_mode_btn("upload", "Upload file", "upload"),
            _seg_btn(
                "hub_dataset",
                _hf_icon(16),
                rx.text("HF Hub dataset", font_size="0.82rem", font_weight="500"),
                extra_event=FinetuneState.load_hub_recommendations,
            ),
            _data_mode_btn("generate", "Generate with AI", "sparkles"),
            spacing="0",
            padding="3px",
            border_radius="8px",
            border="1px solid var(--gray-5)",
            background="var(--gray-3)",
            margin_bottom="16px",
            display="inline-flex",
        ),
        rx.match(
            FinetuneState.data_source,
            ("upload", _upload_panel()),
            ("hub_dataset", _hub_dataset_panel()),
            ("generate", _generate_panel()),
            _upload_panel(),
        ),
        # Formatted template preview — shown once a dataset is loaded
        rx.cond(
            FinetuneState.dataset_row_count > 0,
            rx.vstack(
                rx.button(
                    rx.hstack(
                        rx.icon("eye", size=14),
                        rx.text("Preview formatted sample"),
                        spacing="2",
                        align="center",
                    ),
                    on_click=FinetuneState.preview_dataset_sample,
                    variant="ghost",
                    size="1",
                    color_scheme="blue",
                ),
                rx.cond(
                    FinetuneState.dataset_template_preview.length() > 0,
                    rx.vstack(
                        rx.text(
                            "Rows after template applied (what the trainer sees):",
                            font_size="0.76rem",
                            color=c("text_muted"),
                            font_weight="600",
                        ),
                        rx.foreach(
                            FinetuneState.dataset_template_preview,
                            lambda sample: rx.box(
                                rx.text(
                                    sample,
                                    font_size="0.75rem",
                                    color=c("text_secondary"),
                                    white_space="pre-wrap",
                                    font_family="monospace",
                                ),
                                background=c("bg_input"),
                                border_radius="8px",
                                padding="12px",
                                border="1px solid",
                                border_color=c("border"),
                                width="100%",
                            ),
                        ),
                        spacing="2",
                        width="100%",
                    ),
                    rx.fragment(),
                ),
                spacing="2",
                width="100%",
                align_items="flex-start",
            ),
            rx.fragment(),
        ),
        _nav_buttons(
            next_label="Next: Configure →",
            next_disabled=~FinetuneState.can_go_to_configure,
        ),
        spacing="0",
        width="100%",
        align_items="flex-start",
    )
