"""TuneOS — Fine-tuning wizard page (/finetune)."""

from __future__ import annotations

import reflex as rx

from app.components.loss_chart import loss_chart
from app.state.finetune_state import FinetuneState
from app.state.job_state import JobState
from app.styles import c

# ── Supported models ─────────────────────────────────────────────
_MODELS = [
    {
        "id": "mistralai/Mistral-7B-v0.1",
        "name": "Mistral 7B",
        "size": "7B params",
        "notes": "Primary target, well-tested with QLoRA",
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
]


# ── Shared helpers ────────────────────────────────────────────────
def _card(
    *children,
    padding: str = "20px",
    width: str = "100%",
    **props,
) -> rx.Component:
    return rx.box(
        *children,
        background=c("bg_card"),
        border="1px solid",
        border_color=c("border"),
        border_radius="12px",
        padding=padding,
        width=width,
        **props,
    )


def _label(text: str) -> rx.Component:
    return rx.text(
        text,
        font_size="0.82rem",
        font_weight="500",
        color=c("text_secondary"),
        margin_bottom="6px",
    )


def _section_heading(text: str) -> rx.Component:
    return rx.text(
        text,
        font_size="1.05rem",
        font_weight="600",
        color=c("text_primary"),
        margin_bottom="16px",
    )


# ── Progress bar ─────────────────────────────────────────────────
_STEP_LABELS = ["Model", "Dataset", "Configure", "Train", "Results"]


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
                    font_size="0.75rem",
                    font_weight="600",
                    color=rx.cond(is_active, "white", c("text_muted")),
                ),
            ),
            width="28px",
            height="28px",
            border_radius="50%",
            background=rx.cond(
                is_done,
                c("success"),
                rx.cond(is_active, c("accent"), c("bg_input")),
            ),
            border="2px solid",
            border_color=rx.cond(
                is_active | is_done, c("accent"), c("border")
            ),
            display="flex",
            align_items="center",
            justify_content="center",
        ),
        rx.text(
            _STEP_LABELS[index],
            font_size="0.75rem",
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
                    min_width="40px",
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
        max_width="600px",
        align="center",
        justify="center",
        margin_bottom="32px",
    )


# ── Step 1: Model + Technique ─────────────────────────────────────
def _model_card(m: dict) -> rx.Component:
    is_selected = FinetuneState.selected_model_id == m["id"]
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(
                    m["name"],
                    font_size="0.95rem",
                    font_weight="600",
                    color=c("text_primary"),
                ),
                rx.cond(
                    m["token_required"],
                    rx.badge("HF Token", color_scheme="orange", size="1"),
                    rx.fragment(),
                ),
                justify="between",
                width="100%",
            ),
            rx.text(m["size"], font_size="0.8rem", color=c("text_secondary")),
            rx.text(m["notes"], font_size="0.82rem", color=c("text_muted")),
            spacing="1",
            align_items="flex-start",
            width="100%",
        ),
        background=rx.cond(is_selected, c("accent_soft"), c("bg_card")),
        border="2px solid",
        border_color=rx.cond(is_selected, c("accent"), c("border")),
        border_radius="10px",
        padding="16px",
        cursor="pointer",
        width="100%",
        on_click=FinetuneState.select_model(m["id"], m["name"]),
        _hover={"border_color": c("accent"), "background": c("accent_soft")},
    )


def _technique_btn(technique: str, label: str, description: str, coming_soon: bool = False) -> rx.Component:
    is_active = FinetuneState.selected_technique == technique
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(
                    label,
                    font_size="0.88rem",
                    font_weight="500",
                    color=rx.cond(
                        coming_soon,
                        c("text_muted"),
                        rx.cond(is_active, c("accent"), c("text_primary")),
                    ),
                ),
                rx.cond(
                    coming_soon,
                    rx.badge("Soon", color_scheme="gray", size="1"),
                    rx.cond(
                        is_active,
                        rx.icon("check-circle", size=14, color=c("accent")),
                        rx.fragment(),
                    ),
                ),
                spacing="2",
                align="center",
            ),
            rx.text(
                description,
                font_size="0.78rem",
                color=c("text_muted"),
            ),
            spacing="1",
            align_items="flex-start",
        ),
        background=rx.cond(
            is_active & ~coming_soon, c("accent_soft"), c("bg_input")
        ),
        border="1px solid",
        border_color=rx.cond(
            is_active & ~coming_soon, c("accent"), c("border")
        ),
        border_radius="8px",
        padding="12px 14px",
        cursor=rx.cond(coming_soon, "not-allowed", "pointer"),
        opacity=rx.cond(coming_soon, "0.5", "1"),
        on_click=rx.cond(
            coming_soon,
            rx.prevent_default,
            FinetuneState.select_technique(technique),
        ),
        flex="1",
        min_width="160px",
    )


def _step1() -> rx.Component:
    return rx.vstack(
        _section_heading("Pick a base model"),
        rx.text(
            "Choose the model you want to fine-tune. This is the starting point — your dataset will teach it a new skill.",
            font_size="0.88rem",
            color=c("text_secondary"),
            margin_bottom="16px",
        ),
        rx.grid(
            *[_model_card(m) for m in _MODELS],
            columns="2",
            spacing="3",
            width="100%",
        ),
        rx.box(height="24px"),
        _section_heading("Choose technique"),
        rx.text(
            "The technique determines how the model weights are updated during training.",
            font_size="0.88rem",
            color=c("text_secondary"),
            margin_bottom="12px",
        ),
        rx.flex(
            _technique_btn(
                "qlora",
                "QLoRA",
                "Compressed mode. Works on 12 GB+ GPU.",
            ),
            _technique_btn(
                "lora",
                "LoRA",
                "Float16 mode. Needs ~16 GB GPU.",
            ),
            _technique_btn(
                "full",
                "Full Fine-tune",
                "All weights updated. Needs 80 GB+ GPU.",
                coming_soon=True,
            ),
            _technique_btn(
                "dpo",
                "DPO",
                "Preference tuning for alignment.",
                coming_soon=True,
            ),
            wrap="wrap",
            gap="10px",
            width="100%",
        ),
        rx.box(height="24px"),
        rx.hstack(
            rx.button(
                "Next: Upload Dataset →",
                on_click=FinetuneState.next_step,
                disabled=~FinetuneState.can_go_to_dataset,
                size="3",
                color_scheme="blue",
            ),
            justify="end",
            width="100%",
        ),
        spacing="0",
        width="100%",
        align_items="flex-start",
    )


# ── Step 2: Dataset ───────────────────────────────────────────────
def _preview_row(row: dict) -> rx.Component:
    return rx.table.row(
        rx.table.cell(
            rx.text(
                row["instruction"],
                font_size="0.8rem",
                color=c("text_primary"),
                white_space="nowrap",
                overflow="hidden",
                text_overflow="ellipsis",
                max_width="320px",
            )
        ),
        rx.table.cell(
            rx.text(
                row["output"],
                font_size="0.8rem",
                color=c("text_secondary"),
                white_space="nowrap",
                overflow="hidden",
                text_overflow="ellipsis",
                max_width="260px",
            )
        ),
    )


def _step2() -> rx.Component:
    return rx.vstack(
        _section_heading("Upload your dataset"),
        rx.text(
            "Your dataset teaches the model the new skill. Each row needs an 'instruction' and an 'output'.",
            font_size="0.88rem",
            color=c("text_secondary"),
            margin_bottom="16px",
        ),
        # Upload dropzone
        _card(
            rx.vstack(
                _label("Upload new file (.jsonl, .json, .csv)"),
                rx.upload(
                    rx.vstack(
                        rx.icon("upload", size=24, color=c("text_muted")),
                        rx.text(
                            "Drag & drop or click to select",
                            font_size="0.88rem",
                            color=c("text_secondary"),
                        ),
                        rx.text(
                            "Required columns: instruction, output",
                            font_size="0.78rem",
                            color=c("text_muted"),
                        ),
                        spacing="2",
                        align="center",
                    ),
                    id="finetune_dataset_upload",
                    multiple=False,
                    accept={
                        "application/json": [".jsonl", ".json"],
                        "text/csv": [".csv"],
                    },
                    max_files=1,
                    padding="24px",
                    border="2px dashed",
                    border_color=c("border"),
                    border_radius="8px",
                    width="100%",
                    cursor="pointer",
                    _hover={"border_color": c("accent")},
                ),
                rx.button(
                    rx.cond(
                        FinetuneState.is_uploading,
                        rx.hstack(rx.spinner(size="2"), rx.text("Uploading..."), spacing="2"),
                        rx.text("Upload"),
                    ),
                    on_click=FinetuneState.handle_dataset_upload(
                        rx.upload_files(upload_id="finetune_dataset_upload")
                    ),
                    color_scheme="blue",
                    variant="soft",
                    size="2",
                    disabled=FinetuneState.is_uploading,
                ),
                spacing="3",
                align_items="flex-start",
                width="100%",
            )
        ),
        # Existing datasets
        rx.cond(
            FinetuneState.existing_datasets.length() > 0,
            _card(
                rx.vstack(
                    _label("Or reuse an existing dataset"),
                    rx.flex(
                        rx.foreach(
                            FinetuneState.existing_datasets,
                            lambda f: rx.box(
                                rx.text(f, font_size="0.82rem", color=c("text_primary")),
                                background=rx.cond(
                                    FinetuneState.dataset_filename == f,
                                    c("accent_soft"),
                                    c("bg_input"),
                                ),
                                border="1px solid",
                                border_color=rx.cond(
                                    FinetuneState.dataset_filename == f,
                                    c("accent"),
                                    c("border"),
                                ),
                                border_radius="6px",
                                padding="6px 12px",
                                cursor="pointer",
                                on_click=FinetuneState.select_existing_dataset(f),
                                _hover={"border_color": c("accent")},
                            ),
                        ),
                        wrap="wrap",
                        gap="8px",
                    ),
                    spacing="2",
                    width="100%",
                    align_items="flex-start",
                )
            ),
            rx.fragment(),
        ),
        # Validation error
        rx.cond(
            FinetuneState.dataset_error != "",
            rx.callout(
                FinetuneState.dataset_error,
                icon="triangle-alert",
                color_scheme="red",
                width="100%",
            ),
            rx.fragment(),
        ),
        # Preview table
        rx.cond(
            FinetuneState.dataset_preview.length() > 0,
            _card(
                rx.vstack(
                    _label("Preview (first 5 rows)"),
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                rx.table.column_header_cell("instruction"),
                                rx.table.column_header_cell("output"),
                            )
                        ),
                        rx.table.body(
                            rx.foreach(FinetuneState.dataset_preview, _preview_row)
                        ),
                    ),
                    spacing="2",
                    width="100%",
                    overflow_x="auto",
                )
            ),
            rx.fragment(),
        ),
        # Navigation
        rx.box(height="8px"),
        rx.hstack(
            rx.button(
                "← Back",
                on_click=FinetuneState.prev_step,
                variant="soft",
                color_scheme="gray",
                size="3",
            ),
            rx.button(
                "Next: Configure →",
                on_click=FinetuneState.next_step,
                disabled=~FinetuneState.can_go_to_configure,
                size="3",
                color_scheme="blue",
            ),
            justify="between",
            width="100%",
        ),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


# ── Step 3: Configure ─────────────────────────────────────────────
def _slider_field(label: str, hint: str, value: rx.Var, min_val: int, max_val: int, on_change) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            _label(label),
            rx.text(value, font_size="0.82rem", font_weight="600", color=c("accent")),
            justify="between",
            width="100%",
        ),
        rx.slider(
            default_value=value,
            min=min_val,
            max=max_val,
            step=1,
            on_change=on_change,
            color_scheme="blue",
            width="100%",
        ),
        rx.text(hint, font_size="0.75rem", color=c("text_muted")),
        spacing="1",
        width="100%",
        align_items="flex-start",
    )


def _step3() -> rx.Component:
    return rx.vstack(
        # Config pill
        rx.hstack(
            rx.badge(
                FinetuneState.selected_model_name,
                color_scheme="blue",
                size="2",
            ),
            rx.badge(
                FinetuneState.technique_label,
                color_scheme="green",
                size="2",
            ),
            spacing="2",
            margin_bottom="8px",
        ),
        _section_heading("Configure training"),
        rx.text(
            "The defaults work well for most cases. Adjust if you have specific needs.",
            font_size="0.88rem",
            color=c("text_secondary"),
            margin_bottom="16px",
        ),
        # LoRA params
        _card(
            rx.vstack(
                _label("LoRA Parameters"),
                _slider_field(
                    "Rank (r)",
                    "Controls adapter size. 16 is a good default.",
                    FinetuneState.lora_r,
                    4, 64,
                    FinetuneState.set_lora_r,
                ),
                _slider_field(
                    "Alpha",
                    "Scaling factor. Usually set to 2× rank.",
                    FinetuneState.lora_alpha,
                    8, 128,
                    FinetuneState.set_lora_alpha,
                ),
                spacing="4",
                width="100%",
            )
        ),
        # Training params
        _card(
            rx.vstack(
                _label("Training Parameters"),
                rx.grid(
                    rx.vstack(
                        _label("Epochs"),
                        rx.input(
                            value=FinetuneState.epochs.to_string(),
                            on_change=FinetuneState.set_epochs,
                            type="number",
                            min="1",
                            max="20",
                            width="100%",
                            background=c("bg_input"),
                            border_color=c("border"),
                            color=c("text_primary"),
                        ),
                        rx.text("How many times to train on your full dataset.", font_size="0.75rem", color=c("text_muted")),
                        spacing="1",
                        align_items="flex-start",
                    ),
                    rx.vstack(
                        _label("Learning Rate"),
                        rx.select(
                            ["1e-4", "2e-4", "5e-4"],
                            value=FinetuneState.learning_rate,
                            on_change=FinetuneState.set_learning_rate,
                            width="100%",
                        ),
                        rx.text("How fast the model adapts. 2e-4 is standard.", font_size="0.75rem", color=c("text_muted")),
                        spacing="1",
                        align_items="flex-start",
                    ),
                    rx.vstack(
                        _label("Batch Size"),
                        rx.select(
                            ["1", "2", "4", "8"],
                            value=FinetuneState.batch_size.to_string(),
                            on_change=FinetuneState.set_batch_size,
                            width="100%",
                        ),
                        rx.text("Samples processed per step. Lower = less VRAM.", font_size="0.75rem", color=c("text_muted")),
                        spacing="1",
                        align_items="flex-start",
                    ),
                    rx.vstack(
                        _label("Max Sequence Length"),
                        rx.select(
                            ["256", "512", "1024", "2048"],
                            value=FinetuneState.max_seq_length.to_string(),
                            on_change=FinetuneState.set_max_seq_length,
                            width="100%",
                        ),
                        rx.text("Max tokens per sample. 512 fits most use cases.", font_size="0.75rem", color=c("text_muted")),
                        spacing="1",
                        align_items="flex-start",
                    ),
                    columns="2",
                    spacing="4",
                    width="100%",
                ),
                spacing="3",
                width="100%",
            )
        ),
        # Navigation
        rx.hstack(
            rx.button("← Back", on_click=FinetuneState.prev_step, variant="soft", color_scheme="gray", size="3"),
            rx.button(
                rx.cond(
                    FinetuneState.is_starting,
                    rx.hstack(rx.spinner(size="2"), rx.text("Starting..."), spacing="2"),
                    rx.text("Start Training"),
                ),
                on_click=FinetuneState.start_training,
                disabled=FinetuneState.is_starting | ~FinetuneState.can_start_training,
                size="3",
                color_scheme="blue",
            ),
            justify="between",
            width="100%",
        ),
        rx.cond(
            FinetuneState.start_error != "",
            rx.callout(FinetuneState.start_error, icon="triangle-alert", color_scheme="red"),
            rx.fragment(),
        ),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


# ── Step 4: Training progress ─────────────────────────────────────
def _status_badge() -> rx.Component:
    return rx.match(
        JobState.status,
        ("running", rx.badge("Running", color_scheme="yellow", size="2")),
        ("done", rx.badge("Complete", color_scheme="green", size="2")),
        ("failed", rx.badge("Failed", color_scheme="red", size="2")),
        ("cancelled", rx.badge("Cancelled", color_scheme="gray", size="2")),
        rx.badge("Queued", color_scheme="blue", size="2"),
    )


def _log_entry(entry: dict) -> rx.Component:
    return rx.text(
        rx.el.span(f"[step {entry['step']}]", color=c("text_muted")),
        rx.el.span(f"  loss={entry['loss']}  epoch={entry['epoch']}"),
        font_size="0.78rem",
        font_family="monospace",
        color=c("text_secondary"),
        white_space="nowrap",
    )


def _step4() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            _section_heading("Training in progress"),
            _status_badge(),
            justify="between",
            width="100%",
            align="center",
        ),
        rx.text(
            "Your model is learning. This typically takes 10–60 minutes depending on dataset size and GPU.",
            font_size="0.88rem",
            color=c("text_secondary"),
            margin_bottom="8px",
        ),
        # Loss chart
        _card(
            rx.vstack(
                _label("Loss curve"),
                rx.cond(
                    JobState.loss_history.length() > 0,
                    loss_chart(),
                    rx.box(
                        rx.text("Waiting for first step...", font_size="0.85rem", color=c("text_muted")),
                        height="120px",
                        display="flex",
                        align_items="center",
                        justify_content="center",
                    ),
                ),
                spacing="2",
                width="100%",
            )
        ),
        # Live log stream
        _card(
            rx.vstack(
                _label("Live log"),
                rx.box(
                    rx.cond(
                        JobState.loss_history.length() > 0,
                        rx.vstack(
                            rx.foreach(JobState.loss_history, _log_entry),
                            spacing="0",
                            width="100%",
                            align_items="flex-start",
                        ),
                        rx.text("No logs yet...", font_size="0.78rem", color=c("text_muted"), font_family="monospace"),
                    ),
                    height="160px",
                    overflow_y="auto",
                    width="100%",
                    padding="12px",
                    background=c("bg_input"),
                    border_radius="6px",
                    border="1px solid",
                    border_color=c("border"),
                ),
                spacing="2",
                width="100%",
            )
        ),
        # Action row
        rx.hstack(
            rx.button(
                "Stop Training",
                on_click=FinetuneState.prev_step,
                variant="soft",
                color_scheme="red",
                size="3",
            ),
            rx.cond(
                JobState.status == "done",
                rx.button(
                    "View Results →",
                    on_click=[
                        FinetuneState.go_to_step(5),
                        FinetuneState.run_eval,
                    ],
                    size="3",
                    color_scheme="green",
                ),
                rx.cond(
                    JobState.status == "failed",
                    rx.callout(
                        JobState.error_msg,
                        icon="triangle-alert",
                        color_scheme="red",
                    ),
                    rx.fragment(),
                ),
            ),
            justify="between",
            width="100%",
            align="center",
        ),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


# ── Step 5: Results ───────────────────────────────────────────────
def _result_card(title: str, icon: str, *children) -> rx.Component:
    return _card(
        rx.vstack(
            rx.hstack(
                rx.icon(icon, size=18, color=c("accent")),
                rx.text(title, font_size="0.95rem", font_weight="600", color=c("text_primary")),
                spacing="2",
                align="center",
            ),
            rx.divider(margin_y="10px"),
            *children,
            spacing="3",
            width="100%",
            align_items="flex-start",
        )
    )


def _step5() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.icon("party-popper", size=22, color=c("success")),
            _section_heading("Training complete!"),
            spacing="2",
            align="center",
            margin_bottom="4px",
        ),
        rx.text(
            "Your fine-tuned adapter is ready. Choose what to do with it below.",
            font_size="0.88rem",
            color=c("text_secondary"),
            margin_bottom="16px",
        ),
        rx.grid(
            # Card 1: Download
            _result_card(
                "Download Adapter",
                "download",
                rx.text(
                    "Download the LoRA adapter weights as a zip file (~50 MB). Use them locally with the base model.",
                    font_size="0.82rem",
                    color=c("text_secondary"),
                ),
                rx.button(
                    rx.hstack(rx.icon("download", size=16), rx.text("Download adapter.zip"), spacing="2"),
                    on_click=FinetuneState.download_adapter,
                    color_scheme="blue",
                    variant="soft",
                    size="2",
                ),
            ),
            # Card 2: Push to HF Hub
            _result_card(
                "Push to Hugging Face Hub",
                "upload-cloud",
                rx.text(
                    "Publish your adapter to your HF account as a private repo.",
                    font_size="0.82rem",
                    color=c("text_secondary"),
                ),
                rx.vstack(
                    rx.input(
                        placeholder="username/my-adapter",
                        value=FinetuneState.hf_repo_name,
                        on_change=FinetuneState.set_hf_repo_name,
                        width="100%",
                        background=c("bg_input"),
                        border_color=c("border"),
                        color=c("text_primary"),
                    ),
                    rx.input(
                        placeholder="HF token (hf_...)",
                        value=FinetuneState.hf_token_input,
                        on_change=FinetuneState.set_hf_token_input,
                        type="password",
                        width="100%",
                        background=c("bg_input"),
                        border_color=c("border"),
                        color=c("text_primary"),
                    ),
                    rx.button(
                        rx.cond(
                            FinetuneState.push_status == "pushing",
                            rx.hstack(rx.spinner(size="2"), rx.text("Pushing..."), spacing="2"),
                            rx.text("Push to Hub"),
                        ),
                        on_click=FinetuneState.push_to_hub,
                        disabled=(FinetuneState.push_status == "pushing") | (FinetuneState.hf_repo_name == ""),
                        color_scheme="blue",
                        variant="soft",
                        size="2",
                    ),
                    rx.cond(
                        FinetuneState.push_status == "done",
                        rx.hstack(
                            rx.icon("check-circle", size=14, color=c("success")),
                            rx.text(
                                FinetuneState.push_repo_url,
                                font_size="0.78rem",
                                color=c("success"),
                            ),
                            spacing="1",
                        ),
                        rx.cond(
                            FinetuneState.push_error != "",
                            rx.text(FinetuneState.push_error, font_size="0.78rem", color=c("error")),
                            rx.fragment(),
                        ),
                    ),
                    spacing="2",
                    width="100%",
                ),
            ),
            # Card 3: Evaluation
            _result_card(
                "Evaluation Metrics",
                "chart-bar",
                rx.match(
                    FinetuneState.eval_status,
                    ("running", rx.hstack(rx.spinner(size="2"), rx.text("Computing perplexity...", font_size="0.82rem", color=c("text_secondary")), spacing="2")),
                    ("done",
                        rx.vstack(
                            rx.hstack(
                                rx.text("Perplexity", font_size="0.82rem", color=c("text_secondary")),
                                rx.text(
                                    FinetuneState.eval_perplexity.to_string(),
                                    font_size="1.4rem",
                                    font_weight="700",
                                    color=c("accent"),
                                ),
                                justify="between",
                                width="100%",
                            ),
                            rx.text(
                                "Lower is better. Under 10 = good domain fit.",
                                font_size="0.75rem",
                                color=c("text_muted"),
                            ),
                            spacing="1",
                            width="100%",
                        )
                    ),
                    ("not_ready", rx.text("Metrics not available for this job.", font_size="0.82rem", color=c("text_muted"))),
                    rx.text("Waiting for eval results...", font_size="0.82rem", color=c("text_muted")),
                ),
            ),
            # Card 4: Test in chat
            _result_card(
                "Test in Chat",
                "message-circle",
                rx.text(
                    "Try a prompt to see how your fine-tuned model responds. First call loads the model (~60s).",
                    font_size="0.82rem",
                    color=c("text_secondary"),
                ),
                rx.vstack(
                    rx.text_area(
                        placeholder="Enter a prompt to test your model...",
                        value=FinetuneState.chat_input,
                        on_change=FinetuneState.set_chat_input,
                        width="100%",
                        rows="3",
                        background=c("bg_input"),
                        border_color=c("border"),
                        color=c("text_primary"),
                        resize="vertical",
                    ),
                    rx.button(
                        rx.cond(
                            FinetuneState.chat_loading,
                            rx.hstack(rx.spinner(size="2"), rx.text("Generating..."), spacing="2"),
                            rx.text("Generate"),
                        ),
                        on_click=FinetuneState.send_test_chat,
                        disabled=FinetuneState.chat_loading | (FinetuneState.chat_input == ""),
                        color_scheme="blue",
                        variant="soft",
                        size="2",
                    ),
                    rx.cond(
                        FinetuneState.chat_response != "",
                        rx.box(
                            rx.text(
                                FinetuneState.chat_response,
                                font_size="0.85rem",
                                color=c("text_primary"),
                                white_space="pre-wrap",
                            ),
                            padding="12px",
                            background=c("bg_input"),
                            border_radius="6px",
                            border="1px solid",
                            border_color=c("border"),
                            width="100%",
                        ),
                        rx.cond(
                            FinetuneState.chat_error != "",
                            rx.text(FinetuneState.chat_error, font_size="0.78rem", color=c("error")),
                            rx.fragment(),
                        ),
                    ),
                    spacing="2",
                    width="100%",
                ),
            ),
            columns="2",
            spacing="4",
            width="100%",
        ),
        # Start another
        rx.box(height="8px"),
        rx.hstack(
            rx.button(
                "Fine-tune another model",
                on_click=FinetuneState.go_to_step(1),
                variant="soft",
                color_scheme="gray",
                size="3",
            ),
            justify="start",
            width="100%",
        ),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


# ── Page root ────────────────────────────────────────────────────
def finetune_page() -> rx.Component:
    return rx.box(
        rx.vstack(
            # Header
            rx.hstack(
                rx.icon("flask-conical", size=20, color=c("accent")),
                rx.text(
                    "Fine-tune a model",
                    font_size="1.25rem",
                    font_weight="600",
                    color=c("text_primary"),
                ),
                spacing="3",
                align="center",
                margin_bottom="8px",
            ),
            _progress_bar(),
            # Step body
            rx.match(
                FinetuneState.current_step,
                (1, _step1()),
                (2, _step2()),
                (3, _step3()),
                (4, _step4()),
                (5, _step5()),
                rx.text("Unknown step", color=c("text_muted")),
            ),
            spacing="0",
            width="100%",
            max_width="900px",
            align_items="flex-start",
        ),
        padding="40px",
        min_height="100vh",
        background=c("bg_primary"),
        on_mount=FinetuneState.load_existing_datasets,
    )
