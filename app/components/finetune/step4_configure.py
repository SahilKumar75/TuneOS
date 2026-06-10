"""Fine-tune wizard — Step 4: Hyperparameter configuration."""

from __future__ import annotations

import reflex as rx

from app.components.finetune.shared import _card, _label, _nav_buttons, _section_heading
from app.state.finetune_state import FinetuneState
from app.state.training_poller_state import TrainingPollerState
from app.styles import c

_LR_PRESETS = [
    ("1e-4", "Conservative — small datasets / sensitive tasks"),
    ("2e-4", "Balanced — good for most datasets (recommended)"),
    ("5e-4", "Aggressive — large datasets, fast convergence"),
]

_COMPUTE_BACKENDS = [
    ("local", "Local GPU", "Train on this machine — uses your detected device"),
    (
        "modal",
        "Modal",
        "Free T4 cloud GPU (~$30/mo credits) — needs MODAL_TOKEN_ID + MODAL_TOKEN_SECRET",
    ),
    ("hf_spaces", "HF Spaces", "ZeroGPU A100 — 30 min/job limit when deployed there"),
]


def _compute_option(value: str, title: str, desc: str) -> rx.Component:
    selected = FinetuneState.compute_backend == value
    return rx.box(
        rx.vstack(
            rx.text(title, font_size="0.86rem", font_weight="600", color=c("text_primary")),
            rx.text(desc, font_size="0.72rem", color=c("text_muted")),
            spacing="1",
            align_items="flex-start",
        ),
        on_click=FinetuneState.set_compute_backend(value),
        cursor="pointer",
        padding="12px 14px",
        border_radius="10px",
        border=rx.cond(selected, f"1.5px solid {c('accent')}", f"1px solid {c('border')}"),
        background=rx.cond(selected, c("bg_input"), "transparent"),
        flex="1",
        transition="border 0.15s ease",
    )


def _compute_section() -> rx.Component:
    return _card(
        rx.vstack(
            rx.text(
                "Compute backend",
                font_size="0.82rem",
                font_weight="600",
                color=c("text_secondary"),
                margin_bottom="8px",
            ),
            rx.hstack(
                *[_compute_option(v, t, d) for v, t, d in _COMPUTE_BACKENDS],
                spacing="3",
                width="100%",
                align_items="stretch",
            ),
            spacing="0",
            width="100%",
        )
    )


def _step4() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            _section_heading("Training configuration"),
            rx.spacer(),
            rx.hstack(
                rx.text("Simple", font_size="0.82rem", color=c("text_secondary")),
                rx.switch(
                    checked=FinetuneState.ui_mode == "advanced",
                    on_change=lambda v: FinetuneState.set_ui_mode(rx.cond(v, "advanced", "simple")),
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
                        _label("Batch size"),
                        rx.select.root(
                            rx.select.trigger(width="100%"),
                            rx.select.content(
                                *[rx.select.item(str(v), value=str(v)) for v in [1, 2, 4, 8]],
                            ),
                            value=FinetuneState.batch_size.to_string(),
                            on_change=FinetuneState.set_batch_size,
                        ),
                        rx.text("Samples per GPU step", font_size="0.72rem", color=c("text_muted")),
                        spacing="1",
                    ),
                    rx.vstack(
                        _label("Max sequence length"),
                        rx.select.root(
                            rx.select.trigger(width="100%"),
                            rx.select.content(
                                *[
                                    rx.select.item(str(v), value=str(v))
                                    for v in [256, 512, 1024, 2048]
                                ],
                            ),
                            value=FinetuneState.max_seq_length.to_string(),
                            on_change=FinetuneState.set_max_seq_length,
                        ),
                        rx.text("Tokens per sample", font_size="0.72rem", color=c("text_muted")),
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
                        rx.button(
                            "← Edit in Step 1",
                            on_click=FinetuneState.go_to_step(1),
                            variant="ghost",
                            size="1",
                            color_scheme="blue",
                            padding="0",
                            height="auto",
                        ),
                        spacing="1",
                    ),
                    columns="2",
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
                    # Section 1 — LoRA
                    rx.text(
                        "LoRA Adapter",
                        font_size="0.78rem",
                        font_weight="600",
                        color=c("text_secondary"),
                        margin_top="8px",
                    ),
                    rx.divider(margin_y="6px"),
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
                        columns="3",
                        spacing="4",
                        width="100%",
                    ),
                    # Section 2 — Batch & Memory
                    rx.text(
                        "Batch & Memory",
                        font_size="0.78rem",
                        font_weight="600",
                        color=c("text_secondary"),
                        margin_top="16px",
                    ),
                    rx.divider(margin_y="6px"),
                    rx.grid(
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
                        columns="3",
                        spacing="4",
                        width="100%",
                    ),
                    # Section 2b — Data formatting
                    rx.text(
                        "Data formatting",
                        font_size="0.78rem",
                        font_weight="600",
                        color=c("text_secondary"),
                        margin_top="16px",
                    ),
                    rx.divider(margin_y="6px"),
                    rx.grid(
                        rx.vstack(
                            _label("Prompt template"),
                            rx.select.root(
                                rx.select.trigger(width="100%"),
                                rx.select.content(
                                    *[
                                        rx.select.item(v, value=v)
                                        for v in ["alpaca", "chatml", "llama3", "phi3", "zephyr"]
                                    ],
                                ),
                                value=FinetuneState.prompt_template,
                                on_change=FinetuneState.set_prompt_template,
                            ),
                            rx.text(
                                "How prompts are wrapped for the model",
                                font_size="0.72rem",
                                color=c("text_muted"),
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Sample packing"),
                            rx.switch(
                                checked=FinetuneState.packing,
                                on_change=FinetuneState.set_packing,
                                size="2",
                            ),
                            rx.text(
                                "Concatenate examples to fill the sequence — faster on GPU",
                                font_size="0.72rem",
                                color=c("text_muted"),
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Auto-detect all linear layers (recommended)"),
                            rx.switch(
                                checked=FinetuneState.use_all_linear,
                                on_change=FinetuneState.set_use_all_linear,
                                size="2",
                            ),
                            rx.text(
                                "Passes target_modules='all-linear' to PEFT — no architecture map needed",
                                font_size="0.72rem",
                                color=c("text_muted"),
                            ),
                            spacing="1",
                        ),
                        columns="3",
                        spacing="4",
                        width="100%",
                    ),
                    # Section 3 — Scheduler & Tracking
                    rx.text(
                        "Scheduler & Tracking",
                        font_size="0.78rem",
                        font_weight="600",
                        color=c("text_secondary"),
                        margin_top="16px",
                    ),
                    rx.divider(margin_y="6px"),
                    rx.grid(
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
                            _label("Experiment name"),
                            rx.input(
                                placeholder="my-run-1",
                                value=FinetuneState.experiment_name,
                                on_change=FinetuneState.set_experiment_name,
                                width="100%",
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Eval split ratio"),
                            rx.slider(
                                min=0.0,
                                max=0.3,
                                step=0.05,
                                default_value=[FinetuneState.eval_split_ratio],
                                on_value_commit=FinetuneState.set_eval_split_ratio,
                            ),
                            rx.text(
                                FinetuneState.eval_split_ratio.to_string(),
                                font_size="0.82rem",
                                color=c("text_secondary"),
                            ),
                            rx.text(
                                "Fraction held out for validation",
                                font_size="0.72rem",
                                color=c("text_muted"),
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Early stopping patience"),
                            rx.input(
                                value=FinetuneState.early_stopping_patience.to_string(),
                                on_change=FinetuneState.set_early_stopping_patience,
                                type="number",
                                width="100%",
                            ),
                            rx.text("0 = disabled", font_size="0.72rem", color=c("text_muted")),
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
        # VRAM warning
        rx.cond(
            (FinetuneState.batch_size * FinetuneState.max_seq_length) > 4096,
            rx.callout(
                "⚠ High memory config — estimated >12GB VRAM. Reduce batch size or sequence length if you hit OOM.",
                color_scheme="orange",
                size="1",
            ),
            rx.fragment(),
        ),
        # Compute backend selector
        _compute_section(),
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
            next_event=TrainingPollerState.start_training,
        ),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )
