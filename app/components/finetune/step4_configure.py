"""Fine-tune wizard — Step 4: Hyperparameter configuration."""

from __future__ import annotations

import reflex as rx

from app.state.finetune_state import FinetuneState
from app.styles import c
from app.components.finetune.shared import _card, _label, _section_heading, _nav_buttons

_LR_PRESETS = [
    ("1e-4", "Slow & careful"),
    ("2e-4", "Balanced (recommended)"),
    ("5e-4", "Fast learning"),
]


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
                spacing="2", align="center",
            ),
        ),

        # Simple mode
        _card(
            rx.vstack(
                rx.grid(
                    rx.vstack(
                        _label("Epochs"),
                        rx.input(value=FinetuneState.epochs.to_string(),
                                 on_change=FinetuneState.set_epochs,
                                 type="number", width="100%"),
                        rx.text("One full pass through your dataset",
                                font_size="0.72rem", color=c("text_muted")),
                        spacing="1",
                    ),
                    rx.vstack(
                        _label("Learning rate"),
                        rx.select.root(
                            rx.select.trigger(width="100%"),
                            rx.select.content(
                                *[rx.select.item(f"{lr} — {desc}", value=lr)
                                  for lr, desc in _LR_PRESETS],
                            ),
                            value=FinetuneState.learning_rate,
                            on_change=FinetuneState.set_learning_rate,
                        ),
                        spacing="1",
                    ),
                    rx.vstack(
                        _label("Technique"),
                        rx.text(FinetuneState.technique_label, font_size="0.88rem",
                                font_weight="500", color=c("accent")),
                        rx.text("Change in Step 1", font_size="0.72rem", color=c("text_muted")),
                        spacing="1",
                    ),
                    columns="3", spacing="4", width="100%",
                ),
                spacing="0",
            )
        ),

        # Advanced mode
        rx.cond(
            FinetuneState.ui_mode == "advanced",
            _card(
                rx.vstack(
                    rx.text("Advanced hyperparameters", font_size="0.88rem", font_weight="600",
                            color=c("text_primary"), margin_bottom="12px"),
                    rx.grid(
                        rx.vstack(
                            _label("LoRA rank (r)"),
                            rx.slider(min=4, max=128, step=4,
                                      default_value=[FinetuneState.lora_r],
                                      on_value_commit=FinetuneState.set_lora_r),
                            rx.text(FinetuneState.lora_r, font_size="0.82rem",
                                    color=c("text_secondary")),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("LoRA alpha"),
                            rx.input(value=FinetuneState.lora_alpha.to_string(),
                                     on_change=FinetuneState.set_lora_alpha,
                                     type="number", width="100%"),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("LoRA dropout"),
                            rx.slider(min=0.0, max=0.3, step=0.01,
                                      default_value=[FinetuneState.lora_dropout],
                                      on_value_commit=FinetuneState.set_lora_dropout),
                            rx.text(FinetuneState.lora_dropout, font_size="0.82rem",
                                    color=c("text_secondary")),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Batch size"),
                            rx.select.root(
                                rx.select.trigger(width="100%"),
                                rx.select.content(
                                    *[rx.select.item(str(v), value=str(v))
                                      for v in [1, 2, 4, 8, 16]],
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
                                    *[rx.select.item(str(v), value=str(v))
                                      for v in [128, 256, 512, 1024, 2048]],
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
                                    *[rx.select.item(str(v), value=str(v))
                                      for v in [1, 2, 4, 8, 16]],
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
                                    *[rx.select.item(v, value=v)
                                      for v in ["cosine", "linear", "constant",
                                                "cosine_with_restarts"]],
                                ),
                                value=FinetuneState.lr_scheduler,
                                on_change=FinetuneState.set_lr_scheduler,
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("BF16 mode (A100/H100 only)"),
                            rx.switch(checked=FinetuneState.bf16,
                                      on_change=FinetuneState.set_bf16, size="2"),
                            rx.text("Better precision than FP16 on Ampere+ GPUs",
                                    font_size="0.72rem", color=c("text_muted")),
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
                        columns="3", spacing="4", width="100%",
                    ),
                    spacing="0",
                )
            ),
            rx.fragment(),
        ),

        # Run summary
        _card(
            rx.vstack(
                rx.text("Run summary", font_size="0.82rem", font_weight="600",
                        color=c("text_secondary"), margin_bottom="8px"),
                rx.grid(
                    rx.vstack(
                        rx.text("Model", font_size="0.72rem", color=c("text_muted")),
                        rx.text(FinetuneState.effective_model_name, font_size="0.84rem",
                                font_weight="500", color=c("text_primary")),
                        spacing="0",
                    ),
                    rx.vstack(
                        rx.text("Dataset", font_size="0.72rem", color=c("text_muted")),
                        rx.text(FinetuneState.dataset_name, font_size="0.84rem",
                                font_weight="500", color=c("text_primary")),
                        spacing="0",
                    ),
                    rx.vstack(
                        rx.text("Technique", font_size="0.72rem", color=c("text_muted")),
                        rx.text(FinetuneState.technique_label, font_size="0.84rem",
                                font_weight="500", color=c("text_primary")),
                        spacing="0",
                    ),
                    rx.vstack(
                        rx.text("Training", font_size="0.72rem", color=c("text_muted")),
                        rx.text(
                            FinetuneState.epochs.to_string() + " epochs · lr=" +
                            FinetuneState.learning_rate + " · batch=" +
                            FinetuneState.batch_size.to_string(),
                            font_size="0.82rem", font_weight="500", color=c("text_primary"),
                        ),
                        spacing="0",
                    ),
                    columns="2", spacing="4", width="100%",
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
        spacing="4", width="100%", align_items="flex-start",
    )
