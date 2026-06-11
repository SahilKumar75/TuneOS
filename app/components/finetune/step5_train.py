"""Fine-tune wizard — Step 5: Training dashboard."""

from __future__ import annotations

import reflex as rx

from app.components.finetune.shared import _badge_status, _card
from app.components.loss_chart import loss_chart
from app.state.training_poller_state import TrainingPollerState
from app.styles import c


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


def step5_train() -> rx.Component:
    """Linear wizard layout for Step 5 (training dashboard)."""
    return rx.vstack(
        rx.hstack(
            rx.vstack(
                rx.text("Training", font_size="1.1rem", font_weight="700", color=c("text_primary")),
                rx.text(
                    TrainingPollerState.effective_model_name,
                    font_size="0.82rem",
                    color=c("text_muted"),
                ),
                spacing="0",
            ),
            rx.spacer(),
            _badge_status(TrainingPollerState.training_status),
            align="center",
        ),
        rx.cond(
            TrainingPollerState.start_error != "",
            rx.callout(TrainingPollerState.start_error, color_scheme="red"),
            rx.fragment(),
        ),
        rx.grid(
            _metric_tile("Epoch", TrainingPollerState.current_epoch_display),
            _metric_tile("Steps", TrainingPollerState.current_total_steps_display),
            _metric_tile("Elapsed", TrainingPollerState.elapsed_time_display),
            _metric_tile("GPU Memory", TrainingPollerState.gpu_memory_display),
            columns="4",
            spacing="3",
            width="100%",
        ),
        rx.vstack(
            rx.hstack(
                rx.text("Epoch progress", font_size="0.76rem", color=c("text_muted")),
                rx.spacer(),
                rx.text(
                    TrainingPollerState.epoch_progress_pct.to_string() + "%",
                    font_size="0.76rem",
                    color=c("text_secondary"),
                ),
            ),
            rx.progress(
                value=TrainingPollerState.epoch_progress_pct,
                max=100,
                width="100%",
                color_scheme="blue",
            ),
            width="100%",
            spacing="1",
        ),
        _card(loss_chart()),
        rx.cond(
            TrainingPollerState.ai_commentary != "",
            _card(
                rx.hstack(
                    rx.icon("sparkles", size=16, color=c("accent")),
                    rx.text(
                        TrainingPollerState.ai_commentary,
                        font_size="0.86rem",
                        color=c("text_primary"),
                    ),
                    spacing="2",
                    align="start",
                )
            ),
            rx.fragment(),
        ),
        rx.cond(
            TrainingPollerState.epoch_log.length() > 0,
            _card(
                rx.vstack(
                    rx.text(
                        "Epoch log",
                        font_size="0.78rem",
                        font_weight="600",
                        color=c("text_secondary"),
                        margin_bottom="8px",
                    ),
                    rx.foreach(TrainingPollerState.epoch_log, _epoch_log_row),
                    spacing="2",
                )
            ),
            rx.fragment(),
        ),
        rx.cond(
            TrainingPollerState.training_status == "done",
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
                    on_click=TrainingPollerState.go_to_step(6),
                    color_scheme="green",
                    size="3",
                ),
                spacing="3",
            ),
            rx.fragment(),
        ),
        rx.cond(
            TrainingPollerState.training_status == "failed",
            rx.callout(
                rx.vstack(
                    rx.text("Training failed", font_weight="600"),
                    rx.text(TrainingPollerState.error_msg, font_size="0.82rem"),
                ),
                color_scheme="red",
            ),
            rx.fragment(),
        ),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


def step5_panel() -> rx.Component:
    """Workspace panel layout for Step 5 (right panel of workspace)."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.hstack(
                    rx.icon("activity", size=14, color=c("accent")),
                    rx.text(
                        "Training",
                        font_size="0.82rem",
                        font_weight="600",
                        color=c("text_primary"),
                    ),
                    rx.text(
                        TrainingPollerState.effective_model_name,
                        font_size="0.76rem",
                        color=c("text_muted"),
                    ),
                    spacing="2",
                    align="center",
                ),
                rx.spacer(),
                rx.cond(
                    TrainingPollerState.compute_backend == "modal",
                    rx.badge(
                        rx.hstack(
                            rx.icon("cloud", size=12),
                            rx.text("Modal T4"),
                            spacing="1",
                            align="center",
                        ),
                        color_scheme="violet",
                        variant="soft",
                        size="1",
                    ),
                    rx.fragment(),
                ),
                _badge_status(TrainingPollerState.training_status),
                width="100%",
                align="center",
                margin_bottom="14px",
            ),
            # Idle waiting state
            rx.cond(
                (TrainingPollerState.training_status == "idle") & ~TrainingPollerState.is_starting,
                rx.vstack(
                    rx.box(
                        rx.box(
                            width="52px",
                            height="52px",
                            border_radius="50%",
                            border="2px solid",
                            border_color=c("border"),
                            class_name="idle-pulse-ring",
                        ),
                        rx.box(
                            rx.icon("activity", size=22, color=c("text_muted")),
                            position="absolute",
                            top="50%",
                            left="50%",
                            transform="translate(-50%,-50%)",
                        ),
                        position="relative",
                        width="52px",
                        height="52px",
                    ),
                    rx.text(
                        "Waiting for training to start…",
                        font_size="0.86rem",
                        color=c("text_secondary"),
                        text_align="center",
                    ),
                    rx.text(
                        'Configure hyperparameters and click "Start Training →"',
                        font_size="0.76rem",
                        color=c("text_muted"),
                        text_align="center",
                        max_width="260px",
                    ),
                    spacing="3",
                    align="center",
                    width="100%",
                    padding="48px 0",
                ),
                rx.fragment(),
            ),
            # Initializing state
            rx.cond(
                TrainingPollerState.is_starting,
                rx.vstack(
                    rx.spinner(size="3"),
                    rx.text(
                        "Initializing training job…",
                        font_size="0.86rem",
                        color=c("text_secondary"),
                    ),
                    spacing="3",
                    align="center",
                    padding="48px 0",
                ),
                rx.fragment(),
            ),
            # Start error
            rx.cond(
                TrainingPollerState.start_error != "",
                rx.callout(TrainingPollerState.start_error, color_scheme="red"),
                rx.fragment(),
            ),
            # Active / done / failed training content
            rx.cond(
                (TrainingPollerState.training_status == "running")
                | (TrainingPollerState.training_status == "done")
                | (TrainingPollerState.training_status == "failed"),
                rx.vstack(
                    rx.grid(
                        _metric_tile("Epoch", TrainingPollerState.current_epoch_display),
                        _metric_tile("Steps", TrainingPollerState.current_total_steps_display),
                        _metric_tile("Elapsed", TrainingPollerState.elapsed_time_display),
                        _metric_tile("GPU Mem", TrainingPollerState.gpu_memory_display),
                        columns="2",
                        spacing="3",
                        width="100%",
                    ),
                    rx.vstack(
                        rx.hstack(
                            rx.text("Epoch progress", font_size="0.76rem", color=c("text_muted")),
                            rx.spacer(),
                            rx.text(
                                TrainingPollerState.epoch_progress_pct.to_string() + "%",
                                font_size="0.76rem",
                                color=c("text_secondary"),
                            ),
                        ),
                        rx.progress(
                            value=TrainingPollerState.epoch_progress_pct,
                            max=100,
                            width="100%",
                            color_scheme="blue",
                        ),
                        width="100%",
                        spacing="1",
                    ),
                    _card(loss_chart()),
                    rx.cond(
                        TrainingPollerState.ai_commentary != "",
                        _card(
                            rx.hstack(
                                rx.icon("sparkles", size=16, color=c("accent")),
                                rx.text(
                                    TrainingPollerState.ai_commentary,
                                    font_size="0.86rem",
                                    color=c("text_primary"),
                                ),
                                spacing="2",
                                align="start",
                            )
                        ),
                        rx.fragment(),
                    ),
                    rx.cond(
                        TrainingPollerState.epoch_log.length() > 0,
                        _card(
                            rx.vstack(
                                rx.text(
                                    "Epoch log",
                                    font_size="0.78rem",
                                    font_weight="600",
                                    color=c("text_secondary"),
                                    margin_bottom="8px",
                                ),
                                rx.foreach(TrainingPollerState.epoch_log, _epoch_log_row),
                                spacing="2",
                            )
                        ),
                        rx.fragment(),
                    ),
                    rx.cond(
                        TrainingPollerState.training_status == "done",
                        rx.callout(
                            rx.hstack(
                                rx.icon("circle-check", size=16),
                                rx.text("Training complete! Results are below ↓"),
                                spacing="2",
                            ),
                            color_scheme="green",
                        ),
                        rx.fragment(),
                    ),
                    rx.cond(
                        TrainingPollerState.training_status == "failed",
                        rx.callout(
                            rx.vstack(
                                rx.text("Training failed", font_weight="600"),
                                rx.text(TrainingPollerState.error_msg, font_size="0.82rem"),
                            ),
                            color_scheme="red",
                        ),
                        rx.fragment(),
                    ),
                    spacing="4",
                    width="100%",
                    align_items="flex-start",
                ),
                rx.fragment(),
            ),
            spacing="0",
            width="100%",
            align_items="flex-start",
        ),
        padding="20px",
        width="50%",
        overflow_y="auto",
        height="100%",
    )
