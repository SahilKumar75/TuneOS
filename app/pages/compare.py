"""Multi-run comparison page — overlaid loss curves for selected experiments."""

from __future__ import annotations

import reflex as rx

from app.components.loss_chart import comparison_loss_chart
from app.state.experiment_state import ExperimentRun, ExperimentState
from app.styles import c


def _run_row(run: ExperimentRun) -> rx.Component:
    selected = ExperimentState.selected_run_ids.contains(run.id)
    return rx.hstack(
        rx.checkbox(
            checked=selected,
            on_change=lambda _v: ExperimentState.toggle_run_selection(run.id),
        ),
        rx.text(
            rx.cond(run.name != "", run.name, run.id),
            font_size="0.85rem",
            font_weight="500",
            color=c("text_primary"),
            flex="1",
            overflow="hidden",
            text_overflow="ellipsis",
            white_space="nowrap",
        ),
        rx.badge(run.technique, variant="soft", color_scheme="gray", size="1"),
        rx.text(
            "loss " + run.final_loss.to_string(),
            font_size="0.76rem",
            color=c("text_muted"),
        ),
        rx.text(
            "ppl " + run.perplexity.to_string(),
            font_size="0.76rem",
            color=c("text_muted"),
        ),
        spacing="3",
        align="center",
        width="100%",
        padding="10px 14px",
        border="1px solid",
        border_color=rx.cond(selected, c("accent"), c("border")),
        border_radius="10px",
        background=rx.cond(selected, c("bg_input"), "transparent"),
        transition="border 0.15s ease",
    )


def compare_page() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading(
                "Compare runs", font_size="1.4rem", font_weight="600", color=c("text_primary")
            ),
            rx.text(
                "Select runs and overlay their training curves.",
                font_size="0.9rem",
                color=c("text_secondary"),
            ),
            rx.hstack(
                rx.select(
                    ["loss", "eval_loss", "learning_rate"],
                    value=ExperimentState.compare_metric,
                    on_change=ExperimentState.set_compare_metric,
                    size="2",
                ),
                rx.button(
                    "Compare",
                    on_click=ExperimentState.load_comparison,
                    color_scheme="blue",
                    size="2",
                    disabled=ExperimentState.selected_run_ids.length() < 1,
                ),
                spacing="3",
                align="center",
            ),
            rx.cond(
                ExperimentState.compare_data.length() > 0,
                rx.box(
                    comparison_loss_chart(ExperimentState.compare_data),
                    rx.hstack(
                        rx.foreach(
                            ExperimentState.compare_labels,
                            lambda label, i: rx.text(
                                "Run " + (i + 1).to_string() + " — " + label,
                                font_size="0.74rem",
                                color=c("text_muted"),
                            ),
                        ),
                        spacing="4",
                        wrap="wrap",
                        margin_top="8px",
                    ),
                    width="100%",
                    padding="16px",
                    background=c("bg_card"),
                    border="1px solid",
                    border_color=c("border"),
                    border_radius="12px",
                ),
                rx.fragment(),
            ),
            rx.cond(
                ExperimentState.completed_runs.length() > 0,
                rx.vstack(
                    rx.foreach(ExperimentState.completed_runs, _run_row),
                    spacing="2",
                    width="100%",
                ),
                rx.text(
                    "No completed runs yet — train a model first.",
                    font_size="0.88rem",
                    color=c("text_muted"),
                ),
            ),
            spacing="4",
            width="100%",
            max_width="900px",
            align_items="stretch",
            padding="32px",
        ),
        on_mount=ExperimentState.load_runs,
        width="100%",
        min_height="100vh",
        background=c("bg_primary"),
        overflow_y="auto",
    )
