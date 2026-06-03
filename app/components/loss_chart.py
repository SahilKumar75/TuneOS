"""Loss curve chart — single-run (live) and multi-run comparison overlay."""

from __future__ import annotations

import reflex as rx

from app.state.finetune_state import FinetuneState

# Palette for up to 6 overlaid runs in the comparison chart.
_COMPARE_COLORS = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#a855f7", "#14b8a6"]


def loss_chart() -> rx.Component:
    """Dual-series chart for a single live run: train loss + eval loss + LR."""
    return rx.recharts.composed_chart(
        rx.recharts.line(
            data_key="loss",
            stroke="#3b82f6",
            stroke_width=2,
            dot=False,
            name="Train Loss",
            y_axis_id="left",
        ),
        rx.recharts.line(
            data_key="eval_loss",
            stroke="#22c55e",
            stroke_width=2,
            stroke_dasharray="5 5",
            dot=False,
            name="Eval Loss",
            y_axis_id="left",
        ),
        rx.recharts.line(
            data_key="learning_rate",
            stroke="#f59e0b",
            stroke_width=1,
            dot=False,
            name="Learning Rate",
            y_axis_id="right",
        ),
        rx.recharts.x_axis(data_key="step"),
        rx.recharts.y_axis(y_axis_id="left", width=60),
        rx.recharts.y_axis(y_axis_id="right", orientation="right", width=70),
        rx.recharts.cartesian_grid(stroke_dasharray="3 3", opacity=0.3),
        rx.recharts.legend(),
        rx.recharts.graphing_tooltip(),
        data=FinetuneState.loss_history_chart_data,
        width="100%",
        height=280,
    )


def comparison_loss_chart(
    compare_data: list[dict],
    run_labels: list[str] | None = None,
) -> rx.Component:
    """Multi-run overlaid loss chart for the experiment comparison view.

    ``compare_data`` must be a flat list of dicts with keys
    ``step``, ``value``, ``run_id``.  The chart pivots by ``run_id`` and
    renders one colored line per run.

    Pass ``run_labels`` (same length as the number of distinct run_ids) to
    override the legend labels; defaults to the run_id values.
    """
    # Build one rx.recharts.line per run_id using the passed-in palette.
    # Because this is called at compile time we generate a fixed number of
    # series — at most len(_COMPARE_COLORS).
    labels = run_labels or []
    lines = [
        rx.recharts.line(
            data_key=f"run{i}",
            stroke=_COMPARE_COLORS[i % len(_COMPARE_COLORS)],
            stroke_width=2,
            dot=False,
            name=labels[i] if i < len(labels) else f"Run {i + 1}",
            y_axis_id="left",
        )
        for i in range(len(_COMPARE_COLORS))
    ]
    return rx.recharts.composed_chart(
        *lines,
        rx.recharts.x_axis(data_key="step"),
        rx.recharts.y_axis(y_axis_id="left", width=60),
        rx.recharts.cartesian_grid(stroke_dasharray="3 3", opacity=0.3),
        rx.recharts.legend(),
        rx.recharts.graphing_tooltip(),
        data=compare_data,
        width="100%",
        height=280,
    )
