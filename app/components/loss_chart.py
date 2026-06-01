import reflex as rx

from app.state.finetune_state import FinetuneState


def loss_chart() -> rx.Component:
    """Dual-series chart: training loss (blue) + learning rate (amber) + eval loss (green dashed)."""
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
        data=FinetuneState.loss_history,
        width="100%",
        height=280,
    )
