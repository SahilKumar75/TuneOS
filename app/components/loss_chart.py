import reflex as rx
from app.state.job_state import JobState

def loss_chart() -> rx.Component:
    return rx.recharts.line_chart(
        rx.recharts.line(
            data_key="loss",
            stroke="#8B5CF6",
            stroke_width=2,
            dot=False,
        ),
        rx.recharts.x_axis(data_key="step", label="Step"),
        rx.recharts.y_axis(label="Loss"),
        rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
        rx.recharts.graphing_tooltip(),
        data=JobState.loss_history,
        width="100%",
        height=300,
    )
