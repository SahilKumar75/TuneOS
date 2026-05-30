import reflex as rx

from app.components.loss_chart import loss_chart
from app.state.job_state import JobState


def training_page() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("Training in progress", size="6"),
            rx.badge(
                JobState.status,
                color_scheme=rx.cond(
                    JobState.status == "done",
                    "green",
                    rx.cond(JobState.status == "failed", "red", "yellow"),
                ),
            ),
            loss_chart(),
            rx.cond(
                JobState.status == "done",
                rx.button(
                    "View results",
                    on_click=rx.redirect("/results"),
                    color_scheme="green",
                ),
            ),
            rx.cond(
                JobState.status == "failed",
                rx.callout(JobState.error_msg, color_scheme="red"),
            ),
            spacing="4",
            padding="2em",
            align_items="center",
        )
    )
