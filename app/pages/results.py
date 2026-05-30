import reflex as rx

from app.state.job_state import JobState


def results_page() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("Training Completed", size="8"),
            rx.text("Your LoRA adapter is ready."),
            rx.card(
                rx.vstack(
                    rx.text("Output Path:"),
                    rx.code(JobState.output_path),
                    rx.hstack(
                        rx.button("Download Adapter", color_scheme="blue"),
                        rx.button("Merge and Export Full Model", color_scheme="blue"),
                        spacing="4",
                    ),
                ),
                padding="2em",
                width="100%",
            ),
            rx.button(
                "Train Another Model",
                on_click=rx.redirect("/"),
                color_scheme="gray",
                margin_top="2em",
            ),
            spacing="4",
            padding="2em",
            align_items="center",
        )
    )
