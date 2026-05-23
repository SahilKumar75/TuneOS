import reflex as rx
from app.state.model_state import ModelState
from app.components.config_form import config_form

def configure_page() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("Configure fine-tuning", size="6"),
            
            config_form(),

            rx.button(
                "Start training",
                on_click=ModelState.start_training,
                color_scheme="blue",
                size="3",
                margin_top="2em"
            ),
            spacing="4",
            padding="2em",
            align_items="center",
        )
    )
