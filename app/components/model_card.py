import reflex as rx

from app.state.model_state import ModelState


def model_card(name: str, hf_id: str, description: str) -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading(name, size="4"),
            rx.text(hf_id, font_size="sm", color="gray"),
            rx.text(description, font_size="sm"),
            rx.button(
                "Select Model",
                on_click=ModelState.set_model_name(hf_id),
                color_scheme=rx.cond(ModelState.model_name == hf_id, "green", "blue"),
                margin_top="1em",
            ),
        ),
        padding="1em",
        border_width="1px",
        border_radius="md",
        width="100%",
    )
