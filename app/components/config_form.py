import reflex as rx
from app.state.model_state import ModelState

def config_form() -> rx.Component:
    return rx.vstack(
        rx.text("LoRA parameters", font_weight="500"),
        rx.hstack(
            rx.vstack(
                rx.text("Rank (r)"),
                rx.slider(min_=4, max_=64, step=4,
                          value=ModelState.lora_r,
                          on_change=ModelState.set_lora_r),
                rx.text(ModelState.lora_r[0]),
            ),
            rx.vstack(
                rx.text("Alpha"),
                rx.slider(min_=8, max_=128, step=8,
                          value=ModelState.lora_alpha,
                          on_change=ModelState.set_lora_alpha),
                rx.text(ModelState.lora_alpha[0]),
            ),
        ),

        rx.text("Training parameters", font_weight="500"),
        rx.hstack(
            rx.vstack(
                rx.text("Epochs"),
                rx.input(
                    type="number",
                    value=ModelState.epochs.to(str),
                    on_change=ModelState.set_epochs,
                    min="1", max="20",
                ),
            ),
            rx.vstack(
                rx.text("Learning rate"),
                rx.select(
                    ["1e-4", "2e-4", "5e-4", "1e-3"],
                    value=ModelState.learning_rate,
                    on_change=ModelState.set_learning_rate,
                ),
            ),
        ),
        spacing="4",
        width="100%"
    )
