"""Fine-tune wizard progress bar and step dots."""

from __future__ import annotations

import reflex as rx

from app.state.finetune_state import FinetuneState
from app.styles import c

_STEP_LABELS = ["Model", "Intent", "Data", "Configure", "Train", "Results", "Deploy"]


def _step_dot(index: int) -> rx.Component:
    step_num = index + 1
    is_done = FinetuneState.current_step > step_num
    is_active = FinetuneState.current_step == step_num
    return rx.vstack(
        rx.box(
            rx.cond(
                is_done,
                rx.icon("check", size=12, color="white"),
                rx.text(
                    str(step_num),
                    font_size="0.72rem",
                    font_weight="600",
                    color=rx.cond(is_active, "white", c("text_muted")),
                ),
            ),
            width="26px",
            height="26px",
            border_radius="50%",
            background=rx.cond(
                is_done, c("success"), rx.cond(is_active, c("accent"), c("bg_input"))
            ),
            border="2px solid",
            border_color=rx.cond(is_active | is_done, c("accent"), c("border")),
            display="flex",
            align_items="center",
            justify_content="center",
        ),
        rx.text(
            _STEP_LABELS[index],
            font_size="0.68rem",
            color=rx.cond(is_active, c("text_primary"), c("text_muted")),
            font_weight=rx.cond(is_active, "500", "400"),
        ),
        spacing="1",
        align="center",
    )


def _progress_bar() -> rx.Component:
    return rx.hstack(
        *[
            rx.hstack(
                _step_dot(i),
                rx.box(
                    height="2px",
                    flex="1",
                    background=rx.cond(
                        FinetuneState.current_step > i + 1, c("accent"), c("border")
                    ),
                    min_width="20px",
                )
                if i < len(_STEP_LABELS) - 1
                else rx.fragment(),
                spacing="0",
                align="center",
                flex="1" if i < len(_STEP_LABELS) - 1 else "0",
            )
            for i in range(len(_STEP_LABELS))
        ],
        width="100%",
        max_width="680px",
        align="center",
        justify="center",
        margin_bottom="32px",
    )
