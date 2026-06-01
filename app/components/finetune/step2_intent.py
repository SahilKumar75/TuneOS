"""Fine-tune wizard — Step 2: Intent / use-case description."""

from __future__ import annotations

import reflex as rx

from app.components.finetune.shared import _card, _label, _nav_buttons, _section_heading
from app.state.finetune_state import FinetuneState
from app.styles import c

_INTENT_IDEAS = [
    "Health chatbot for diabetes patients",
    "Python code review assistant",
    "Customer support for SaaS products",
    "Legal document summarizer",
    "Recipe recommendation assistant",
    "Scientific paper Q&A bot",
    "SQL query generator",
    "Children's education tutor",
]


def _step2() -> rx.Component:
    return rx.vstack(
        _section_heading("What are you building?"),
        rx.text(
            "Describe your use-case in plain English. TuneOS uses this to generate starter data, "
            "guide the training dashboard, and pre-fill the system prompt for testing.",
            font_size="0.86rem",
            color=c("text_secondary"),
            margin_bottom="16px",
        ),
        _card(
            rx.vstack(
                _label("Your goal (1–3 sentences)"),
                rx.text_area(
                    placeholder="e.g. A health chatbot that answers questions for people with Type 2 diabetes in simple language.",
                    value=FinetuneState.user_intent,
                    on_change=FinetuneState.set_user_intent,
                    rows="4",
                    width="100%",
                    resize="vertical",
                ),
                rx.text("Quick ideas:", font_size="0.76rem", color=c("text_muted")),
                rx.flex(
                    *[
                        rx.badge(
                            idea,
                            cursor="pointer",
                            on_click=FinetuneState.set_user_intent(idea),
                            color_scheme="blue",
                            variant="soft",
                            size="1",
                        )
                        for idea in _INTENT_IDEAS
                    ],
                    wrap="wrap",
                    gap="6px",
                ),
                spacing="3",
            )
        ),
        _nav_buttons(
            next_label="Next: Add Data →",
            next_disabled=FinetuneState.user_intent == "",
        ),
        spacing="0",
        width="100%",
        align_items="flex-start",
    )
