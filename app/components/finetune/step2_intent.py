"""Fine-tune wizard — Step 2: Structured intent questionnaire (3-phase)."""

from __future__ import annotations

import reflex as rx

from app.components.finetune.shared import _card, _label, _section_heading
from app.state.finetune_state import FinetuneState
from app.styles import c

# ── Static data (not in state) ────────────────────────────────────────────────

_FILTER_USE_FOR = [("personal", "Personal"), ("company", "Company product")]
_FILTER_DOMAIN = [
    ("healthcare", "Healthcare"),
    ("finance", "Finance"),
    ("education", "Education"),
    ("legal", "Legal"),
    ("creative", "Creative"),
]
_FILTER_TASK = [
    ("text", "Text generation"),
    ("vision", "Image / Vision"),
    ("audio", "Audio / Speech"),
    ("code", "Code"),
]

_QUESTIONS = [
    {
        "heading": "What is the primary goal of this model?",
        "options": [
            "Answer questions / provide information",
            "Generate or transform content",
            "Classify, analyze, or extract data",
        ],
    },
    {
        "heading": "Who is the target audience?",
        "options": [
            "General public / consumers",
            "Domain professionals",
            "Internal team / developers",
        ],
    },
    {
        "heading": "What is the primary input format?",
        "options": [
            "Free-form text / conversations",
            "Structured data or documents",
            "Mixed / varies",
        ],
    },
    {
        "heading": "What tone and style should the model use?",
        "options": [
            "Formal and precise",
            "Friendly and conversational",
            "Concise and direct",
        ],
    },
    {
        "heading": "How will you measure success?",
        "options": [
            "Accuracy / factual correctness",
            "User satisfaction / engagement",
            "Task completion / automation rate",
        ],
    },
]


# ── Phase A helpers ───────────────────────────────────────────────────────────


def _filter_chip(
    value: str,
    label: str,
    current_var,
    event_handler,
    color: str = "blue",
) -> rx.Component:
    return rx.badge(
        label,
        cursor="pointer",
        on_click=event_handler(value),
        color_scheme=color,
        variant=rx.cond(current_var == value, "solid", "soft"),
        size="2",
        padding="6px 12px",
    )


def _filter_row(row_label: str, chips: list) -> rx.Component:
    return rx.vstack(
        _label(row_label),
        rx.flex(*chips, wrap="wrap", gap="8px"),
        spacing="2",
        width="100%",
    )


def _phase_a() -> rx.Component:
    return rx.vstack(
        rx.text("Step 1 of 3 — Context filters", font_size="0.75rem", color=c("text_muted"), margin_bottom="8px"),
        _card(
            rx.vstack(
                rx.text(
                    "All fields are optional — skip any you're not sure about.",
                    font_size="0.82rem",
                    color=c("text_muted"),
                    margin_bottom="4px",
                ),
                _filter_row(
                    "Use for?",
                    [
                        _filter_chip(v, l, FinetuneState.intent_use_for, FinetuneState.set_intent_use_for, color="blue")
                        for v, l in _FILTER_USE_FOR
                    ],
                ),
                rx.text("Who will benefit from this model?", font_size="0.75rem", color=c("text_muted")),
                rx.divider(),
                _filter_row(
                    "Domain?",
                    [
                        _filter_chip(v, l, FinetuneState.intent_domain, FinetuneState.set_intent_domain, color="violet")
                        for v, l in _FILTER_DOMAIN
                    ],
                ),
                rx.text("What industry or field?", font_size="0.75rem", color=c("text_muted")),
                rx.divider(),
                _filter_row(
                    "Task type?",
                    [
                        _filter_chip(v, l, FinetuneState.intent_task_type, FinetuneState.set_intent_task_type, color="green")
                        for v, l in _FILTER_TASK
                    ],
                ),
                rx.text("What kind of output does it produce?", font_size="0.75rem", color=c("text_muted")),
                spacing="4",
                width="100%",
            )
        ),
        rx.button(
            "Continue →",
            on_click=FinetuneState.intent_next_phase,
            color_scheme="blue",
            size="3",
            width="100%",
        ),
        spacing="3",
        width="100%",
    )


# ── Phase B helpers ───────────────────────────────────────────────────────────


def _question_option_btn(q_idx: int, option_text: str) -> rx.Component:
    is_selected = FinetuneState.intent_answers[q_idx] == option_text
    return rx.button(
        option_text,
        on_click=FinetuneState.set_intent_answer(q_idx, option_text),
        variant=rx.cond(is_selected, "solid", "outline"),
        color_scheme="blue",
        size="2",
        width="100%",
        text_align="left",
        justify_content="flex-start",
    )


def _question_other_input(q_idx: int) -> rx.Component:
    is_open = FinetuneState.intent_is_custom[q_idx]
    return rx.vstack(
        rx.button(
            rx.hstack(
                rx.icon(
                    rx.cond(is_open, "chevron-down", "chevron-right"),
                    size=14,
                ),
                rx.text("Other..."),
                spacing="2",
                align="center",
            ),
            on_click=FinetuneState.toggle_intent_custom(q_idx),
            variant=rx.cond(is_open, "solid", "ghost"),
            color_scheme="blue",
            size="2",
        ),
        rx.cond(
            is_open,
            rx.input(
                placeholder="Describe in your own words...",
                value=FinetuneState.intent_custom_answers[q_idx],
                on_change=lambda v: FinetuneState.set_intent_custom_answer(q_idx, v),
                width="100%",
                auto_focus=True,
            ),
            rx.fragment(),
        ),
        spacing="2",
        width="100%",
    )


def _progress_dots() -> rx.Component:
    return rx.hstack(
        *[
            rx.box(
                width="28px",
                height="4px",
                border_radius="2px",
                background=rx.cond(
                    FinetuneState.intent_question_idx >= i,
                    c("accent"),
                    c("border"),
                ),
            )
            for i in range(5)
        ],
        spacing="2",
    )


def _phase_b_question(q_idx: int) -> rx.Component:
    q = _QUESTIONS[q_idx]
    is_answered = (FinetuneState.intent_answers[q_idx] != "") | FinetuneState.intent_is_custom[q_idx]
    is_last = q_idx == 4
    return _card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    f"Question {q_idx + 1} of 5",
                    font_size="0.75rem",
                    color=c("text_muted"),
                ),
                rx.spacer(),
                _progress_dots(),
                width="100%",
                align="center",
            ),
            rx.box(height="4px"),
            rx.text(
                q["heading"],
                font_size="0.97rem",
                font_weight="600",
                color=c("text_primary"),
            ),
            rx.box(height="6px"),
            *[_question_option_btn(q_idx, opt) for opt in q["options"]],
            _question_other_input(q_idx),
            rx.box(height="6px"),
            rx.hstack(
                rx.button(
                    "← Back",
                    on_click=FinetuneState.intent_prev_question,
                    variant="soft",
                    color_scheme="gray",
                    size="2",
                ),
                rx.spacer(),
                rx.button(
                    rx.cond(is_last, "Preview & Confirm →", "Next →"),
                    on_click=FinetuneState.intent_next_question,
                    disabled=~is_answered,
                    color_scheme="blue",
                    size="3",
                ),
                width="100%",
                align="center",
            ),
            spacing="2",
            width="100%",
        )
    )


def _phase_b() -> rx.Component:
    return rx.match(
        FinetuneState.intent_question_idx,
        (0, _phase_b_question(0)),
        (1, _phase_b_question(1)),
        (2, _phase_b_question(2)),
        (3, _phase_b_question(3)),
        (4, _phase_b_question(4)),
        _phase_b_question(0),
    )


# ── Phase C ───────────────────────────────────────────────────────────────────


def _phase_c() -> rx.Component:
    return rx.vstack(
        _section_heading("Review your intent profile"),
        rx.text(
            "This profile will guide data generation, training configuration, and system prompt scaffolding.",
            font_size="0.85rem",
            color=c("text_secondary"),
            margin_bottom="8px",
        ),
        _card(
            rx.vstack(
                rx.hstack(
                    rx.icon("check-circle", size=16, color=c("accent")),
                    rx.text("Intent profile ready", font_size="0.88rem", font_weight="600", color=c("text_primary")),
                    spacing="2", align="center",
                ),
                rx.box(height="8px"),
                rx.box(
                    rx.markdown(FinetuneState.intent_md),
                    max_height="380px",
                    overflow_y="auto",
                    padding="4px",
                    width="100%",
                ),
                spacing="0",
                width="100%",
            ),
            padding="16px",
        ),
        rx.box(height="8px"),
        rx.hstack(
            rx.button(
                "← Edit",
                on_click=FinetuneState.intent_prev_phase,
                variant="soft",
                color_scheme="gray",
                size="2",
            ),
            rx.spacer(),
            rx.button(
                "Approve & Continue →",
                on_click=FinetuneState.approve_intent,
                color_scheme="blue",
                size="3",
            ),
            width="100%",
            align="center",
        ),
        spacing="0",
        width="100%",
        align_items="flex-start",
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def _step2() -> rx.Component:
    return rx.vstack(
        _section_heading("Tell us about your use case"),
        rx.cond(
            FinetuneState.intent_phase == 1,
            rx.button(
                "← Back to Model",
                on_click=FinetuneState.prev_step,
                variant="soft",
                color_scheme="gray",
                size="2",
                margin_bottom="12px",
            ),
            rx.fragment(),
        ),
        rx.cond(
            FinetuneState.intent_phase == 1,
            _phase_a(),
            rx.cond(
                FinetuneState.intent_phase == 2,
                _phase_b(),
                _phase_c(),
            ),
        ),
        spacing="0",
        width="100%",
        align_items="flex-start",
    )
