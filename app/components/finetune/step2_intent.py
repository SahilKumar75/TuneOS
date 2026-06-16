"""Fine-tune wizard — Step 2: Structured intent questionnaire (3-phase) with iOS-style design."""

from __future__ import annotations

import reflex as rx

from app.components.finetune.shared import _card, _section_heading
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
    ("technology", "Technology"),
]
_FILTER_TASK = [
    ("text", "Text generation"),
    ("vision", "Image / Vision"),
    ("audio", "Audio / Speech"),
    ("code", "Code"),
]


# ── Phase A helpers ───────────────────────────────────────────────────────────


def _filter_chip(
    value: str,
    label: str,
    current_var,
    event_handler,
    color: str = "blue",  # kept for compat, ignored
) -> rx.Component:
    is_selected = current_var == value
    return rx.box(
        rx.text(label, font_size="0.83rem", font_weight="500"),
        cursor="pointer",
        on_click=event_handler(value),
        padding="7px 14px",
        border_radius="8px",
        background=rx.cond(is_selected, "var(--blue-9)", "var(--gray-3)"),
        color=rx.cond(is_selected, "white", "var(--gray-11)"),
        border=rx.cond(
            is_selected,
            "1.5px solid var(--blue-9)",
            "1.5px solid var(--gray-5)",
        ),
        style={
            "transition": "all 0.15s ease",
            "user-select": "none",
            ":hover": {"border-color": "var(--blue-7)", "color": "var(--gray-12)"},
        },
    )


def _filter_row(row_label: str, chips: list, subtitle: str = "") -> rx.Component:
    """iOS-style filter row with label and chips."""
    return rx.vstack(
        rx.vstack(
            rx.text(
                row_label,
                font_size="0.95rem",
                font_weight="600",
                color="var(--gray-12)",
            ),
            rx.cond(
                subtitle != "",
                rx.text(
                    subtitle,
                    font_size="0.8rem",
                    color="var(--gray-10)",
                ),
                rx.fragment(),
            ),
            spacing="1",
            align="start",
        ),
        rx.flex(*chips, wrap="wrap", gap="10px"),
        spacing="3",
        width="100%",
    )


def _phase_a() -> rx.Component:
    """Phase A: iOS-style intent collection with chips and text inputs."""
    return rx.vstack(
        rx.vstack(
            rx.text(
                "Tell us about your project",
                font_size="1.15rem",
                font_weight="600",
                color="var(--gray-12)",
            ),
            rx.text(
                "All fields are optional — we'll generate personalized questions based on what you share",
                font_size="0.85rem",
                color="var(--gray-10)",
            ),
            spacing="1",
            align_items="flex-start",
            margin_bottom="16px",
        ),
        _card(
            rx.vstack(
                # Project basics
                rx.vstack(
                    rx.text(
                        "Project Name",
                        font_size="0.9rem",
                        font_weight="600",
                        color="var(--gray-12)",
                    ),
                    rx.input(
                        placeholder="e.g., Medical Q&A Assistant",
                        value=FinetuneState.intent_project_name,
                        on_change=FinetuneState.set_intent_project_name,
                        size="3",
                        width="100%",
                        style={
                            "border-radius": "10px",
                            "border": "1.5px solid var(--gray-5)",
                            ":focus": {
                                "border-color": "var(--blue-8)",
                                "box-shadow": "0 0 0 3px var(--blue-a3)",
                            },
                        },
                    ),
                    spacing="2",
                    width="100%",
                ),
                rx.vstack(
                    rx.text(
                        "Description",
                        font_size="0.9rem",
                        font_weight="600",
                        color="var(--gray-12)",
                    ),
                    rx.text_area(
                        placeholder="Briefly describe what your model will do...",
                        value=FinetuneState.intent_description,
                        on_change=FinetuneState.set_intent_description,
                        rows="3",
                        width="100%",
                        style={
                            "border-radius": "10px",
                            "border": "1.5px solid var(--gray-5)",
                            ":focus": {
                                "border-color": "var(--blue-8)",
                                "box-shadow": "0 0 0 3px var(--blue-a3)",
                            },
                        },
                    ),
                    spacing="2",
                    width="100%",
                ),
                rx.divider(margin="4px 0"),
                # Context filters
                _filter_row(
                    "Use Case",
                    [
                        _filter_chip(
                            v, lbl,
                            FinetuneState.intent_use_for,
                            FinetuneState.set_intent_use_for,
                        )
                        for v, lbl in _FILTER_USE_FOR
                    ],
                    "Who will use this model?",
                ),
                _filter_row(
                    "Domain",
                    [
                        _filter_chip(
                            v, lbl,
                            FinetuneState.intent_domain,
                            FinetuneState.set_intent_domain,
                        )
                        for v, lbl in _FILTER_DOMAIN
                    ],
                    "What industry or field?",
                ),
                _filter_row(
                    "Task Type",
                    [
                        _filter_chip(
                            v, lbl,
                            FinetuneState.intent_task_type,
                            FinetuneState.set_intent_task_type,
                        )
                        for v, lbl in _FILTER_TASK
                    ],
                    "What kind of output?",
                ),
                spacing="5",
                width="100%",
            ),
            style={
                "border-radius": "16px",
                "box-shadow": "0 2px 8px rgba(0,0,0,0.04)",
            },
        ),
        rx.button(
            rx.hstack(
                rx.text("Continue to Questions"),
                rx.icon("arrow-right", size=18),
                spacing="2",
            ),
            on_click=FinetuneState.intent_next_phase,
            color_scheme="blue",
            size="3",
            width="100%",
            style={
                "border-radius": "12px",
                "padding": "16px",
                "font-weight": "600",
                "background": "var(--blue-9)",
                ":hover": {
                    "background": "var(--blue-10)",
                    "transform": "translateY(-1px)",
                    "box-shadow": "0 4px 12px rgba(0,0,0,0.15)",
                },
                "transition": "all 0.2s ease",
            },
        ),
        spacing="4",
        width="100%",
    )


# ── Phase B helpers ───────────────────────────────────────────────────────────


def _question_option_btn(q_idx: int, option_text: str) -> rx.Component:
    """iOS-style option button with smooth animations."""
    is_selected = FinetuneState.intent_answers[q_idx] == option_text
    return rx.button(
        rx.hstack(
            rx.icon(
                "check-circle",
                size=18,
                color=rx.cond(is_selected, "white", "transparent"),
            ),
            rx.text(option_text, flex="1"),
            spacing="3",
            align="center",
            width="100%",
        ),
        on_click=FinetuneState.set_intent_answer(q_idx, option_text),
        variant="surface",
        color_scheme=rx.cond(is_selected, "blue", "gray"),
        size="3",
        width="100%",
        text_align="left",
        justify_content="flex-start",
        style={
            "background": rx.cond(
                is_selected,
                "var(--blue-9)",
                "var(--gray-2)",
            ),
            "color": rx.cond(is_selected, "white", "var(--gray-12)"),
            "border": rx.cond(
                is_selected,
                "2px solid var(--blue-9)",
                "2px solid var(--gray-5)",
            ),
            "border-radius": "12px",
            "padding": "16px",
            "transition": "all 0.2s ease",
            "cursor": "pointer",
            ":hover": {
                "transform": "translateY(-2px)",
                "box-shadow": "0 4px 12px rgba(0,0,0,0.1)",
                "border-color": rx.cond(is_selected, "var(--blue-9)", "var(--blue-7)"),
            },
            ":active": {
                "transform": "translateY(0)",
            },
        },
    )


def _question_other_input(q_idx: int) -> rx.Component:
    """iOS-style custom input with smooth expansion."""
    is_open = FinetuneState.intent_is_custom[q_idx]
    return rx.vstack(
        rx.button(
            rx.hstack(
                rx.icon(
                    rx.cond(is_open, "chevron-down", "chevron-right"),
                    size=16,
                ),
                rx.text("Other...", font_weight="500"),
                spacing="2",
                align="center",
            ),
            on_click=FinetuneState.toggle_intent_custom(q_idx),
            variant="ghost",
            color_scheme="blue",
            size="3",
            style={
                "border-radius": "10px",
                "transition": "all 0.2s ease",
            },
        ),
        rx.cond(
            is_open,
            rx.box(
                rx.input(
                    placeholder="Describe in your own words...",
                    value=FinetuneState.intent_custom_answers[q_idx],
                    on_change=lambda v: FinetuneState.set_intent_custom_answer(q_idx, v),
                    width="100%",
                    auto_focus=True,
                    size="3",
                    style={
                        "border-radius": "12px",
                        "border": "2px solid var(--blue-7)",
                        "padding": "12px 16px",
                        "transition": "all 0.2s ease",
                        ":focus": {
                            "outline": "none",
                            "border-color": "var(--blue-9)",
                            "box-shadow": "0 0 0 3px var(--blue-a4)",
                        },
                    },
                ),
                style={
                    "animation": "slideDown 0.2s ease",
                    "@keyframes slideDown": {
                        "from": {
                            "opacity": "0",
                            "transform": "translateY(-10px)",
                        },
                        "to": {
                            "opacity": "1",
                            "transform": "translateY(0)",
                        },
                    },
                },
            ),
            rx.fragment(),
        ),
        spacing="3",
        width="100%",
    )


def _progress_dots() -> rx.Component:
    """iOS-style progress indicator with smooth animations."""
    total_questions = rx.cond(
        FinetuneState.intent_questions.length() > 0,
        FinetuneState.intent_questions.length(),
        5,
    )
    return rx.hstack(
        rx.foreach(
            rx.Var.range(total_questions),
            lambda i: rx.box(
                width="32px",
                height="5px",
                border_radius="3px",
                background=rx.cond(
                    FinetuneState.intent_question_idx >= i,
                    "var(--blue-9)",
                    "var(--gray-5)",
                ),
                style={
                    "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                },
            ),
        ),
        spacing="2",
    )


def _phase_b_question(q_idx: int) -> rx.Component:
    """Render a single question card with iOS-style design and dynamic content."""
    # Use dynamic questions from state
    q = FinetuneState.intent_questions[q_idx]
    is_answered = (FinetuneState.intent_answers[q_idx] != "") | FinetuneState.intent_is_custom[
        q_idx
    ]
    total_q = FinetuneState.intent_questions.length()
    is_last = q_idx >= (total_q - 1)

    return _card(
        rx.vstack(
            # Header with progress
            rx.hstack(
                rx.text(
                    f"Question {q_idx + 1} of {total_q.to(str)}",
                    font_size="0.8rem",
                    color="var(--gray-10)",
                    font_weight="500",
                ),
                rx.spacer(),
                _progress_dots(),
                width="100%",
                align="center",
            ),
            rx.box(height="8px"),
            # Question heading
            rx.text(
                q.heading,
                font_size="1.1rem",
                font_weight="600",
                color="var(--gray-12)",
                line_height="1.4",
            ),
            rx.box(height="12px"),
            # Options with dynamic rendering
            rx.foreach(
                q.options,
                lambda opt: _question_option_btn(q_idx, opt),
            ),
            _question_other_input(q_idx),
            rx.box(height="12px"),
            # Navigation buttons
            rx.hstack(
                rx.button(
                    rx.hstack(
                        rx.icon("arrow-left", size=16),
                        rx.text("Back"),
                        spacing="2",
                    ),
                    on_click=FinetuneState.intent_prev_question,
                    variant="soft",
                    color_scheme="gray",
                    size="3",
                    style={
                        "border-radius": "10px",
                        "padding": "12px 20px",
                    },
                ),
                rx.spacer(),
                rx.button(
                    rx.hstack(
                        rx.text(rx.cond(is_last, "Preview", "Continue")),
                        rx.icon("arrow-right", size=16),
                        spacing="2",
                    ),
                    on_click=FinetuneState.intent_next_question,
                    disabled=~is_answered,
                    color_scheme="blue",
                    size="3",
                    style={
                        "border-radius": "10px",
                        "padding": "12px 24px",
                        "background": rx.cond(is_answered, "var(--blue-9)", "var(--gray-5)"),
                    },
                ),
                width="100%",
                align="center",
            ),
            spacing="0",
            width="100%",
        ),
        style={
            "border-radius": "16px",
            "box-shadow": "0 2px 8px rgba(0,0,0,0.05)",
        },
    )


def _phase_b() -> rx.Component:
    """Phase B: Dynamic questionnaire with live plan preview."""
    return rx.cond(
        FinetuneState.intent_is_generating_questions,
        # Loading state while generating questions
        _card(
            rx.vstack(
                rx.spinner(size="3", color="blue"),
                rx.text(
                    "Generating personalized questions...",
                    font_size="0.95rem",
                    font_weight="500",
                    color="var(--gray-11)",
                ),
                rx.text(
                    "Based on your project details",
                    font_size="0.82rem",
                    color="var(--gray-9)",
                ),
                spacing="4",
                align="center",
                padding="48px 24px",
            ),
            style={"border-radius": "16px"},
        ),
        # Show dynamic questions when loaded
        rx.vstack(
            # Live plan preview (if available)
            rx.cond(
                FinetuneState.intent_live_plan != "",
                _card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("lightbulb", size=18, color="var(--amber-9)"),
                            rx.text(
                                "Your Plan",
                                font_size="0.85rem",
                                font_weight="600",
                                color="var(--gray-12)",
                            ),
                            spacing="2",
                            align="center",
                        ),
                        rx.text(
                            FinetuneState.intent_live_plan,
                            font_size="0.88rem",
                            color="var(--gray-11)",
                            line_height="1.5",
                        ),
                        spacing="3",
                        width="100%",
                    ),
                    style={
                        "background": "var(--amber-2)",
                        "border": "1px solid var(--amber-6)",
                        "border-radius": "12px",
                        "margin-bottom": "16px",
                    },
                ),
                rx.fragment(),
            ),
            # Current question
            rx.foreach(
                FinetuneState.intent_questions,
                lambda q, idx: rx.cond(
                    FinetuneState.intent_question_idx == idx,
                    _phase_b_question(idx),
                    rx.fragment(),
                ),
            ),
            width="100%",
            spacing="0",
        ),
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
                    rx.text(
                        "Intent profile ready",
                        font_size="0.88rem",
                        font_weight="600",
                        color=c("text_primary"),
                    ),
                    spacing="2",
                    align="center",
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


def _mode_card(mode: str, icon: str, title: str, subtitle: str) -> rx.Component:
    """One clickable card in the training-goal selector."""
    selected = FinetuneState.training_mode == mode
    return rx.box(
        rx.vstack(
            # Icon badge
            rx.box(
                rx.icon(
                    icon,
                    size=20,
                    color=rx.cond(selected, "white", "var(--gray-11)"),
                ),
                padding="10px",
                border_radius="10px",
                background=rx.cond(
                    selected,
                    "rgba(255,255,255,0.18)",
                    "var(--gray-4)",
                ),
                display="inline-flex",
                align_items="center",
                justify_content="center",
                style={"transition": "background 0.18s ease"},
            ),
            # Title
            rx.text(
                title,
                font_size="0.9rem",
                font_weight="700",
                color=rx.cond(selected, "white", "var(--gray-12)"),
                line_height="1.3",
                style={"transition": "color 0.18s ease"},
            ),
            # Subtitle
            rx.text(
                subtitle,
                font_size="0.78rem",
                color=rx.cond(selected, "rgba(255,255,255,0.75)", "var(--gray-10)"),
                line_height="1.5",
                style={"transition": "color 0.18s ease"},
            ),
            spacing="3",
            align_items="flex-start",
        ),
        on_click=FinetuneState.set_training_mode(mode),
        cursor="pointer",
        padding="20px",
        border_radius="14px",
        background=rx.cond(selected, "var(--blue-9)", "var(--gray-2)"),
        border=rx.cond(
            selected,
            "2px solid var(--blue-9)",
            "1.5px solid var(--gray-5)",
        ),
        flex="1",
        min_width="200px",
        style={
            "transition": "all 0.18s ease",
            ":hover": {"border-color": "var(--gray-7)"},
        },
    )


def _training_goal_card() -> rx.Component:
    """Top-of-step-2 card: pick SFT / DPO / KD before answering intent questions."""
    return rx.vstack(
        rx.hstack(
            rx.vstack(
                rx.text(
                    "Training goal",
                    font_size="0.9rem",
                    font_weight="700",
                    color="var(--gray-12)",
                ),
                rx.text(
                    "Pick the right paradigm — this gates which backend API and data format is used.",
                    font_size="0.78rem",
                    color="var(--gray-10)",
                ),
                spacing="1",
                align_items="flex-start",
            ),
            rx.spacer(),
            rx.hstack(
                rx.cond(
                    FinetuneState.training_goal_help_error,
                    rx.text(
                        "Fill in project details first",
                        font_size="0.75rem",
                        font_weight="500",
                        color="var(--red-10)",
                    ),
                    rx.fragment(),
                ),
                rx.tooltip(
                    rx.button(
                        rx.text("?", font_size="0.8rem", font_weight="700"),
                        on_click=FinetuneState.ask_training_goal_help,
                        variant="soft",
                        color_scheme="gray",
                        size="1",
                        width="26px",
                        height="26px",
                        border_radius="50%",
                        cursor="pointer",
                        style={"flex-shrink": "0"},
                    ),
                    content="Not sure? Ask the assistant",
                ),
                spacing="2",
                align="center",
            ),
            width="100%",
            align="center",
        ),
        rx.flex(
            _mode_card(
                "sft",
                "book-open",
                "Supervised Fine-Tuning",
                "Teach the model new tasks with instruction/output pairs",
            ),
            _mode_card(
                "dpo",
                "git-compare",
                "Preference Alignment (DPO)",
                "Align model to human preferences via chosen/rejected pairs",
            ),
            _mode_card(
                "kd",
                "minimize-2",
                "Knowledge Distillation",
                "Compress a large teacher into a smaller student model",
            ),
            gap="12px",
            wrap="wrap",
            width="100%",
        ),
        spacing="3",
        width="100%",
        padding="16px",
        background="var(--gray-2)",
        border_radius="14px",
        border="1px solid var(--gray-6)",
        margin_bottom="16px",
    )


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
        # Training goal selector — always visible in phase 1
        rx.cond(FinetuneState.intent_phase == 1, _training_goal_card(), rx.fragment()),
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
