"""Fine-tune wizard — Step 2: Intent questionnaire — cascading AI-generated questions."""

from __future__ import annotations

import reflex as rx

from app.components.brand.loader import metaball_loader
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
                            v,
                            lbl,
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
                            v,
                            lbl,
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
                            v,
                            lbl,
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

_LOADING_MESSAGES = [
    "Analysing your model choice…",
    "Reading your project context…",
    "Crafting targeted questions…",
]


def _skeleton_question_card() -> rx.Component:
    """Placeholder skeleton card mimicking a real question card."""
    shimmer = "linear-gradient(90deg, var(--gray-3) 25%, var(--gray-4) 50%, var(--gray-3) 75%)"
    return rx.box(
        rx.vstack(
            # question title bar
            rx.box(
                height="14px",
                width="65%",
                border_radius="6px",
                background=shimmer,
                background_size="200% 100%",
                style={"animation": "shimmer 1.5s infinite linear"},
            ),
            rx.vstack(
                *[
                    rx.box(
                        height="38px",
                        width="100%",
                        border_radius="8px",
                        background=shimmer,
                        background_size="200% 100%",
                        style={"animation": f"shimmer 1.5s {i * 0.15}s infinite linear"},
                    )
                    for i in range(3)
                ],
                spacing="2",
                width="100%",
            ),
            spacing="3",
            width="100%",
        ),
        padding="20px",
        border_radius="12px",
        border="1px solid var(--gray-4)",
        width="100%",
    )


def _loading_state() -> rx.Component:
    """Skeleton loader shown while AI generates questions."""
    return rx.vstack(
        # header + spinner
        rx.hstack(
            metaball_loader(36, color=True),
            rx.vstack(
                rx.text(
                    "Generating your questions",
                    font_size="0.95rem",
                    font_weight="600",
                    color="var(--gray-12)",
                ),
                rx.text(
                    "Tailoring 3 questions to your model, technique & project context — please wait",
                    font_size="0.78rem",
                    color="var(--gray-10)",
                ),
                spacing="1",
                align_items="start",
            ),
            align="center",
            spacing="3",
            width="100%",
            padding="16px 20px",
            border_radius="12px",
            background="var(--gray-2)",
            border="1px solid var(--gray-4)",
        ),
        # 3 skeleton question cards
        _skeleton_question_card(),
        _skeleton_question_card(),
        _skeleton_question_card(),
        rx.html(
            "<style>@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}</style>"
        ),
        spacing="3",
        width="100%",
    )


def _answered_question_card(q_idx: int) -> rx.Component:
    """Compact summary row for an already-answered question."""
    q = FinetuneState.intent_questions[q_idx]
    answer = FinetuneState.intent_answers[q_idx]
    return rx.box(
        rx.hstack(
            rx.box(
                rx.icon("check", size=14, color="white"),
                width="22px",
                height="22px",
                border_radius="50%",
                background="var(--green-9)",
                display="flex",
                align_items="center",
                justify_content="center",
                flex_shrink="0",
            ),
            rx.vstack(
                rx.text(
                    q.heading,
                    font_size="0.82rem",
                    color="var(--gray-10)",
                    font_weight="500",
                    line_height="1.3",
                ),
                rx.text(
                    answer,
                    font_size="0.9rem",
                    color="var(--gray-12)",
                    font_weight="600",
                ),
                spacing="0",
                align_items="flex-start",
            ),
            rx.spacer(),
            rx.button(
                "Edit",
                on_click=FinetuneState.intent_edit_question(q_idx),
                variant="ghost",
                color_scheme="blue",
                size="1",
                style={"font-size": "0.78rem", "padding": "4px 10px"},
            ),
            spacing="3",
            align="center",
            width="100%",
        ),
        padding="14px 16px",
        border_radius="12px",
        background="var(--gray-2)",
        border="1px solid var(--gray-4)",
        width="100%",
        style={"transition": "background 0.15s ease"},
    )


def _active_question_card(q_idx: int) -> rx.Component:
    """Full expanded card for the currently active question."""
    q = FinetuneState.intent_questions[q_idx]
    total_q = FinetuneState.intent_questions.length()
    is_answered = (FinetuneState.intent_answers[q_idx] != "") | FinetuneState.intent_is_custom[
        q_idx
    ]
    is_last = q_idx >= (total_q - 1)
    is_open = FinetuneState.intent_is_custom[q_idx]

    return rx.box(
        rx.vstack(
            # Question number badge + heading
            rx.hstack(
                rx.box(
                    rx.text(
                        f"{q_idx + 1}",
                        font_size="0.72rem",
                        font_weight="700",
                        color="white",
                    ),
                    width="22px",
                    height="22px",
                    border_radius="50%",
                    background="var(--blue-9)",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    flex_shrink="0",
                ),
                rx.text(
                    q.heading,
                    font_size="1.05rem",
                    font_weight="700",
                    color="var(--gray-12)",
                    line_height="1.35",
                ),
                spacing="3",
                align="start",
                width="100%",
            ),
            rx.box(height="4px"),
            # Options
            rx.vstack(
                rx.foreach(
                    q.options,
                    lambda opt: rx.box(
                        rx.hstack(
                            rx.box(
                                rx.cond(
                                    FinetuneState.intent_answers[q_idx] == opt,
                                    rx.icon("check", size=13, color="white"),
                                    rx.box(
                                        width="13px",
                                        height="13px",
                                        border_radius="50%",
                                        border="2px solid var(--gray-7)",
                                    ),
                                ),
                                width="20px",
                                height="20px",
                                border_radius="50%",
                                background=rx.cond(
                                    FinetuneState.intent_answers[q_idx] == opt,
                                    "var(--blue-9)",
                                    "transparent",
                                ),
                                display="flex",
                                align_items="center",
                                justify_content="center",
                                flex_shrink="0",
                                style={"transition": "all 0.15s ease"},
                            ),
                            rx.text(
                                opt,
                                font_size="0.9rem",
                                color=rx.cond(
                                    FinetuneState.intent_answers[q_idx] == opt,
                                    "var(--gray-12)",
                                    "var(--gray-11)",
                                ),
                                font_weight=rx.cond(
                                    FinetuneState.intent_answers[q_idx] == opt,
                                    "600",
                                    "400",
                                ),
                            ),
                            spacing="3",
                            align="center",
                        ),
                        on_click=FinetuneState.set_intent_answer(q_idx, opt),
                        padding="12px 14px",
                        border_radius="10px",
                        border=rx.cond(
                            FinetuneState.intent_answers[q_idx] == opt,
                            "1.5px solid var(--blue-7)",
                            "1.5px solid var(--gray-4)",
                        ),
                        background=rx.cond(
                            FinetuneState.intent_answers[q_idx] == opt,
                            "var(--blue-2)",
                            "var(--gray-1)",
                        ),
                        cursor="pointer",
                        width="100%",
                        style={
                            "transition": "all 0.15s ease",
                            ":hover": {
                                "border-color": "var(--blue-6)",
                                "background": "var(--blue-1)",
                            },
                        },
                    ),
                ),
                # Other... toggle
                rx.vstack(
                    rx.button(
                        rx.hstack(
                            rx.icon(
                                rx.cond(is_open, "chevron-down", "chevron-right"),
                                size=14,
                            ),
                            rx.text("Other…", font_size="0.85rem"),
                            spacing="1",
                            align="center",
                        ),
                        on_click=FinetuneState.toggle_intent_custom(q_idx),
                        variant="ghost",
                        color_scheme="gray",
                        size="2",
                        padding="6px 10px",
                    ),
                    rx.cond(
                        is_open,
                        rx.input(
                            placeholder="Describe in your own words…",
                            value=FinetuneState.intent_custom_answers[q_idx],
                            on_change=lambda v: FinetuneState.set_intent_custom_answer(q_idx, v),
                            width="100%",
                            size="2",
                            auto_focus=True,
                            style={
                                "border-radius": "8px",
                                "border": "1.5px solid var(--blue-6)",
                            },
                        ),
                        rx.fragment(),
                    ),
                    spacing="2",
                    width="100%",
                    align_items="flex-start",
                ),
                spacing="2",
                width="100%",
            ),
            # Continue / Preview button (only shown when answered)
            rx.cond(
                is_answered,
                rx.box(
                    rx.button(
                        rx.hstack(
                            rx.text(rx.cond(is_last, "Preview intent →", "Next question →")),
                            spacing="2",
                            align="center",
                        ),
                        on_click=FinetuneState.intent_next_question,
                        color_scheme="blue",
                        size="3",
                        width="100%",
                        style={
                            "border-radius": "10px",
                            "font-weight": "600",
                            "background": "var(--blue-9)",
                            "margin-top": "8px",
                            ":hover": {"background": "var(--blue-10)"},
                        },
                    ),
                    width="100%",
                    style={
                        "animation": "fadeSlideUp 0.2s ease",
                        "@keyframes fadeSlideUp": {
                            "from": {"opacity": "0", "transform": "translateY(6px)"},
                            "to": {"opacity": "1", "transform": "translateY(0)"},
                        },
                    },
                ),
                rx.fragment(),
            ),
            spacing="3",
            width="100%",
        ),
        id=f"intent-q-{q_idx}",
        padding="20px",
        border_radius="14px",
        background="var(--gray-1)",
        border="1.5px solid var(--blue-6)",
        box_shadow="0 4px 16px rgba(0,0,0,0.08)",
        width="100%",
        style={
            "animation": "cascadeIn 0.25s cubic-bezier(0.4,0,0.2,1)",
            "@keyframes cascadeIn": {
                "from": {"opacity": "0", "transform": "translateY(12px)"},
                "to": {"opacity": "1", "transform": "translateY(0)"},
            },
        },
    )


def _phase_b() -> rx.Component:
    """Phase B: cascading question reveal — answered questions collapse above, active expands below."""
    return rx.cond(
        FinetuneState.intent_is_generating_questions,
        _loading_state(),
        rx.vstack(
            # Answered questions (compact, stacked above)
            rx.foreach(
                FinetuneState.intent_questions,
                lambda q, idx: rx.cond(
                    FinetuneState.intent_question_idx > idx,
                    _answered_question_card(idx),
                    rx.fragment(),
                ),
            ),
            # Active (current) question — full expanded
            rx.foreach(
                FinetuneState.intent_questions,
                lambda q, idx: rx.cond(
                    FinetuneState.intent_question_idx == idx,
                    _active_question_card(idx),
                    rx.fragment(),
                ),
            ),
            # Back link
            rx.button(
                rx.hstack(
                    rx.icon("arrow-left", size=14),
                    rx.text("Back to project details"),
                    spacing="2",
                    align="center",
                ),
                on_click=FinetuneState.intent_prev_phase,
                variant="ghost",
                color_scheme="gray",
                size="2",
                margin_top="4px",
            ),
            spacing="3",
            width="100%",
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
