"""Context-aware chat assistant panel — embedded in the fine-tune wizard."""

from __future__ import annotations

import reflex as rx

from app.state.app_state import AppState
from app.state.finetune_state import FinetuneState
from app.styles import c

_STEP_HINTS: dict[int, list[str]] = {
    1: [
        "Which model is best for instruction following?",
        "Compare Mistral 7B vs Llama 3 8B for fine-tuning",
        "What VRAM do I need for QLoRA on a 7B model?",
    ],
    2: [
        "What use case fits a medical Q&A dataset?",
        "How many training examples do I need?",
        "What's the difference between instruction tuning and chat tuning?",
    ],
    3: [
        "Is my dataset large enough to fine-tune?",
        "What format does my CSV need to be in?",
        "How do I generate synthetic data for my use case?",
    ],
    4: [
        "What learning rate should I use?",
        "Will my config fit in 16GB VRAM?",
        "What LoRA rank is good for a 7B model?",
    ],
    5: [
        "Why is my loss not decreasing?",
        "Is this training curve normal?",
        "How do I know when to stop training?",
    ],
    6: [
        "What does my perplexity score mean?",
        "How do I interpret ROUGE-1 scores?",
        "Is my model good enough to deploy?",
    ],
    7: [
        "How do I run my model with Ollama?",
        "How do I use my adapter with transformers?",
        "What's the difference between adapter and merged model?",
    ],
}


def _hint_chip(text: str) -> rx.Component:
    return rx.button(
        text,
        on_click=AppState.set_chat_input(text),
        size="1",
        variant="soft",
        color_scheme="blue",
        cursor="pointer",
        text_align="left",
        white_space="normal",
        height="auto",
        padding="6px 10px",
    )


def _hints_for_step(step: int) -> rx.Component:
    hints = _STEP_HINTS.get(step, _STEP_HINTS[1])
    return rx.vstack(
        rx.text("Suggestions", font_size="0.72rem", color=c("text_muted"), font_weight="500"),
        *[_hint_chip(h) for h in hints],
        spacing="2",
        width="100%",
        padding="12px",
    )


def _dynamic_hints() -> rx.Component:
    return rx.match(
        FinetuneState.current_step,
        (1, _hints_for_step(1)),
        (2, _hints_for_step(2)),
        (3, _hints_for_step(3)),
        (4, _hints_for_step(4)),
        (5, _hints_for_step(5)),
        (6, _hints_for_step(6)),
        (7, _hints_for_step(7)),
        _hints_for_step(1),
    )


def _message_bubble(msg: dict) -> rx.Component:
    is_user = msg["role"] == "user"
    return rx.box(
        rx.markdown(
            msg["text"],
            font_size="0.82rem",
            color=rx.cond(is_user, "white", c("text_primary")),
        ),
        background=rx.cond(is_user, c("accent"), c("bg_input")),
        border_radius=rx.cond(is_user, "12px 12px 4px 12px", "12px 12px 12px 4px"),
        padding="8px 12px",
        max_width="88%",
        align_self=rx.cond(is_user, "flex-end", "flex-start"),
    )


def _messages_area() -> rx.Component:
    return rx.cond(
        AppState.chat_messages.length() == 0,
        # Empty state
        rx.vstack(
            rx.box(
                rx.icon("bot", size=28, color=c("text_muted")),
                padding="12px",
                border_radius="50%",
                background=c("bg_input"),
            ),
            rx.text(
                "TuneOS Assistant",
                font_size="0.9rem",
                font_weight="600",
                color=c("text_primary"),
            ),
            rx.text(
                "Knows your current model, dataset, and config. Ask anything.",
                font_size="0.78rem",
                color=c("text_muted"),
                text_align="center",
                max_width="200px",
            ),
            spacing="2",
            align="center",
            padding="24px 12px 8px",
            width="100%",
        ),
        # Message list
        rx.vstack(
            rx.foreach(AppState.chat_messages, _message_bubble),
            spacing="2",
            padding="12px",
            width="100%",
            align_items="stretch",
        ),
    )


def _input_row() -> rx.Component:
    return rx.hstack(
        rx.input(
            placeholder="Ask about your fine-tune...",
            value=AppState.chat_input,
            on_change=AppState.set_chat_input,
            on_key_down=AppState.handle_chat_key,
            disabled=AppState.is_chat_loading,
            flex="1",
            size="2",
            border_radius="8px",
        ),
        rx.button(
            rx.cond(
                AppState.is_chat_loading,
                rx.spinner(size="2"),
                rx.icon("send", size=14),
            ),
            on_click=AppState.send_chat_message,
            disabled=AppState.is_chat_loading | (AppState.chat_input == ""),
            size="2",
            color_scheme="blue",
            border_radius="8px",
        ),
        spacing="2",
        padding="8px 12px",
        border_top=f"1px solid {c('border')}",
        background=c("bg_primary"),
        width="100%",
    )


def _has_started() -> rx.Var:
    """True once the user has selected a model or moved past step 1."""
    return (FinetuneState.selected_model_id != "") | (FinetuneState.current_step > 1)


def chat_panel() -> rx.Component:
    """Collapsible context-aware chat panel — icon appears only after user starts working."""
    return rx.cond(
        _has_started(),
        rx.cond(
            AppState.chat_open,
            rx.box(
            rx.vstack(
                # Header
                rx.hstack(
                    rx.hstack(
                        rx.icon("bot", size=14, color=c("accent")),
                        rx.text("Assistant", font_size="0.82rem", font_weight="600", color=c("text_primary")),
                        spacing="2",
                        align="center",
                    ),
                    rx.spacer(),
                    rx.button(
                        rx.icon("x", size=14),
                        on_click=AppState.toggle_chat,
                        variant="ghost",
                        size="1",
                        color_scheme="gray",
                    ),
                    width="100%",
                    align="center",
                    padding="10px 12px 6px",
                    border_bottom=f"1px solid {c('border')}",
                ),
                # Messages or empty state + hints
                rx.box(
                    rx.vstack(
                        _messages_area(),
                        rx.cond(
                            AppState.chat_messages.length() == 0,
                            _dynamic_hints(),
                            rx.fragment(),
                        ),
                        spacing="0",
                        width="100%",
                    ),
                    flex="1",
                    overflow_y="auto",
                    width="100%",
                ),
                # Input
                _input_row(),
                spacing="0",
                height="100%",
                width="100%",
            ),
            width="272px",
            min_width="272px",
            height="100vh",
            border_left=f"1px solid {c('border')}",
            background=c("bg_primary"),
            flex_shrink="0",
            display="flex",
            flex_direction="column",
            overflow="hidden",
        ),
        # Collapsed: just show toggle button
        rx.box(
            rx.button(
                rx.icon("bot", size=16),
                on_click=AppState.toggle_chat,
                variant="soft",
                color_scheme="blue",
                size="2",
                border_radius="8px 0 0 8px",
            ),
            position="fixed",
            right="0",
            top="50%",
            transform="translateY(-50%)",
            z_index="100",
        ),
        ),
        rx.fragment(),  # nothing shown before user starts working
    )
