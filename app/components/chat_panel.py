"""Single consolidated chat assistant panel.

Keeps the original landing chat's neutral, ChatGPT-style look (model selector,
plain message bubbles, copy action, minimal input) and folds in the newer
features that were briefly split into a second panel:

  - step-aware suggestions that follow the fine-tune wizard,
  - a collapsible panel with an edge toggle,
  - gated visibility (only after the user starts working).

One component, mounted once in the app shell (`layout.two_panel_layout`). The
panel reads/writes the shared `AppState` chat state, and `FinetuneState` only for
context (current step / whether a model is selected).
"""

from __future__ import annotations

import reflex as rx

from app.state.app_state import AppState
from app.state.finetune_state import FinetuneState
from app.styles import c

# Strong ease-out curve for press feedback (Emil Kowalski design-eng guidance).
_EASE_OUT = "cubic-bezier(0.23, 1, 0.32, 1)"

# Step-aware suggestions for the fine-tune wizard (steps 1–7).
_STEP_HINTS: dict[int, list[str]] = {
    1: [
        "Which model is best for instruction following?",
        "Compare Mistral 7B vs Llama 3 8B for fine-tuning",
        "What VRAM do I need for QLoRA on a 7B model?",
    ],
    2: [
        "What use case fits a medical Q&A dataset?",
        "How many training examples do I need?",
        "Instruction tuning vs chat tuning — which fits?",
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
        "Adapter vs merged model — what's the difference?",
    ],
}

# Suggestions when browsing a model on the landing/preview view.
_MODEL_HINTS = [
    "What LoRA rank works best for this model?",
    "Suggest a training config for my dataset size",
    "How much VRAM do I need to fine-tune this?",
]


def _model_option(m: rx.Var[dict]) -> rx.Component:
    return rx.select.item(m["label"], value=m["id"])


def _action_icon_btn(icon_name: str, on_click=None) -> rx.Component:
    extra = {"on_click": on_click} if on_click is not None else {}
    return rx.icon_button(
        rx.icon(icon_name, size=14),
        variant="ghost",
        size="1",
        color=c("text_muted"),
        border_radius="6px",
        cursor="pointer",
        _hover={"color": c("text_primary"), "background": c("hover")},
        **extra,
    )


def _streaming_cursor() -> rx.Component:
    """Animated streaming cursor for AI responses."""
    return rx.box(
        class_name="streaming-cursor",
        width="2px",
        height="1em",
        background=c("text_primary"),
        display="inline-block",
        margin_left="2px",
        vertical_align="text-bottom",
        style={
            "@keyframes blink": {
                "0%, 50%": {"opacity": "1"},
                "51%, 100%": {"opacity": "0"},
            },
            "animation": "blink 1s ease-in-out infinite",
        },
    )


def _chat_message(msg: rx.Var[dict[str, str]]) -> rx.Component:
    is_user = msg["role"] == "user"
    is_streaming = (msg["role"] == "assistant") & (AppState.is_chat_loading)
    
    return rx.cond(
        is_user,
        # User message — right-aligned subtle pill (original neutral look).
        rx.hstack(
            rx.spacer(),
            rx.box(
                rx.text(
                    msg["text"], font_size="0.9rem", line_height="1.55", color=c("text_primary")
                ),
                padding="10px 16px",
                background=rx.color_mode_cond(light="#f0f0f0", dark="#2a2a2a"),
                border_radius="20px",
                max_width="72%",
            ),
            width="100%",
            align="start",
        ),
        # Assistant message — plain text left, copy action below, with streaming effect.
        rx.vstack(
            rx.box(
                rx.markdown(
                    msg["text"],
                    font_size="0.9rem",
                    color=c("text_primary"),
                    class_name=rx.cond(is_streaming, "streaming-text", ""),
                ),
                rx.cond(
                    is_streaming,
                    _streaming_cursor(),
                    rx.fragment(),
                ),
                display="flex",
                align_items="flex-end",
                width="100%",
            ),
            rx.hstack(
                _action_icon_btn("copy", on_click=rx.set_clipboard(msg["text"])),
                spacing="0",
                align="center",
            ),
            spacing="1",
            align="start",
            width="100%",
            padding_left="2px",
        ),
    )


def _hint_rows(hints: list[str]) -> rx.Component:
    """Neutral suggestion card (original style) — clickable rows that prefill input."""
    n = len(hints)
    return rx.box(
        rx.vstack(
            *[
                rx.text(
                    h,
                    font_size="0.8rem",
                    color=c("text_secondary"),
                    line_height="1.4",
                    width="100%",
                    padding_y="8px",
                    border_bottom="1px solid" if i < n - 1 else "none",
                    border_color=c("border"),
                    cursor="pointer",
                    on_click=AppState.set_chat_input(h),
                    transition="color 120ms ease",
                    _hover={"color": c("text_primary")},
                )
                for i, h in enumerate(hints)
            ],
            spacing="0",
            width="100%",
        ),
        width="100%",
        padding="4px 14px",
        background=c("bg_card"),
        border="1px solid",
        border_color=c("border"),
        border_radius="12px",
    )


def _suggestions() -> rx.Component:
    """Step-aware in the wizard; model-preview hints otherwise — same neutral card."""
    return rx.cond(
        AppState.active_tab_is_finetune,
        rx.match(
            FinetuneState.current_step,
            *[(step, _hint_rows(hints)) for step, hints in _STEP_HINTS.items()],
            _hint_rows(_STEP_HINTS[1]),
        ),
        _hint_rows(_MODEL_HINTS),
    )


def _header() -> rx.Component:
    return rx.hstack(
        rx.select.root(
            rx.select.trigger(
                placeholder="Auto (smart route)",
                size="1",
                variant="ghost",
                color=c("text_secondary"),
                font_size="0.78rem",
                cursor="pointer",
            ),
            rx.select.content(
                rx.foreach(AppState.CHAT_MODELS, _model_option),
                position="popper",
            ),
            value=AppState.chat_model,
            on_change=AppState.set_chat_model,
        ),
        rx.spacer(),
        rx.cond(
            AppState.last_used_model != "",
            rx.badge(
                AppState.last_used_model,
                variant="soft",
                color_scheme="gray",
                size="1",
                font_size="0.68rem",
                max_width="140px",
                overflow="hidden",
                text_overflow="ellipsis",
                white_space="nowrap",
            ),
            rx.fragment(),
        ),
        rx.icon_button(
            rx.icon("panel-right-close", size=15),
            on_click=AppState.toggle_chat,
            variant="ghost",
            size="1",
            color=c("text_muted"),
            cursor="pointer",
            transition=f"transform 140ms {_EASE_OUT}",
            _hover={"color": c("text_primary"), "background": c("hover")},
            _active={"transform": "scale(0.96)"},
        ),
        align="center",
        width="100%",
        padding_bottom="10px",
        border_bottom="1px solid",
        border_color=c("border"),
        spacing="2",
    )


def _empty_state() -> rx.Component:
    return rx.vstack(
        rx.vstack(
            rx.icon("bot", size=28, color=c("text_muted")),
            rx.text(
                "Ask about your fine-tune",
                font_size="0.92rem",
                font_weight="500",
                color=c("text_primary"),
            ),
            rx.text(
                "Knows your current model, dataset, and config. "
                "Ask about architecture, training, or dataset prep.",
                font_size="0.82rem",
                color=c("text_muted"),
                text_align="center",
                line_height="1.5",
            ),
            spacing="2",
            align="center",
        ),
        _suggestions(),
        spacing="5",
        align="center",
        justify="center",
        flex="1",
        width="100%",
        padding_y="24px",
    )


def _message_list() -> rx.Component:
    return rx.vstack(
        rx.foreach(AppState.chat_messages, _chat_message),
        spacing="4",
        width="100%",
        flex="1",
        overflow_y="auto",
        padding_y="8px",
        padding_x="4px",
    )


def _input_area() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.input(
                placeholder="Ask anything",
                value=AppState.chat_input,
                on_change=AppState.set_chat_input,
                on_key_down=AppState.handle_chat_key,
                size="2",
                width="100%",
                background="transparent",
                border="none",
                box_shadow="none",
                outline="none",
                font_size="0.9rem",
                color=c("text_primary"),
                _placeholder={"color": c("text_muted")},
                _focus={"outline": "none", "box_shadow": "none"},
            ),
            rx.hstack(
                rx.icon_button(
                    rx.icon("plus", size=15),
                    variant="ghost",
                    size="1",
                    color=c("text_secondary"),
                    border_radius="6px",
                    cursor="pointer",
                    _hover={"background": c("hover"), "color": c("text_primary")},
                ),
                rx.spacer(),
                rx.cond(
                    AppState.is_chat_loading,
                    rx.icon_button(
                        rx.icon("square", size=13, fill="currentColor"),
                        variant="solid",
                        size="2",
                        border_radius="999px",
                        background=rx.color_mode_cond(light="#171717", dark="#ededed"),
                        color=rx.color_mode_cond(light="#ffffff", dark="#171717"),
                        cursor="not-allowed",
                        disabled=True,
                    ),
                    rx.icon_button(
                        rx.icon("arrow-up", size=15),
                        on_click=AppState.send_chat_message,
                        variant="solid",
                        size="2",
                        border_radius="999px",
                        background=rx.color_mode_cond(light="#171717", dark="#ededed"),
                        color=rx.color_mode_cond(light="#ffffff", dark="#171717"),
                        cursor="pointer",
                        transition=f"transform 140ms {_EASE_OUT}",
                        _hover={"background": rx.color_mode_cond(light="#000000", dark="#ffffff")},
                        _active={"transform": "scale(0.96)"},
                    ),
                ),
                spacing="1",
                align="center",
                width="100%",
            ),
            spacing="1",
            width="100%",
        ),
        width="100%",
        padding="10px 12px",
        background=c("bg_input"),
        border_radius="16px",
        border="none",
        box_shadow="none",
    )


def _open_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            _header(),
            rx.cond(
                AppState.chat_messages.length() == 0,
                _empty_state(),
                _message_list(),
            ),
            _input_area(),
            spacing="3",
            height="100%",
            width="100%",
        ),
        width="380px",
        min_width="360px",
        height="100vh",
        padding_top="10px",
        padding_x="16px",
        padding_bottom="16px",
        background=c("bg_sidebar"),
        border_left="1px solid",
        border_color=c("border"),
        flex_shrink="0",
    )


def _collapsed() -> rx.Component:
    return rx.box(
        rx.icon_button(
            rx.icon("bot", size=18),
            on_click=AppState.toggle_chat,
            variant="soft",
            color_scheme="gray",
            size="2",
            border_radius="8px 0 0 8px",
            cursor="pointer",
            transition=f"transform 160ms {_EASE_OUT}",
            _active={"transform": "scale(0.96)"},
        ),
        position="fixed",
        right="0",
        top="50%",
        transform="translateY(-50%)",
        z_index="50",
    )


def _has_started() -> rx.Var:
    """Show the assistant only while the user is in the workspace.

    ``workspace_active`` is the real lifecycle signal — true once a project is
    opened, reset on return to the start screen — so the panel tracks the
    workspace instead of sticky FinetuneState fields.
    """
    return AppState.workspace_active


def chat_panel() -> rx.Component:
    """Single collapsible, context-aware assistant — mounted once in the shell."""
    return rx.cond(
        _has_started(),
        rx.cond(AppState.chat_open, _open_panel(), _collapsed()),
        rx.fragment(),
    )
