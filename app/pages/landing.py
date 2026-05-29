"""TuneOS start screen with a focused software-style composer."""
import reflex as rx

from app.state.app_state import AppState
from app.styles import c


def _dropdown_item(icon_name: str, label: str, on_click) -> rx.Component:
    return rx.hstack(
        rx.icon(icon_name, size=16, color=c("text_secondary")),
        rx.text(label, font_size="0.875rem", color=c("text_primary")),
        spacing="3",
        align="center",
        width="100%",
        padding_x="12px",
        padding_y="9px",
        border_radius="8px",
        cursor="pointer",
        _hover={"background": c("hover")},
        on_click=on_click,
    )


def _permission_selector() -> rx.Component:
    return rx.cond(
        AppState.show_permission_selector,
        rx.box(
            rx.vstack(
                _dropdown_item("chart-no-axes-column", "Analytics", AppState.select_permission_mode("analytics")),
                _dropdown_item("activity", "Training", AppState.select_permission_mode("training")),
                _dropdown_item("sliders-horizontal", "Fine-tuning", AppState.select_permission_mode("finetuning")),
                spacing="1",
                width="100%",
            ),
            position="absolute",
            top="100%",
            left="12px",
            margin_top="8px",
            z_index="20",
            width="210px",
            padding="6px",
            background=c("menu_bg"),
            border="1px solid",
            border_color=c("border_strong"),
            border_radius="12px",
            box_shadow=rx.color_mode_cond(
                light="0 14px 40px rgba(0,0,0,0.12)",
                dark="0 14px 40px rgba(0,0,0,0.38)",
            ),
        ),
    )


def _model_selector_item(icon_name: str, label: str, tab: str) -> rx.Component:
    active = AppState.active_tab == tab
    return rx.hstack(
        rx.icon(icon_name, size=15, color=c("text_secondary")),
        rx.text(label, font_size="0.85rem", color=c("text_primary"), flex="1"),
        rx.cond(active, rx.icon("check", size=15, color=c("accent")), rx.fragment()),
        spacing="2",
        align="center",
        width="100%",
        padding_x="12px",
        padding_y="9px",
        border_radius="8px",
        cursor="pointer",
        _hover={"background": c("hover")},
        on_click=AppState.select_tab_from_menu(tab),
    )


def _model_selector() -> rx.Component:
    return rx.cond(
        AppState.show_model_selector,
        rx.box(
            rx.vstack(
                _model_selector_item("globe", "Hugging Face", "huggingface"),
                _model_selector_item("git-branch", "GitHub", "github"),
                _model_selector_item("hard-drive", "Local", "local"),
                spacing="1",
                width="100%",
            ),
            position="absolute",
            top="100%",
            right="96px",
            margin_top="8px",
            z_index="20",
            width="190px",
            padding="6px",
            background=c("menu_bg"),
            border="1px solid",
            border_color=c("border_strong"),
            border_radius="12px",
            box_shadow=rx.color_mode_cond(
                light="0 14px 40px rgba(0,0,0,0.12)",
                dark="0 14px 40px rgba(0,0,0,0.38)",
            ),
        ),
    )


def _composer() -> rx.Component:
    return rx.box(
        _permission_selector(),
        _model_selector(),
        rx.vstack(
            rx.input(
                placeholder="Paste a Hugging Face link, GitHub URL, or local model path...",
                value=AppState.current_input_value,
                on_change=AppState.handle_input_change,
                variant="soft",
                size="3",
                width="100%",
                min_height="88px",
                align_items="flex-start",
                background="transparent",
                border="none",
                color=c("text_primary"),
                font_size="0.98rem",
                _placeholder={"color": c("text_muted")},
                _focus={"outline": "none", "box_shadow": "none"},
            ),
            rx.hstack(
                rx.icon_button(
                    rx.icon("plus", size=17),
                    on_click=AppState.start_project,
                    variant="ghost",
                    size="2",
                    color=c("text_secondary"),
                    border_radius="8px",
                    cursor="pointer",
                    _hover={"background": c("hover"), "color": c("text_primary")},
                ),
                rx.spacer(),
                rx.icon_button(
                    rx.icon("arrow-up", size=20),
                    on_click=AppState.start_project,
                    variant="solid",
                    size="3",
                    background=rx.color_mode_cond(light="#171717", dark="#ededed"),
                    color=rx.color_mode_cond(light="#ffffff", dark="#171717"),
                    border_radius="999px",
                    cursor="pointer",
                    _hover={
                        "background": rx.color_mode_cond(light="#000000", dark="#ffffff"),
                    },
                ),
                spacing="2",
                align="center",
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        position="relative",
        width="min(780px, calc(100vw - 80px))",
        background=c("input_bg"),
        border="1px solid",
        border_color=c("input_border"),
        border_radius="22px",
        padding="12px",
        box_shadow=rx.color_mode_cond(
            light="0 8px 28px rgba(0,0,0,0.06)",
            dark="0 8px 28px rgba(0,0,0,0.24)",
        ),
    )


def _preview_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.badge(AppState.preview_source_label, color_scheme="blue", variant="soft"),
                rx.spacer(),
                rx.button(
                    "Change",
                    on_click=AppState.cancel_preview,
                    variant="ghost",
                    size="2",
                    color=c("text_secondary"),
                    cursor="pointer",
                ),
                spacing="3",
                align="center",
                width="100%",
            ),
            rx.heading(AppState.preview_title, font_size="1.2rem", font_weight="600", color=c("text_primary")),
            rx.text(AppState.preview_meta, font_size="0.88rem", color=c("text_secondary")),
            rx.text(
                AppState.preview_summary,
                font_size="0.95rem",
                line_height="1.55",
                color=c("text_primary"),
            ),
            rx.text(AppState.preview_url, font_size="0.82rem", color=c("text_muted")),
            rx.hstack(
                rx.button(
                    "Yes, use this",
                    on_click=AppState.confirm_preview,
                    size="2",
                    background=rx.color_mode_cond(light="#171717", dark="#ededed"),
                    color=rx.color_mode_cond(light="#ffffff", dark="#171717"),
                    border_radius="999px",
                    cursor="pointer",
                ),
                rx.button(
                    "Cancel",
                    on_click=AppState.cancel_preview,
                    variant="outline",
                    size="2",
                    border_radius="999px",
                    cursor="pointer",
                ),
                spacing="3",
                align="center",
            ),
            spacing="3",
            align_items="flex-start",
            width="100%",
        ),
        width="min(780px, calc(100vw - 80px))",
        padding="18px",
        background=c("bg_card"),
        border="1px solid",
        border_color=c("border"),
        border_radius="16px",
        box_shadow=rx.color_mode_cond(
            light="0 8px 28px rgba(0,0,0,0.06)",
            dark="0 8px 28px rgba(0,0,0,0.22)",
        ),
    )


def _preview_status() -> rx.Component:
    return rx.cond(
        AppState.preview_loading,
        rx.hstack(
            rx.spinner(size="2"),
            rx.text("Fetching link information...", font_size="0.92rem", color=c("text_secondary")),
            spacing="3",
            align="center",
            justify="center",
            width="min(780px, calc(100vw - 80px))",
            padding="14px",
        ),
        rx.cond(
            AppState.preview_ready,
            _preview_panel(),
            rx.cond(
                AppState.preview_error != "",
                rx.text(
                    AppState.preview_error,
                    font_size="0.92rem",
                    color=c("error"),
                    width="min(780px, calc(100vw - 80px))",
                    text_align="center",
                ),
                rx.fragment(),
            ),
        ),
    )


def _tag_pill(tag: rx.Var[str]) -> rx.Component:
    return rx.badge(
        tag,
        variant="soft",
        color_scheme="gray",
        size="1",
        font_size="0.72rem",
        border_radius="999px",
    )


def _action_card(icon_name: str, label: str, desc: str) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.icon(icon_name, size=22, color=c("text_secondary")),
            rx.text(label, font_size="0.88rem", font_weight="600", color=c("text_primary")),
            rx.text(desc, font_size="0.75rem", color=c("text_muted"), line_height="1.3"),
            spacing="1",
            align_items="flex-start",
        ),
        padding="14px",
        background=c("bg_card"),
        border="1px solid",
        border_color=c("border"),
        border_radius="12px",
        cursor="pointer",
        _hover={"border_color": c("border_strong"), "background": c("hover")},
        transition="all 0.15s ease",
        flex="1",
        min_width="140px",
    )


def _stat_item(label: str, value: rx.Var[str]) -> rx.Component:
    return rx.vstack(
        rx.text(value, font_size="0.95rem", font_weight="600", color=c("text_primary")),
        rx.text(label, font_size="0.72rem", color=c("text_muted")),
        spacing="0",
        align_items="center",
    )


def _chat_message(msg: rx.Var[dict[str, str]]) -> rx.Component:
    is_user = msg["role"] == "user"
    return rx.box(
        rx.text(
            msg["text"],
            font_size="0.88rem",
            line_height="1.5",
        ),
        padding="10px 14px",
        background=rx.cond(
            is_user,
            rx.color_mode_cond(light="#171717", dark="#ededed"),
            c("hover"),
        ),
        color=rx.cond(
            is_user,
            rx.color_mode_cond(light="#ffffff", dark="#171717"),
            c("text_primary"),
        ),
        border_radius=rx.cond(is_user, "16px 16px 4px 16px", "16px 16px 16px 4px"),
        max_width="88%",
        align_self=rx.cond(is_user, "flex-end", "flex-start"),
    )


def _chat_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.icon("message-square", size=16, color=c("accent")),
                rx.text("Chat", font_size="0.95rem", font_weight="600", color=c("text_primary")),
                rx.spacer(),
                rx.icon("sparkles", size=14, color=c("accent")),
                align="center",
                spacing="2",
                width="100%",
                padding_bottom="10px",
                border_bottom="1px solid",
                border_color=c("border"),
            ),
            rx.vstack(
                rx.cond(
                    AppState.chat_messages.length() == 0,
                    rx.vstack(
                        rx.icon("bot", size=28, color=c("text_muted")),
                        rx.text(
                            "Ask about this repository",
                            font_size="0.92rem",
                            font_weight="500",
                            color=c("text_secondary"),
                        ),
                        rx.text(
                            "I can help with datasets, LoRA config, training setup, and model analysis.",
                            font_size="0.82rem",
                            color=c("text_muted"),
                            line_height="1.4",
                            text_align="center",
                        ),
                        spacing="2",
                        align="center",
                        justify="center",
                        padding="32px 16px",
                        width="100%",
                    ),
                    rx.foreach(
                        AppState.chat_messages,
                        _chat_message,
                    ),
                ),
                spacing="2",
                width="100%",
                flex="1",
                overflow_y="auto",
                align_items="stretch",
            ),
            rx.box(
                rx.hstack(
                    rx.input(
                        placeholder="Ask about this model...",
                        value=AppState.chat_input,
                        on_change=AppState.set_chat_input,
                        size="2",
                        flex="1",
                        background="transparent",
                        border="none",
                        _focus={"outline": "none", "box_shadow": "none"},
                    ),
                    rx.icon_button(
                        rx.icon("send", size=15),
                        on_click=AppState.send_chat_message,
                        variant="ghost",
                        size="2",
                        color=c("text_secondary"),
                        cursor="pointer",
                        _hover={"color": c("accent")},
                    ),
                    spacing="2",
                    align="center",
                    width="100%",
                ),
                width="100%",
                padding="8px 12px",
                background=c("bg_input"),
                border="1px solid",
                border_color=c("border"),
                border_radius="12px",
            ),
            spacing="3",
            height="100%",
            width="100%",
        ),
        width="360px",
        min_width="340px",
        height="100vh",
        padding="18px",
        background=c("bg_sidebar"),
        border_left="1px solid",
        border_color=c("border"),
    )


def _overview_panel() -> rx.Component:
    return rx.vstack(
        # ── Title row with actions ──────────────────────────────
        rx.hstack(
            rx.vstack(
                rx.hstack(
                    rx.badge(AppState.preview_source_label, color_scheme="blue", variant="soft", size="1"),
                    rx.cond(
                        AppState.preview_license != "",
                        rx.badge(AppState.preview_license, color_scheme="green", variant="soft", size="1"),
                        rx.fragment(),
                    ),
                    spacing="2",
                ),
                rx.heading(
                    AppState.preview_title,
                    font_size="1.6rem",
                    font_weight="700",
                    line_height="1.2",
                    color=c("text_primary"),
                ),
                spacing="2",
                align_items="flex-start",
                flex="1",
            ),
            rx.link(
                rx.icon("external-link", size=16, color=c("text_muted")),
                href=AppState.preview_url,
                is_external=True,
            ),
            align="flex-start",
            width="100%",
        ),
        # ── Stats row ───────────────────────────────────────────
        rx.hstack(
            rx.cond(
                AppState.preview_pipeline != "",
                rx.hstack(
                    rx.icon("cpu", size=14, color=c("text_muted")),
                    rx.text(AppState.preview_pipeline, font_size="0.82rem", color=c("text_secondary")),
                    spacing="1",
                    align="center",
                ),
                rx.fragment(),
            ),
            rx.cond(
                AppState.preview_downloads != "",
                rx.hstack(
                    rx.icon("download", size=14, color=c("text_muted")),
                    rx.text(AppState.preview_downloads, font_size="0.82rem", color=c("text_secondary")),
                    spacing="1",
                    align="center",
                ),
                rx.fragment(),
            ),
            rx.cond(
                AppState.preview_likes != "",
                rx.hstack(
                    rx.icon("heart", size=14, color=c("text_muted")),
                    rx.text(AppState.preview_likes, font_size="0.82rem", color=c("text_secondary")),
                    spacing="1",
                    align="center",
                ),
                rx.fragment(),
            ),
            spacing="4",
            flex_wrap="wrap",
        ),
        # ── Tags ────────────────────────────────────────────────
        rx.cond(
            AppState.preview_tags.length() > 0,
            rx.hstack(
                rx.foreach(AppState.preview_tags, _tag_pill),
                spacing="1",
                flex_wrap="wrap",
            ),
            rx.fragment(),
        ),
        # ── Description card ────────────────────────────────────
        rx.box(
            rx.text(
                AppState.preview_summary,
                font_size="0.92rem",
                line_height="1.65",
                color=c("text_primary"),
            ),
            width="100%",
            padding="16px",
            background=c("bg_card"),
            border="1px solid",
            border_color=c("border"),
            border_radius="12px",
        ),
        # ── URL ─────────────────────────────────────────────────
        rx.text(AppState.preview_url, font_size="0.75rem", color=c("text_muted")),
        # ── Action grid ─────────────────────────────────────────
        rx.hstack(
            _action_card("zap", "Train", "Fine-tune with LoRA/QLoRA"),
            _action_card("bar-chart-2", "Analyze", "Model architecture & metrics"),
            _action_card("refresh-cw", "Convert", "GGUF, SafeTensors formats"),
            _action_card("notebook", "Notebook", "Open Jupyter explorer"),
            spacing="3",
            width="100%",
            flex_wrap="wrap",
        ),
        spacing="4",
        align_items="flex-start",
        width="100%",
        padding="28px 32px",
        max_width="800px",
    )


def _workspace_content() -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.button(
                        rx.icon("arrow-left", size=16),
                        rx.text("Back", font_size="0.88rem"),
                        on_click=AppState.cancel_preview,
                        variant="ghost",
                        size="2",
                        color=c("text_secondary"),
                        border_radius="8px",
                        cursor="pointer",
                        _hover={"background": c("hover"), "color": c("text_primary")},
                    ),
                    rx.spacer(),
                    rx.text(
                        AppState.preview_title,
                        font_size="0.85rem",
                        color=c("text_muted"),
                        overflow="hidden",
                        text_overflow="ellipsis",
                        white_space="nowrap",
                    ),
                    rx.spacer(),
                    width="100%",
                    padding="12px 28px",
                    border_bottom="1px solid",
                    border_color=c("border"),
                    align="center",
                ),
                _overview_panel(),
                spacing="0",
                align_items="flex-start",
                width="100%",
            ),
            flex="1",
            height="100vh",
            overflow_y="auto",
            background=c("bg_primary"),
        ),
        _chat_panel(),
        spacing="0",
        width="100%",
        height="100vh",
        overflow="hidden",
    )


def _start_content() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading(
                "What should we work on in TuneOS?",
                font_size="2.15rem",
                font_weight="500",
                line_height="1.2",
                color=c("text_primary"),
                text_align="center",
            ),
            _composer(),
            _preview_status(),
            spacing="6",
            align="center",
            justify="center",
            min_height="100vh",
            width="100%",
            padding_x="32px",
            padding_y="48px",
        ),
        background=c("bg_primary"),
        min_height="100vh",
        width="100%",
        overflow_y="auto",
    )


def landing_content() -> rx.Component:
    return rx.cond(
        AppState.workspace_active,
        _workspace_content(),
        _start_content(),
    )
