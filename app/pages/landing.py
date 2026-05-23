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
                placeholder=AppState.input_placeholder,
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
                rx.button(
                    rx.icon("hand", size=16),
                    rx.text(AppState.permission_label, font_size="0.88rem"),
                    rx.icon("chevron-down", size=15),
                    on_click=AppState.toggle_permission_selector,
                    variant="ghost",
                    size="2",
                    color=c("text_secondary"),
                    border_radius="8px",
                    cursor="pointer",
                    _hover={"background": c("hover"), "color": c("text_primary")},
                ),
                rx.spacer(),
                rx.button(
                    rx.text(AppState.tab_label, font_size="0.88rem"),
                    rx.icon("chevron-down", size=15),
                    on_click=AppState.toggle_model_selector,
                    variant="ghost",
                    size="2",
                    color=c("text_secondary"),
                    border_radius="8px",
                    cursor="pointer",
                    _hover={"background": c("hover"), "color": c("text_primary")},
                ),
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


def _chat_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("Chat", font_size="1rem", font_weight="600", color=c("text_primary")),
                rx.spacer(),
                rx.icon("message-square", size=17, color=c("text_secondary")),
                align="center",
                width="100%",
            ),
            rx.vstack(
                rx.box(
                    rx.text(
                        "I can help configure datasets, techniques, LoRA settings, and training runs for this model.",
                        font_size="0.9rem",
                        color=c("text_primary"),
                        line_height="1.45",
                    ),
                    padding="12px",
                    background=c("hover"),
                    border_radius="12px",
                    width="100%",
                ),
                spacing="3",
                width="100%",
                flex="1",
                overflow_y="auto",
            ),
            rx.hstack(
                rx.input(
                    placeholder="Ask about this model...",
                    value=AppState.chat_input,
                    on_change=AppState.set_chat_input,
                    size="2",
                    width="100%",
                    background=c("bg_input"),
                    border="1px solid",
                    border_color=c("border"),
                ),
                rx.icon_button(
                    rx.icon("arrow-up", size=17),
                    variant="solid",
                    size="2",
                    border_radius="999px",
                    background=rx.color_mode_cond(light="#171717", dark="#ededed"),
                    color=rx.color_mode_cond(light="#ffffff", dark="#171717"),
                ),
                spacing="2",
                align="center",
                width="100%",
            ),
            spacing="4",
            height="100%",
            width="100%",
        ),
        width="340px",
        min_width="320px",
        height="100vh",
        padding="18px",
        background=c("bg_sidebar"),
        border_left="1px solid",
        border_color=c("border"),
    )


def _workspace_content() -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.vstack(
                rx.badge(AppState.preview_source_label, color_scheme="blue", variant="soft"),
                rx.heading(
                    AppState.preview_title,
                    font_size="2rem",
                    font_weight="600",
                    line_height="1.15",
                    color=c("text_primary"),
                    text_align="center",
                ),
                rx.text(AppState.preview_meta, font_size="0.95rem", color=c("text_secondary"), text_align="center"),
                rx.box(
                    rx.text(
                        AppState.preview_summary,
                        font_size="1rem",
                        line_height="1.6",
                        color=c("text_primary"),
                        text_align="center",
                    ),
                    width="min(680px, calc(100vw - 520px))",
                    padding="20px",
                    background=c("bg_card"),
                    border="1px solid",
                    border_color=c("border"),
                    border_radius="16px",
                ),
                rx.text(AppState.preview_url, font_size="0.82rem", color=c("text_muted"), text_align="center"),
                spacing="4",
                align="center",
                justify="center",
                min_height="100vh",
                width="100%",
                padding="48px",
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
