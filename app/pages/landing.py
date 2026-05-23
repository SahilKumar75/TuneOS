"""TuneOS start screen with a focused software-style composer."""
import reflex as rx

from app.state.app_state import AppState
from app.styles import c


def _action_menu_item(icon_name: str, label: str, tab: str) -> rx.Component:
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
        on_click=AppState.select_tab_from_menu(tab),
    )


def _action_menu() -> rx.Component:
    return rx.cond(
        AppState.show_action_menu,
        rx.box(
            rx.vstack(
                _action_menu_item("upload", "Upload dataset", "huggingface"),
                _action_menu_item("globe", "Use Hugging Face model", "huggingface"),
                _action_menu_item("git-branch", "Import from GitHub", "github"),
                _action_menu_item("folder-open", "Use local model", "local"),
                spacing="1",
                width="100%",
            ),
            position="absolute",
            bottom="48px",
            left="14px",
            z_index="20",
            width="236px",
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
            bottom="48px",
            right="96px",
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
        _action_menu(),
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
                rx.icon_button(
                    rx.cond(
                        AppState.show_action_menu,
                        rx.icon("x", size=18),
                        rx.icon("plus", size=18),
                    ),
                    on_click=AppState.toggle_action_menu,
                    variant="ghost",
                    size="2",
                    color=c("text_secondary"),
                    border_radius="999px",
                    cursor="pointer",
                    _hover={"background": c("hover"), "color": c("text_primary")},
                ),
                rx.button(
                    rx.icon("hand", size=16),
                    rx.text("Default permissions", font_size="0.88rem"),
                    rx.icon("chevron-down", size=15),
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
                    rx.icon("mic", size=17),
                    variant="ghost",
                    size="2",
                    color=c("text_secondary"),
                    border_radius="999px",
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


def landing_content() -> rx.Component:
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
            rx.hstack(
                rx.icon("folder", size=16, color=c("text_secondary")),
                rx.text("TuneOS", font_size="0.9rem", color=c("text_secondary")),
                rx.icon("chevron-down", size=15, color=c("text_muted")),
                spacing="2",
                align="center",
                width="min(780px, calc(100vw - 80px))",
                padding_x="16px",
                padding_y="12px",
                background=c("hover"),
                border_bottom_radius="22px",
                margin_top="-8px",
            ),
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
