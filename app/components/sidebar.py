"""Neutral TuneOS sidebar."""
import reflex as rx

from app.state.app_state import AppState
from app.styles import c


def _nav_item(icon_name: str, label: str, active: bool = False, on_click=None) -> rx.Component:
    click_props = {"on_click": on_click} if on_click is not None else {}
    return rx.hstack(
        rx.icon(icon_name, size=18, color=c("text_secondary")),
        rx.text(label, font_size="0.95rem", color=c("text_primary")),
        spacing="3",
        align="center",
        width="100%",
        padding_x="14px",
        padding_y="10px",
        border_radius="8px",
        background=rx.cond(active, c("hover_strong"), "transparent"),
        cursor="pointer",
        _hover={"background": c("hover")},
        **click_props,
    )


def _project_row_static(name: str, created_at: str) -> rx.Component:
    return rx.hstack(
        rx.icon("folder", size=17, color=c("text_secondary")),
        rx.vstack(
            rx.text(
                name,
                font_size="0.92rem",
                color=c("text_primary"),
                overflow="hidden",
                text_overflow="ellipsis",
                white_space="nowrap",
                max_width="210px",
            ),
            rx.text(
                created_at,
                font_size="0.78rem",
                color=c("text_muted"),
                line_height="1.1",
            ),
            spacing="0",
            align_items="flex-start",
            min_width="0",
        ),
        spacing="3",
        align="center",
        width="100%",
        padding_x="14px",
        padding_y="9px",
        border_radius="8px",
        cursor="pointer",
        _hover={"background": c("hover")},
    )


def _project_row_dynamic(project: rx.Var) -> rx.Component:
    return rx.hstack(
        rx.icon("folder", size=17, color=c("text_secondary")),
        rx.vstack(
            rx.text(
                project.name,
                font_size="0.92rem",
                color=c("text_primary"),
                overflow="hidden",
                text_overflow="ellipsis",
                white_space="nowrap",
                max_width="210px",
            ),
            rx.text(
                project.created_at,
                font_size="0.78rem",
                color=c("text_muted"),
                line_height="1.1",
            ),
            spacing="0",
            align_items="flex-start",
            min_width="0",
        ),
        spacing="3",
        align="center",
        width="100%",
        padding_x="14px",
        padding_y="9px",
        border_radius="8px",
        cursor="pointer",
        _hover={"background": c("hover")},
    )


def _section_label(label: str) -> rx.Component:
    return rx.text(
        label,
        font_size="0.82rem",
        font_weight="500",
        color=c("text_muted"),
        padding_x="14px",
        padding_top="18px",
        padding_bottom="6px",
    )


def _panel_button() -> rx.Component:
    return rx.icon_button(
        rx.icon("panel-left", size=17),
        on_click=AppState.toggle_sidebar,
        variant="ghost",
        size="2",
        color=c("text_secondary"),
        border_radius="8px",
        cursor="pointer",
        _hover={"background": c("hover"), "color": c("text_primary")},
    )


def _expanded_sidebar() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            _panel_button(),
            rx.text("TuneOS", font_size="0.98rem", font_weight="600", color=c("text_primary")),
            spacing="3",
            align="center",
            width="100%",
            padding="14px",
        ),
        rx.vstack(
            _nav_item("square-pen", "New chat", active=True, on_click=AppState.new_project),
            _nav_item("database", "Datasets", on_click=rx.redirect("/datasets")),
            _nav_item("flask-conical", "Techniques"),
            spacing="1",
            width="100%",
            padding_x="8px",
        ),
        rx.vstack(
            _section_label("Projects"),
            rx.cond(
                AppState.projects.length() == 0,
                rx.text(
                    "No projects yet",
                    font_size="0.82rem",
                    color=c("text_muted"),
                    padding_x="14px",
                    padding_y="8px",
                ),
                rx.foreach(
                    AppState.projects,
                    _project_row_dynamic,
                ),
            ),
            spacing="1",
            width="100%",
            overflow_y="auto",
            flex="1",
            padding_x="8px",
            padding_bottom="12px",
        ),
        rx.hstack(
            rx.icon("settings", size=19, color=c("text_secondary")),
            rx.text("Settings", font_size="0.95rem", color=c("text_primary")),
            spacing="3",
            align="center",
            width="100%",
            padding="14px",
            border_top="1px solid",
            border_color=c("border"),
        ),
        spacing="0",
        height="100%",
        width="100%",
    )


def _collapsed_icon_btn(icon_name: str, on_click=None) -> rx.Component:
    click_props = {"on_click": on_click} if on_click is not None else {}
    return rx.icon_button(
        rx.icon(icon_name, size=18),
        variant="ghost",
        size="2",
        color=c("text_secondary"),
        border_radius="8px",
        cursor="pointer",
        _hover={"background": c("hover"), "color": c("text_primary")},
        **click_props,
    )


def _collapsed_sidebar() -> rx.Component:
    return rx.vstack(
        _panel_button(),
        _collapsed_icon_btn("square-pen", on_click=AppState.new_project),
        _collapsed_icon_btn("database", on_click=rx.redirect("/datasets")),
        _collapsed_icon_btn("flask-conical"),
        rx.spacer(),
        _collapsed_icon_btn("settings"),
        spacing="4",
        align="center",
        height="100%",
        width="100%",
        padding_y="14px",
    )


def sidebar() -> rx.Component:
    return rx.box(
        rx.cond(
            AppState.sidebar_collapsed,
            _collapsed_sidebar(),
            _expanded_sidebar(),
        ),
        background=c("bg_sidebar"),
        height="100vh",
        width="100%",
        border_right="1px solid",
        border_color=c("border"),
        overflow="hidden",
    )
