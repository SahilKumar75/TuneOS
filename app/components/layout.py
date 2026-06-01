"""TuneOS application shell."""

import reflex as rx

from app.components.settings_panel import settings_panel
from app.components.sidebar import sidebar
from app.pages.datasets import datasets_page
from app.pages.landing import landing_content
from app.state.app_state import AppState
from app.styles import c


def _models_placeholder() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading("Models", font_size="1.4rem", font_weight="600", color=c("text_primary")),
            rx.text(
                "Browse models from Hugging Face and GitHub — coming soon.",
                font_size="0.95rem",
                color=c("text_secondary"),
            ),
            spacing="3",
            align="center",
            justify="center",
            height="100%",
            width="100%",
        ),
        width="100%",
        height="100vh",
        display="flex",
        align_items="center",
        justify_content="center",
        background=c("bg_primary"),
    )


def _center_panel() -> rx.Component:
    return rx.cond(
        AppState.current_view == "datasets",
        datasets_page(),
        rx.cond(
            AppState.current_view == "models",
            _models_placeholder(),
            rx.cond(
                AppState.current_view == "settings",
                settings_panel(),
                landing_content(),
            ),
        ),
    )


def _resize_handle() -> rx.Component:
    return rx.box(
        width="1px",
        height="100vh",
        cursor="col-resize",
        background=c("border"),
        flex_shrink="0",
        _hover={"background": c("border_strong")},
    )


def two_panel_layout() -> rx.Component:
    return rx.hstack(
        rx.box(
            sidebar(),
            width=rx.cond(AppState.sidebar_collapsed, "56px", "280px"),
            min_width=rx.cond(AppState.sidebar_collapsed, "56px", "240px"),
            max_width=rx.cond(AppState.sidebar_collapsed, "56px", "340px"),
            height="100vh",
            flex_shrink="0",
            overflow="hidden",
            transition="width 0.25s ease, min-width 0.25s ease, max-width 0.25s ease",
        ),
        _resize_handle(),
        rx.box(
            _center_panel(),
            flex="1",
            height="100vh",
            overflow="hidden",
        ),
        spacing="0",
        width="100vw",
        height="100vh",
        overflow="hidden",
    )
