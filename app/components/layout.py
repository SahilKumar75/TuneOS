"""TuneOS application shell."""
import reflex as rx
from app.components.sidebar import sidebar
from app.pages.landing import landing_content
from app.state.app_state import AppState
from app.styles import c


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
        ),
        _resize_handle(),
        rx.box(
            landing_content(),
            flex="1",
            height="100vh",
            overflow="hidden",
        ),
        spacing="0",
        width="100vw",
        height="100vh",
        overflow="hidden",
    )
