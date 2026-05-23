"""TuneOS application shell."""
import reflex as rx
from app.components.sidebar import sidebar
from app.pages.landing import landing_content
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
            width="280px",
            min_width="240px",
            max_width="340px",
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
