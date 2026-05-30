"""VS Code-style workspace tab bar."""

import reflex as rx

from app.state.app_state import AppState, WorkspaceTab
from app.styles import c


def _tab_item(tab: WorkspaceTab) -> rx.Component:
    is_active = AppState.active_workspace_tab_id == tab.id
    is_editing = AppState.editing_tab_id == tab.id
    return rx.box(
        rx.hstack(
            # Icon driven by tab type
            rx.cond(
                tab.tab_type == "notebook",
                rx.icon(
                    "notebook-pen",
                    size=13,
                    color=rx.cond(is_active, c("accent"), c("text_muted")),
                    flex_shrink="0",
                ),
                rx.icon(
                    "cpu",
                    size=13,
                    color=rx.cond(is_active, c("accent"), c("text_muted")),
                    flex_shrink="0",
                ),
            ),
            # Title: inline input when editing, plain text otherwise
            rx.cond(
                is_editing,
                rx.input(
                    value=AppState.editing_tab_title,
                    on_change=AppState.update_editing_title,
                    on_blur=AppState.save_tab_title,
                    on_key_down=AppState.handle_tab_rename_key,
                    auto_focus=True,
                    size="1",
                    font_size="0.78rem",
                    width="130px",
                    background="transparent",
                    border="1px solid",
                    border_color=c("accent"),
                    border_radius="4px",
                    padding_x="4px",
                    color=c("text_primary"),
                    _focus={"outline": "none", "box_shadow": "none"},
                ),
                rx.text(
                    tab.title,
                    font_size="0.78rem",
                    color=rx.cond(is_active, c("text_primary"), c("text_secondary")),
                    max_width="160px",
                    overflow="hidden",
                    text_overflow="ellipsis",
                    white_space="nowrap",
                    on_double_click=AppState.start_editing_tab(tab.id, tab.title),
                    cursor="text",
                ),
            ),
            # Close button (only for closeable tabs)
            rx.cond(
                tab.closeable,
                rx.box(
                    rx.icon("x", size=11),
                    padding="2px",
                    border_radius="3px",
                    cursor="pointer",
                    color=c("text_muted"),
                    flex_shrink="0",
                    _hover={"color": c("text_primary"), "background": c("hover_strong")},
                    on_click=AppState.close_workspace_tab(tab.id),
                ),
                rx.fragment(),
            ),
            spacing="2",
            align="center",
            height="100%",
        ),
        # Tab container
        height="36px",
        padding_x="12px",
        display="flex",
        align_items="center",
        border_right="1px solid",
        border_color=c("border"),
        border_top_width="2px",
        border_top_style="solid",
        border_top_color=rx.cond(is_active, c("accent"), "transparent"),
        background=rx.cond(is_active, c("bg_primary"), c("bg_sidebar")),
        cursor="pointer",
        on_click=AppState.set_active_workspace_tab(tab.id),
        transition="background 0.12s",
        flex_shrink="0",
    )


def workspace_tab_bar() -> rx.Component:
    return rx.hstack(
        rx.foreach(AppState.workspace_tabs, _tab_item),
        height="36px",
        min_height="36px",
        max_height="36px",
        width="100%",
        border_bottom="1px solid",
        border_color=c("border"),
        background=c("bg_sidebar"),
        overflow_x="auto",
        overflow_y="hidden",
        spacing="0",
        align="center",
        flex_shrink="0",
    )
