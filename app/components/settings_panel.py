"""TuneOS — Settings panel shown when current_view == 'settings'."""

import reflex as rx

from app.state.theme_state import ThemeState
from app.styles import c


def _theme_card(label: str, icon: str, pref: str) -> rx.Component:
    is_active = ThemeState.theme_preference == pref
    return rx.box(
        rx.vstack(
            rx.icon(
                icon,
                size=24,
                color=rx.cond(is_active, c("accent"), c("text_secondary")),
            ),
            rx.text(
                label,
                font_size="0.9rem",
                font_weight=rx.cond(is_active, "600", "400"),
                color=rx.cond(is_active, c("text_primary"), c("text_secondary")),
            ),
            spacing="2",
            align="center",
        ),
        padding="20px 24px",
        border_radius="10px",
        border="1.5px solid",
        border_color=rx.cond(is_active, c("accent"), c("border")),
        background=rx.cond(is_active, c("hover_strong"), c("bg_card")),
        cursor="pointer",
        min_width="130px",
        _hover={"border_color": c("accent"), "background": c("hover")},
        on_click=ThemeState.set_theme(pref),
        transition="border-color 0.15s ease, background 0.15s ease",
    )


def settings_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading(
                "Settings",
                font_size="1.4rem",
                font_weight="600",
                color=c("text_primary"),
            ),
            rx.text(
                "Appearance",
                font_size="0.85rem",
                font_weight="500",
                color=c("text_muted"),
                padding_top="24px",
                padding_bottom="12px",
            ),
            rx.hstack(
                _theme_card("Light", "sun", "light"),
                _theme_card("Dark", "moon", "dark"),
                _theme_card("System Preference", "monitor", "system"),
                spacing="4",
                wrap="wrap",
            ),
            spacing="0",
            align_items="flex-start",
            width="100%",
            max_width="680px",
        ),
        width="100%",
        height="100vh",
        padding="40px 48px",
        background=c("bg_primary"),
        overflow_y="auto",
    )
