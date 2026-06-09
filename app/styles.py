"""
TuneOS neutral app styling.
The palette intentionally avoids the previous purple landing-page treatment.
"""

import reflex as rx

STYLESHEETS = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap",
    "/streaming.css",
]


COLORS = {
    "bg_primary": {"dark": "#111111", "light": "#ffffff"},
    "bg_sidebar": {"dark": "#181818", "light": "#f4f4f4"},
    "bg_card": {"dark": "#1f1f1f", "light": "#ffffff"},
    "bg_input": {"dark": "#232323", "light": "#ffffff"},
    "accent": {"dark": "#3b82f6", "light": "#2563eb"},
    "accent_soft": {"dark": "rgba(59,130,246,0.16)", "light": "rgba(37,99,235,0.10)"},
    "text_primary": {"dark": "#f2f2f2", "light": "#171717"},
    "text_secondary": {"dark": "#a1a1a1", "light": "#555555"},
    "text_muted": {"dark": "#707070", "light": "#a3a3a3"},
    "border": {"dark": "rgba(255,255,255,0.10)", "light": "rgba(0,0,0,0.08)"},
    "border_strong": {"dark": "rgba(255,255,255,0.18)", "light": "rgba(0,0,0,0.14)"},
    "hover": {"dark": "rgba(255,255,255,0.06)", "light": "rgba(0,0,0,0.04)"},
    "hover_strong": {"dark": "rgba(255,255,255,0.10)", "light": "rgba(0,0,0,0.07)"},
    "success": {"dark": "#22c55e", "light": "#16a34a"},
    "warning": {"dark": "#f59e0b", "light": "#d97706"},
    "error": {"dark": "#f87171", "light": "#dc2626"},
    "input_bg": {"dark": "#202020", "light": "#ffffff"},
    "input_border": {"dark": "rgba(255,255,255,0.14)", "light": "rgba(0,0,0,0.12)"},
    "menu_bg": {"dark": "#242424", "light": "#ffffff"},
}


def c(token: str):
    """Return a theme-aware color token."""
    pair = COLORS[token]
    return rx.color_mode_cond(light=pair["light"], dark=pair["dark"])


GLOBAL_STYLES = {
    "body": {
        "font_family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "margin": "0",
        "padding": "0",
    },
    "*": {
        "box_sizing": "border-box",
    },
    "::selection": {
        "background": "rgba(37,99,235,0.18)",
    },
    "::-webkit-scrollbar": {
        "width": "7px",
        "height": "7px",
    },
    "::-webkit-scrollbar-track": {
        "background": "transparent",
    },
    "::-webkit-scrollbar-thumb": {
        "background": "rgba(115,115,115,0.35)",
        "border_radius": "999px",
    },
    "@keyframes pulse": {
        "0%, 100%": {"opacity": "1"},
        "50%": {"opacity": "0.5"},
    },
    # Benchmark card — make tables fit the fixed-width column
    ".bench-card table": {
        "width": "100%",
        "border_collapse": "collapse",
        "font_size": "0.78rem",
        "table_layout": "fixed",
    },
    ".bench-card th, .bench-card td": {
        "padding": "4px 6px",
        "text_align": "left",
        "word_break": "break-word",
        "white_space": "normal",
        "vertical_align": "top",
        "border_bottom": "1px solid rgba(128,128,128,0.15)",
    },
    ".bench-card th": {
        "font_weight": "600",
        "font_size": "0.72rem",
        "color": "inherit",
    },
    ".bench-card p, .bench-card h1, .bench-card h2, .bench-card h3": {
        "font_size": "0.82rem",
        "margin_bottom": "6px",
    },
}
