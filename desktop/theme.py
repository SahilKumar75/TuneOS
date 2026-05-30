"""
TuneOS Desktop — Dark theme stylesheet (QSS).

Matches the Reflex UI color palette from app/styles.py:
- bg_primary: #111111
- bg_sidebar: #181818
- bg_card:    #1f1f1f
- accent:     #3b82f6
- text:       #f2f2f2 / #a1a1a1 / #707070
- border:     rgba(255,255,255,0.10)
"""

# ── Color Tokens ──────────────────────────────────────────────────
COLORS = {
    "bg_base": "#0e0e0e",
    "bg_primary": "#111111",
    "bg_sidebar": "#181818",
    "bg_card": "#1f1f1f",
    "bg_input": "#232323",
    "bg_titlebar": "#0e0e0e",
    "bg_statusbar": "#1a1a2e",
    "bg_hover": "#2a2a2a",
    "bg_pressed": "#333333",
    "bg_splash": "#0a0a0a",
    "accent": "#3b82f6",
    "accent_hover": "#2563eb",
    "accent_soft": "rgba(59,130,246,0.16)",
    "text_primary": "#f2f2f2",
    "text_secondary": "#a1a1a1",
    "text_muted": "#707070",
    "text_inverse": "#111111",
    "border": "rgba(255,255,255,0.10)",
    "border_strong": "rgba(255,255,255,0.18)",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "error": "#f87171",
    "close_hover": "#e81123",
    "close_pressed": "#c10e1e",
}


# ── Title Bar QSS ─────────────────────────────────────────────────
TITLE_BAR_QSS = f"""
QWidget#TitleBar {{
    background-color: {COLORS["bg_titlebar"]};
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}}

QLabel#TitleLabel {{
    color: {COLORS["text_secondary"]};
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
    font-weight: 500;
    padding-left: 4px;
}}

QLabel#AppIcon {{
    padding: 0px;
    margin: 0px;
}}

QPushButton#TitleBarBtn {{
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 6px;
    margin: 2px 1px;
    min-width: 36px;
    max-width: 36px;
    min-height: 28px;
    max-height: 28px;
    color: {COLORS["text_secondary"]};
    font-size: 14px;
}}

QPushButton#TitleBarBtn:hover {{
    background-color: {COLORS["bg_hover"]};
    color: {COLORS["text_primary"]};
}}

QPushButton#TitleBarBtn:pressed {{
    background-color: {COLORS["bg_pressed"]};
}}

QPushButton#CloseBtn {{
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 6px;
    margin: 2px 1px;
    min-width: 36px;
    max-width: 36px;
    min-height: 28px;
    max-height: 28px;
    color: {COLORS["text_secondary"]};
    font-size: 14px;
}}

QPushButton#CloseBtn:hover {{
    background-color: {COLORS["close_hover"]};
    color: #ffffff;
}}

QPushButton#CloseBtn:pressed {{
    background-color: {COLORS["close_pressed"]};
    color: #ffffff;
}}
"""


# ── Status Bar QSS ────────────────────────────────────────────────
STATUS_BAR_QSS = f"""
QWidget#StatusBar {{
    background-color: {COLORS["bg_statusbar"]};
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    min-height: 26px;
    max-height: 26px;
}}

QLabel#StatusLabel {{
    color: {COLORS["text_secondary"]};
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 11px;
    padding: 0 8px;
}}

QLabel#StatusAccent {{
    color: {COLORS["accent"]};
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 11px;
    font-weight: 600;
    padding: 0 8px;
}}

QLabel#StatusSuccess {{
    color: {COLORS["success"]};
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 11px;
    padding: 0 8px;
}}

QLabel#StatusWarning {{
    color: {COLORS["warning"]};
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 11px;
    padding: 0 8px;
}}

QLabel#StatusError {{
    color: {COLORS["error"]};
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 11px;
    padding: 0 8px;
}}
"""


# ── Splash Screen QSS ────────────────────────────────────────────
SPLASH_QSS = f"""
QWidget#SplashScreen {{
    background-color: {COLORS["bg_splash"]};
}}

QLabel#SplashTitle {{
    color: {COLORS["text_primary"]};
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
}}

QLabel#SplashSubtitle {{
    color: {COLORS["text_muted"]};
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
    font-weight: 400;
}}

QLabel#SplashStatus {{
    color: {COLORS["text_secondary"]};
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 12px;
}}

QProgressBar#SplashProgress {{
    background-color: rgba(255, 255, 255, 0.08);
    border: none;
    border-radius: 3px;
    min-height: 5px;
    max-height: 5px;
    text-align: center;
    color: transparent;
}}

QProgressBar#SplashProgress::chunk {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS["accent"]},
        stop:1 #8b5cf6
    );
    border-radius: 3px;
}}
"""


# ── Main Window QSS ──────────────────────────────────────────────
MAIN_WINDOW_QSS = f"""
QMainWindow {{
    background-color: {COLORS["bg_primary"]};
}}

QWidget#CentralWidget {{
    background-color: {COLORS["bg_primary"]};
}}
"""


# ── Docker Check Dialog QSS ──────────────────────────────────────
DIALOG_QSS = f"""
QDialog {{
    background-color: {COLORS["bg_card"]};
}}

QLabel#DialogTitle {{
    color: {COLORS["text_primary"]};
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 18px;
    font-weight: 600;
}}

QLabel#DialogText {{
    color: {COLORS["text_secondary"]};
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
    line-height: 1.5;
}}

QPushButton#DialogBtn {{
    background-color: {COLORS["accent"]};
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
    font-weight: 600;
}}

QPushButton#DialogBtn:hover {{
    background-color: {COLORS["accent_hover"]};
}}

QPushButton#DialogBtnSecondary {{
    background-color: transparent;
    color: {COLORS["text_secondary"]};
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 8px;
    padding: 10px 24px;
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
    font-weight: 500;
}}

QPushButton#DialogBtnSecondary:hover {{
    background-color: {COLORS["bg_hover"]};
    color: {COLORS["text_primary"]};
}}
"""


def get_full_stylesheet() -> str:
    """Return the complete application stylesheet."""
    return "\n".join(
        [
            MAIN_WINDOW_QSS,
            TITLE_BAR_QSS,
            STATUS_BAR_QSS,
            SPLASH_QSS,
            DIALOG_QSS,
        ]
    )
