"""
TuneOS Desktop — Startup splash screen.

Displayed while Docker containers and the Reflex dev-server are booting.
Features:
- Frameless, rounded-corner widget centred on the primary screen
- Animated gradient progress bar (#3b82f6 → #8b5cf6)
- Live status text that the launcher updates as each service comes up
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainterPath, QRegion
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import COLORS, SPLASH_QSS

# ── Constants ────────────────────────────────────────────────────
_WIDTH = 480
_HEIGHT = 320
_CORNER_RADIUS = 16


class SplashScreen(QWidget):
    """Full-screen-centred startup splash for TuneOS.

    Usage::

        splash = SplashScreen()
        splash.show()
        splash.set_status("Checking Docker...")
        splash.set_progress(25)
        ...
        splash.close_splash()
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SplashScreen")
        self.setFixedSize(_WIDTH, _HEIGHT)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.SplashScreen
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setStyleSheet(SPLASH_QSS)

        self._apply_rounded_mask()
        self._setup_ui()
        self._centre_on_screen()

    # ── UI Construction ──────────────────────────────────────────
    def _setup_ui(self) -> None:
        """Build the splash layout: logo → subtitle → spacer → progress → status."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 48, 40, 32)
        layout.setSpacing(0)

        # ⬢ TuneOS  (hex icon in accent, text in primary)
        self._title = QLabel()
        self._title.setObjectName("SplashTitle")
        self._title.setTextFormat(Qt.TextFormat.RichText)
        self._title.setText(f'<span style="color:{COLORS["accent"]};">⬢</span> TuneOS')
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title)

        layout.addSpacing(8)

        # Subtitle
        self._subtitle = QLabel("Fine-tune LLMs locally")
        self._subtitle.setObjectName("SplashSubtitle")
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._subtitle)

        # Flexible spacer pushes the progress section to the bottom
        layout.addStretch(1)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setObjectName("SplashProgress")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        layout.addSpacing(12)

        # Status label
        self._status = QLabel("Initialising…")
        self._status.setObjectName("SplashStatus")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

    # ── Rounded Corners ──────────────────────────────────────────
    def _apply_rounded_mask(self) -> None:
        """Clip the widget to a rounded rectangle."""
        path = QPainterPath()
        path.addRoundedRect(
            0.0,
            0.0,
            float(_WIDTH),
            float(_HEIGHT),
            _CORNER_RADIUS,
            _CORNER_RADIUS,
        )
        region = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)

    # ── Positioning ──────────────────────────────────────────────
    def _centre_on_screen(self) -> None:
        """Place the splash in the centre of the primary screen."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        screen_geo = screen.availableGeometry()
        x = screen_geo.x() + (screen_geo.width() - _WIDTH) // 2
        y = screen_geo.y() + (screen_geo.height() - _HEIGHT) // 2
        self.move(x, y)

    # ── Public API ───────────────────────────────────────────────
    def set_status(self, text: str) -> None:
        """Update the status message shown below the progress bar."""
        self._status.setText(text)
        QApplication.processEvents()

    def set_progress(self, value: int) -> None:
        """Set the progress bar value (0 – 100)."""
        self._progress.setValue(max(0, min(value, 100)))
        QApplication.processEvents()

    def close_splash(self) -> None:
        """Fade-out placeholder — immediately closes for now."""
        self.close()
