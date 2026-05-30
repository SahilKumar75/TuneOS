"""
TuneOS Desktop — System tray icon with context menu.

Features:
- Right-click menu: Show TuneOS, Hide, ─── separator, Quit TuneOS
- Single-click toggles the main window visibility
- Notification bubbles for training completion and other events
"""

from PyQt6.QtCore import QObject
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from desktop.theme import COLORS


def _create_tray_icon_pixmap() -> QPixmap:
    """Render a simple 64×64 blue-hex icon for the system tray."""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))  # transparent background

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor(COLORS["accent"]))
    font = QFont("Segoe UI Symbol", 40)
    font.setWeight(QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), 0x0084, "⬢")  # AlignCenter
    painter.end()

    return pixmap


class SystemTray(QObject):
    """System tray icon with a context menu bound to *main_window*.

    Usage::

        tray = SystemTray(main_window)
        tray.show_notification("Training Complete", "Job #42 finished.")
    """

    def __init__(self, main_window: "QMainWindow", parent: QObject | None = None) -> None:  # noqa: F821
        super().__init__(parent)
        self._window = main_window

        # ── Tray Icon ────────────────────────────────────────────
        self._tray = QSystemTrayIcon(QIcon(_create_tray_icon_pixmap()), self)
        self._tray.setToolTip("TuneOS — LLM Fine-Tuning Platform")
        self._tray.activated.connect(self._on_tray_activated)

        # ── Context Menu ─────────────────────────────────────────
        menu = QMenu()
        menu.setStyleSheet(
            f"""
            QMenu {{
                background-color: {COLORS["bg_card"]};
                color: {COLORS["text_primary"]};
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 6px;
                padding: 4px 0;
                font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
                font-size: 13px;
            }}
            QMenu::item {{
                padding: 6px 24px;
            }}
            QMenu::item:selected {{
                background-color: {COLORS["bg_hover"]};
            }}
            QMenu::separator {{
                height: 1px;
                background: rgba(255,255,255,0.10);
                margin: 4px 8px;
            }}
            """
        )

        action_show = QAction("Show TuneOS", self)
        action_show.triggered.connect(self._show_window)
        menu.addAction(action_show)

        action_hide = QAction("Hide", self)
        action_hide.triggered.connect(self._hide_window)
        menu.addAction(action_hide)

        menu.addSeparator()

        action_quit = QAction("Quit TuneOS", self)
        action_quit.triggered.connect(self._quit_app)
        menu.addAction(action_quit)

        self._tray.setContextMenu(menu)
        self._tray.show()

    # ── Slot Handlers ────────────────────────────────────────────
    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Toggle window visibility on a single-click / double-click."""
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            if self._window.isVisible():
                self._hide_window()
            else:
                self._show_window()

    def _show_window(self, checked: bool = False) -> None:
        if self._window:
            self._window.showNormal()
            self._window.activateWindow()
            self._window.raise_()

    def _hide_window(self, checked: bool = False) -> None:
        if self._window:
            self._window.hide()

    @staticmethod
    def _quit_app(checked: bool = False) -> None:
        QApplication.quit()

    # ── Public API ───────────────────────────────────────────────
    def show_notification(self, title: str, message: str) -> None:
        """Display an OS-native notification bubble via the tray icon.

        Args:
            title:   Notification title (e.g. "Training Complete").
            message: Notification body text.
        """
        if self._tray.supportsMessages():
            self._tray.showMessage(
                title,
                message,
                QSystemTrayIcon.MessageIcon.Information,
                5000,  # display for 5 seconds
            )
