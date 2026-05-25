"""
TuneOS Desktop — Custom frameless title bar (VS Code / Codex style).

Features:
- Draggable area to move the window
- App icon + title label
- Window control buttons: minimize, maximize/restore, close
- Double-click to maximize/restore
- Dark theme matching the Reflex UI
"""
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QMouseEvent, QIcon, QPainter, QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
)


class _WindowButton(QPushButton):
    """A single window control button (minimize / maximize / close)."""

    def __init__(self, icon_char: str, tooltip: str, obj_name: str = "TitleBarBtn"):
        super().__init__(icon_char)
        self.setObjectName(obj_name)
        self.setToolTip(tooltip)
        self.setFixedSize(36, 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)


class TitleBar(QWidget):
    """Custom dark title bar widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TitleBar")
        self.setFixedHeight(38)
        self._drag_pos: QPoint | None = None
        self._setup_ui()

    # ── UI Setup ──────────────────────────────────────────────────
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 6, 0)
        layout.setSpacing(0)

        # App icon (circle dot)
        self._icon_label = QLabel("⬢")
        self._icon_label.setObjectName("AppIcon")
        self._icon_label.setStyleSheet(
            "color: #3b82f6; font-size: 16px; padding-right: 6px;"
        )
        layout.addWidget(self._icon_label)

        # Title text
        self._title = QLabel("TuneOS")
        self._title.setObjectName("TitleLabel")
        layout.addWidget(self._title)

        # Spacer — this area is draggable
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        spacer.setStyleSheet("background: transparent;")
        layout.addWidget(spacer)

        # Window control buttons
        self._btn_minimize = _WindowButton("─", "Minimize")
        self._btn_maximize = _WindowButton("□", "Maximize")
        self._btn_close = _WindowButton("✕", "Close", obj_name="CloseBtn")

        self._btn_minimize.clicked.connect(self._on_minimize)
        self._btn_maximize.clicked.connect(self._on_maximize)
        self._btn_close.clicked.connect(self._on_close)

        layout.addWidget(self._btn_minimize)
        layout.addWidget(self._btn_maximize)
        layout.addWidget(self._btn_close)

    # ── Window Actions ────────────────────────────────────────────
    def _on_minimize(self):
        window = self.window()
        if window:
            window.showMinimized()

    def _on_maximize(self):
        window = self.window()
        if not window:
            return
        if window.isMaximized():
            window.showNormal()
            self._btn_maximize.setText("□")
            self._btn_maximize.setToolTip("Maximize")
        else:
            window.showMaximized()
            self._btn_maximize.setText("❐")
            self._btn_maximize.setToolTip("Restore")

    def _on_close(self):
        window = self.window()
        if window:
            window.close()

    # ── Drag to Move ──────────────────────────────────────────────
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.window().pos()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_pos is not None and event.buttons() == Qt.MouseButton.LeftButton:
            window = self.window()
            if window and window.isMaximized():
                # Un-maximize when dragging from maximized state
                window.showNormal()
                self._btn_maximize.setText("□")
                # Adjust drag position for the new window size
                self._drag_pos = QPoint(
                    window.width() // 2,
                    self.height() // 2,
                )
            self.window().move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_pos = None
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double-click to maximize / restore."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_maximize()
            event.accept()

    def update_title(self, title: str):
        """Update the title text displayed in the title bar."""
        self._title.setText(title)
