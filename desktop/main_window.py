"""
TuneOS Desktop — Main application window.

Frameless QMainWindow that hosts:
  - Custom TitleBar (drag, min/max/close)
  - QWebEngineView (loads the Reflex app at localhost:3000)
  - StatusBar (GPU · Docker · job info)

The window supports edge-resize grips (8 px from each border) so the
user can freely resize the frameless window.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QPoint, QSize, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QCursor
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QApplication,
)

from desktop.theme import get_full_stylesheet
from desktop.title_bar import TitleBar
from desktop.status_bar import StatusBar

# ── Constants ────────────────────────────────────────────────────
_DEFAULT_SIZE = QSize(1400, 900)
_MINIMUM_SIZE = QSize(900, 600)
_DEFAULT_URL = "http://localhost:3000"
_GRIP_SIZE = 8  # pixels from the edge used for resize detection


class MainWindow(QMainWindow):
    """Frameless main window for TuneOS.

    Signals:
        closing: Emitted from ``closeEvent`` so the launcher can
                 perform cleanup (e.g. ``process_manager.stop()``).
    """

    closing = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setMinimumSize(_MINIMUM_SIZE)
        self.resize(_DEFAULT_SIZE)
        self.setStyleSheet(get_full_stylesheet())
        self.setWindowTitle("TuneOS")

        # Resize state
        self._resize_edge: str | None = None
        self._resize_origin: QPoint | None = None
        self._origin_geometry = None

        self._setup_ui()
        self._centre_on_screen()

    # ── UI Setup ─────────────────────────────────────────────────
    def _setup_ui(self) -> None:
        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title bar
        self._title_bar = TitleBar(self)
        layout.addWidget(self._title_bar)

        # Web view
        self._web_view = QWebEngineView()
        self._web_view.setUrl(_DEFAULT_URL)  # type: ignore[arg-type]
        layout.addWidget(self._web_view, stretch=1)

        # Status bar
        self._status_bar = StatusBar(self)
        layout.addWidget(self._status_bar)

    # ── Public API ───────────────────────────────────────────────
    def load_url(self, url: str) -> None:
        """Navigate the embedded browser to *url*."""
        self._web_view.setUrl(url)  # type: ignore[arg-type]

    @property
    def title_bar(self) -> TitleBar:
        """Direct access to the custom title bar widget."""
        return self._title_bar

    @property
    def status_bar_widget(self) -> StatusBar:
        """Direct access to the custom status bar widget."""
        return self._status_bar

    # ── Centering ────────────────────────────────────────────────
    def _centre_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + (geo.height() - self.height()) // 2
        self.move(x, y)

    # ── Frameless resize helpers ─────────────────────────────────
    def _edge_at(self, pos: QPoint) -> str | None:
        """Return a compass direction string if *pos* is within the grip area."""
        rect = self.rect()
        x, y = pos.x(), pos.y()
        on_left = x < _GRIP_SIZE
        on_right = x > rect.width() - _GRIP_SIZE
        on_top = y < _GRIP_SIZE
        on_bottom = y > rect.height() - _GRIP_SIZE

        if on_top and on_left:
            return "tl"
        if on_top and on_right:
            return "tr"
        if on_bottom and on_left:
            return "bl"
        if on_bottom and on_right:
            return "br"
        if on_top:
            return "t"
        if on_bottom:
            return "b"
        if on_left:
            return "l"
        if on_right:
            return "r"
        return None

    @staticmethod
    def _cursor_for_edge(edge: str | None) -> Qt.CursorShape:
        _MAP = {
            "t": Qt.CursorShape.SizeVerCursor,
            "b": Qt.CursorShape.SizeVerCursor,
            "l": Qt.CursorShape.SizeHorCursor,
            "r": Qt.CursorShape.SizeHorCursor,
            "tl": Qt.CursorShape.SizeFDiagCursor,
            "br": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor,
            "bl": Qt.CursorShape.SizeBDiagCursor,
        }
        return _MAP.get(edge, Qt.CursorShape.ArrowCursor)  # type: ignore[arg-type]

    # ── Mouse events for edge resize ─────────────────────────────
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            edge = self._edge_at(event.position().toPoint())
            if edge:
                self._resize_edge = edge
                self._resize_origin = event.globalPosition().toPoint()
                self._origin_geometry = self.geometry()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # Update cursor when hovering near edges
        if self._resize_edge is None:
            edge = self._edge_at(event.position().toPoint())
            self.setCursor(self._cursor_for_edge(edge))

        # Resize while dragging
        if self._resize_edge and self._resize_origin and self._origin_geometry:
            delta = event.globalPosition().toPoint() - self._resize_origin
            geo = self._origin_geometry
            new_geo = geo.__class__(geo)  # copy

            if "r" in self._resize_edge:
                new_geo.setRight(geo.right() + delta.x())
            if "b" in self._resize_edge:
                new_geo.setBottom(geo.bottom() + delta.y())
            if "l" in self._resize_edge:
                new_geo.setLeft(geo.left() + delta.x())
            if "t" in self._resize_edge:
                new_geo.setTop(geo.top() + delta.y())

            # Enforce minimum size
            if new_geo.width() < _MINIMUM_SIZE.width():
                if "l" in self._resize_edge:
                    new_geo.setLeft(new_geo.right() - _MINIMUM_SIZE.width())
                else:
                    new_geo.setRight(new_geo.left() + _MINIMUM_SIZE.width())
            if new_geo.height() < _MINIMUM_SIZE.height():
                if "t" in self._resize_edge:
                    new_geo.setTop(new_geo.bottom() - _MINIMUM_SIZE.height())
                else:
                    new_geo.setBottom(new_geo.top() + _MINIMUM_SIZE.height())

            self.setGeometry(new_geo)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._resize_edge = None
        self._resize_origin = None
        self._origin_geometry = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    # ── Window state handling ────────────────────────────────────
    def changeEvent(self, event) -> None:
        """Keep title-bar maximize button text in sync with window state."""
        super().changeEvent(event)
        if event.type() == event.Type.WindowStateChange:
            if self.isMaximized():
                self._title_bar._btn_maximize.setText("❐")
                self._title_bar._btn_maximize.setToolTip("Restore")
            else:
                self._title_bar._btn_maximize.setText("□")
                self._title_bar._btn_maximize.setToolTip("Maximize")

    # ── Close ────────────────────────────────────────────────────
    def closeEvent(self, event) -> None:
        """Emit ``closing`` so the launcher can shut down backend services."""
        self.closing.emit()
        super().closeEvent(event)
