"""
TuneOS Desktop — Application entry point.

Orchestrates the startup sequence:
  1. Enforce single-instance via QLockFile
  2. Show splash screen
  3. Check Docker availability (prompt if missing)
  4. Start backend services (docker-compose + Reflex) in a QThread
  5. Wait for the server to become ready
  6. Show the main window and system tray
  7. Close the splash
  8. Cleanup on exit

Uses QTimer so all UI updates remain responsive while the backend
is booting in the background.
"""
from __future__ import annotations

import logging
import sys
import tempfile
import os

from PyQt6.QtCore import QTimer, QThread, pyqtSignal, QLockFile
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PyQt6.QtWidgets import QApplication, QMessageBox

from desktop.splash_screen import SplashScreen
from desktop.docker_check import DockerRequiredDialog
from desktop.process_manager import ProcessManager
from desktop.main_window import MainWindow
from desktop.system_tray import SystemTray

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("tuneos.main")


def _build_app_icon() -> QIcon:
    """Create a simple blue-hex pixmap icon for the application."""
    size = 128
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor("#3b82f6"))
    font = QFont("Segoe UI Symbol", 80)
    font.setWeight(QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), 0x0084, "⬢")  # Qt.AlignCenter
    painter.end()
    return QIcon(pixmap)


class _StartupThread(QThread):
    """Runs process_manager.start() off the main thread to avoid freezing the UI."""

    finished = pyqtSignal(bool)  # True = started OK

    def __init__(self, process_manager: ProcessManager) -> None:
        super().__init__()
        self._pm = process_manager

    def run(self) -> None:
        ok = self._pm.start()
        self.finished.emit(ok)


class _Launcher:
    """Coordinates the multi-step, timer-driven startup sequence.

    Keeps references to every long-lived object so they are not
    garbage-collected until the application exits.
    """

    def __init__(self, app: QApplication) -> None:
        self._app = app
        self._splash: SplashScreen | None = None
        self._process_manager: ProcessManager | None = None
        self._main_window: MainWindow | None = None
        self._tray: SystemTray | None = None
        self._docker_available = False

    # ── Step 1: Splash ───────────────────────────────────────────
    def start(self) -> None:
        """Show the splash and kick off step 2 after a short delay."""
        self._splash = SplashScreen()
        self._splash.show()
        self._splash.set_status("Initialising…")
        self._splash.set_progress(5)

        # Give the event loop one tick to paint the splash
        QTimer.singleShot(100, self._step_docker_check)

    # ── Step 2: Docker Check ─────────────────────────────────────
    def _step_docker_check(self) -> None:
        if self._splash:
            self._splash.set_status("Checking Docker…")
            self._splash.set_progress(15)

        self._docker_available = DockerRequiredDialog.check_and_prompt()
        log.info("Docker available: %s", self._docker_available)

        QTimer.singleShot(50, self._step_start_backend)

    # ── Step 3: Start Backend ────────────────────────────────────
    def _step_start_backend(self) -> None:
        if self._splash:
            self._splash.set_status("Starting services…")
            self._splash.set_progress(30)

        self._process_manager = ProcessManager()
        self._process_manager.status_changed.connect(self._on_status_changed)
        self._process_manager.server_ready.connect(self._on_server_ready)
        self._process_manager.server_failed.connect(self._on_server_failed)

        # Run start() in a background thread so the Qt event loop stays
        # responsive (subprocess.run can block for up to 60 s).
        self._startup_thread = _StartupThread(self._process_manager)
        self._startup_thread.finished.connect(self._on_startup_thread_done)
        self._startup_thread.start()

    def _on_startup_thread_done(self, ok: bool) -> None:
        if not ok:
            log.warning("Backend start reported failure; polling anyway.")

        # Begin polling for the server in a non-blocking fashion
        self._poll_timer = QTimer()
        self._poll_timer.setInterval(500)
        self._poll_count = 0
        self._poll_max = 60  # 30 seconds at 500 ms intervals
        self._poll_timer.timeout.connect(self._poll_server)
        self._poll_timer.start()

    def _poll_server(self) -> None:
        """Check if the Reflex server is responding; update splash."""
        self._poll_count += 1
        progress = min(30 + int(self._poll_count / self._poll_max * 60), 90)
        if self._splash:
            self._splash.set_progress(progress)
            self._splash.set_status("Loading UI…")

        if self._process_manager and self._process_manager.is_server_ready():
            self._poll_timer.stop()
            self._on_server_ready()
            return

        if self._poll_count >= self._poll_max:
            self._poll_timer.stop()
            self._on_server_failed()

    # ── Step 4: Server Ready → show window ───────────────────────
    def _on_server_ready(self) -> None:
        log.info("Server is ready.")
        if self._splash:
            self._splash.set_status("Ready!")
            self._splash.set_progress(100)

        self._show_main_window()

    def _on_server_failed(self) -> None:
        log.warning("Server did not start in time — showing window anyway.")
        if self._splash:
            self._splash.set_status("Server slow — opening anyway…")
            self._splash.set_progress(100)

        self._show_main_window()

    def _show_main_window(self) -> None:
        self._main_window = MainWindow()
        self._main_window.closing.connect(self._cleanup)
        self._main_window.show()

        if self._process_manager and self._process_manager.is_server_ready():
            self._main_window.status_bar_widget.set_server_status("Running", is_ready=True)
        else:
            self._main_window.status_bar_widget.set_server_status("Starting…", is_ready=False)

        # System tray
        self._tray = SystemTray(self._main_window)

        # Close splash
        if self._splash:
            self._splash.close_splash()
            self._splash = None

    # ── Signal Handlers ──────────────────────────────────────────
    def _on_status_changed(self, text: str) -> None:
        if self._splash:
            self._splash.set_status(text)

    # ── Cleanup ──────────────────────────────────────────────────
    def _cleanup(self) -> None:
        """Gracefully stop backend services."""
        log.info("Cleaning up…")
        if self._process_manager:
            self._process_manager.stop()


# ── Main ─────────────────────────────────────────────────────────
def main() -> None:
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("TuneOS")
    app.setOrganizationName("TuneOS")
    app.setWindowIcon(_build_app_icon())
    app.setQuitOnLastWindowClosed(False)

    # ── Single-instance guard ────────────────────────────────────
    lock_path = os.path.join(tempfile.gettempdir(), "tuneos.lock")
    lock_file = QLockFile(lock_path)
    if not lock_file.tryLock(100):
        QMessageBox.warning(
            None,
            "TuneOS already running",
            "Another instance of TuneOS is already open.\n"
            "Please use the existing window.",
        )
        sys.exit(0)

    launcher = _Launcher(app)
    launcher.start()

    exit_code = app.exec()

    # Ensure backend processes are cleaned up even if the window was
    # force-closed without the closeEvent signal firing.
    launcher._cleanup()
    lock_file.unlock()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
