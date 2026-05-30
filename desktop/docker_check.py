"""
TuneOS Desktop — Docker availability checker and prompt dialog.

When Docker Desktop is not reachable the user is shown a dialog
explaining the dependency and offering a one-click download link.
"""

import platform
import subprocess
import webbrowser

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import DIALOG_QSS

# ── Download URLs per platform ───────────────────────────────────
_DOCKER_URLS: dict[str, str] = {
    "Darwin": "https://docs.docker.com/desktop/install/mac-install/",
    "Windows": "https://docs.docker.com/desktop/install/windows-install/",
    "Linux": "https://docs.docker.com/desktop/install/linux/",
}


def _is_docker_available() -> bool:
    """Return *True* if `docker info` succeeds (daemon is reachable)."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


class DockerRequiredDialog(QDialog):
    """Modal dialog shown when Docker is not available.

    Provides two options:
    - **Download Docker** — opens the correct platform URL in the browser.
    - **Continue without Docker** — closes the dialog; caller gets *False*.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Docker Required")
        self.setFixedSize(440, 280)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet(DIALOG_QSS)
        self._docker_available = False
        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(0)

        # Warning icon + title row
        icon_label = QLabel("⚠️")
        icon_label.setStyleSheet("font-size: 28px; padding-right: 8px;")

        title = QLabel("Docker Required")
        title.setObjectName("DialogTitle")

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_row.addWidget(icon_label)
        title_row.addWidget(title)
        title_row.addStretch()
        layout.addLayout(title_row)

        layout.addSpacing(16)

        # Explanation text
        message = QLabel(
            "TuneOS needs Docker to run Redis and the Celery worker that "
            "power background training jobs.\n\n"
            "Please install Docker Desktop and make sure it is running, "
            "then restart TuneOS."
        )
        message.setObjectName("DialogText")
        message.setWordWrap(True)
        layout.addWidget(message)

        layout.addStretch(1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        btn_continue = QPushButton("Continue without Docker")
        btn_continue.setObjectName("DialogBtnSecondary")
        btn_continue.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_continue.clicked.connect(self._on_continue)
        btn_row.addWidget(btn_continue)

        btn_row.addStretch()

        btn_download = QPushButton("Download Docker")
        btn_download.setObjectName("DialogBtn")
        btn_download.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_download.clicked.connect(self._on_download)
        btn_row.addWidget(btn_download)

        layout.addLayout(btn_row)

    # ── Handlers ─────────────────────────────────────────────────
    def _on_download(self, checked: bool = False) -> None:
        """Open the Docker download page for the current platform."""
        url = _DOCKER_URLS.get(platform.system(), _DOCKER_URLS["Linux"])
        webbrowser.open(url)

    def _on_continue(self, checked: bool = False) -> None:
        """Close dialog without Docker — caller will get *False*."""
        self._docker_available = False
        self.accept()

    # ── Result ───────────────────────────────────────────────────
    @property
    def docker_available(self) -> bool:
        """Whether Docker was found to be available."""
        return self._docker_available

    # ── Static Entry Point ───────────────────────────────────────
    @staticmethod
    def check_and_prompt(parent: QWidget | None = None) -> bool:
        """Check Docker and, if missing, show the prompt dialog.

        Returns:
            *True* if Docker is available and running, *False* otherwise.
        """
        if _is_docker_available():
            return True

        dialog = DockerRequiredDialog(parent)
        dialog.exec()
        return dialog.docker_available
