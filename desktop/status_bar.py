"""
TuneOS Desktop — Status bar with GPU, job, and Docker health indicators.

Shows at the bottom of the window (like VS Code's blue status bar).
"""

import platform
import subprocess

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget


def _detect_gpu() -> str:
    """Detect available GPU at startup."""
    system = platform.system()

    # Check NVIDIA (Windows / Linux)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0]
            return f"GPU: {gpu_name}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check Apple Silicon MPS (macOS)
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                cpu = result.stdout.strip()
                if "Apple" in cpu:
                    return f"GPU: {cpu} (MPS)"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return "GPU: CPU only"


def _check_docker_running() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class StatusBar(QWidget):
    """Native status bar at the bottom of the window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusBar")
        self.setFixedHeight(26)
        self._setup_ui()
        self._start_polling()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(0)

        # Left side: Docker status + GPU info
        self._docker_label = QLabel("⬤  Docker")
        self._docker_label.setObjectName("StatusLabel")
        layout.addWidget(self._docker_label)

        self._separator1 = QLabel("│")
        self._separator1.setObjectName("StatusLabel")
        self._separator1.setStyleSheet("color: rgba(255,255,255,0.15); padding: 0 6px;")
        layout.addWidget(self._separator1)

        self._gpu_label = QLabel(_detect_gpu())
        self._gpu_label.setObjectName("StatusLabel")
        layout.addWidget(self._gpu_label)

        self._separator2 = QLabel("│")
        self._separator2.setObjectName("StatusLabel")
        self._separator2.setStyleSheet("color: rgba(255,255,255,0.15); padding: 0 6px;")
        layout.addWidget(self._separator2)

        # Job status
        self._job_label = QLabel("No active jobs")
        self._job_label.setObjectName("StatusLabel")
        layout.addWidget(self._job_label)

        # Spacer
        spacer = QWidget()
        spacer.setStyleSheet("background: transparent;")
        from PyQt6.QtWidgets import QSizePolicy

        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(spacer)

        # Right side: Reflex server status
        self._server_label = QLabel("Reflex: Starting...")
        self._server_label.setObjectName("StatusLabel")
        layout.addWidget(self._server_label)

    def _start_polling(self):
        """Poll Docker status every 10 seconds."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_docker_status)
        self._timer.start(10_000)
        # Initial check
        self._update_docker_status()

    def _update_docker_status(self):
        if _check_docker_running():
            self._docker_label.setText("⬤  Docker: Running")
            self._docker_label.setObjectName("StatusSuccess")
        else:
            self._docker_label.setText("⬤  Docker: Stopped")
            self._docker_label.setObjectName("StatusError")
        # Re-apply stylesheet for the object name change
        self._docker_label.setStyleSheet(self._docker_label.styleSheet())
        self.style().unpolish(self._docker_label)
        self.style().polish(self._docker_label)

    def set_server_status(self, status: str, is_ready: bool = False):
        """Update the Reflex server status label."""
        if is_ready:
            self._server_label.setText(f"Reflex: {status}")
            self._server_label.setObjectName("StatusSuccess")
        else:
            self._server_label.setText(f"Reflex: {status}")
            self._server_label.setObjectName("StatusWarning")
        self.style().unpolish(self._server_label)
        self.style().polish(self._server_label)

    def set_job_status(self, status: str):
        """Update the job status label."""
        self._job_label.setText(status)

    def set_gpu_info(self, info: str):
        """Update the GPU info label."""
        self._gpu_label.setText(info)
