"""
TuneOS Desktop — Process manager for Docker services and Reflex server.

Owns the full lifecycle:
  1. ``docker-compose up -d redis worker``  (background containers)
  2. ``python -m reflex run``               (foreground subprocess)
  3. Waits for ``localhost:3000`` to respond
  4. Graceful shutdown in reverse order on exit

All heavy I/O is guarded by timeouts so the GUI thread is never blocked
for an unreasonable amount of time.  Status updates are emitted through
Qt signals.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from PyQt6.QtCore import QObject, pyqtSignal

log = logging.getLogger(__name__)

# ── Paths & Commands ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DOCKER_COMPOSE_UP = ["docker-compose", "up", "-d", "redis", "worker"]
_DOCKER_COMPOSE_DOWN = ["docker-compose", "down"]
_REFLEX_RUN = [sys.executable, "-m", "reflex", "run"]
_SERVER_URL = "http://localhost:3000"


class ProcessManager(QObject):
    """Manages Docker containers and the Reflex dev-server subprocess.

    Signals:
        server_ready:       Emitted once ``localhost:3000`` responds.
        server_failed:      Emitted when the server could not start.
        status_changed(str): Emitted with a human-readable status string
                            whenever the startup phase changes.
    """

    # ── Signals ──────────────────────────────────────────────────
    server_ready = pyqtSignal()
    server_failed = pyqtSignal()
    status_changed = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._reflex_process: subprocess.Popen | None = None
        self._docker_started: bool = False
        self._local_redis_process: subprocess.Popen | None = None
        self._local_celery_process: subprocess.Popen | None = None

    # ── Docker helpers ───────────────────────────────────────────
    @staticmethod
    def is_docker_available() -> bool:
        """Return *True* if the Docker daemon is reachable."""
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

    def _start_docker_services(self) -> bool:
        """Run ``docker-compose up -d redis worker``.

        Returns *True* on success.
        """
        self.status_changed.emit("Starting Docker services…")
        try:
            result = subprocess.run(
                _DOCKER_COMPOSE_UP,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                log.info("Docker services started successfully.")
                self._docker_started = True
                return True
            log.warning("docker-compose up failed: %s", result.stderr)
            return False
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            log.error("Failed to start Docker services: %s", exc)
            return False

    def _stop_docker_services(self) -> None:
        """Run ``docker-compose down``."""
        if not self._docker_started:
            return
        try:
            subprocess.run(
                _DOCKER_COMPOSE_DOWN,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            log.info("Docker services stopped.")
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            log.warning("Error stopping Docker services: %s", exc)
        finally:
            self._docker_started = False

    # ── Local fallback helpers (no Docker) ──────────────────────
    def _start_local_redis(self) -> bool:
        """Start a Redis server as a local subprocess fallback."""
        if not shutil.which("redis-server"):
            log.warning("redis-server not found in PATH; skipping local Redis.")
            self.status_changed.emit(
                "redis-server not found. Install Redis or start Docker."
            )
            return False
        self.status_changed.emit("Starting local Redis server…")
        try:
            self._local_redis_process = subprocess.Popen(
                ["redis-server", "--daemonize", "no"],
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Brief pause to let Redis bind its port
            time.sleep(1.0)
            log.info("Local Redis started (PID %s).", self._local_redis_process.pid)
            return True
        except OSError as exc:
            log.error("Failed to start local Redis: %s", exc)
            return False

    def _start_local_celery(self) -> bool:
        """Start a Celery worker as a local subprocess fallback."""
        self.status_changed.emit("Starting local Celery worker…")
        try:
            self._local_celery_process = subprocess.Popen(
                [
                    sys.executable, "-m", "celery",
                    "-A", "workers.celery_app",
                    "worker",
                    "--loglevel=info",
                    "--concurrency=1",
                    "--without-heartbeat",
                ],
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info("Local Celery worker started (PID %s).", self._local_celery_process.pid)
            return True
        except OSError as exc:
            log.error("Failed to start local Celery worker: %s", exc)
            return False

    def _stop_local_services(self) -> None:
        """Terminate locally-spawned Redis and Celery processes."""
        for name, proc in [
            ("Celery worker", self._local_celery_process),
            ("Redis", self._local_redis_process),
        ]:
            if proc is None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=8)
                log.info("Local %s stopped.", name)
            except subprocess.TimeoutExpired:
                proc.kill()
                log.warning("Local %s killed after timeout.", name)
            except OSError as exc:
                log.warning("Error stopping local %s: %s", name, exc)
        self._local_redis_process = None
        self._local_celery_process = None

    # ── Reflex server helpers ────────────────────────────────────
    def _start_reflex(self) -> bool:
        """Launch the Reflex dev-server as a child process.

        Returns *True* if the process was spawned successfully.
        """
        self.status_changed.emit("Starting Reflex server…")
        try:
            self._reflex_process = subprocess.Popen(
                _REFLEX_RUN,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log.info("Reflex process started (PID %s).", self._reflex_process.pid)
            return True
        except (FileNotFoundError, OSError) as exc:
            log.error("Failed to start Reflex: %s", exc)
            self._reflex_process = None
            return False

    def _stop_reflex(self) -> None:
        """Gracefully terminate the Reflex subprocess."""
        if self._reflex_process is None:
            return
        try:
            self._reflex_process.terminate()
            self._reflex_process.wait(timeout=10)
            log.info("Reflex process terminated.")
        except subprocess.TimeoutExpired:
            self._reflex_process.kill()
            log.warning("Reflex process killed after timeout.")
        except OSError as exc:
            log.warning("Error stopping Reflex: %s", exc)
        finally:
            self._reflex_process = None

    # ── Server readiness ─────────────────────────────────────────
    @staticmethod
    def is_server_ready() -> bool:
        """Return *True* if ``localhost:3000`` responds with HTTP 200."""
        try:
            with urlopen(_SERVER_URL, timeout=2) as resp:
                return resp.status == 200
        except (URLError, OSError, ValueError):
            return False

    def wait_for_server(self, timeout: int = 30) -> bool:
        """Poll ``is_server_ready()`` every 0.5 s up to *timeout* seconds.

        Emits ``server_ready`` or ``server_failed`` when done.

        Returns:
            *True* if the server became ready within the timeout.
        """
        self.status_changed.emit("Waiting for server…")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_server_ready():
                self.status_changed.emit("Ready!")
                self.server_ready.emit()
                return True
            time.sleep(0.5)

        self.status_changed.emit("Server failed to start.")
        self.server_failed.emit()
        return False

    # ── Public lifecycle API ─────────────────────────────────────
    def start(self) -> bool:
        """Start Docker services then the Reflex server.

        Returns *True* if both were launched (does **not** wait for
        the server to be ready — call ``wait_for_server()`` for that).
        """
        docker_ok = True
        if self.is_docker_available():
            docker_ok = self._start_docker_services()
        else:
            self.status_changed.emit(
                "Docker not available — starting Redis & Celery locally…"
            )
            log.warning("Docker not available; falling back to local subprocesses.")
            redis_ok = self._start_local_redis()
            celery_ok = self._start_local_celery()
            docker_ok = redis_ok and celery_ok
            if not docker_ok:
                self.status_changed.emit(
                    "⚠ Could not start Redis/Celery. "
                    "Run manually: celery -A workers.celery_app worker"
                )

        reflex_ok = self._start_reflex()

        if not reflex_ok:
            self.server_failed.emit()
            return False

        return docker_ok

    def stop(self) -> None:
        """Stop the Reflex server, Docker containers, and any local services."""
        self.status_changed.emit("Shutting down…")
        self._stop_reflex()
        self._stop_docker_services()
        self._stop_local_services()
        self.status_changed.emit("Stopped.")
