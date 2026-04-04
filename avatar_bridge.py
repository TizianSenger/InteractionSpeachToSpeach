from __future__ import annotations

import importlib
import math
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable

DEFAULT_VRM_RELATIVE_PATH = "runtime_assets/model/vrm_AvatarSample_S.vrm"


class AvatarBridge:
    def __init__(
        self,
        base_dir: Path,
        http_session: Any,
        lipsync_enabled_getter: Callable[[], bool],
        set_status: Callable[[str], None],
        log_exception: Callable[[str, Exception], None],
        logger: Any,
        vrm_relative_path: str = DEFAULT_VRM_RELATIVE_PATH,
    ) -> None:
        self.base_dir = base_dir
        self.http_session = http_session
        self.lipsync_enabled_getter = lipsync_enabled_getter
        self.set_status = set_status
        self.log_exception = log_exception
        self.logger = logger
        self.vrm_relative_path = vrm_relative_path

        self.http_port: int | None = None
        self.server_module: Any | None = None
        self.viewer_process: subprocess.Popen[str] | None = None
        self.last_push_at = 0.0
        self.auto_start_attempted = False

    def is_running(self) -> bool:
        return self.viewer_process is not None and self.viewer_process.poll() is None

    def update_button_state(self, button: Any) -> None:
        if self.is_running():
            button.configure(text="Avatar stoppen", fg_color="#991b1b", hover_color="#7f1d1d")
        else:
            button.configure(text="Avatar starten", fg_color="#1d4ed8", hover_color="#1e40af")

    def _build_viewer_url(self) -> str:
        vrm_file = self.base_dir / self.vrm_relative_path
        if not vrm_file.exists():
            raise FileNotFoundError(f"VRM-Datei nicht gefunden: {vrm_file}")

        rel_vrm = self.vrm_relative_path.replace("\\", "/")
        vrm_url = f"http://127.0.0.1:{self.http_port}/{urllib.parse.quote(rel_vrm, safe='/')}"
        query = urllib.parse.urlencode({"vrm": vrm_url})
        return f"http://127.0.0.1:{self.http_port}/web/vrm_viewer.html?{query}"

    def start(self, button: Any | None = None) -> bool:
        if self.is_running():
            if button is not None:
                self.update_button_state(button)
            return True

        if not self.base_dir.exists():
            self.set_status(f"Avatar-Ordner fehlt: {self.base_dir}")
            return False

        try:
            if self.server_module is None:
                self.server_module = importlib.import_module("local_http_server")

            self.http_port = int(self.server_module.start(str(self.base_dir)))
            viewer_url = self._build_viewer_url()
            viewer_script = self.base_dir / "viewer_process.py"
            self.viewer_process = subprocess.Popen(
                [sys.executable, str(viewer_script), viewer_url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self.auto_start_attempted = True
            if button is not None:
                self.update_button_state(button)
            self.set_status("Avatar-Viewer gestartet")
            self.logger.info("Avatar-Viewer gestartet (Port %s)", self.http_port)
            return True
        except Exception as exc:
            self.log_exception("Avatar-Viewer Start", exc)
            self.viewer_process = None
            if button is not None:
                self.update_button_state(button)
            self.set_status(f"Avatar-Viewer Start fehlgeschlagen: {exc}")
            return False

    def stop(self, button: Any | None = None) -> None:
        self.reset_lipsync()
        proc = self.viewer_process
        self.viewer_process = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

        if self.server_module is not None:
            try:
                self.server_module.stop()
            except Exception:
                pass

        self.http_port = None
        if button is not None:
            self.update_button_state(button)
        self.set_status("Avatar-Viewer gestoppt")

    def toggle(self, button: Any | None = None) -> None:
        if self.is_running():
            self.stop(button)
            return
        self.start(button)

    def post_lipsync(self, active: bool, energy: float = 0.0, force: bool = False) -> None:
        if not self.lipsync_enabled_getter():
            return

        if self.http_port is None or not self.is_running():
            return

        now = time.perf_counter()
        if not force and (now - self.last_push_at) < 0.045:
            return

        self.last_push_at = now
        try:
            self.http_session.post(
                f"http://127.0.0.1:{self.http_port}/api/lipsync",
                json={"active": bool(active), "energy": max(0.0, min(1.0, float(energy)))},
                timeout=(0.5, 0.5),
            )
        except Exception:
            pass

    def reset_lipsync(self) -> None:
        if self.http_port is None or not self.is_running():
            return
        try:
            self.http_session.post(
                f"http://127.0.0.1:{self.http_port}/api/lipsync",
                json={"active": False, "energy": 0.0},
                timeout=(0.5, 0.5),
            )
        except Exception:
            pass

    @staticmethod
    def estimate_lipsync_energy(text: str, elapsed_seconds: float) -> float:
        vowel_count = sum(1 for ch in text.lower() if ch in "aeiouäöüy")
        density = vowel_count / max(1, len(text))
        text_factor = max(0.35, min(0.95, 0.45 + density * 2.6))
        pulse = 0.55 + 0.45 * abs(math.sin(elapsed_seconds * 11.5))
        return max(0.12, min(1.0, text_factor * pulse))

    def start_lipsync_background(self, text: str) -> tuple[threading.Event, threading.Thread | None]:
        stop_event = threading.Event()

        if not self.lipsync_enabled_getter() or self.http_port is None:
            return stop_event, None

        def worker() -> None:
            started_at = time.perf_counter()
            while not stop_event.is_set():
                elapsed = time.perf_counter() - started_at
                self.post_lipsync(True, self.estimate_lipsync_energy(text, elapsed))
                time.sleep(0.05)

            self.reset_lipsync()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return stop_event, thread

    def ensure_for_lipsync(self, button: Any | None = None) -> None:
        if not self.lipsync_enabled_getter():
            return
        if self.is_running() and self.http_port is not None:
            return
        if self.auto_start_attempted:
            return
        self.start(button)
