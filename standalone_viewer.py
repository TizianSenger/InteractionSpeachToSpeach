"""Standalone VRM viewer launcher.

Starts the VRM viewer directly without a selection UI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.parse
from typing import Optional

from local_http_server import start as start_http


class ViewerLauncher:
    def __init__(self) -> None:
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = self._resolve_project_root()
        self.viewer_process_script = os.path.join(self.base_dir, "viewer_process.py")
        self.runtime_asset_dir = os.path.join(self.project_root, "runtime_assets")
        os.makedirs(self.runtime_asset_dir, exist_ok=True)

        self.vrm_path = os.path.join(
            self.base_dir,
            "runtime_assets",
            "model",
            "vrm_AvatarSample_S.vrm",
        )
        self.proc: Optional[subprocess.Popen] = None

    def _resolve_project_root(self) -> str:
        """Find the folder that contains web/vrm_viewer.html."""
        candidates = [
            self.base_dir,
            os.path.dirname(self.base_dir),
            os.path.join(os.path.dirname(self.base_dir), "CompanionApp"),
        ]

        for root in candidates:
            viewer_html = os.path.join(root, "web", "vrm_viewer.html")
            if os.path.isfile(viewer_html):
                return root

        return self.base_dir

    def _stage_asset(self, abs_path: str, prefix: str) -> str:
        """Copy an asset into a served runtime folder and return relative web path."""
        normalized = os.path.abspath(abs_path)
        filename = os.path.basename(normalized)
        safe_name = f"{prefix}_{filename}"
        target = os.path.join(self.runtime_asset_dir, safe_name)
        shutil.copy2(normalized, target)
        return os.path.relpath(target, self.project_root).replace("\\", "/")

    def _to_url(self, abs_path: str, port: int, prefix: str) -> str:
        normalized = os.path.abspath(abs_path)

        try:
            rel = os.path.relpath(normalized, self.project_root).replace("\\", "/")
            if rel.startswith("../"):
                rel = self._stage_asset(normalized, prefix)
        except ValueError:
            # Different drive letters on Windows -> always stage into runtime folder.
            rel = self._stage_asset(normalized, prefix)

        return f"http://127.0.0.1:{port}/{urllib.parse.quote(rel, safe='/')}"

    def _build_viewer_url(self, port: int) -> str:
        base = f"http://127.0.0.1:{port}/web/vrm_viewer.html"
        parts: dict[str, str] = {}

        parts["vrm"] = self._to_url(self.vrm_path, port, "vrm")

        return base if not parts else base + "?" + urllib.parse.urlencode(parts)

    def open_viewer(self) -> None:
        if not os.path.isfile(self.vrm_path):
            raise FileNotFoundError(f"VRM file not found: {self.vrm_path}")

        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

        port = start_http(self.project_root)
        url = self._build_viewer_url(port)

        try:
            self.proc = subprocess.Popen(
                [sys.executable, self.viewer_process_script, url],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
            raise

        self._wait_for_start_feedback()

    def _wait_for_start_feedback(self) -> None:
        if not self.proc or not self.proc.stdout:
            return

        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("HWND:"):
                print("Viewer is running", flush=True)
                return
            if line.startswith("ERROR:"):
                print(line, flush=True)
                return

    def run(self) -> None:
        self.open_viewer()
        if self.proc:
            self.proc.wait()


def main() -> int:
    try:
        app = ViewerLauncher()
        app.run()
        return 0
    except Exception as exc:
        print(f"Failed to launch viewer: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
