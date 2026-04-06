"""Local HTTP server for standalone VRM viewer.

Serves files from the project root so the standalone launcher can access:
- web/vrm_viewer.html
- web/vendor/*
- assets/*
"""

from __future__ import annotations

import functools
import http.server
import json
import socket
import threading
import time
from typing import Optional


_server: Optional[http.server.HTTPServer] = None
_port: int = 18456
_lock = threading.Lock()
_lipsync_state_lock = threading.Lock()
_lipsync_state: dict[str, float | bool] = {
    "active": False,
    "energy": 0.0,
    "updated_at": 0.0,
}


class _Handler(http.server.SimpleHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_lipsync_get(self) -> None:
        with _lipsync_state_lock:
            payload = dict(_lipsync_state)
        self._send_json(200, payload)

    _MAX_REQUEST_BODY = 4096  # bytes – prevents memory exhaustion via oversized requests

    def _handle_lipsync_post(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            self._send_json(400, {"ok": False, "error": "missing_body"})
            return
        if content_length > self._MAX_REQUEST_BODY:
            self._send_json(413, {"ok": False, "error": "request_too_large"})
            return

        try:
            raw_body = self.rfile.read(content_length)
            parsed = json.loads(raw_body.decode("utf-8"))
            active = bool(parsed.get("active", False))
            energy = float(parsed.get("energy", 0.0))
        except Exception:
            self._send_json(400, {"ok": False, "error": "invalid_json"})
            return

        with _lipsync_state_lock:
            _lipsync_state["active"] = active
            _lipsync_state["energy"] = max(0.0, min(1.0, energy))
            _lipsync_state["updated_at"] = time.time()

        self._send_json(200, {"ok": True})

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/api/lipsync"):
            self._handle_lipsync_get()
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith("/api/lipsync"):
            self._handle_lipsync_post()
            return
        self.send_error(404)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        return


def _find_free_port(start: int = 18456) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def start(root_dir: str) -> int:
    global _server, _port

    with _lock:
        if _server is not None:
            return _port

        _port = _find_free_port()
        with _lipsync_state_lock:
            _lipsync_state["active"] = False
            _lipsync_state["energy"] = 0.0
            _lipsync_state["updated_at"] = time.time()
        handler = functools.partial(_Handler, directory=root_dir)
        _server = http.server.HTTPServer(("127.0.0.1", _port), handler)

        thread = threading.Thread(target=_server.serve_forever, daemon=True)
        thread.name = "vrm-standalone-http"
        thread.start()

    return _port


def stop() -> None:
    global _server
    with _lock:
        if _server:
            _server.shutdown()
            _server = None
