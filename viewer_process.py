"""Launch Microsoft Edge in app mode for the standalone VRM viewer.

Usage:
    python viewer_process.py <url>
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional


_EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def _find_edge() -> Optional[str]:
    for path in _EDGE_CANDIDATES:
        if os.path.isfile(path):
            return path

    found = shutil.which("msedge") or shutil.which("msedge.exe")
    if found:
        return found

    for pattern in (
        r"C:\Program Files (x86)\Microsoft\EdgeCore\*\msedge.exe",
        r"C:\Program Files\Microsoft\EdgeCore\*\msedge.exe",
    ):
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            return matches[0]

    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe",
        )
        value, _ = winreg.QueryValueEx(key, "")
        winreg.CloseKey(key)
        if value and os.path.isfile(value):
            return value
    except Exception:
        pass

    return None


def _find_edge_hwnd(pid: int, timeout: float = 12.0) -> Optional[int]:
    user32 = ctypes.windll.user32
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    buf = ctypes.create_unicode_buffer(256)
    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:
        matches: list[int] = []

        def _cb(hwnd: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True
            if user32.GetParent(hwnd):
                return True
            user32.GetClassNameW(hwnd, buf, 256)
            if buf.value not in ("Chrome_WidgetWin_1", "Chrome_WidgetWin_0"):
                return True

            proc_id = ctypes.c_ulong(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
            if proc_id.value == pid:
                matches.append(hwnd)
            return True

        user32.EnumWindows(enum_proc(_cb), 0)
        if matches:
            return matches[0]
        time.sleep(0.1)

    return None


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "about:blank"
    edge = _find_edge()
    if not edge:
        print("ERROR:no_edge", flush=True)
        return 1

    profile_dir = tempfile.mkdtemp(prefix="vrm_edge_")
    cmd = [
        edge,
        f"--app={url}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-default-apps",
        "--disable-sync",
        "--disable-signin",
        "--no-pings",
        "--disable-features=msEdgeEnableNTPAppsSyncApps,msEdgeSyncFeature,IdentityConsistency,SignedExchange,EdgeWelcomePage,EdgeSignInPromo,EdgeAccountCTAInNewTab",
        "--edge-no-first-run-if-not-default",
        "--suppress-message-center-popups",
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"ERROR:{exc}", flush=True)
        return 1

    def _report() -> None:
        time.sleep(2.0)
        hwnd = _find_edge_hwnd(proc.pid)
        if hwnd:
            print(f"HWND:{hwnd}", flush=True)
        else:
            print("ERROR:hwnd_not_found", flush=True)

    threading.Thread(target=_report, daemon=True).start()

    try:
        proc.wait()
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
