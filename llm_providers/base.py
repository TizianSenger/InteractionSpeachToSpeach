from __future__ import annotations

from typing import Any, Callable, Protocol
import threading


class ChatProvider(Protocol):
    def check_connection(self, model_name: str) -> None:
        ...

    def list_models(self) -> list[str]:
        ...

    def send_chat(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        timeout: tuple[float, float] = (10, 60),
        keep_alive: str | None = None,
    ) -> str:
        ...

    def stream_chat(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        on_chunk: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
        options: dict[str, Any] | None = None,
        timeout: tuple[float, float] = (10, 300),
        keep_alive: str | None = None,
        active_response_setter: Callable[[Any | None], None] | None = None,
    ) -> str:
        ...
