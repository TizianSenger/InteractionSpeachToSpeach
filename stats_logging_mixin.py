"""Stats and debug-log UI helpers for VoiceAssistantUI."""

from __future__ import annotations

import queue
import time
from datetime import datetime


class StatsLoggingMixin:
    def _extract_level_token(self, line: str) -> str:
        parts = line.split("|")
        if len(parts) >= 2:
            return parts[1].strip().upper()
        return "INFO"

    def _matches_selected_log_level(self, line: str) -> bool:
        log_level = self.debug_log_level_var.get().strip().upper() or "INFO"
        level_order = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        selected_level = level_order.get(log_level, 20)
        line_level = self._extract_level_token(line)
        return level_order.get(line_level, 20) >= selected_level

    def _insert_debug_log_line(self, line: str) -> None:
        if not hasattr(self, "debug_log_box"):
            return

        level_token = self._extract_level_token(line)
        tag_name = {
            "DEBUG": "log_debug",
            "INFO": "log_info",
            "WARNING": "log_warning",
            "ERROR": "log_error",
            "CRITICAL": "log_critical",
        }.get(level_token, "log_info")
        self.debug_log_box.insert("end", f"{line}\n", tag_name)

    def _pump_logs_into_ui(self) -> None:
        if not hasattr(self, "debug_log_box"):
            return

        inserted_any = False
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break

            self.debug_log_history.append(line)
            if self._matches_selected_log_level(line):
                self._insert_debug_log_line(line)
                inserted_any = True

        if inserted_any:
            self.debug_log_box.see("end")

            current_end = self.debug_log_box.index("end-1c")
            try:
                line_count = int(float(current_end.split(".")[0]))
            except Exception:
                line_count = 0
            overflow = max(0, line_count - self.max_ui_log_lines)
            if overflow > 0:
                self.debug_log_box.delete("1.0", f"{overflow + 1}.0")

    def _track_event(self, label: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.event_history.append(f"[{timestamp}] {label}")
        self._refresh_event_box()

    def _increment_counter(self, name: str, amount: int = 1) -> None:
        current = self.counters.get(name, 0)
        self.counters[name] = current + amount

    def _add_metric_sample(self, name: str, value: float) -> None:
        samples = self.metric_samples.get(name)
        if samples is None:
            return
        samples.append(value)
        if len(samples) > 300:
            del samples[: len(samples) - 300]

    def _format_metric(self, name: str) -> str:
        values = self.metric_samples.get(name, [])
        if not values:
            return "-"
        avg = sum(values) / len(values)
        return f"{avg:.2f}s (n={len(values)})"

    def _schedule_stats_refresh(self) -> None:
        self._refresh_stats_view()
        self.stats_refresh_job_id = self.after(1200, self._schedule_stats_refresh)

    def _refresh_event_box(self) -> None:
        if not hasattr(self, "events_box"):
            return
        history = list(self.event_history)
        current_len = len(history)
        prev_len = getattr(self, "_events_displayed", 0)
        if current_len == prev_len:
            return
        if current_len < prev_len or prev_len == 0:
            # First render or deque rotation: rebuild full content.
            self.events_box.delete("1.0", "end")
            self.events_box.insert("1.0", "\n".join(history) if history else "Noch keine Events")
        else:
            # Append only new entries to avoid full redraw every tick.
            for entry in history[prev_len:]:
                self.events_box.insert("end", f"\n{entry}")
        self._events_displayed = current_len
        self.events_box.see("end")

    def _refresh_stats_view(self) -> None:
        uptime_seconds = max(0, int(time.time() - self.app_started_at))
        minutes, seconds = divmod(uptime_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        summary = (
            f"Uptime: {uptime}\n"
            f"Recordings: {self.counters.get('recordings_started', 0)} gestartet / "
            f"{self.counters.get('recordings_finished', 0)} beendet\n"
            f"Transkriptionen: {self.counters.get('transcriptions', 0)}\n"
            f"Ollama Requests: {self.counters.get('ollama_requests', 0)}\n"
            f"Abbrueche: {self.counters.get('ollama_cancels', 0)}\n"
            f"TTS Chunks: {self.counters.get('tts_chunks', 0)}\n"
            f"Fehler: {self.counters.get('errors', 0)}"
        )
        self.stats_summary_var.set(summary)

        latency_text = (
            f"Transkription: {self._format_metric('transcription_seconds')} | "
            f"First token: {self._format_metric('ollama_first_token_seconds')} | "
            f"First audio: {self._format_metric('first_audio_seconds')} | "
            f"Ollama gesamt: {self._format_metric('ollama_total_seconds')} | "
            f"TTS Chunk: {self._format_metric('tts_chunk_seconds')}"
        )
        self.stats_latency_var.set(latency_text)

    def clear_debug_logs(self) -> None:
        if hasattr(self, "debug_log_box"):
            self.debug_log_box.delete("1.0", "end")

    def on_debug_log_level_changed(self, _selected: str) -> None:
        self.clear_debug_logs()
        for line in self.debug_log_history:
            if self._matches_selected_log_level(line):
                self._insert_debug_log_line(line)
        if hasattr(self, "debug_log_box"):
            self.debug_log_box.see("end")

    def _log_exception(self, context: str, exc: Exception) -> None:
        self._increment_counter("errors")
        self._track_event(f"Fehler in {context}: {exc}")
        self.logger.exception("%s: %s", context, exc)
