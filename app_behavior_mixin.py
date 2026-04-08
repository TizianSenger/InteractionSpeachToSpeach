"""Application behavior helpers for VoiceAssistantUI."""

from __future__ import annotations

import subprocess
import sys

import customtkinter as ctk


class AppBehaviorMixin:
    def on_appearance_mode_changed(self, selected_mode: str) -> None:
        normalized = selected_mode.strip().capitalize() if selected_mode else "Dark"
        if normalized not in {"Dark", "Light", "System"}:
            normalized = "Dark"

        self.appearance_mode_var.set(normalized)
        ctk.set_appearance_mode(normalized)

        # Only restart when the user changed the theme after startup
        active = getattr(self, "_active_appearance_mode", None)
        if active is not None and normalized != active:
            try:
                self.save_profile(notify=False)
            except Exception:
                pass
            self.after(80, self._restart_process)

    def _restart_process(self) -> None:
        try:
            self._closing = True
            self._stop_bg_stream()
            self.stop_wake_word_listener()
        except Exception:
            pass
        subprocess.Popen([sys.executable] + sys.argv)
        self.destroy()

    def _persona_instruction(self, key: str, value: float) -> str:
        if key == "flirty":
            if value >= 70:
                return "dezent flirtend und charmant"
            if value >= 40:
                return "warm, spielerisch und freundlich"
            return "neutral-professionell"

        if key == "humor":
            if value >= 70:
                return "mit leichtem, intelligentem Humor"
            if value >= 40:
                return "locker und auflockernd"
            return "ohne Witze"

        if key == "serious":
            if value >= 70:
                return "sachlich, praezise und fokussiert"
            if value >= 40:
                return "ausgewogen zwischen locker und sachlich"
            return "locker-konversationell"

        if key == "dominance":
            if value >= 70:
                return "klar fuehrend mit konkreten Vorschlaegen"
            if value >= 40:
                return "leitend, aber kooperativ"
            return "zurueckhaltend und fragend"

        if key == "empathy":
            if value >= 70:
                return "sehr empathisch und validierend"
            if value >= 40:
                return "freundlich und verstaendnisvoll"
            return "direkt ohne viel Emotionalisierung"

        if value >= 70:
            return "dynamisch und energisch"
        if value >= 40:
            return "ausgeglichen im Tempo"
        return "ruhig und kontrolliert"

    def _build_persona_system_prompt(self) -> str:
        flirty = self._persona_instruction("flirty", float(self.persona_flirty_var.get()))
        humor = self._persona_instruction("humor", float(self.persona_humor_var.get()))
        serious = self._persona_instruction("serious", float(self.persona_serious_var.get()))
        dominance = self._persona_instruction("dominance", float(self.persona_dominance_var.get()))
        empathy = self._persona_instruction("empathy", float(self.persona_empathy_var.get()))
        temperament = self._persona_instruction("temperament", float(self.persona_temperament_var.get()))
        concise_instruction = "Antworte in 1-3 Saetzen." if self.concise_reply_var.get() else "Antworte so detailliert wie noetig."
        return (
            "Du bist ein persoenlicher Sprachassistent auf Deutsch. "
            "Sei menschlich, charmant und kontextbewusst."
            f"Stil: {flirty}; {humor}; {serious}; {dominance}; {empathy}; {temperament}. "
            "Verwende KEINE Emojis in deinen Antworten. "
            f"{concise_instruction}"
        )
