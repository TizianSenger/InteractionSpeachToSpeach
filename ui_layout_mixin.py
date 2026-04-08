"""UI layout and settings popup construction for VoiceAssistantUI."""

from __future__ import annotations

import customtkinter as ctk

from constants import *


class UiLayoutMixin:
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        topbar = ctk.CTkFrame(self)
        topbar.grid(row=0, column=0, padx=12, pady=(10, 6), sticky="ew")
        topbar.grid_columnconfigure(6, weight=1)

        ctk.CTkLabel(topbar, text="Voice Studio", font=(FONT_FAMILY, 20, "bold")).grid(
            row=0, column=0, padx=(10, 12), pady=8, sticky="w"
        )
        ctk.CTkButton(topbar, text="Links", width=72, command=lambda: self._toggle_column("left")).grid(
            row=0, column=1, padx=4, pady=8
        )
        ctk.CTkButton(topbar, text="Input Wave", width=92, command=lambda: self._toggle_column("middle_left")).grid(
            row=0, column=2, padx=4, pady=8
        )
        ctk.CTkButton(topbar, text="Output Wave", width=98, command=lambda: self._toggle_column("middle_right")).grid(
            row=0, column=3, padx=4, pady=8
        )
        ctk.CTkButton(topbar, text="Rechts", width=72, command=lambda: self._toggle_column("right")).grid(
            row=0, column=4, padx=8, pady=8
        )
        ctk.CTkButton(topbar, text="Einstellungen", width=120, command=self.open_settings_popup).grid(
            row=0, column=5, padx=8, pady=8
        )

        # ── Status-Anzeige (streckt sich, Echtzeit-Feedback) ───────────
        self.status_label = ctk.CTkLabel(
            topbar, textvariable=self.status_var,
            font=(FONT_FAMILY, 13), text_color="#94a3b8", anchor="w",
        )
        self.status_label.grid(row=0, column=6, padx=(16, 8), pady=8, sticky="ew")

        # ── Pipeline-Indikator (rechts) ──────────────────────────────────
        import tkinter as tk
        self.pipeline_canvas = tk.Canvas(
            topbar, bg="#1a2236", highlightthickness=0, height=36, width=340
        )
        self.pipeline_canvas.grid(row=0, column=7, padx=(0, 8), pady=6, sticky="e")

        body = ctk.CTkFrame(self)
        body.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.body_frame = body
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=5)
        body.grid_columnconfigure(1, weight=2)
        body.grid_columnconfigure(2, weight=2)
        body.grid_columnconfigure(3, weight=5)

        left_col = ctk.CTkFrame(body)
        left_col.grid(row=0, column=0, padx=(0, 6), pady=0, sticky="nsew")
        left_col.grid_rowconfigure(1, weight=1)
        left_col.grid_columnconfigure(0, weight=1)

        middle_left_col = ctk.CTkFrame(body)
        middle_left_col.grid(row=0, column=1, padx=6, pady=0, sticky="nsew")
        middle_left_col.grid_rowconfigure(1, weight=1)
        middle_left_col.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            middle_left_col,
            text="User Input Wave",
            font=(FONT_FAMILY, 14, "bold"),
        ).grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")

        waveform_host = ctk.CTkFrame(middle_left_col, fg_color="#0d1117")
        waveform_host.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="nsew")
        waveform_host.grid_columnconfigure(0, weight=1)
        waveform_host.grid_rowconfigure(0, weight=1)
        self.waveform_canvas = tk.Canvas(
            waveform_host,
            bg="#0d1117",
            highlightthickness=0,
        )
        self.waveform_canvas.grid(row=0, column=0, sticky="nsew")
        self._schedule_waveform_draw()

        middle_right_col = ctk.CTkFrame(body)
        middle_right_col.grid(row=0, column=2, padx=6, pady=0, sticky="nsew")
        middle_right_col.grid_rowconfigure(1, weight=1)
        middle_right_col.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            middle_right_col,
            text="Model Output Wave",
            font=(FONT_FAMILY, 14, "bold"),
        ).grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")

        output_waveform_host = ctk.CTkFrame(middle_right_col, fg_color="#0d1117")
        output_waveform_host.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="nsew")
        output_waveform_host.grid_columnconfigure(0, weight=1)
        output_waveform_host.grid_rowconfigure(0, weight=1)
        self.output_waveform_canvas = tk.Canvas(
            output_waveform_host,
            bg="#0d1117",
            highlightthickness=0,
        )
        self.output_waveform_canvas.grid(row=0, column=0, sticky="nsew")

        right_col = ctk.CTkFrame(body)
        right_col.grid(row=0, column=3, padx=(6, 0), pady=0, sticky="nsew")
        right_col.grid_rowconfigure(1, weight=1)
        right_col.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(right_col, text="Model Viewer", font=(FONT_FAMILY, 16, "bold")).grid(
            row=0, column=0, padx=12, pady=(10, 6), sticky="w"
        )

        self.viewer_host_frame = ctk.CTkFrame(right_col, fg_color="#101524")
        self.viewer_host_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        ctk.CTkLabel(
            self.viewer_host_frame,
            text="Avatar starten, um den Viewer hier einzubetten",
            text_color="gray70",
        ).place(relx=0.5, rely=0.5, anchor="center")
        self.viewer_host_frame.bind("<Configure>", lambda _evt: self._resize_docked_viewer())

        self.column_frames = {
            "left": left_col,
            "middle_left": middle_left_col,
            "middle_right": middle_right_col,
            "right": right_col,
        }
        self._refresh_column_layout()

        workflow = ctk.CTkFrame(left_col)
        workflow.grid(row=0, column=0, padx=8, pady=(8, 6), sticky="ew")
        for i in range(6):
            workflow.grid_columnconfigure(i, weight=1)

        self.start_btn = ctk.CTkButton(workflow, text="Mic Start", command=self.start_recording)
        self.start_btn.grid(row=0, column=0, padx=4, pady=6, sticky="ew")

        self.stop_btn = ctk.CTkButton(workflow, text="Mic Stop", command=self.stop_recording, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=4, pady=6, sticky="ew")

        self.transcribe_btn = ctk.CTkButton(
            workflow,
            text="Transkribieren",
            command=self.transcribe_recording,
            state="disabled",
            fg_color="#1f6aa5",
        )
        self.transcribe_btn.grid(row=0, column=2, padx=4, pady=6, sticky="ew")

        self.send_btn = ctk.CTkButton(
            workflow,
            text="Senden",
            command=self.send_to_ollama,
            state="disabled",
            fg_color="#0f766e",
        )
        self.send_btn.grid(row=0, column=3, padx=4, pady=6, sticky="ew")

        self.cancel_btn = ctk.CTkButton(
            workflow,
            text="Abbrechen",
            command=self.cancel_current_response,
            state="disabled",
            fg_color="#b42318",
            hover_color="#8f1d15",
        )
        self.cancel_btn.grid(row=0, column=4, padx=4, pady=6, sticky="ew")

        self.avatar_btn = ctk.CTkButton(
            workflow,
            text="Avatar starten",
            command=self.toggle_avatar_viewer,
            fg_color="#1d4ed8",
            hover_color="#1e40af",
        )
        self.avatar_btn.grid(row=0, column=5, padx=4, pady=6, sticky="ew")

        self.text_send_btn = ctk.CTkButton(
            workflow,
            text="Text senden",
            command=self.send_to_ollama,
            fg_color="#7c3aed",
        )
        self.text_send_btn.grid(row=1, column=0, padx=4, pady=6, sticky="ew")

        tabs = ctk.CTkTabview(left_col)
        tabs.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="nsew")

        chat_tab = tabs.add("Chat")
        chat_tab.grid_columnconfigure(0, weight=1)
        chat_tab.grid_rowconfigure(1, weight=1)
        chat_tab.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(chat_tab, text="Transkript", font=(FONT_FAMILY, 13, "bold")).grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w"
        )
        self.transcript_box = ctk.CTkTextbox(chat_tab, wrap="word")
        self.transcript_box.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="nsew")
        self.transcript_box.insert("1.0", "Du kannst hier auch direkt Text eintippen und dann auf 'Text senden' klicken.")

        ctk.CTkLabel(chat_tab, text="Antwort", font=(FONT_FAMILY, 13, "bold")).grid(
            row=2, column=0, padx=10, pady=(4, 2), sticky="w"
        )
        # Thinking indicator (hidden by default, shown while waiting for first token)
        self.thinking_canvas = tk.Canvas(
            chat_tab, bg="#0d1117", highlightthickness=0, height=18
        )
        self.thinking_canvas.grid(row=3, column=0, padx=10, pady=(0, 2), sticky="ew")
        self.thinking_canvas.grid_remove()   # hidden until needed
        self.answer_box = ctk.CTkTextbox(chat_tab, wrap="word")
        self.answer_box.grid(row=4, column=0, padx=10, pady=(0, 10), sticky="nsew")

        history_tab = tabs.add("Verlauf")
        history_tab.grid_rowconfigure(1, weight=1)
        history_tab.grid_columnconfigure(0, weight=1)

        history_header = ctk.CTkFrame(history_tab, fg_color="transparent")
        history_header.grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        history_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(history_header, text="Gesprächsverlauf", font=(FONT_FAMILY, 16, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.clear_history_btn = ctk.CTkButton(
            history_header,
            text="Verlauf leeren",
            width=130,
            command=self.clear_history,
        )
        self.clear_history_btn.grid(row=0, column=1, sticky="e")

        self.history_box = ctk.CTkTextbox(history_tab, wrap="word")
        self.history_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.history_box.insert("1.0", "Der Verlauf wird hier mit Zeitstempel angezeigt.\n")

        stats_tab = tabs.add("Statistik")
        stats_tab.grid_rowconfigure(2, weight=1)
        stats_tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(stats_tab, text="Laufzeit und Kennzahlen", font=(FONT_FAMILY, 16, "bold")).grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w"
        )
        ctk.CTkLabel(stats_tab, textvariable=self.stats_summary_var, justify="left").grid(
            row=1, column=0, padx=10, pady=(0, 8), sticky="w"
        )
        ctk.CTkLabel(stats_tab, textvariable=self.stats_latency_var, justify="left", text_color="gray70").grid(
            row=2, column=0, padx=10, pady=(0, 8), sticky="w"
        )

        self.events_box = ctk.CTkTextbox(stats_tab, wrap="word", height=220)
        self.events_box.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.events_box.insert("1.0", "Noch keine Events")

        debug_tab = tabs.add("Debug Logs")
        debug_tab.grid_rowconfigure(1, weight=1)
        debug_tab.grid_columnconfigure(0, weight=1)

        debug_controls = ctk.CTkFrame(debug_tab, fg_color="transparent")
        debug_controls.grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        debug_controls.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(debug_controls, text="Log-Level").grid(row=0, column=0, padx=(0, 6), pady=0, sticky="w")
        self.debug_level_menu = ctk.CTkOptionMenu(
            debug_controls,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            variable=self.debug_log_level_var,
            command=self.on_debug_log_level_changed,
            width=120,
        )
        self.debug_level_menu.grid(row=0, column=1, padx=(0, 8), pady=0, sticky="w")

        self.clear_debug_btn = ctk.CTkButton(debug_controls, text="Logs leeren", command=self.clear_debug_logs, width=120)
        self.clear_debug_btn.grid(row=0, column=2, padx=(0, 8), pady=0, sticky="w")

        self.debug_log_box = ctk.CTkTextbox(debug_tab, wrap="none")
        self.debug_log_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.debug_log_box.tag_config("log_debug", foreground="#8f9bb3")
        self.debug_log_box.tag_config("log_info", foreground="#d7dce2")
        self.debug_log_box.tag_config("log_warning", foreground="#f4c86a")
        self.debug_log_box.tag_config("log_error", foreground="#ff6b6b")
        self.debug_log_box.tag_config("log_critical", foreground="#ff4d4d")

        self._build_settings_popup()

    def _build_settings_popup(self) -> None:
        popup = ctk.CTkToplevel(self)
        popup.title("Einstellungen")
        popup.geometry("660x820")
        popup.minsize(560, 600)
        popup.withdraw()
        popup.protocol("WM_DELETE_WINDOW", popup.withdraw)
        self.settings_popup = popup

        # ── Titelzeile ────────────────────────────────────────────────────
        header = ctk.CTkFrame(popup, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(18, 4))
        ctk.CTkLabel(
            header, text="⚙  Einstellungen",
            font=(FONT_FAMILY, 22, "bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="Konfiguriere Voice Studio nach deinen Wünschen.",
            text_color="gray60",
            font=(FONT_FAMILY, 12),
        ).pack(anchor="w", pady=(2, 0))

        # ── Tab-Leiste ────────────────────────────────────────────────────
        tabs = ctk.CTkTabview(popup, corner_radius=10, border_width=0)
        tabs.pack(fill="both", expand=True, padx=12, pady=(8, 12))

        for tab_name in ("Workflow", "STT", "Modell", "Persona", "Audio", "TTS", "Avatar"):
            tabs.add(tab_name)

        # Helper: section header + separator line
        def section(parent: ctk.CTkScrollableFrame, title: str) -> None:
            ctk.CTkLabel(
                parent, text=title.upper(),
                font=(FONT_FAMILY, 10, "bold"),
                text_color="#64748b",
                anchor="w",
            ).pack(fill="x", padx=14, pady=(16, 0))
            ctk.CTkFrame(parent, height=1, fg_color="#334155").pack(
                fill="x", padx=14, pady=(3, 8)
            )

        # Helper: label + widget pair
        def lbl(parent: ctk.CTkScrollableFrame, text: str) -> None:
            ctk.CTkLabel(parent, text=text, anchor="w", font=(FONT_FAMILY, 13)).pack(
                fill="x", padx=14, pady=(0, 2)
            )

        # ── Tab: Workflow ─────────────────────────────────────────────────
        wf = ctk.CTkScrollableFrame(tabs.tab("Workflow"), fg_color="transparent")
        wf.pack(fill="both", expand=True)

        section(wf, "Darstellung")
        lbl(wf, "Theme-Modus")
        self.appearance_mode_menu = ctk.CTkOptionMenu(
            wf, values=["Dark", "Light", "System"],
            variable=self.appearance_mode_var,
            command=self.on_appearance_mode_changed,
        )
        self.appearance_mode_menu.pack(fill="x", padx=14, pady=(0, 6))

        section(wf, "Automatisierung")
        self.speak_switch = ctk.CTkSwitch(
            wf, text="Antwort automatisch vorlesen",
            variable=self.auto_speak_var,
        )
        self.speak_switch.pack(anchor="w", padx=14, pady=(0, 8))
        self.auto_pipeline_switch = ctk.CTkSwitch(
            wf, text="Auto-Workflow  (Aufnahme → STT → Ollama)",
            variable=self.auto_pipeline_var,
        )
        self.auto_pipeline_switch.pack(anchor="w", padx=14, pady=(0, 8))
        self.avatar_lipsync_switch = ctk.CTkSwitch(
            wf, text="LipSync (Avatar-Lippenbewegung)",
            variable=self.avatar_lipsync_var,
        )
        self.avatar_lipsync_switch.pack(anchor="w", padx=14, pady=(0, 10))

        section(wf, "Tests & Diagnose")
        self.test_btn = ctk.CTkButton(
            wf, text="  Ollama-Verbindung testen",
            command=self.test_ollama,
            height=38, anchor="w",
        )
        self.test_btn.pack(fill="x", padx=14, pady=(0, 8))
        self.light_test_btn = ctk.CTkButton(
            wf, text="  Licht-Controller testen",
            command=self.open_light_test_popup,
            fg_color="#a16207", hover_color="#854d0e",
            height=38, anchor="w",
        )
        self.light_test_btn.pack(fill="x", padx=14, pady=(0, 10))

        # ── Tab: STT ──────────────────────────────────────────────────────
        stt = ctk.CTkScrollableFrame(tabs.tab("STT"), fg_color="transparent")
        stt.pack(fill="both", expand=True)

        section(stt, "Whisper-Modell")
        lbl(stt, "STT-Provider")
        self.stt_provider_menu = ctk.CTkOptionMenu(
            stt,
            values=STT_PROVIDER_OPTIONS,
            variable=self.stt_provider_var,
        )
        self.stt_provider_menu.pack(fill="x", padx=14, pady=(0, 8))
        self.gemini_stt_test_btn = ctk.CTkButton(
            stt,
            text="Gemini STT testen",
            command=self.run_gemini_stt_test_async,
            height=34,
        )
        self.gemini_stt_test_btn.pack(fill="x", padx=14, pady=(0, 8))
        lbl(stt, "Modell")
        self.whisper_menu = ctk.CTkOptionMenu(
            stt, values=WHISPER_MODEL_OPTIONS,
            variable=self.whisper_model_var,
            command=self.on_whisper_model_changed,
        )
        self.whisper_menu.pack(fill="x", padx=14, pady=(0, 8))
        lbl(stt, "Erkennungssprache")
        self.whisper_language_menu = ctk.CTkOptionMenu(
            stt, values=list(WHISPER_LANGUAGE_OPTIONS.keys()),
            variable=self.whisper_language_var,
        )
        self.whisper_language_menu.pack(fill="x", padx=14, pady=(0, 8))
        lbl(stt, "Geschwindigkeitsmodus")
        self.whisper_speed_menu = ctk.CTkOptionMenu(
            stt, values=WHISPER_SPEED_OPTIONS,
            variable=self.whisper_speed_var,
        )
        self.whisper_speed_menu.pack(fill="x", padx=14, pady=(0, 10))

        section(stt, "Ladefortschritt")
        self.stt_progress_bar = ctk.CTkProgressBar(stt, height=14)
        self.stt_progress_bar.pack(fill="x", padx=14, pady=(0, 6))
        self.stt_progress_bar.set(0)
        self.stt_progress_label = ctk.CTkLabel(
            stt, textvariable=self.stt_progress_var,
            text_color="gray60", anchor="w",
        )
        self.stt_progress_label.pack(fill="x", padx=14, pady=(0, 10))

        # ── Tab: Modell ───────────────────────────────────────────────────
        mdl = ctk.CTkScrollableFrame(tabs.tab("Modell"), fg_color="transparent")
        mdl.pack(fill="both", expand=True)

        section(mdl, "LLM Provider")
        lbl(mdl, "Provider")
        self.llm_provider_menu = ctk.CTkOptionMenu(
            mdl,
            values=LLM_PROVIDER_OPTIONS,
            variable=self.llm_provider_var,
            command=self.on_llm_provider_changed,
        )
        self.llm_provider_menu.pack(fill="x", padx=14, pady=(0, 8))

        self.provider_frames: dict[str, ctk.CTkFrame] = {}

        ollama_frame = ctk.CTkFrame(mdl, fg_color="transparent")
        self.provider_frames["Ollama"] = ollama_frame
        lbl(ollama_frame, "Modell")
        self.ollama_model_menu = ctk.CTkOptionMenu(
            ollama_frame,
            values=OLLAMA_MODEL_OPTIONS,
            variable=self.ollama_model_var,
            command=self.on_active_provider_model_changed,
        )
        self.ollama_model_menu.pack(fill="x", padx=0, pady=(0, 6))
        self.refresh_ollama_btn = ctk.CTkButton(
            ollama_frame,
            text="Modelle vom Provider laden",
            command=self.refresh_ollama_models,
            height=36,
        )
        self.refresh_ollama_btn.pack(fill="x", padx=0, pady=(0, 8))
        lbl(ollama_frame, "API-URL")
        self.ollama_url_entry = ctk.CTkEntry(ollama_frame, textvariable=self.ollama_url_var, height=36)
        self.ollama_url_entry.pack(fill="x", padx=0, pady=(0, 10))

        openai_frame = ctk.CTkFrame(mdl, fg_color="transparent")
        self.provider_frames["OpenAI"] = openai_frame
        lbl(openai_frame, "Modell")
        self.openai_model_menu = ctk.CTkOptionMenu(
            openai_frame,
            values=[self.openai_model_var.get().strip() or "gpt-4o-mini"],
            variable=self.openai_model_var,
            command=self.on_active_provider_model_changed,
        )
        self.openai_model_menu.pack(fill="x", padx=0, pady=(0, 6))
        self.refresh_openai_btn = ctk.CTkButton(
            openai_frame,
            text="Modelle vom Provider laden",
            command=self.refresh_ollama_models,
            height=36,
        )
        self.refresh_openai_btn.pack(fill="x", padx=0, pady=(0, 8))
        lbl(openai_frame, "Base URL")
        self.openai_base_url_entry = ctk.CTkEntry(openai_frame, textvariable=self.openai_base_url_var, height=36)
        self.openai_base_url_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(openai_frame, "API Key")
        self.openai_api_key_entry = ctk.CTkEntry(openai_frame, textvariable=self.openai_api_key_var, show="*", height=36)
        self.openai_api_key_entry.pack(fill="x", padx=0, pady=(0, 6))
        self.openai_api_test_btn = ctk.CTkButton(
            openai_frame,
            text="API-Key testen",
            command=lambda: self.run_provider_api_key_test_async("OpenAI"),
            height=34,
        )
        self.openai_api_test_btn.pack(fill="x", padx=0, pady=(0, 10))

        groq_frame = ctk.CTkFrame(mdl, fg_color="transparent")
        self.provider_frames["Groq"] = groq_frame
        lbl(groq_frame, "Modell")
        self.groq_model_menu = ctk.CTkOptionMenu(
            groq_frame,
            values=[self.groq_model_var.get().strip() or "llama-3.3-70b-versatile"],
            variable=self.groq_model_var,
            command=self.on_active_provider_model_changed,
        )
        self.groq_model_menu.pack(fill="x", padx=0, pady=(0, 6))
        self.refresh_groq_btn = ctk.CTkButton(
            groq_frame,
            text="Modelle vom Provider laden",
            command=self.refresh_ollama_models,
            height=36,
        )
        self.refresh_groq_btn.pack(fill="x", padx=0, pady=(0, 8))
        lbl(groq_frame, "Base URL")
        self.groq_base_url_entry = ctk.CTkEntry(groq_frame, textvariable=self.groq_base_url_var, height=36)
        self.groq_base_url_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(groq_frame, "API Key")
        self.groq_api_key_entry = ctk.CTkEntry(groq_frame, textvariable=self.groq_api_key_var, show="*", height=36)
        self.groq_api_key_entry.pack(fill="x", padx=0, pady=(0, 6))
        self.groq_api_test_btn = ctk.CTkButton(
            groq_frame,
            text="API-Key testen",
            command=lambda: self.run_provider_api_key_test_async("Groq"),
            height=34,
        )
        self.groq_api_test_btn.pack(fill="x", padx=0, pady=(0, 10))

        gemini_frame = ctk.CTkFrame(mdl, fg_color="transparent")
        self.provider_frames["Google Gemini"] = gemini_frame
        lbl(gemini_frame, "Modell")
        self.gemini_model_menu = ctk.CTkOptionMenu(
            gemini_frame,
            values=GEMINI_MODEL_OPTIONS,
            variable=self.gemini_model_var,
            command=self.on_active_provider_model_changed,
        )
        self.gemini_model_menu.pack(fill="x", padx=0, pady=(0, 6))
        self.refresh_gemini_btn = ctk.CTkButton(
            gemini_frame,
            text="Modelle vom Provider laden",
            command=self.refresh_ollama_models,
            height=36,
        )
        self.refresh_gemini_btn.pack(fill="x", padx=0, pady=(0, 8))
        lbl(gemini_frame, "Base URL")
        self.gemini_base_url_entry = ctk.CTkEntry(gemini_frame, textvariable=self.gemini_base_url_var, height=36)
        self.gemini_base_url_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(gemini_frame, "API Key")
        self.gemini_api_key_entry = ctk.CTkEntry(gemini_frame, textvariable=self.gemini_api_key_var, show="*", height=36)
        self.gemini_api_key_entry.pack(fill="x", padx=0, pady=(0, 6))
        self.gemini_api_test_btn = ctk.CTkButton(
            gemini_frame,
            text="API-Key testen",
            command=lambda: self.run_provider_api_key_test_async("Google Gemini"),
            height=34,
        )
        self.gemini_api_test_btn.pack(fill="x", padx=0, pady=(0, 10))

        azure_frame = ctk.CTkFrame(mdl, fg_color="transparent")
        self.provider_frames["Azure OpenAI"] = azure_frame
        lbl(azure_frame, "Deployment")
        self.azure_openai_model_menu = ctk.CTkOptionMenu(
            azure_frame,
            values=[self.azure_openai_deployment_var.get().strip() or "my-deployment"],
            variable=self.azure_openai_deployment_var,
            command=self.on_active_provider_model_changed,
        )
        self.azure_openai_model_menu.pack(fill="x", padx=0, pady=(0, 6))
        self.refresh_azure_btn = ctk.CTkButton(
            azure_frame,
            text="Modelle vom Provider laden",
            command=self.refresh_ollama_models,
            height=36,
        )
        self.refresh_azure_btn.pack(fill="x", padx=0, pady=(0, 8))
        lbl(azure_frame, "Endpoint")
        self.azure_openai_endpoint_entry = ctk.CTkEntry(
            azure_frame,
            textvariable=self.azure_openai_endpoint_var,
            height=36,
        )
        self.azure_openai_endpoint_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(azure_frame, "API Version")
        self.azure_openai_api_version_entry = ctk.CTkEntry(
            azure_frame,
            textvariable=self.azure_openai_api_version_var,
            height=36,
        )
        self.azure_openai_api_version_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(azure_frame, "API Key")
        self.azure_openai_api_key_entry = ctk.CTkEntry(
            azure_frame,
            textvariable=self.azure_openai_api_key_var,
            show="*",
            height=36,
        )
        self.azure_openai_api_key_entry.pack(fill="x", padx=0, pady=(0, 6))
        self.azure_api_test_btn = ctk.CTkButton(
            azure_frame,
            text="API-Key testen",
            command=lambda: self.run_provider_api_key_test_async("Azure OpenAI"),
            height=34,
        )
        self.azure_api_test_btn.pack(fill="x", padx=0, pady=(0, 10))

        anthropic_frame = ctk.CTkFrame(mdl, fg_color="transparent")
        self.provider_frames["Anthropic"] = anthropic_frame
        lbl(anthropic_frame, "Modell")
        self.anthropic_model_menu = ctk.CTkOptionMenu(
            anthropic_frame,
            values=[self.anthropic_model_var.get().strip() or "claude-3-5-sonnet-latest"],
            variable=self.anthropic_model_var,
            command=self.on_active_provider_model_changed,
        )
        self.anthropic_model_menu.pack(fill="x", padx=0, pady=(0, 6))
        self.refresh_anthropic_btn = ctk.CTkButton(
            anthropic_frame,
            text="Modelle vom Provider laden",
            command=self.refresh_ollama_models,
            height=36,
        )
        self.refresh_anthropic_btn.pack(fill="x", padx=0, pady=(0, 8))
        lbl(anthropic_frame, "Base URL")
        self.anthropic_base_url_entry = ctk.CTkEntry(
            anthropic_frame,
            textvariable=self.anthropic_base_url_var,
            height=36,
        )
        self.anthropic_base_url_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(anthropic_frame, "API Version")
        self.anthropic_api_version_entry = ctk.CTkEntry(
            anthropic_frame,
            textvariable=self.anthropic_api_version_var,
            height=36,
        )
        self.anthropic_api_version_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(anthropic_frame, "API Key")
        self.anthropic_api_key_entry = ctk.CTkEntry(
            anthropic_frame,
            textvariable=self.anthropic_api_key_var,
            show="*",
            height=36,
        )
        self.anthropic_api_key_entry.pack(fill="x", padx=0, pady=(0, 6))
        self.anthropic_api_test_btn = ctk.CTkButton(
            anthropic_frame,
            text="API-Key testen",
            command=lambda: self.run_provider_api_key_test_async("Anthropic"),
            height=34,
        )
        self.anthropic_api_test_btn.pack(fill="x", padx=0, pady=(0, 10))

        self.on_llm_provider_changed(self.llm_provider_var.get())

        section(mdl, "Diagnostik")
        self.provider_diagnostics_btn = ctk.CTkButton(
            mdl,
            text="Provider-Diagnostik",
            command=self.run_provider_diagnostics_async,
            height=36,
        )
        self.provider_diagnostics_btn.pack(fill="x", padx=14, pady=(0, 6))
        transfer_row = ctk.CTkFrame(mdl, fg_color="transparent")
        transfer_row.pack(fill="x", padx=14, pady=(0, 6))
        transfer_row.grid_columnconfigure((0, 1), weight=1)
        self.export_provider_cfg_btn = ctk.CTkButton(
            transfer_row,
            text="Provider-Config export",
            command=self.export_provider_config,
            height=34,
        )
        self.export_provider_cfg_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.import_provider_cfg_btn = ctk.CTkButton(
            transfer_row,
            text="Provider-Config import",
            command=self.import_provider_config,
            height=34,
        )
        self.import_provider_cfg_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")
        self.store_api_keys_switch = ctk.CTkSwitch(
            mdl,
            text="API-Keys im Profil speichern",
            variable=self.store_api_keys_var,
        )
        self.store_api_keys_switch.pack(anchor="w", padx=14, pady=(0, 6))
        ctk.CTkLabel(
            mdl,
            textvariable=self.provider_diagnostics_var,
            text_color="gray60",
            wraplength=560,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 10))

        section(mdl, "Generierungs-Parameter")
        self.concise_reply_switch = ctk.CTkSwitch(
            mdl, text="Kurze Voice-Antworten bevorzugen",
            variable=self.concise_reply_var,
        )
        self.concise_reply_switch.pack(anchor="w", padx=14, pady=(0, 10))
        lbl(mdl, "Max Tokens  (num_predict)")
        self.reply_max_tokens_entry = ctk.CTkEntry(mdl, textvariable=self.reply_max_tokens_var, height=36)
        self.reply_max_tokens_entry.pack(fill="x", padx=14, pady=(0, 8))
        lbl(mdl, "Temperatur")
        self.reply_temperature_entry = ctk.CTkEntry(mdl, textvariable=self.reply_temperature_var, height=36)
        self.reply_temperature_entry.pack(fill="x", padx=14, pady=(0, 10))

        # ── Tab: Persona ──────────────────────────────────────────────────
        per = ctk.CTkScrollableFrame(tabs.tab("Persona"), fg_color="transparent")
        per.pack(fill="both", expand=True)

        section(per, "Persönlichkeit der KI")
        ctk.CTkLabel(
            per,
            text="Schieberegler bestimmen den Charakter der Assistenz-Antworten.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 10))

        def add_persona_slider(label: str, variable: ctk.DoubleVar, label_var: ctk.StringVar) -> None:
            row = ctk.CTkFrame(per, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=(6, 0))
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(row, text=label, anchor="w", font=(FONT_FAMILY, 13)).grid(
                row=0, column=0, sticky="w"
            )
            ctk.CTkLabel(row, textvariable=label_var, width=42, anchor="e", font=(FONT_FAMILY, 13, "bold")).grid(
                row=0, column=1, sticky="e"
            )
            ctk.CTkSlider(
                per, from_=0, to=100, number_of_steps=100,
                variable=variable,
                command=self._on_persona_slider_changed,
            ).pack(fill="x", padx=14, pady=(4, 2))

        add_persona_slider("Flirty", self.persona_flirty_var, self.persona_flirty_label_var)
        add_persona_slider("Humor / Sarkasmus", self.persona_humor_var, self.persona_humor_label_var)
        add_persona_slider("Ernsthaftigkeit", self.persona_serious_var, self.persona_serious_label_var)
        add_persona_slider("Dominanz", self.persona_dominance_var, self.persona_dominance_label_var)
        add_persona_slider("Empathie / Wärme", self.persona_empathy_var, self.persona_empathy_label_var)
        add_persona_slider("Temperament", self.persona_temperament_var, self.persona_temperament_label_var)

        self.save_profile_btn = ctk.CTkButton(
            per, text="  Profil speichern",
            command=self.save_profile,
            fg_color="#1d4ed8", hover_color="#1e40af",
            height=40, anchor="w",
        )
        self.save_profile_btn.pack(fill="x", padx=14, pady=(16, 10))
        self._refresh_persona_labels()

        # ── Tab: Audio ────────────────────────────────────────────────────
        aud = ctk.CTkScrollableFrame(tabs.tab("Audio"), fg_color="transparent")
        aud.pack(fill="both", expand=True)

        section(aud, "Mikrofon")
        lbl(aud, "Gerät")
        self.mic_menu = ctk.CTkOptionMenu(
            aud, values=[NO_MIC_DEVICES_LABEL],
            variable=self.mic_device_var,
            command=self.on_mic_selection_changed,
        )
        self.mic_menu.pack(fill="x", padx=14, pady=(0, 6))
        self.refresh_mic_btn = ctk.CTkButton(
            aud, text="Geräteliste aktualisieren",
            command=self.refresh_input_devices, height=36,
        )
        self.refresh_mic_btn.pack(fill="x", padx=14, pady=(0, 8))
        lbl(aud, "Sample Rate (Hz)")
        self.sample_rate_entry = ctk.CTkEntry(aud, textvariable=self.sample_rate_var, height=36)
        self.sample_rate_entry.pack(fill="x", padx=14, pady=(0, 8))
        self.mic_level_bar = ctk.CTkProgressBar(aud, height=14)
        self.mic_level_bar.pack(fill="x", padx=14, pady=(2, 4))
        self.mic_level_bar.set(0)
        self.mic_level_label = ctk.CTkLabel(
            aud, textvariable=self.mic_level_text_var,
            text_color="gray60", anchor="w",
        )
        self.mic_level_label.pack(fill="x", padx=14, pady=(0, 6))

        section(aud, "VAD – Automatischer Stopp")
        self.vad_switch = ctk.CTkSwitch(
            aud, text="Auto-Stop aktivieren  (energiebasiert)",
            variable=self.vad_enabled_var,
        )
        self.vad_switch.pack(anchor="w", padx=14, pady=(0, 10))
        vad_row = ctk.CTkFrame(aud, fg_color="transparent")
        vad_row.pack(fill="x", padx=14, pady=(0, 4))
        vad_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(vad_row, text="Aggressivität  (0–3)", font=(FONT_FAMILY, 13)).grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        self.vad_aggressiveness_menu = ctk.CTkOptionMenu(
            vad_row, values=["0", "1", "2", "3"],
            variable=self.vad_aggressiveness_var, width=90,
        )
        self.vad_aggressiveness_menu.grid(row=0, column=1, sticky="w")
        lbl(aud, "Stille-Timeout (Sekunden)")
        self.vad_silence_entry = ctk.CTkEntry(aud, textvariable=self.vad_silence_timeout_var, height=36)
        self.vad_silence_entry.pack(fill="x", padx=14, pady=(0, 10))

        section(aud, "Wake-Word")
        ctk.CTkLabel(
            aud,
            text="Sprich das Aktivierungswort – das Mikrofon startet automatisch.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 8))
        self.ww_switch = ctk.CTkSwitch(
            aud, text="Wake-Word aktivieren",
            variable=self.wake_word_enabled_var,
            command=self.on_wake_word_toggle,
        )
        self.ww_switch.pack(anchor="w", padx=14, pady=(0, 8))
        lbl(aud, "Aktivierungswort")
        ww_model_row = ctk.CTkFrame(aud, fg_color="transparent")
        ww_model_row.pack(fill="x", padx=14, pady=(0, 6))
        ww_model_row.grid_columnconfigure(0, weight=1)
        from wake_word_mixin import OWW_MODEL_DISPLAY_NAMES
        self.ww_model_menu = ctk.CTkOptionMenu(
            ww_model_row, values=OWW_MODEL_DISPLAY_NAMES,
            variable=self.wake_word_model_var,
        )
        self.ww_model_menu.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(
            ww_model_row, text="Neu starten", width=110,
            command=self.start_wake_word_listener,
        ).grid(row=0, column=1)
        lbl(aud, "Status")
        ctk.CTkLabel(
            aud, textvariable=self.wake_word_status_var,
            text_color="#22d3ee", anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 10))

        section(aud, "TTS Streaming")
        lbl(aud, "Latenz-/Puffer-Profil")
        self.realtime_mode_menu = ctk.CTkOptionMenu(
            aud,
            values=REALTIME_MODE_OPTIONS,
            variable=self.realtime_mode_var,
            command=self.on_realtime_mode_changed,
        )
        self.realtime_mode_menu.pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkLabel(
            aud,
            text="Aggressiv = fruehere Audio-Ausgabe, Stabil = groessere Puffer fuer ruhigeres Streaming.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 10))

        # ── Tab: TTS ──────────────────────────────────────────────────────
        tts = ctk.CTkScrollableFrame(tabs.tab("TTS"), fg_color="transparent")
        tts.pack(fill="both", expand=True)

        section(tts, "Engine & Stimme")
        lbl(tts, "TTS-Engine")
        tts_engines = [
            "edge-tts (natürlich)",
            "gemini-tts (cloud)",
            "piper (lokal, natürlich)",
            "pyttsx3 (lokal)",
        ]
        self.tts_engine_menu = ctk.CTkOptionMenu(
            tts, values=tts_engines,
            variable=self.tts_engine_var,
            command=self.on_tts_engine_changed,
        )
        self.tts_engine_menu.pack(fill="x", padx=14, pady=(0, 8))
        lbl(tts, "Stimme")
        self.tts_voice_menu = ctk.CTkOptionMenu(
            tts, values=list(EDGE_VOICE_OPTIONS.keys()),
            variable=self.tts_voice_var,
            command=self.on_tts_voice_changed,
        )
        self.tts_voice_menu.pack(fill="x", padx=14, pady=(0, 8))
        lbl(tts, "Emotion")
        self.tts_emotion_menu = ctk.CTkOptionMenu(
            tts, values=list(EMOTION_PRESETS.keys()),
            variable=self.tts_emotion_var,
        )
        self.tts_emotion_menu.pack(fill="x", padx=14, pady=(0, 8))
        lbl(tts, "Sprechgeschwindigkeit (Rate)")
        self.tts_rate_entry = ctk.CTkEntry(tts, textvariable=self.tts_rate_var, height=36)
        self.tts_rate_entry.pack(fill="x", padx=14, pady=(0, 10))

        section(tts, "Gemini TTS")
        lbl(tts, "Gemini TTS Modell")
        self.gemini_tts_model_entry = ctk.CTkEntry(tts, textvariable=self.gemini_tts_model_var, height=36)
        self.gemini_tts_model_entry.pack(fill="x", padx=14, pady=(0, 8))
        ctk.CTkLabel(
            tts,
            text="Nutze z.B. gemini-2.5-flash-preview-tts. Base URL/API-Key kommen aus dem Gemini-Provider.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 10))
        self.gemini_tts_test_btn = ctk.CTkButton(
            tts,
            text="Gemini TTS testen",
            command=self.run_gemini_tts_test_async,
            height=34,
        )
        self.gemini_tts_test_btn.pack(fill="x", padx=14, pady=(0, 10))

        section(tts, "Piper – Lokale Stimme")
        lbl(tts, "Modell-Pfad  (.onnx)")
        self.piper_model_entry = ctk.CTkEntry(tts, textvariable=self.piper_model_path_var, height=36)
        self.piper_model_entry.pack(fill="x", padx=14, pady=(0, 8))
        lbl(tts, "Config-Pfad  (.json, optional)")
        self.piper_config_entry = ctk.CTkEntry(tts, textvariable=self.piper_config_path_var, height=36)
        self.piper_config_entry.pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkLabel(
            tts,
            text="Tipp: Lade eine .onnx-Stimmdatei herunter und trage den Pfad ein.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 10))

        # ── Tab: Avatar ───────────────────────────────────────────────────
        av = ctk.CTkScrollableFrame(tabs.tab("Avatar"), fg_color="transparent")
        av.pack(fill="both", expand=True)

        section(av, "VRM-Modell auswählen")
        lbl(av, "Installiertes Modell")
        vrm_names = self._scan_vrm_models()
        self.vrm_model_menu = ctk.CTkOptionMenu(
            av, values=vrm_names if vrm_names else ["(Kein Modell gefunden)"],
            variable=self.vrm_model_var,
        )
        self.vrm_model_menu.pack(fill="x", padx=14, pady=(0, 6))

        # Refresh + Browse row
        btn_row = ctk.CTkFrame(av, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(0, 8))
        btn_row.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(
            btn_row, text="↻  Ordner neu einlesen",
            command=self._refresh_vrm_model_list,
            height=36,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ctk.CTkButton(
            btn_row, text="＋  VRM-Datei importieren",
            command=self._browse_vrm_model,
            height=36,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        ctk.CTkLabel(
            av,
            text="Importierte Dateien werden in den Modell-Ordner kopiert.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 10))

        section(av, "Modell anwenden")
        ctk.CTkButton(
            av, text="Viewer neu starten mit gewähltem Modell",
            command=self._apply_vrm_model,
            height=40, fg_color="#1d4ed8", hover_color="#1e40af",
        ).pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkLabel(
            av,
            text="Der Viewer wird gestoppt und sofort mit dem neuen Modell neu gestartet.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 10))

    def open_settings_popup(self) -> None:
        if self.settings_popup is None:
            return
        self.settings_popup.deiconify()
        self.settings_popup.lift()
        self.settings_popup.focus()

    def _toggle_column(self, column_name: str) -> None:
        aliases = {
            "middle": "middle_left",
            "input": "middle_left",
            "output": "middle_right",
        }
        column_name = aliases.get(column_name, column_name)
        frame = self.column_frames.get(column_name)
        if frame is None:
            return
        visible = self.column_visible.get(column_name, True)
        if visible and sum(1 for value in self.column_visible.values() if value) == 1:
            return
        self.column_visible[column_name] = not visible
        self._refresh_column_layout()

    def _refresh_column_layout(self) -> None:
        body = self.body_frame
        if body is None:
            return

        # Compatibility defaults after layout refactor (3-col -> 4-col).
        if "middle_left" not in self.column_visible:
            self.column_visible["middle_left"] = True
        if "middle_right" not in self.column_visible:
            self.column_visible["middle_right"] = True

        order = ["left", "middle_left", "middle_right", "right"]
        visible_columns = [name for name in order if self.column_visible.get(name, True)]
        if not visible_columns:
            self.column_visible["left"] = True
            visible_columns = ["left"]

        for slot in range(4):
            body.grid_columnconfigure(slot, weight=0)

        for name, frame in self.column_frames.items():
            if name not in visible_columns:
                frame.grid_remove()

        for slot, name in enumerate(visible_columns):
            frame = self.column_frames[name]
            left_pad = 0 if slot == 0 else 6
            right_pad = 0 if slot == (len(visible_columns) - 1) else 6
            frame.grid(row=0, column=slot, padx=(left_pad, right_pad), pady=0, sticky="nsew")
            body.grid_columnconfigure(slot, weight=self.column_weights.get(name, 1))

