"""
Main hands-free reviewer that orchestrates the review process.
"""

import threading
from typing import Optional

from aqt import mw
from aqt.reviewer import Reviewer
from aqt.utils import showWarning
from anki.cards import Card
from anki.hooks import wrap

from .config import Config
from .services import TTSService, STTService, OCRService, ScoringService
from .services.answer_flow import (
    AUTO_RATING,
    OUTCOME_EXACT,
    OUTCOME_UNCLEAR,
    OUTCOME_WRONG,
    classify_answer,
    should_auto_rate,
)


def _clean_field_value(value: str) -> str:
    """Remove sound tags and HTML from a note field, and collapse whitespace."""
    import re

    if not value:
        return ""
    value = re.sub(r'\[sound:[^\]]+\]', '', value)
    value = re.sub(r'<[^>]+>', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def _normalize_spoken_text(text: str) -> str:
    """Lower-case spoken text and drop punctuation, keeping letters and marks.

    Combining marks are kept because Devanagari vowel signs are marks, not
    letters; dropping them would turn "द्वे" into "दव".
    """
    import re
    import unicodedata

    kept = [
        char for char in text.lower()
        if char.isalnum() or char == "_" or char.isspace()
        or unicodedata.category(char).startswith("M")
    ]
    return re.sub(r'\s+', ' ', "".join(kept)).strip()


class HandsFreeReviewer:
    """
    Orchestrates the hands-free review process.

    Flow:
    1. When a card is shown, extract text (including OCR)
    2. Read the front of the card via TTS
    3. Wait for user's verbal answer via STT
    4. Compare answer to back of card
    5. Score and apply rating
    6. Move to next card
    """

    def __init__(self, config: Config):
        self.config = config
        self.is_active = False

        # Debug panel state
        # (timestamp, level, message)
        self._debug_logs: list[tuple[str, str, str]] = []
        self._debug_panel_visible = False

        # Initialize services
        self.tts = TTSService(config.tts)
        self.stt = STTService(config.stt, config.llm)
        self.ocr = OCRService(config.ocr, config.llm.openai_api_key)
        self.scoring = ScoringService(config)

        # Connect debug loggers to services
        self.tts.set_debug_logger(self.debug_log)
        self.stt.set_debug_logger(self.debug_log)
        self.scoring.set_debug_logger(self.debug_log)

        # State
        self._current_card: Optional[Card] = None
        self._front_text: str = ""
        self._back_text: str = ""
        self._processing = False
        self._review_thread: Optional[threading.Thread] = None

        # Original reviewer methods (for restoration)
        self._original_show_question = None
        self._original_show_answer = None

    def debug_log(self, message: str, level: str = "info"):
        """
        Log a message to the debug panel and file.

        Args:
            message: The message to log
            level: One of 'info', 'success', 'warning', 'error', 'ai', 'tts', 'stt'
        """
        from datetime import datetime
        from .utils.logger import log_debug, log_info, log_warning, log_error

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Log to file based on level
        log_msg = f"[{level.upper()}] {message}"
        if level in ("error",):
            log_error(message)
        elif level in ("warning",):
            log_warning(message)
        else:
            log_debug(message)

        # Also print to console
        print(f"[HF {level.upper()}] {message}")

        # Store in memory (keep last 100 logs)
        self._debug_logs.append((timestamp, level, message))
        if len(self._debug_logs) > 100:
            self._debug_logs.pop(0)

        # Update panel if visible
        if self._debug_panel_visible:
            self._append_debug_log(timestamp, level, message)

    def start(self):
        """Start hands-free review mode."""
        if self.is_active:
            return

        self.is_active = True
        self._debug_logs.clear()

        # Show debug panel if enabled
        if self.config.display.show_debug_console:
            self._show_debug_panel()
        self.debug_log("Hands-free mode started", "success")

        # Hook into the reviewer
        self._hook_reviewer()

        # Auto-calibrate microphone
        self._auto_calibrate_mic()

        # Show OCR disabled indicator if needed
        if not self.ocr.is_available():
            self._show_ocr_disabled_indicator()

        # If we're already in review, process current card
        if mw.state == "review" and mw.reviewer.card:
            self._on_question_shown()

    def _auto_calibrate_mic(self):
        """Automatically calibrate microphone threshold."""
        self.debug_log("Starting microphone auto-calibration...", "stt")
        try:
            # Show calibration message
            def show_calibrating():
                if mw and mw.web:
                    mw.web.eval('''
                        (function() {
                            // Remove any existing indicator first
                            let old = document.getElementById("hf-calibration-indicator");
                            if (old) old.remove();
                            
                            let ind = document.createElement("div");
                            ind.id = "hf-calibration-indicator";
                            ind.style.cssText = "position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:rgba(0,0,0,0.9);color:white;padding:30px 50px;border-radius:15px;font-weight:bold;z-index:9999;text-align:center;";
                            ind.innerHTML = "<div style='font-size:24px;margin-bottom:10px;'>🎤 Calibrating...</div><div style='font-size:14px;'>Stay quiet for a moment</div>";
                            document.body.appendChild(ind);
                        })();
                    ''')
            mw.taskman.run_on_main(show_calibrating)

            # Wait for UI to update
            import time
            time.sleep(0.8)

            self.debug_log("Measuring ambient noise...", "stt")
            ambient, speech = self.stt.calibrate(duration=1.5)

            # Set threshold slightly above ambient
            new_threshold = max(300, int(ambient * 1.5))
            self.config.stt.energy_threshold = new_threshold
            self.stt.update_threshold(
                new_threshold, self.config.stt.dynamic_threshold)

            self.debug_log(
                f"Calibration complete: ambient={ambient}, threshold={new_threshold}", "success")

            # Show result
            def show_result():
                if mw and mw.web:
                    mw.web.eval(f'''
                        (function() {{
                            let ind = document.getElementById("hf-calibration-indicator");
                            if (ind) {{
                                ind.innerHTML = "<div style='font-size:24px;margin-bottom:10px;'>✅ Calibrated</div><div style='font-size:14px;'>Threshold: {new_threshold}</div>";
                                setTimeout(() => ind.remove(), 1500);
                            }}
                        }})();
                    ''')
            mw.taskman.run_on_main(show_result)
            time.sleep(1.5)

        except Exception as e:
            self.debug_log(f"Calibration FAILED: {e}", "error")
            print(f"[Hands-Free] Auto-calibration failed: {e}")
            # Hide indicator on error

            def hide_indicator():
                if mw and mw.web:
                    mw.web.eval(
                        'document.getElementById("hf-calibration-indicator")?.remove();')
            mw.taskman.run_on_main(hide_indicator)

    def stop(self):
        """Stop hands-free review mode."""
        self.is_active = False
        self._processing = False

        self.debug_log("Hands-free mode stopped", "info")

        # Hide debug panel
        self._hide_debug_panel()

        # Hide OCR disabled indicator
        self._hide_ocr_disabled_indicator()

        # Stop any ongoing TTS/STT
        self.tts.stop()
        self.stt.stop_recording()

        # Unhook reviewer
        self._unhook_reviewer()

    def _hook_reviewer(self):
        """Hook into Anki's reviewer to intercept card display."""
        from aqt.gui_hooks import reviewer_did_show_question, reviewer_did_show_answer

        reviewer_did_show_question.append(self._on_question_shown)
        reviewer_did_show_answer.append(self._on_answer_shown)

    def _unhook_reviewer(self):
        """Remove hooks from reviewer."""
        from aqt.gui_hooks import reviewer_did_show_question, reviewer_did_show_answer

        try:
            reviewer_did_show_question.remove(self._on_question_shown)
        except ValueError:
            pass

        try:
            reviewer_did_show_answer.remove(self._on_answer_shown)
        except ValueError:
            pass

    def _on_question_shown(self, card: Optional[Card] = None):
        """Called when a question is shown."""
        if not self.is_active:
            return

        if card is None:
            card = mw.reviewer.card

        if card is None:
            return

        self._current_card = card
        self._processing = True

        # Run in background thread to not block UI
        self._review_thread = threading.Thread(
            target=self._process_question,
            args=(card,),
            daemon=True
        )
        self._review_thread.start()

    def _on_answer_shown(self, card: Optional[Card] = None):
        """Called when an answer is shown (shouldn't happen in hands-free mode)."""
        # In hands-free mode, we handle everything ourselves
        pass

    def _process_question(self, card: Card):
        """Process a question card - read it and get user's answer."""
        try:
            self.debug_log(f"Processing card #{card.id}", "info")

            # Get card content
            note = card.note()

            # Get front template
            template = card.template()

            # Check for deck-specific settings
            deck_id = card.did
            deck = mw.col.decks.get(deck_id)
            deck_name = deck.get("name", "") if deck else ""

            # Get deck-specific language
            deck_language = self.config.deck_settings.get_language_for_deck(
                deck_name)
            if deck_language:
                self.debug_log(
                    f"Using deck language: {deck_language} (deck: {deck_name})", "info")
                self._current_language = deck_language
            else:
                self._current_language = self.config.tts.language

            # Get deck-specific field settings
            deck_fields = self.config.deck_settings.get_fields_for_deck(
                deck_name)
            self._current_deck_fields = deck_fields  # Store for extraction

            # Get card content from note fields (cleaner than HTML)
            self._front_text, self._back_text = self._extract_card_text(
                card, note, deck_fields)

            self.debug_log(
                f"Front: {self._front_text[:50]}...", "info") if self._front_text else None
            self.debug_log(
                f"Back: {self._back_text[:50]}...", "info") if self._back_text else None

            # Remove front text from back (Anki includes front in answer)
            if self._front_text and self._back_text.startswith(self._front_text):
                self._back_text = self._back_text[len(
                    self._front_text):].strip()

            # Set TTS language (deck-specific or default)
            self.tts.set_language(self._current_language)
            self.stt.set_language(self._current_language)

            # Read card ease if configured
            if self.config.tts.read_card_ease:
                ease_text = self._get_ease_description(card)
                if ease_text:
                    self.debug_log(f"TTS: Speaking ease '{ease_text}'", "tts")
                    # Default rate for ease
                    self.tts.set_rate(self.config.tts.rate)
                    self.tts.speak(ease_text, blocking=True)

            # Read the question (front rate)
            if self._front_text:
                self.debug_log(f"TTS: Speaking question", "tts")
                self.tts.set_rate(self.config.tts.front_rate)
                spoken = self.tts.speak(self._front_text, blocking=True)
                if not spoken and self.tts._is_sanskrit_language(self._current_language):
                    self._handle_error(
                        "Local Sanskrit TTS could not generate or play audio. "
                        "Check the configured Sanskrit model runtime and local model files."
                    )
                    return

            if not self.is_active:
                return

            # Start listening for user's answer (no prompt - just listen)
            self.debug_log(f"STT: Listening for answer...", "stt")
            user_answer = self.stt.listen_and_recognize(
                on_recording_start=self._on_recording_start,
                on_recording_end=self._on_recording_end,
            )
            if self.stt.last_error:
                self._handle_error(f"Local speech recognition failed: {self.stt.last_error}")
                return

            # Show what was recognized on screen (debug)
            self._show_recognized_text(user_answer)
            self.debug_log(
                f"STT: Recognized: '{user_answer or '(nothing)'}'", "stt")

            if not self.is_active:
                return

            # Check for voice commands (STOP, SKIP, NEXT, DISABLE)
            voice_command = self._parse_voice_command(
                user_answer) if user_answer else None
            if voice_command:
                self.debug_log(f"Voice command: {voice_command}", "info")
                if voice_command == "disable":
                    # Fully disable the plugin
                    self.config.enabled = False
                    self.config.save()
                    self.stop()
                    self.debug_log(
                        "Plugin disabled by voice command", "warning")
                    return
                elif voice_command == "stop":
                    self.stop()
                    return
                elif voice_command in ("skip", "next"):
                    self._skip_card()
                    return

            # Check if user said a rating number instead of an answer
            voice_rating = self._parse_voice_rating(
                user_answer) if user_answer else None

            if voice_rating:
                # User said "1", "2", "3", or "4" - use as manual rating
                self.debug_log(
                    f"Manual rating detected: {voice_rating}", "info")
                self._show_rating_indicator(voice_rating)

                # Always read the final answer so the learner hears it
                self._speak_final_answer()

                self._apply_rating(voice_rating)

            elif user_answer:
                try:
                    graded_status = ""
                    if self.tts._is_sanskrit_language(self._current_language):
                        from .services.sanskrit_grading import grade_sanskrit_answer

                        note_fields = dict(zip(note.keys(), note.fields))
                        deck_mode = self.config.deck_settings.get_grading_mode_for_deck(
                            deck_name
                        ) or "strict"
                        result = grade_sanskrit_answer(note_fields, user_answer, deck_mode)
                        if result.status == "manual":
                            self.debug_log("Manual Sanskrit rating required", "scoring")
                            self._manual_rate_current_card()
                            return
                        score = result.score or 0.0
                        rating = result.rating or 1
                        graded_status = result.status
                    else:
                        self.debug_log(
                            f"Scoring answer with method: {self.config.scoring.method}", "scoring")
                        score, rating = self.scoring.score_answer(
                            user_answer,
                            self._back_text,
                            self._front_text
                        )
                    self.debug_log(
                        f"Score: {score:.2f} → Rating: {rating}", "scoring")
                except Exception as e:
                    self.debug_log(f"Grading ERROR: {e}", "error")
                    self.tts.speak(self._get_localized_prompt(
                        "grading_error"), blocking=True)
                    self._skip_card()
                    return

                if graded_status:
                    outcome = classify_answer(
                        user_answer, self._expected_answer_text(note), graded_status)
                else:
                    outcome = (
                        OUTCOME_EXACT if rating >= AUTO_RATING else OUTCOME_WRONG)

                if should_auto_rate(outcome):
                    # Announce the applied result, then read the answer, then rate.
                    if self.config.announcement.enabled:
                        result_text = self._get_rating_announcement(AUTO_RATING)
                        if result_text:
                            self.tts.speak(result_text, blocking=True)
                self._finish_answer_attempt(outcome)
            else:
                self._finish_answer_attempt(OUTCOME_UNCLEAR)

        except Exception as e:
            from .utils.logger import log_exception
            log_exception(e, f"_process_question card={card.id}")
            self.debug_log(f"Error processing question: {e}", "error")
            try:
                self.tts.speak(self._get_localized_prompt(
                    "error_skip"), blocking=True)
                self._skip_card()
            except Exception as e2:
                log_exception(e2, "_process_question - error recovery")

    def _expected_answer_text(self, note) -> str:
        """Canonical final answer: the primary answer field, else the card back text."""
        field_names = list(note.keys())
        field_values = list(note.fields)
        for name in ("Answer-Devanagari", "Answer-IAST", "Answer-ISO15919"):
            if name in field_names:
                value = _clean_field_value(field_values[field_names.index(name)])
                if value:
                    return value
        return self._back_text

    def _speak_final_answer(self) -> bool:
        """Speak the canonical final answer. Returns whether anything was spoken."""
        if not self._back_text:
            return False
        answer_prefix = self._get_localized_prompt("answer_was")
        self.tts.set_rate(self.config.tts.back_rate)
        self.tts.speak(f"{answer_prefix} {self._back_text}", blocking=True)
        return True

    def _reveal_answer(self) -> None:
        """Reveal the answer side of the card in the UI."""
        import time

        mw.taskman.run_on_main(self._show_answer)
        time.sleep(0.5)

    def _request_rating(self) -> None:
        """Ask for a spoken Anki rating and apply it, or skip when nothing is heard."""
        self.tts.speak(self._get_localized_prompt("say_rating_or_skip"), blocking=True)
        rating_response = self.stt.listen_and_recognize(
            on_recording_start=self._on_recording_start,
            on_recording_end=self._on_recording_end,
        )
        if self.stt.last_error:
            self._handle_error(f"Local speech recognition failed: {self.stt.last_error}")
            return
        self._show_recognized_text(rating_response)
        if not self.is_active:
            return
        rating = self._parse_voice_rating(rating_response) if rating_response else None
        if rating:
            self._show_rating_indicator(rating)
            self._apply_rating_after_answer(rating)
        else:
            self.tts.speak(self._get_localized_prompt("skipping"), blocking=True)
            self._skip_card()

    def _finish_answer_attempt(self, outcome: str) -> None:
        """Read the final answer, then auto-rate only a clearly exact answer.

        Every outcome speaks the canonical answer first. Only ``exact`` is graded
        automatically (Good); near-match, unclear, and wrong answers are never
        auto-rated and fall back to the spoken/manual rating prompt.
        """
        self.debug_log(f"Answer outcome: {outcome}", "scoring")
        if not should_auto_rate(outcome):
            self._reveal_answer()
        if outcome == OUTCOME_UNCLEAR:
            self.tts.speak(self._get_localized_prompt("didnt_catch"), blocking=True)
        self._speak_final_answer()
        if should_auto_rate(outcome):
            self._show_rating_indicator(AUTO_RATING)
            self._apply_rating(AUTO_RATING)
            return
        self._request_rating()

    def _manual_rate_current_card(self) -> None:
        """Show the answer and accept a spoken Anki rating without semantic grading."""
        self._reveal_answer()
        self._speak_final_answer()
        self._request_rating()

    def _extract_card_text(self, card: Card, note, include_fields: list[str] | None = None) -> tuple[str, str]:
        """
        Extract front and back text from card using note fields.
        Filters fields based on include_fields (per-deck) or default skip list.

        Args:
            card: The Anki card
            note: The note associated with the card
            include_fields: Optional list of field names to include (from deck settings).
                           If set, only these fields are read.
                           If None/empty, uses default skip list.
        """
        field_names = list(note.keys())
        field_values = list(note.fields)

        # Get default skip fields
        default_skip = [f.lower()
                        for f in self.config.deck_settings.get_default_skip_fields()]

        # Normalize include_fields
        include_set = [f.lower()
                       for f in include_fields] if include_fields else []

        def should_include_field(name: str) -> bool:
            name_lower = name.lower()
            # If include_fields is set for this deck, only include those
            if include_set:
                return name_lower in include_set
            # Otherwise, include unless in default skip list
            return name_lower not in default_skip

        # Build text from filtered fields
        filtered_texts = []
        for name, value in zip(field_names, field_values):
            if should_include_field(name):
                cleaned = _clean_field_value(value)
                if cleaned:
                    filtered_texts.append(cleaned)

        combined_text = " ".join(filtered_texts)

        # Get the card template to determine front/back split
        template = card.template()
        front_template = template.get("qfmt", "")
        back_template = template.get("afmt", "")

        # Find which fields are on front vs back by checking template references
        front_fields = []
        back_fields = []

        for name, value in zip(field_names, field_values):
            if not should_include_field(name):
                continue
            cleaned = _clean_field_value(value)
            if not cleaned:
                continue
            # Check if field is referenced in templates
            field_ref = f"{{{{{name}}}}}"
            if field_ref in front_template:
                front_fields.append(cleaned)
            if field_ref in back_template and field_ref not in front_template:
                back_fields.append(cleaned)

        # Fallback: if no template parsing worked, use first field as front, rest as back
        if not front_fields and not back_fields and filtered_texts:
            front_fields = [filtered_texts[0]]
            back_fields = filtered_texts[1:] if len(filtered_texts) > 1 else []

        front_text = " ".join(front_fields)
        back_text = " ".join(back_fields)

        return front_text, back_text

    def _get_ease_description(self, card: Card) -> str:
        """Get description of card ease/status."""
        is_german = self.config.tts.language == "de"
        is_sanskrit = self.tts._is_sanskrit_language(
            getattr(self, "_current_language", self.config.tts.language)
        )

        if card.type == 0:  # New card
            if is_sanskrit:
                return "नवीनं पत्रम्।"
            return "Neue Karte." if is_german else "New card."
        elif card.type == 1:  # Learning
            if is_sanskrit:
                return "अध्ययनपत्रम्।"
            return "Lernkarte." if is_german else "Learning card."
        elif card.type == 2:  # Review
            if card.factor:
                ease_percent = card.factor / 10
                if ease_percent >= 250:
                    if is_sanskrit:
                        return "सुपरिचितं पत्रम्।"
                    return "Gut bekannte Karte." if is_german else "Well known card."
                elif ease_percent >= 200:
                    if is_sanskrit:
                        return "परिचितं पत्रम्।"
                    return "Bekannte Karte." if is_german else "Familiar card."
                else:
                    if is_sanskrit:
                        return "कठिनं पत्रम्।"
                    return "Schwierige Karte." if is_german else "Difficult card."
        elif card.type == 3:  # Relearning
            if is_sanskrit:
                return "पुनरभ्यासपत्रम्।"
            return "Wiederholungskarte." if is_german else "Relearning card."
        return ""

    def _on_recording_start(self):
        """Called when recording starts."""
        # Show recording indicator (with debug level display if enabled)
        show_recording = self.config.display.show_recording_indicator
        show_audio_monitor = self.config.display.show_audio_level_monitor
        threshold = self.config.stt.energy_threshold

        def show_indicator():
            if mw and mw.web:
                debug_html = ""
                if show_audio_monitor:
                    debug_html = f'''
                        <div id="hf-debug-panel" style="position:fixed;top:50px;right:10px;background:rgba(0,0,0,0.85);color:white;padding:15px;border-radius:10px;font-family:monospace;font-size:12px;z-index:9999;min-width:200px;">
                            <div style="margin-bottom:8px;font-weight:bold;">🔊 Audio Debug</div>
                            <div>Threshold: <span style="color:#e74c3c;">{threshold}</span></div>
                            <div>Level: <span id="hf-audio-level">0</span></div>
                            <div style="margin-top:8px;background:#333;border-radius:5px;height:20px;overflow:hidden;">
                                <div id="hf-level-bar" style="height:100%;width:0%;background:#27ae60;transition:width 0.1s;"></div>
                            </div>
                            <div id="hf-level-history" style="margin-top:8px;height:60px;background:#222;border-radius:5px;display:flex;align-items:flex-end;padding:2px;gap:1px;"></div>
                        </div>
                    '''

                if not show_recording:
                    return

                mw.web.eval(f'''
                    (function() {{
                        let ind = document.getElementById("hf-recording-indicator");
                        if (!ind) {{
                            ind = document.createElement("div");
                            ind.id = "hf-recording-indicator";
                            ind.style.cssText = "position:fixed;top:10px;right:10px;background:#e74c3c;color:white;padding:10px 20px;border-radius:20px;font-weight:bold;z-index:9999;animation:pulse 1s infinite;";
                            ind.innerHTML = "🎤 Recording...";
                            document.body.appendChild(ind);
                            
                            if (!document.getElementById("hf-pulse-style")) {{
                                let style = document.createElement("style");
                                style.id = "hf-pulse-style";
                                style.textContent = "@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}";
                                document.head.appendChild(style);
                            }}
                        }}
                        ind.style.display = "block";
                        
                        // Add debug panel if enabled
                        let debugPanel = document.getElementById("hf-debug-panel");
                        if (debugPanel) debugPanel.remove();
                        
                        if ({str(show_audio_monitor).lower()}) {{
                            document.body.insertAdjacentHTML("beforeend", `{debug_html}`);
                        }}
                    }})();
                ''')
        mw.taskman.run_on_main(show_indicator)

        # Start audio level monitoring if enabled
        if show_audio_monitor:
            self._start_debug_audio_monitor()

    def _on_recording_end(self):
        """Called when recording ends."""
        # Stop debug monitor
        self._stop_debug_audio_monitor()

        # Hide recording indicator and debug panel
        def hide_indicator():
            if mw and mw.web:
                mw.web.eval('''
                    (function() {
                        let ind = document.getElementById("hf-recording-indicator");
                        if (ind) ind.style.display = "none";
                        let debug = document.getElementById("hf-debug-panel");
                        if (debug) debug.remove();
                    })();
                ''')
        mw.taskman.run_on_main(hide_indicator)

    def _show_recognized_text(self, text: Optional[str]):
        """Show the recognized text on screen for debugging."""
        # Check if display is enabled
        if not self.config.display.show_user_answer:
            print(f"[STT Debug] Recognized: {text if text else '(nothing)'}")
            return
            
        recognized = text if text else "(nothing recognized)"
        print(f"[STT Debug] Recognized: {recognized}")

        def show_text():
            if mw and mw.web:
                # Escape text for JS
                escaped = recognized.replace("\\", "\\\\").replace(
                    '"', '\\"').replace("'", "\\'")
                escaped = escaped.replace("\n", " ").replace("\r", "")

                color = "#27ae60" if text else "#e74c3c"

                mw.web.eval(f'''
                    (function() {{
                        // Remove old indicator
                        let old = document.getElementById("hf-recognized-text");
                        if (old) old.remove();
                        
                        let ind = document.createElement("div");
                        ind.id = "hf-recognized-text";
                        ind.style.cssText = "position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.9);color:white;padding:15px 30px;border-radius:10px;font-size:16px;z-index:9999;max-width:80%;text-align:center;border-left:4px solid {color};";
                        ind.innerHTML = '<div style="font-size:12px;color:#888;margin-bottom:5px;">🎤 You said:</div><div style="color:{color};font-weight:bold;">{escaped}</div>';
                        document.body.appendChild(ind);
                        
                        // Auto-hide after 3 seconds
                        setTimeout(() => ind.remove(), 3000);
                    }})();
                ''')
        mw.taskman.run_on_main(show_text)

    def _start_debug_audio_monitor(self):
        """Start monitoring audio levels for debug display."""
        self._debug_monitor_active = True

        def monitor_thread():
            import audioop
            import time

            try:
                with self.stt._get_microphone() as source:
                    threshold = self.config.stt.energy_threshold

                    while self._debug_monitor_active and self.stt.is_recording:
                        try:
                            buffer = source.stream.read(source.CHUNK)
                            if buffer:
                                energy = audioop.rms(
                                    buffer, source.SAMPLE_WIDTH)
                                self._update_debug_display(energy, threshold)
                        except Exception:
                            break
                        time.sleep(0.05)
            except Exception as e:
                print(f"Debug monitor error: {e}")

        self._debug_thread = threading.Thread(
            target=monitor_thread, daemon=True)
        self._debug_thread.start()

    def _stop_debug_audio_monitor(self):
        """Stop the debug audio monitor."""
        self._debug_monitor_active = False

    def _show_debug_panel(self):
        """Show the debug terminal panel on the left side of the screen."""
        self._debug_panel_visible = True

        def show_panel():
            if mw and mw.web:
                mw.web.eval('''
                    (function() {
                        // Remove old panel if exists
                        let old = document.getElementById("hf-debug-terminal");
                        if (old) old.remove();
                        
                        let panel = document.createElement("div");
                        panel.id = "hf-debug-terminal";
                        panel.style.cssText = `
                            position: fixed;
                            left: 10px;
                            top: 10px;
                            bottom: 10px;
                            width: 350px;
                            background: rgba(0, 0, 0, 0.95);
                            color: #00ff00;
                            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                            font-size: 11px;
                            border-radius: 8px;
                            z-index: 9998;
                            display: flex;
                            flex-direction: column;
                            border: 1px solid #333;
                            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                        `;
                        
                        // Header
                        let header = document.createElement("div");
                        header.style.cssText = "padding:10px 15px;background:#1a1a1a;border-bottom:1px solid #333;border-radius:8px 8px 0 0;display:flex;justify-content:space-between;align-items:center;";
                        header.innerHTML = '<span style="color:#fff;font-weight:bold;">🔧 Hands-Free Debug</span><span style="color:#666;font-size:10px;">Press ESC to close</span>';
                        panel.appendChild(header);
                        
                        // Log container
                        let logs = document.createElement("div");
                        logs.id = "hf-debug-logs";
                        logs.style.cssText = "flex:1;overflow-y:auto;padding:10px;line-height:1.6;";
                        panel.appendChild(logs);
                        
                        // Status bar
                        let status = document.createElement("div");
                        status.id = "hf-debug-status";
                        status.style.cssText = "padding:8px 15px;background:#1a1a1a;border-top:1px solid #333;border-radius:0 0 8px 8px;font-size:10px;color:#666;";
                        status.innerHTML = 'Ready';
                        panel.appendChild(status);
                        
                        document.body.appendChild(panel);
                    })();
                ''')
        mw.taskman.run_on_main(show_panel)

    def _hide_debug_panel(self):
        """Hide the debug terminal panel."""
        self._debug_panel_visible = False

        def hide_panel():
            if mw and mw.web:
                mw.web.eval(
                    'document.getElementById("hf-debug-terminal")?.remove();')
        mw.taskman.run_on_main(hide_panel)

    def _append_debug_log(self, timestamp: str, level: str, message: str):
        """Append a log entry to the debug panel."""
        # Color mapping for different log levels
        colors = {
            "info": "#888",
            "success": "#27ae60",
            "warning": "#f39c12",
            "error": "#e74c3c",
            "ai": "#9b59b6",
            "tts": "#3498db",
            "stt": "#1abc9c",
            "scoring": "#e67e22"
        }

        color = colors.get(level, "#888")

        # Escape message for JS
        escaped = message.replace("\\", "\\\\").replace(
            '"', '\\"').replace("'", "\\'")
        escaped = escaped.replace("\n", "<br>").replace("\r", "")

        # Icon mapping
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "ai": "🤖",
            "tts": "🔊",
            "stt": "🎤",
            "scoring": "📊"
        }
        icon = icons.get(level, "•")

        def append_log():
            if mw and mw.web:
                mw.web.eval(f'''
                    (function() {{
                        let logs = document.getElementById("hf-debug-logs");
                        if (!logs) return;
                        
                        let entry = document.createElement("div");
                        entry.style.cssText = "margin-bottom:6px;padding:4px 0;border-bottom:1px solid #222;";
                        entry.innerHTML = '<span style="color:#555;">[{timestamp}]</span> <span>{icon}</span> <span style="color:{color};">{escaped}</span>';
                        logs.appendChild(entry);
                        
                        // Auto-scroll to bottom
                        logs.scrollTop = logs.scrollHeight;
                        
                        // Update status
                        let status = document.getElementById("hf-debug-status");
                        if (status) status.innerHTML = 'Last: {timestamp} - {level}';
                    }})();
                ''')
        mw.taskman.run_on_main(append_log)

    def _update_debug_display(self, level: int, threshold: int):
        """Update the debug audio display."""
        def update():
            if mw and mw.web:
                # Calculate percentage (cap at 4000)
                pct = min(level / 4000 * 100, 100)
                color = "#27ae60" if level > threshold else "#e74c3c"

                mw.web.eval(f'''
                    (function() {{
                        let levelSpan = document.getElementById("hf-audio-level");
                        let levelBar = document.getElementById("hf-level-bar");
                        let history = document.getElementById("hf-level-history");
                        
                        if (levelSpan) levelSpan.textContent = "{level}";
                        if (levelBar) {{
                            levelBar.style.width = "{pct}%";
                            levelBar.style.background = "{color}";
                        }}
                        
                        // Add to history graph
                        if (history) {{
                            let bar = document.createElement("div");
                            bar.style.cssText = "flex:1;min-width:2px;max-width:4px;background:{color};height:" + Math.max(2, {pct} * 0.6) + "%;align-self:flex-end;";
                            history.appendChild(bar);
                            
                            // Keep only last 50 bars
                            while (history.children.length > 50) {{
                                history.removeChild(history.firstChild);
                            }}
                        }}
                    }})();
                ''')
        mw.taskman.run_on_main(update)

    def _parse_voice_rating(self, text: str) -> Optional[int]:
        """
        Parse a voice rating from text.

        Recognizes numbers 1-4 in various forms:
        - Digits: "1", "2", "3", "4"
        - English words: "one", "two", "three", "four"
        - German words: "eins", "zwei", "drei", "vier"
        - Rating names: "again", "hard", "good", "easy"
        - German names: "nochmal", "schwer", "gut", "leicht"

        Returns:
            Rating 1-4 or None if not recognized
        """
        if not text:
            return None

        text = _normalize_spoken_text(text)

        # Direct number matches
        rating_map = {
            # Digits
            "1": 1, "2": 2, "3": 3, "4": 4,
            # English words
            "one": 1, "two": 2, "three": 3, "four": 4,
            # German words
            "eins": 1, "zwei": 2, "drei": 3, "vier": 4,
            # Sanskrit cardinal words (IAST and common unaccented forms)
            "eka": 1, "ekam": 1, "ekaṃ": 1,
            "dve": 2,
            "trīṇi": 3, "trini": 3,
            "catvāri": 4, "catvari": 4,
            # Sanskrit cardinal words (Devanagari)
            "एकम्": 1, "एकं": 1, "द्वे": 2, "त्रीणि": 3, "चत्वारि": 4,
            # English rating names
            "again": 1, "hard": 2, "good": 3, "easy": 4,
            # German rating names
            "nochmal": 1, "schwer": 2, "gut": 3, "leicht": 4,
            # Common mishearings
            "won": 1, "to": 2, "too": 2, "for": 4,
        }

        # Check for exact match
        if text in rating_map:
            return rating_map[text]

        # Check if the text contains a rating keyword
        for keyword, rating in rating_map.items():
            if keyword in text.split():
                return rating

        return None

    def _parse_voice_command(self, text: str) -> Optional[str]:
        """
        Parse voice commands from text using configurable word lists.

        Returns:
            Command name ("stop", "skip", "next", "disable") or None
        """
        if not text:
            return None

        text = _normalize_spoken_text(text)
        words = set(text.split())

        # Get configurable word lists from config
        vc = self.config.voice_commands

        stop_words = set(w.strip().lower()
                         for w in vc.stop_words.split(",") if w.strip())
        skip_words = set(w.strip().lower()
                         for w in vc.skip_words.split(",") if w.strip())
        next_words = set(w.strip().lower()
                         for w in vc.next_words.split(",") if w.strip())
        disable_words = set(w.strip().lower()
                            for w in vc.disable_words.split(",") if w.strip())

        if words & disable_words:
            return "disable"
        if words & stop_words:
            return "stop"
        if words & skip_words:
            return "skip"
        if words & next_words:
            return "next"

        return None

    def _apply_rating(self, rating: int):
        """Apply the rating to the current card."""
        def do_rating():
            if mw.reviewer.card and mw.state == "review":
                # First show the answer (required by Anki)
                mw.reviewer._showAnswer()
                # Then apply the rating
                mw.reviewer._answerCard(rating)

        mw.taskman.run_on_main(do_rating)

    def _apply_rating_after_answer(self, rating: int):
        """Apply the rating when answer is already shown."""
        def do_rating():
            if mw.reviewer.card and mw.state == "review":
                # Answer is already shown, just apply rating
                mw.reviewer._answerCard(rating)

        mw.taskman.run_on_main(do_rating)

    def _show_answer(self):
        """Show the answer in the reviewer."""
        if mw.reviewer.card and mw.state == "review":
            mw.reviewer._showAnswer()

    def _handle_error(self, error_message: str):
        """Handle an error by stopping hands-free mode."""
        def show_error():
            self.stop()
            showWarning(
                f"Hands-Free mode encountered an error and has been disabled:\n\n{error_message}\n\n"
                "Please check your settings and try again."
            )

        mw.taskman.run_on_main(show_error)

    def _skip_card(self):
        """Skip the current card by burying it."""
        def do_skip():
            if mw.reviewer.card and mw.state == "review":
                # Bury the card (will come back tomorrow)
                mw.reviewer.bury_current_card()
        mw.taskman.run_on_main(do_skip)

    def _get_localized_prompt(self, key: str) -> str:
        """Get a localized prompt based on TTS language."""
        language = getattr(self, "_current_language", self.config.tts.language).lower()
        is_german = language.startswith("de")

        if language.startswith("sa"):
            sanskrit_prompts = {
                "your_answer": "भवतः उत्तरम्?",
                "didnt_catch": "न श्रुतम्। उत्तरं दर्शयामि।",
                "answer_was": "उत्तरम् आसीत्:",
                "grading_error": "मूल्याङ्कने दोषः। कार्डं त्यजामि।",
                "error_skip": "दोषः अभवत्। कार्डं त्यजामि।",
                "say_rating_or_skip": "एकम्, द्वे, त्रीणि, चत्वारि वा वदतु; मौने कार्डं त्यजतु।",
                "skipping": "त्यजामि।",
            }
            return sanskrit_prompts.get(key, "")

        prompts = {
            "your_answer": ("Deine Antwort?", "Your answer?"),
            "didnt_catch": ("Das habe ich nicht verstanden. Zeige die Antwort.", "I didn't catch that. Showing the answer."),
            "answer_was": ("Die Antwort war:", "The answer was:"),
            "grading_error": ("Bewertungsfehler. Überspringe Karte.", "Grading error. Skipping card."),
            "error_skip": ("Fehler aufgetreten. Überspringe Karte.", "Error occurred. Skipping card."),
            "say_rating_or_skip": (
                "Sage eins bis vier zum Bewerten, oder schweige zum Überspringen.",
                "Say one to four to rate, or stay silent to skip."
            ),
            "skipping": ("Überspringe.", "Skipping."),
        }

        german, english = prompts.get(key, ("", ""))
        return german if is_german else english

    def _get_rating_announcement(self, rating: int) -> str:
        """Get audio announcement for rating. Uses custom phrases if set, otherwise defaults."""
        import random
        language = getattr(self, "_current_language", self.config.tts.language).lower()
        if language.startswith("sa"):
            return {
                1: "पुनः प्रयत्नं कुरु।",
                2: "किञ्चित् कठिनम्।",
                3: "साधु।",
                4: "उत्तमम्।",
            }.get(rating, "")
        is_german = language.startswith("de")

        # Check for custom phrases first
        custom_phrases = self._get_custom_phrases(rating, is_german)
        if custom_phrases:
            return random.choice(custom_phrases)

        # Default variations for each rating in each language
        if is_german:
            variations = {
                1: [
                    "Nochmal. Lass uns das wiederholen.",
                    "Nochmal. Das üben wir nochmal.",
                    "Nochmal. Kein Problem, wir versuchen es nochmal.",
                    "Nochmal. Das kriegen wir hin.",
                    "Nochmal. Noch eine Runde.",
                    "Nochmal. Übung macht den Meister.",
                    "Nochmal. Bleib dran!",
                    "Nochmal. Das schaffen wir.",
                    "Nochmal. Wir wiederholen das.",
                    "Nochmal. Gleich nochmal versuchen.",
                ],
                2: [
                    "Schwer. Das war knapp.",
                    "Schwer. Fast richtig.",
                    "Schwer. Du warst nah dran.",
                    "Schwer. Ein Teil war richtig.",
                    "Schwer. Guter Versuch.",
                    "Schwer. Das wird noch.",
                    "Schwer. Weiter so.",
                    "Schwer. Du bist auf dem richtigen Weg.",
                    "Schwer. Nicht schlecht.",
                    "Schwer. Das kommt noch.",
                ],
                3: [
                    "Gut! Richtig!",
                    "Gut! Super gemacht!",
                    "Gut! Toll!",
                    "Gut! Weiter so!",
                    "Gut! Genau richtig!",
                    "Gut! Prima!",
                    "Gut! Gut gemacht!",
                    "Gut! Sehr schön!",
                    "Gut! Klasse!",
                    "Gut! Das sitzt!",
                ],
                4: [
                    "Leicht! Perfekt!",
                    "Leicht! Ausgezeichnet!",
                    "Leicht! Hervorragend!",
                    "Leicht! Das war einfach!",
                    "Leicht! Fantastisch!",
                    "Leicht! Wie im Schlaf!",
                    "Leicht! Meisterhaft!",
                    "Leicht! Das kannst du!",
                    "Leicht! Einwandfrei!",
                    "Leicht! Spitze!",
                ],
            }
        else:
            variations = {
                1: [
                    "Again. Let's review this one.",
                    "Again. We'll practice this more.",
                    "Again. No worries, let's try again.",
                    "Again. We'll get it next time.",
                    "Again. One more round.",
                    "Again. Practice makes perfect.",
                    "Again. Keep at it!",
                    "Again. We've got this.",
                    "Again. Let's repeat that.",
                    "Again. Once more.",
                ],
                2: [
                    "Hard. That was close.",
                    "Hard. Almost there.",
                    "Hard. You were close.",
                    "Hard. Partially correct.",
                    "Hard. Good try.",
                    "Hard. Getting there.",
                    "Hard. Keep going.",
                    "Hard. On the right track.",
                    "Hard. Not bad.",
                    "Hard. You're improving.",
                ],
                3: [
                    "Good! Correct!",
                    "Good! Nice work!",
                    "Good! Well done!",
                    "Good! Keep it up!",
                    "Good! That's right!",
                    "Good! Great job!",
                    "Good! Nicely done!",
                    "Good! You got it!",
                    "Good! Solid answer!",
                    "Good! That's it!",
                ],
                4: [
                    "Easy! Perfect!",
                    "Easy! Excellent!",
                    "Easy! Outstanding!",
                    "Easy! That was easy!",
                    "Easy! Fantastic!",
                    "Easy! Like a pro!",
                    "Easy! Masterful!",
                    "Easy! You nailed it!",
                    "Easy! Flawless!",
                    "Easy! Brilliant!",
                ],
            }

        phrases = variations.get(rating, [""])
        return random.choice(phrases)

    def _get_custom_phrases(self, rating: int, is_german: bool) -> list:
        """Get custom phrases for a rating if configured."""
        ann = self.config.announcement

        # Map rating to config field
        if is_german:
            phrase_map = {
                1: ann.phrases_again_de,
                2: ann.phrases_hard_de,
                3: ann.phrases_good_de,
                4: ann.phrases_easy_de,
            }
        else:
            phrase_map = {
                1: ann.phrases_again_en,
                2: ann.phrases_hard_en,
                3: ann.phrases_good_en,
                4: ann.phrases_easy_en,
            }

        custom_text = phrase_map.get(rating, "")
        if not custom_text or not custom_text.strip():
            return []

        # Split by newlines, filter empty lines
        phrases = [p.strip()
                   for p in custom_text.strip().split("\n") if p.strip()]
        return phrases

    def _show_rating_indicator(self, rating: int):
        """Show a visual indicator of the rating."""
        # Check if display is enabled
        if not self.config.display.show_rating_indicator:
            return
            
        colors = {
            1: "#e74c3c",  # Red - Again
            2: "#e67e22",  # Orange - Hard
            3: "#27ae60",  # Green - Good
            4: "#3498db",  # Blue - Easy
        }
        labels = {
            1: "1 - Again",
            2: "2 - Hard",
            3: "3 - Good",
            4: "4 - Easy",
        }

        color = colors.get(rating, "#95a5a6")
        label = labels.get(rating, "")

        def show_indicator():
            if mw and mw.web:
                mw.web.eval(f'''
                    (function() {{
                        let ind = document.getElementById("hf-rating-indicator");
                        if (!ind) {{
                            ind = document.createElement("div");
                            ind.id = "hf-rating-indicator";
                            ind.style.cssText = "position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);padding:30px 50px;border-radius:15px;font-size:32px;font-weight:bold;color:white;z-index:9999;transition:opacity 0.5s;";
                            document.body.appendChild(ind);
                        }}
                        ind.style.backgroundColor = "{color}";
                        ind.innerHTML = "{label}";
                        ind.style.opacity = "1";
                        ind.style.display = "block";
                        
                        setTimeout(function() {{
                            ind.style.opacity = "0";
                            setTimeout(function() {{ ind.style.display = "none"; }}, 500);
                        }}, 1500);
                    }})();
                ''')
        mw.taskman.run_on_main(show_indicator)

    def _show_ocr_disabled_indicator(self):
        """Show a small indicator in top-right that OCR is disabled."""
        def show_indicator():
            if mw and mw.web:
                mw.web.eval('''
                    (function() {
                        let ind = document.getElementById("hf-ocr-disabled");
                        if (!ind) {
                            ind = document.createElement("div");
                            ind.id = "hf-ocr-disabled";
                            ind.style.cssText = "position:fixed;top:10px;right:10px;padding:6px 12px;border-radius:6px;font-size:11px;background:rgba(230,126,34,0.9);color:white;z-index:9998;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,0.2);";
                            ind.innerHTML = "📷 OCR disabled";
                            ind.title = "Tesseract not installed. Images in cards will be skipped.\\nClick to learn how to install.";
                            ind.onclick = function() {
                                pycmd("hf_show_ocr_help");
                            };
                            document.body.appendChild(ind);
                        }
                        ind.style.display = "block";
                    })();
                ''')
        mw.taskman.run_on_main(show_indicator)

    def _hide_ocr_disabled_indicator(self):
        """Hide the OCR disabled indicator."""
        def hide_indicator():
            if mw and mw.web:
                mw.web.eval('''
                    (function() {
                        let ind = document.getElementById("hf-ocr-disabled");
                        if (ind) {
                            ind.style.display = "none";
                        }
                    })();
                ''')
        mw.taskman.run_on_main(hide_indicator)

    def cleanup(self):
        """Clean up all resources."""
        self.stop()
        self.tts.cleanup()
        self.stt.cleanup()
        self.ocr.cleanup()
        self.scoring.cleanup()
