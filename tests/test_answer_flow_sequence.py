import importlib
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ANSWER = "नमस्ते रामः पठति"
RATING_PROMPT = "वदतु"
DIDNT_CATCH = "न श्रुतम्"

aqt = types.ModuleType("aqt")
aqt.mw = types.SimpleNamespace(taskman=types.SimpleNamespace(run_on_main=lambda fn: fn()))
reviewer_module = types.ModuleType("aqt.reviewer")
reviewer_module.Reviewer = type("Reviewer", (), {})
utils_module = types.ModuleType("aqt.utils")
utils_module.showWarning = lambda *args, **kwargs: None
aqt.reviewer = reviewer_module
aqt.utils = utils_module

anki = types.ModuleType("anki")
anki_cards = types.ModuleType("anki.cards")
anki_cards.Card = type("Card", (), {})
anki_hooks = types.ModuleType("anki.hooks")
anki_hooks.wrap = lambda *args, **kwargs: (lambda fn: fn)
anki.cards = anki_cards
anki.hooks = anki_hooks

package = types.ModuleType("hands_free_anki_sanskrit")
package.__path__ = [str(ROOT)]
for name, module in (
    ("aqt", aqt), ("aqt.reviewer", reviewer_module), ("aqt.utils", utils_module),
    ("anki", anki), ("anki.cards", anki_cards), ("anki.hooks", anki_hooks),
    (package.__name__, package),
):
    sys.modules.setdefault(name, module)

reviewer_module_under_test = importlib.import_module(
    f"{package.__name__}.hands_free_reviewer")
answer_flow = importlib.import_module(f"{package.__name__}.services.answer_flow")

HandsFreeReviewer = reviewer_module_under_test.HandsFreeReviewer
AUTO_RATING = answer_flow.AUTO_RATING
OUTCOME_EXACT = answer_flow.OUTCOME_EXACT
OUTCOME_NEAR_MATCH = answer_flow.OUTCOME_NEAR_MATCH
OUTCOME_UNCLEAR = answer_flow.OUTCOME_UNCLEAR
OUTCOME_WRONG = answer_flow.OUTCOME_WRONG


class FakeTTS:
    def __init__(self, calls):
        self.calls = calls

    def set_rate(self, rate):
        self.calls.append(("set_rate", rate))

    def speak(self, text, blocking=False):
        self.calls.append(("speak", text))


class FakeSTT:
    def __init__(self, calls, response=""):
        self.calls = calls
        self.response = response
        self.last_error = None

    def listen_and_recognize(self, **kwargs):
        self.calls.append(("listen", "rating"))
        return self.response


def build_reviewer(calls, stt_response=""):
    """A reviewer with every Anki/TTS/rating collaborator replaced by a double."""
    reviewer = object.__new__(HandsFreeReviewer)
    reviewer.calls = calls
    reviewer.is_active = True
    reviewer._current_language = "sa-IN"
    reviewer._back_text = "नमस्ते रामः पठति"
    reviewer.tts = FakeTTS(calls)
    reviewer.stt = FakeSTT(calls, stt_response)
    reviewer.config = types.SimpleNamespace(
        announcement=types.SimpleNamespace(enabled=False),
        tts=types.SimpleNamespace(back_rate=1.0, language="sa-IN"),
    )
    reviewer._on_recording_start = lambda: None
    reviewer._on_recording_end = lambda: None
    reviewer._reveal_answer = lambda: calls.append(("reveal", "answer"))
    reviewer._show_rating_indicator = lambda rating: calls.append(("indicator", rating))
    reviewer._apply_rating = lambda rating: calls.append(("apply_rating", rating))
    reviewer._apply_rating_after_answer = (
        lambda rating: calls.append(("apply_rating_after_answer", rating)))
    reviewer._skip_card = lambda: calls.append(("skip", "card"))
    reviewer._handle_error = lambda message: calls.append(("error", message))
    reviewer._show_recognized_text = lambda text: calls.append(("heard", text))
    reviewer.debug_log = lambda message, level="info": None
    return reviewer


def spoken_text(calls):
    return [text for kind, text in calls if kind == "speak"]


def answer_speech(calls):
    """The single speak() call that reads the canonical final answer."""
    answers = [text for text in spoken_text(calls) if ANSWER in text]
    assert len(answers) == 1, f"expected one answer speech, got {answers}"
    return answers[0]


def kinds(calls):
    return [kind for kind, _ in calls]


class AnswerFlowTests(unittest.TestCase):
    def test_exact_answer_is_spoken_before_good_is_applied(self):
        calls = []
        build_reviewer(calls)._finish_answer_attempt(OUTCOME_EXACT)

        self.assertEqual(
            kinds(calls), ["set_rate", "speak", "indicator", "apply_rating"])
        self.assertIn(ANSWER, answer_speech(calls))
        self.assertIn(("apply_rating", AUTO_RATING), calls)
        self.assertNotIn("reveal", kinds(calls))

    def test_near_match_speaks_answer_and_asks_for_a_rating(self):
        calls = []
        build_reviewer(calls, stt_response="")._finish_answer_attempt(OUTCOME_NEAR_MATCH)

        self.assertEqual(
            kinds(calls),
            ["reveal", "set_rate", "speak", "speak", "listen", "heard", "speak", "skip"],
        )
        self.assertIn(ANSWER, answer_speech(calls))
        self.assertNotIn("apply_rating", kinds(calls))
        self.assertNotIn("apply_rating_after_answer", kinds(calls))

    def test_near_match_applies_the_spoken_rating(self):
        calls = []
        build_reviewer(calls, stt_response="द्वे")._finish_answer_attempt(OUTCOME_NEAR_MATCH)

        self.assertIn("heard", kinds(calls))
        self.assertIn(("apply_rating_after_answer", 2), calls)

    def test_unclear_and_wrong_speak_the_answer_and_never_auto_rate(self):
        for outcome in (OUTCOME_UNCLEAR, OUTCOME_WRONG):
            with self.subTest(outcome=outcome):
                calls = []
                build_reviewer(calls)._finish_answer_attempt(outcome)
                self.assertIn(ANSWER, answer_speech(calls))
                self.assertNotIn("apply_rating", kinds(calls))
                if outcome == OUTCOME_UNCLEAR:
                    self.assertIn(DIDNT_CATCH, spoken_text(calls)[0])

    def test_answer_is_spoken_before_the_rating_prompt(self):
        calls = []
        build_reviewer(calls, stt_response="")._finish_answer_attempt(OUTCOME_WRONG)

        speak_index = next(
            index for index, call in enumerate(calls)
            if call[0] == "speak" and ANSWER in call[1])
        prompt_index = next(
            index for index, call in enumerate(calls)
            if call[0] == "speak" and RATING_PROMPT in call[1])
        listen_index = kinds(calls).index("listen")

        self.assertLess(speak_index, prompt_index)
        self.assertLess(prompt_index, listen_index)

    def test_speech_failure_falls_back_to_a_visible_rating_prompt(self):
        calls = []
        reviewer = build_reviewer(calls)
        reviewer.tts = FakeTTS(calls)

        def failing_speak(text, blocking=False):
            if ANSWER in text:
                calls.append(("tts_error", text))
                raise RuntimeError("tts unavailable")

        reviewer.tts.speak = failing_speak
        reviewer._speak_final_answer = lambda: False
        reviewer._finish_answer_attempt(OUTCOME_NEAR_MATCH)

        self.assertIn("reveal", kinds(calls))
        self.assertIn("listen", kinds(calls))

    def test_manual_mode_keeps_revealing_speaking_and_asking(self):
        calls = []
        reviewer = build_reviewer(calls)
        reviewer._manual_rate_current_card()

        self.assertEqual(
            kinds(calls),
            ["reveal", "set_rate", "speak", "speak", "listen", "heard", "speak", "skip"],
        )
        self.assertIn(ANSWER, answer_speech(calls))


class SpokenRatingParsingTests(unittest.TestCase):
    """Sanskrit cards are prompted with Devanagari numerals, so they must parse."""

    def setUp(self):
        self.reviewer = build_reviewer([])

    def test_devanagari_ratings_parse(self):
        for text, rating in (
            ("एकम्", 1), ("एकं", 1), ("द्वे", 2), ("त्रीणि", 3), ("चत्वारि", 4),
            ("dve", 2), ("trīṇi", 3), ("catvāri", 4),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.reviewer._parse_voice_rating(text), rating)

    def test_latin_ratings_still_parse(self):
        for text, rating in (
            ("1", 1), ("three", 3), ("4", 4), ("eins", 1), ("gut", 3), ("leicht", 4),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.reviewer._parse_voice_rating(text), rating)

    def test_punctuation_and_case_are_ignored(self):
        self.assertEqual(self.reviewer._parse_voice_rating("Three!"), 3)
        self.assertIsNone(self.reviewer._parse_voice_rating(""))
        self.assertIsNone(self.reviewer._parse_voice_rating("रामः"))


if __name__ == "__main__":
    unittest.main()
