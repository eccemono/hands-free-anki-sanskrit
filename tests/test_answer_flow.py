import importlib
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("answer_flow_test_package")
package.__path__ = [str(ROOT / "services")]
sys.modules[package.__name__] = package
answer_flow = importlib.import_module(f"{package.__name__}.answer_flow")
compare_answers = answer_flow.compare_answers
classify_answer = answer_flow.classify_answer
should_auto_rate = answer_flow.should_auto_rate
AUTO_RATING = answer_flow.AUTO_RATING
MIN_SIMILARITY = answer_flow.MIN_SIMILARITY
MAX_VARIANT_UNITS_PER_WORD = answer_flow.MAX_VARIANT_UNITS_PER_WORD
OUTCOME_EXACT = answer_flow.OUTCOME_EXACT
OUTCOME_NEAR_MATCH = answer_flow.OUTCOME_NEAR_MATCH
OUTCOME_UNCLEAR = answer_flow.OUTCOME_UNCLEAR
OUTCOME_WRONG = answer_flow.OUTCOME_WRONG


class AnswerComparisonTests(unittest.TestCase):
    def test_known_flac_answers_compare_as_near_match_with_documented_variants(self):
        comparison = compare_answers(
            "नमस्ते रामः पठति", "नमस्ते, रामह पत्थति")

        self.assertFalse(comparison.exact)
        self.assertTrue(comparison.near_match)
        self.assertGreaterEqual(comparison.similarity, MIN_SIMILARITY)
        self.assertEqual(
            comparison.variants,
            ["rāmaḥ: ḥ ↔ ha", "paṭhati: ṭha ↔ ttha"],
        )

    def test_exact_answer_in_any_script_is_exact(self):
        for answer in ("रामः पठति", "rāmaḥ paṭhati", "rāmaḥ paṭhati."):
            comparison = compare_answers("रामः पठति", answer)
            self.assertTrue(comparison.exact)
            self.assertTrue(comparison.near_match)
            self.assertEqual(comparison.similarity, 1.0)
            self.assertEqual(comparison.variants, [])

    def test_phonemic_differences_are_never_near_matches(self):
        for expected, recognized in (
            ("रामः", "रामी"),          # different vowel
            ("पठति", "पतति"),          # retroflex vs dental
            ("रामः", "राम्ह"),          # visarga vs visarga with ha
            ("रामः", "नरः"),            # different word
            ("रामः", "रामः पठति"),      # different length
            ("रामः पठति", "रामा पठता"),  # every word different
        ):
            with self.subTest(expected=expected, recognized=recognized):
                self.assertFalse(
                    compare_answers(expected, recognized).near_match)

    def test_near_match_requires_aligned_word_count_and_position(self):
        self.assertFalse(compare_answers("रामः पठति", "पठति रामः").near_match)
        self.assertFalse(compare_answers("रामः", "रामी रामः").near_match)

    def test_near_match_caps_variants_per_word_and_similarity(self):
        two_variants = "राम्ह पठथति"
        self.assertEqual(MAX_VARIANT_UNITS_PER_WORD, 1)
        self.assertFalse(compare_answers("रामः पठति", two_variants).near_match)

    def test_empty_answers_never_match(self):
        comparison = compare_answers("रामः", "")
        self.assertFalse(comparison.exact)
        self.assertFalse(comparison.near_match)


class AnswerClassificationTests(unittest.TestCase):
    def test_only_an_exact_answer_is_auto_rated(self):
        self.assertEqual(AUTO_RATING, 3)
        self.assertTrue(should_auto_rate(OUTCOME_EXACT))
        for outcome in (OUTCOME_NEAR_MATCH, OUTCOME_UNCLEAR, OUTCOME_WRONG):
            self.assertFalse(should_auto_rate(outcome))

    def test_unclear_when_nothing_was_recognized(self):
        self.assertEqual(classify_answer(None), OUTCOME_UNCLEAR)
        self.assertEqual(classify_answer("   "), OUTCOME_UNCLEAR)

    def test_graded_correct_is_exact(self):
        self.assertEqual(
            classify_answer("रामः", "रामः", "correct"), OUTCOME_EXACT)

    def test_graded_incorrect_uses_the_near_match_rule(self):
        self.assertEqual(
            classify_answer("रामह पठति", "रामः पठति", "incorrect"),
            OUTCOME_NEAR_MATCH,
        )
        self.assertEqual(
            classify_answer("रामी", "रामः पठति", "incorrect"), OUTCOME_WRONG)

    def test_graded_manual_never_auto_rates(self):
        self.assertEqual(
            classify_answer("रामः", "रामः", "manual"), OUTCOME_WRONG)


if __name__ == "__main__":
    unittest.main()
