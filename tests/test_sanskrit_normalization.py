import importlib
import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("sanskrit_service_test_package")
package.__path__ = [str(ROOT / "services")]
sys.modules[package.__name__] = package
sanskrit = importlib.import_module(f"{package.__name__}.sanskrit")
sanskrit_grading = importlib.import_module(f"{package.__name__}.sanskrit_grading")
devanagari_to_iast = sanskrit.devanagari_to_iast
iast_to_devanagari = sanskrit.iast_to_devanagari
normalize_sanskrit = sanskrit.normalize_sanskrit
to_devanagari = sanskrit.to_devanagari
grade_sanskrit_answer = sanskrit_grading.grade_sanskrit_answer


class SanskritNormalizationTests(unittest.TestCase):
    def test_iast_and_devanagari_normalize_to_the_same_form(self):
        self.assertEqual(normalize_sanskrit("rāmaḥ"), normalize_sanskrit("रामः"))
        self.assertEqual(to_devanagari("rāmaḥ"), "रामः")
        self.assertEqual(devanagari_to_iast("रामः"), "rāmaḥ")

    def test_iso15919_vocalic_r_and_retroflex_aspiration(self):
        self.assertEqual(to_devanagari("r̥ṣi"), "ऋषि")
        self.assertEqual(to_devanagari("ṭhā"), "ठा")
        self.assertEqual(to_devanagari("ḻa"), "ळ")

    def test_phonemic_distinctions_are_not_collapsed(self):
        self.assertNotEqual(normalize_sanskrit("a"), normalize_sanskrit("ā"))
        self.assertNotEqual(normalize_sanskrit("ta"), normalize_sanskrit("ṭa"))
        self.assertNotEqual(normalize_sanskrit("sa"), normalize_sanskrit("ṣa"))
        self.assertNotEqual(normalize_sanskrit("aṃ"), normalize_sanskrit("aḥ"))

    def test_unicode_combining_marks_are_canonicalized(self):
        self.assertEqual(normalize_sanskrit("r\u0325ma"), normalize_sanskrit("ṛma"))
        self.assertEqual(normalize_sanskrit("saṃskṛtam"), normalize_sanskrit("संस्कृतम्"))


class SanskritGradingTests(unittest.TestCase):
    def setUp(self):
        self.fields = {
            "Answer-Devanagari": "रामः",
            "Answer-IAST": "rāmaḥ",
            "Answer-ISO15919": "rāmaḥ",
            "Accepted answers": "नरः<br>नरो",
        }

    def test_strict_accepts_equivalent_scripts_but_not_unlisted_sandhi(self):
        self.assertEqual(grade_sanskrit_answer(self.fields, "rāmaḥ").status, "correct")
        self.assertEqual(grade_sanskrit_answer(self.fields, "रामः").status, "correct")
        self.assertEqual(grade_sanskrit_answer(self.fields, "नरो").status, "incorrect")

    def test_accepted_mode_accepts_only_explicit_variants(self):
        accepted = {**self.fields, "Answer-Devanagari": "नरः"}
        self.assertEqual(grade_sanskrit_answer(accepted, "नरो", "accepted").status, "correct")
        self.assertEqual(grade_sanskrit_answer(accepted, "नरौ", "accepted").status, "incorrect")

    def test_translation_recitation_and_open_response_are_manual(self):
        for exercise in ("translation", "recitation", "open-ended"):
            fields = dict(self.fields, **{"Exercise Type": exercise})
            self.assertEqual(grade_sanskrit_answer(fields, "रामः").status, "manual")

    def test_missing_answer_fields_fall_back_to_manual(self):
        self.assertEqual(grade_sanskrit_answer({"Prompt": "रामः"}, "rāmaḥ").status, "manual")

    def test_card_can_force_manual_mode(self):
        fields = dict(self.fields, **{"Grading Mode": "manual"})
        self.assertEqual(grade_sanskrit_answer(fields, "रामः", "strict").status, "manual")

    def test_source_controlled_validation_fixture(self):
        fixture_path = ROOT / "tests/fixtures/sanskrit-validation.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        cases = {case["id"]: case for case in fixture["cases"]}

        cross_script = cases["cross-script-rama-visarga"]
        self.assertEqual(
            normalize_sanskrit(cross_script["Answer-Devanagari"]),
            normalize_sanskrit(cross_script["Answer-IAST"]),
        )
        sandhi = cases["listed-sandhi-variant"]
        self.assertEqual(
            grade_sanskrit_answer(sandhi, sandhi["spoken_variant"], sandhi["grading_mode"]).status,
            "correct",
        )
        manual = cases["translation-manual"]
        self.assertEqual(grade_sanskrit_answer(manual, "rāmaḥ").status, "manual")


if __name__ == "__main__":
    unittest.main()
