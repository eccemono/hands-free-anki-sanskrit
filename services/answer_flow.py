"""Answer-outcome classification for hands-free review.

Two concerns live here, both pure so they can be tested without Anki:

* a deliberately narrow Sanskrit near-match comparison, used to decide whether
  a recognized answer is recognizably the expected utterance, and
* the outcome labels that drive what the review flow does next.

Near-match is a *presentation* decision only. It never grades a card correct;
`sanskrit_grading.grade_sanskrit_answer` keeps its exact matching, and any
non-exact outcome falls through to asking the learner for a rating.

The near-match rule is deliberately narrow and fully reproducible:

1. both sides are compared after `normalize_sanskrit`, which folds scripts,
   punctuation, and Unicode combining marks into IAST;
2. the word count must match, and words are compared positionally;
3. each word is split into syllable units with consonant clusters merged, so
   ``paṭhati`` -> ``["pa", "ṭha", "ti"]`` and ``patthati`` -> ``["pa", "ttha", "ti"]``;
4. every unit pair must either be identical or differ only by an approved
   orthographic variant (see `_ONSET_CLASSES`); any other difference, including
   a real phonemic contrast, disqualifies the answer;
5. at most `MAX_VARIANT_UNITS_PER_WORD` variant units may appear per word, and
   `(equal + 0.5 * variants) / units` must reach `MIN_SIMILARITY`.

`AnswerComparison.variants` lists every non-exact token difference, so a rejected
or uncertain answer can always be explained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .sanskrit import normalize_sanskrit


OUTCOME_EXACT = "exact"
OUTCOME_NEAR_MATCH = "near_match"
OUTCOME_UNCLEAR = "unclear"
OUTCOME_WRONG = "wrong"

AUTO_RATING = 3

# A near match allows at most this many orthographic-variant units per word, so
# a word can never be "close" just because it is short.
MAX_VARIANT_UNITS_PER_WORD = 1
# Minimum phoneme-unit similarity for a near match.
MIN_SIMILARITY = 0.6

_VOWELS = (
    "ai", "au", "ā", "ī", "ū", "ṝ", "ḹ", "e", "o", "ṛ", "ḷ", "a", "i", "u",
)
_MARKS = ("ṃ", "ḥ", "ṁ", "m̐")
_CONSONANTS = (
    "kh", "gh", "ṅ", "ch", "jh", "ñ", "ṭh", "ṭ", "ḍh", "ḍ", "ṇ", "th", "dh",
    "ph", "bh", "ś", "ṣ", "t", "d", "p", "b", "m", "n", "y", "r", "l", "ḻ",
    "v", "s", "h", "k", "g", "c", "j",
)

# Approved orthographic variants: distinct spellings of the same sound that a
# speech recognizer may legitimately produce. Every other onset is compared
# exactly, which keeps real phonemic distinctions (retroflex vs dental,
# aspirated vs unaspirated, visarga vs anusvara) out of the near-match class.
_ONSET_CLASSES = {
    "h": "H",
    "ḥ": "H",
    "ṭh": "TTh",
    "tth": "TTh",
}


@dataclass(frozen=True)
class AnswerComparison:
    exact: bool
    near_match: bool
    similarity: float
    expected: str
    recognized: str
    variants: list[str] = field(default_factory=list)


def _syllables(text: str) -> list[str]:
    """Split text into onset+vowel units, merging consonant clusters.

    ``paṭhati`` -> ``["pa", "ṭha", "ti"]`` and ``patthati`` -> ``["pa", "ttha", "ti"]``
    so that a retroflex aspirate and its cluster spelling line up positionally.
    """
    units: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        onset = text[index]
        if onset in _CONSONANTS or onset in _VOWELS or onset in _MARKS:
            index += 1
            if onset in _CONSONANTS:
                while index < length and text[index] in _CONSONANTS:
                    onset += text[index]
                    index += 1
            if index < length and (text[index] in _VOWELS or text[index] in _MARKS):
                onset += text[index]
                index += 1
            units.append(onset)
        else:
            index += 1
            units.append(onset)
    return units


def _onset_class(onset: str) -> str | None:
    """Classify a consonant onset, or None when it is not a Sanskrit onset."""
    if onset in _ONSET_CLASSES:
        return _ONSET_CLASSES[onset]
    if onset in _MARKS:
        return onset
    parts: list[str] = []
    index = 0
    while index < len(onset):
        token = next((item for item in _CONSONANTS if onset.startswith(item, index)), None)
        if token is None:
            return None
        parts.append(_ONSET_CLASSES.get(token, token))
        index += len(token)
    return "+".join(parts) or None


def _unit_key(unit: str) -> tuple[str, str, str] | None:
    """Return (onset, onset class, vowel+marks) for a syllable, or None."""
    onset = unit
    marks = ""
    if onset not in _MARKS:
        while onset and onset[-1] in _MARKS:
            marks = onset[-1] + marks
            onset = onset[:-1]
    vowel = ""
    while onset and onset[-1] in _VOWELS:
        vowel = onset[-1] + vowel
        onset = onset[:-1]
    onset_class = _onset_class(onset)
    if onset_class is None:
        return None
    if vowel in ("", "a"):
        # A bare consonant and the same consonant with its inherent vowel are the
        # same sound; Devanagari writes the visarga without the inherent vowel.
        vowel = "a"
    return (onset, onset_class, vowel + marks)


def compare_answers(expected: str, recognized: str) -> AnswerComparison:
    """Compare an expected answer with a recognized answer.

    ``exact`` is plain normalized equality. ``near_match`` additionally requires
    the same number of words, positional alignment, and that every differing
    syllable pair is an approved orthographic variant.
    """
    expected_norm = normalize_sanskrit(expected)
    recognized_norm = normalize_sanskrit(recognized)
    if expected_norm and expected_norm == recognized_norm:
        return AnswerComparison(True, True, 1.0, expected_norm, recognized_norm)
    if not expected_norm or not recognized_norm:
        return AnswerComparison(False, False, 0.0, expected_norm, recognized_norm)

    expected_words = expected_norm.split()
    recognized_words = recognized_norm.split()
    if len(expected_words) != len(recognized_words):
        return AnswerComparison(False, False, 0.0, expected_norm, recognized_norm)

    total_units = 0
    equal_units = 0
    variants: list[str] = []
    for expected_word, recognized_word in zip(expected_words, recognized_words):
        expected_units = _syllables(expected_word)
        recognized_units = _syllables(recognized_word)
        if len(expected_units) != len(recognized_units):
            return AnswerComparison(False, False, 0.0, expected_norm, recognized_norm)
        word_variants = 0
        for expected_unit, recognized_unit in zip(expected_units, recognized_units):
            total_units += 1
            expected_key = _unit_key(expected_unit)
            recognized_key = _unit_key(recognized_unit)
            if expected_key is None or recognized_key is None:
                if expected_unit == recognized_unit:
                    equal_units += 1
                else:
                    return AnswerComparison(False, False, 0.0, expected_norm, recognized_norm)
                continue
            if expected_key[1:] != recognized_key[1:]:
                return AnswerComparison(False, False, 0.0, expected_norm, recognized_norm)
            if expected_unit == recognized_unit:
                equal_units += 1
            elif (expected_key[0] in _ONSET_CLASSES
                  and recognized_key[0] in _ONSET_CLASSES):
                word_variants += 1
                variants.append(
                    f"{expected_word}: {expected_unit} ↔ {recognized_unit}")
            else:
                return AnswerComparison(False, False, 0.0, expected_norm, recognized_norm)
        if word_variants > MAX_VARIANT_UNITS_PER_WORD:
            return AnswerComparison(False, False, 0.0, expected_norm, recognized_norm)

    if not total_units:
        return AnswerComparison(False, False, 0.0, expected_norm, recognized_norm)
    similarity = (equal_units + 0.5 * len(variants)) / total_units
    near_match = similarity >= MIN_SIMILARITY
    return AnswerComparison(False, near_match, round(similarity, 4), expected_norm, recognized_norm, variants)


def classify_answer(user_answer: str | None, expected: str = "", graded_status: str = "") -> str:
    """Classify a completed answer attempt into a review outcome."""
    if not user_answer or not user_answer.strip():
        return OUTCOME_UNCLEAR
    if graded_status == "correct":
        return OUTCOME_EXACT
    if graded_status == "manual":
        # A manual card is never auto-graded, however the answer compares.
        return OUTCOME_WRONG
    if graded_status and expected:
        comparison = compare_answers(expected, user_answer)
        if comparison.near_match:
            return OUTCOME_NEAR_MATCH
        return OUTCOME_WRONG
    if expected and compare_answers(expected, user_answer).near_match:
        return OUTCOME_NEAR_MATCH
    return OUTCOME_WRONG


def should_auto_rate(outcome: str) -> bool:
    """Only a clearly exact answer is graded automatically."""
    return outcome == OUTCOME_EXACT
