from __future__ import annotations

import html
import re
from dataclasses import dataclass

from .sanskrit import normalize_sanskrit


_MODE_ALIASES = {
    "accepted": "accepted",
    "accepted variants": "accepted",
    "accepted_variants": "accepted",
    "strict": "strict",
    "manual": "manual",
}
_MANUAL_CARD_TYPES = {
    "translation", "free translation", "recitation", "oral recitation",
    "open", "open ended", "open-ended", "free response", "free-form",
}
_ANSWER_FIELDS = ("answer-devanagari", "answer-iast", "answer-iso15919")
_ACCEPTED_FIELDS = ("accepted answers", "accepted-answers", "acceptedanswers")


@dataclass(frozen=True)
class GradeResult:
    mode: str
    status: str
    score: float | None = None
    rating: int | None = None
    matched_answer: str = ""


def _field_key(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().casefold())


def _clean_field(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<\s*br\s*/?\s*>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"</\s*(div|p|li)\s*>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]*>", " ", value)
    return value.strip()


def _get(fields: dict[str, str], candidates: tuple[str, ...]) -> str:
    keyed = {_field_key(key): value for key, value in fields.items()}
    compact = {key.replace(" ", ""): value for key, value in keyed.items()}
    for candidate in candidates:
        if candidate in keyed:
            return _clean_field(keyed[candidate])
        compact_candidate = candidate.replace(" ", "")
        if compact_candidate in compact:
            return _clean_field(compact[compact_candidate])
    return ""


def _split_answers(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?:\r?\n|\s*[|;]\s*)", value) if part.strip()]


def _mode_from_fields(fields: dict[str, str], default_mode: str) -> str:
    requested = _get(fields, ("grading mode", "grading-mode", "gradingmode"))
    mode = _MODE_ALIASES.get(requested.casefold(), _MODE_ALIASES.get(default_mode.casefold(), "strict"))
    card_type = _get(fields, ("exercise type", "card type", "response type")).casefold()
    if card_type in _MANUAL_CARD_TYPES:
        return "manual"
    return mode


def grade_sanskrit_answer(
    fields: dict[str, str],
    user_answer: str,
    default_mode: str = "strict",
) -> GradeResult:
    """Compare a Sanskrit answer across Devanagari, IAST, and ISO 15919.

    This deliberately performs exact normalized matching; it does not infer meaning.
    """
    mode = _mode_from_fields(fields, default_mode)
    if mode == "manual":
        return GradeResult(mode="manual", status="manual")

    expected = [
        value for name, value in fields.items()
        if _field_key(name) in _ANSWER_FIELDS and _clean_field(value)
    ]
    if not expected:
        return GradeResult(mode="manual", status="manual")
    if mode == "accepted":
        accepted = _get(fields, _ACCEPTED_FIELDS)
        expected.extend(_split_answers(accepted))

    normalized_user = normalize_sanskrit(user_answer)
    if not normalized_user:
        return GradeResult(mode=mode, status="incorrect", score=0.0, rating=1)

    for answer in expected:
        if normalize_sanskrit(answer) == normalized_user:
            return GradeResult(mode=mode, status="correct", score=1.0, rating=3, matched_answer=answer)
    return GradeResult(mode=mode, status="incorrect", score=0.0, rating=1)
