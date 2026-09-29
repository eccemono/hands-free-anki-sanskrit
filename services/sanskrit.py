from __future__ import annotations

import html
import re
import unicodedata


_VOWELS = {
    "a": ("अ", ""), "ā": ("आ", "ा"), "i": ("इ", "ि"), "ī": ("ई", "ी"),
    "u": ("उ", "ु"), "ū": ("ऊ", "ू"), "ṛ": ("ऋ", "ृ"), "ṝ": ("ॠ", "ॄ"),
    "ḷ": ("ऌ", "ॢ"), "ḹ": ("ॡ", "ॣ"), "e": ("ए", "े"), "ai": ("ऐ", "ै"),
    "o": ("ओ", "ो"), "au": ("औ", "ौ"),
}
_CONSONANTS = {
    "k": "क", "kh": "ख", "g": "ग", "gh": "घ", "ṅ": "ङ",
    "c": "च", "ch": "छ", "j": "ज", "jh": "झ", "ñ": "ञ",
    "ṭ": "ट", "ṭh": "ठ", "ḍ": "ड", "ḍh": "ढ", "ṇ": "ण",
    "t": "त", "th": "थ", "d": "द", "dh": "ध", "n": "न",
    "p": "प", "ph": "फ", "b": "ब", "bh": "भ", "m": "म",
    "y": "य", "r": "र", "l": "ल", "ḻ": "ळ", "v": "व",
    "ś": "श", "ṣ": "ष", "s": "स", "h": "ह",
}
_INDEPENDENT_VOWELS = {roman: pair[0] for roman, pair in _VOWELS.items()}
_VOWEL_SIGNS = {roman: pair[1] for roman, pair in _VOWELS.items() if pair[1]}
_DEVANAGARI_VOWELS = {value: key for key, value in _INDEPENDENT_VOWELS.items()}
_DEVANAGARI_SIGNS = {value: key for key, value in _VOWEL_SIGNS.items()}
_DEVANAGARI_CONSONANTS = {value: key for key, value in _CONSONANTS.items()}
_SIGNS = {"ṃ": "ं", "ḥ": "ः", "m̐": "ँ"}
_DEVANAGARI_SIGNS_TO_ROMAN = {value: key for key, value in _SIGNS.items()}
_ALIASES = (
    ("r̥̄", "ṝ"), ("l̥̄", "ḹ"), ("r̥", "ṛ"), ("l̥", "ḷ"),
    ("ṁ", "ṃ"), ("ṁ", "ṃ"),
)
_ROMAN_TOKENS = sorted(
    set(_VOWELS) | set(_CONSONANTS) | set(_SIGNS) | {"'", "’"},
    key=len,
    reverse=True,
)
_MARKS = {"ṃ", "ḥ", "m̐"}
_DANDA = {"।", "॥"}


def _canonical_roman(text: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(text)).casefold()
    for source, target in _ALIASES:
        text = text.replace(source, target)
    text = text.replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def devanagari_to_iast(text: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(text))
    output: list[str] = []
    previous_consonant = False
    for char in text:
        if char in _DEVANAGARI_CONSONANTS:
            output.append(_DEVANAGARI_CONSONANTS[char] + "a")
            previous_consonant = True
        elif char in _DEVANAGARI_SIGNS:
            if previous_consonant:
                output[-1] = output[-1][:-1] + _DEVANAGARI_SIGNS[char]
            else:
                output.append(_DEVANAGARI_SIGNS[char])
            previous_consonant = False
        elif char == "्":
            if previous_consonant:
                output[-1] = output[-1][:-1]
            previous_consonant = False
        elif char in _DEVANAGARI_VOWELS:
            output.append(_DEVANAGARI_VOWELS[char])
            previous_consonant = False
        elif char in _DEVANAGARI_SIGNS_TO_ROMAN:
            output.append(_DEVANAGARI_SIGNS_TO_ROMAN[char])
            previous_consonant = False
        elif char == "ऽ":
            output.append("'")
            previous_consonant = False
        elif char == "ॐ":
            output.append("oṃ")
            previous_consonant = False
        else:
            output.append(char)
            previous_consonant = False
    return _canonical_roman("".join(output))


def iast_to_devanagari(text: str) -> str:
    text = _canonical_roman(text)
    output: list[str] = []
    previous_consonant = False
    index = 0
    while index < len(text):
        if text[index].isspace():
            if previous_consonant:
                output.append("्")
                previous_consonant = False
            output.append(text[index])
            index += 1
            continue
        if text[index] in _DANDA or unicodedata.category(text[index]).startswith("P"):
            if previous_consonant:
                output.append("्")
                previous_consonant = False
            output.append("ऽ" if text[index] in {"'", "’"} else text[index])
            index += 1
            continue

        token = next((candidate for candidate in _ROMAN_TOKENS if text.startswith(candidate, index)), None)
        if token is None:
            if previous_consonant:
                output.append("्")
                previous_consonant = False
            output.append(text[index])
            index += 1
            continue

        if token in _CONSONANTS:
            if previous_consonant:
                output.append("्")
            output.append(_CONSONANTS[token])
            previous_consonant = True
        elif token in _VOWELS:
            if previous_consonant:
                if token != "a":
                    output.append(_VOWEL_SIGNS[token])
            else:
                output.append(_INDEPENDENT_VOWELS[token])
            previous_consonant = False
        elif token in _MARKS:
            if previous_consonant:
                output.append("्")
            output.append(_SIGNS[token])
            previous_consonant = False
        else:
            output.append("ऽ" if token in {"'", "’"} else token)
            previous_consonant = False
        index += len(token)

    if previous_consonant:
        output.append("्")
    return unicodedata.normalize("NFC", "".join(output))


def to_devanagari(text: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(text)).strip()
    if any("\u0900" <= char <= "\u097f" for char in text):
        return text
    return iast_to_devanagari(text)


def normalize_sanskrit(text: str) -> str:
    """Return a script-neutral IAST-like form without dropping phonemic distinctions."""
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = re.sub(r"<[^>]*>", " ", text)
    text = devanagari_to_iast(text) if any("\u0900" <= c <= "\u097f" for c in text) else _canonical_roman(text)
    text = text.replace("।", " ").replace("॥", " ")
    text = re.sub(r"[.,;:!?|–—-]+", " ", text)
    text = re.sub(r"[\t\n\r]+", " ", text)
    text = re.sub(r"[ ]+", " ", text)
    return text.strip()
