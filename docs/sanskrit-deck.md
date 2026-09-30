# Sanskrit deck preset and note fields

In **Tools → Hands-Free Anki → Settings → Per-Deck Settings**, add the deck and
select **Apply Sanskrit deck preset**. It sets the deck language to `sa-IN`,
uses strict matching by default, and selects `Prompt` plus `Answer-Devanagari`
as the fields read aloud. IAST and ISO 15919 remain visible on the card but are
not redundantly spoken. Local TTS converts romanized Sanskrit to Devanagari.

For new notes, use these fields (spelling is significant):

| Field | Use |
|---|---|
| `Prompt` | Question or cue; prefer Sanskrit text when the answer is to be spoken in Sanskrit. |
| `Answer-Devanagari` | Canonical Sanskrit answer and preferred TTS source. |
| `Answer-IAST` | Sanskrit in International Alphabet of Sanskrit Transliteration. |
| `Answer-ISO15919` | Sanskrit in ISO 15919, useful for consistency with other Indic scripts. |
| `Gloss` | Concise meaning; display it on the answer side, not as a Sanskrit pronunciation target. |
| `Grammar` | Optional grammatical analysis or form labels. |
| `Accepted answers` | Optional alternatives separated by newlines, `<br>`, `|`, or `;`. Use for explicitly approved sandhi/orthographic variants. |
| `Grading Mode` | Optional `strict`, `accepted`, or `manual` per-card override. |
| `Exercise Type` | Optional `translation`, `recitation`, `open-ended`, or another short kind label. Translation, recitation, and open-ended values are always manually rated. |

Example note type templates:

**Front template**

```html
{{Prompt}}
```

**Back template**

```html
{{FrontSide}}
<hr id="answer">
<div class="devanagari">{{Answer-Devanagari}}</div>
<div>{{Answer-IAST}}</div>
<div>{{Answer-ISO15919}}</div>
<div>{{Gloss}}</div>
<div>{{Grammar}}</div>
```

The reviewer accepts exact normalized answers across Devanagari, IAST, and
ISO 15919. It preserves short/long vowels, retroflex consonants, aspirates,
anusvara, and visarga as distinctions. `strict` ignores `Accepted answers`;
`accepted` also compares only the alternatives written there. If answer fields
are missing, or the note is open-ended/translation/recitation, the add-on asks
you to rate it manually. It does not attempt offline semantic translation
grading.

Keep content in the note as entered: normalization is comparison-only and does
not rewrite fields. Add only alternatives you have checked as correct for that
particular prompt.
