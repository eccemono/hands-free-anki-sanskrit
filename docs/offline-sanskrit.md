# Offline Sanskrit speech on Linux Mint

## What stays local

The Sanskrit deck preset routes card speech to the installed Sanskrit model and
routes recognition to the local Faster-Whisper model with language `sa`. For
Sanskrit, failure is explicit: there is no Hindi gTTS, Google, cloud Whisper,
English, or other voice fallback. Review audio is not uploaded. The one-time
runtime setup below downloads packages and model assets; it is not run while
reviewing. API providers remain in the add-on for other languages, but are not
selected by the Sanskrit route.

Local transcription is set to Faster-Whisper `medium` with `int8_float16` on a
CUDA GPU. The installed runtime uses NVIDIA CUDA 12 cuBLAS/cuDNN wheels and a
local model directory (about 1.5 GB for the model). The RTX 4070 Ti is supported;
if GPU memory is occupied, free memory before starting a review or expect a
visible inference error rather than a cloud fallback.

## Install / set up

1. Install the add-on from the [public fork](https://github.com/eccemono/hands-free-anki-sanskrit). For a source checkout, the Anki
   module folder must be `hands_free_anki_sanskrit` (the GitHub repo is
   hyphenated). Restart Anki after copying the add-on.
2. Keep the separate local Sanskrit/Hindi TTS add-on and its model runtime at
   the configured defaults:
   `~/.local/share/Anki2/addons21/sanskrit_hindi_tts/worker.py` and
   `~/.local/share/Anki2/sanskrit-hindi-tts-runtime/`. This fork invokes only
   its Sanskrit model API and bypasses that add-on's Hindi online fallback.
3. From the fork checkout or installed add-on folder, run the setup command
   once in a terminal:

   ```bash
   python3.12 tools/setup_local_whisper.py
   ```

   It creates an isolated environment at
   `~/.local/share/Anki2/hands-free-anki-runtime/stt-venv`, installs
   Faster-Whisper, PyAV (pinned below 17 for its audio decoder API), CUDA
   libraries, and downloads the `Systran/faster-whisper-medium` model to
   `~/.local/share/Anki2/hands-free-anki-runtime/models/whisper-medium`.
   The script validates that CUDA can load the model. Internet access is needed
   only for this explicit setup operation.
4. In **Tools → Hands-Free Anki → Settings → Per-Deck Settings**, add your deck,
   select **Apply Sanskrit deck preset**, and save. Confirm the microphone in
   **Settings → Speech Recognition**. The preset sets `sa-IN`; transcription
   passes `sa` to Whisper.
5. Run a small due-card review first. The initial model load may take time.

The current Linux Mint profile already contains the separate Sanskrit TTS
runtime and local TTS model. Faster-Whisper medium is installed in the separate
runtime above. Whisper interpreter/model paths and its timeout are available in
Speech Recognition settings. The Sanskrit TTS runtime paths can be changed in
`user_config.json` if your local TTS installation uses another location; the
Sanskrit synthesis timeout is in TTS settings.

## Card study patterns

Use the field definitions and templates in [sanskrit-deck.md](sanskrit-deck.md).

- **Vocabulary**: Prompt on the front; canonical Devanagari answer on the back.
  Use `strict` for one expected form or `accepted` for variants you have checked.
- **Grammar production**: cue the lemma, case/number/person, or required form;
  put the exact target in the three answer script fields. Keep short/long
  vowels, aspirates, and retroflexes significant unless the card explicitly
  permits another form.
- **Script practice**: show one script in the prompt and store the canonical
  response in Devanagari, IAST, and ISO 15919. The grader compares across
  scripts without rewriting the note.
- **Translation**: label the note `Exercise Type=translation`; it will always
  ask for a manual Anki rating. No offline semantic-translation score is
  claimed.
- **Recitation/open response**: label `Exercise Type=recitation` or `open-ended`
  and rate manually. Use pre-recorded audio or your own listening judgment for
  meter, accent, and pronunciation nuance.

Use `Accepted answers` only for verified alternatives, such as a specific
sandhi form. Separate alternatives by line, `<br>`, `|`, or `;`. Manual rating
can also be selected for a whole deck or an individual note. If answer fields
are missing, the grader safely falls back to manual rating.

## Privacy and recovery

- Model files and the venv live outside the add-on source folder and are
  ignored by Git. Do not copy your `User 1` profile or collection into a test
  checkout.
- Review workers set Hugging Face/Transformers offline flags and require a
  local model path. Missing assets and timeouts are surfaced; they do not
  trigger model downloads during review.
- If TTS fails, verify the local Sanskrit runtime path and cached Sanskrit
  model. If STT fails, verify the runtime Python, model directory, available
  GPU memory, and selected microphone. Re-run the one-time setup script only
  when deliberately installing/updating local assets.
- You can use keyboard ratings if the microphone or local speech model is not
  ready. Restarting Anki after changing add-on files/configuration is advised.

## Validation record

Environment: Linux Mint, Anki 26.09.3, NVIDIA GeForce RTX 4070 Ti. The local
Whisper `medium` CUDA model loads and runs with language `sa` and network access
disabled. A synthetic silence test correctly produced an empty transcript. A
local Sanskrit TTS generation returned a 122,958-byte WAV without network
fallback. Whisper returned an empty transcript for that generated sample, so
recognition quality on spoken Sanskrit audio remains unvalidated.

The disposable Anki GUI/profile smoke test could not be run while the existing
Anki GUI was open: a launch with a separate base/profile was intercepted with
“Already running; reusing existing instance.” The live `User 1` session was
left untouched. Therefore add-on GUI startup, in-Anki audio playback, microphone
capture, and actual manual/strict review flow remain pending that smoke test.
Do not treat them as passed based on the unit tests.
