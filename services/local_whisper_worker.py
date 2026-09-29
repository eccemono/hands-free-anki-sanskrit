from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path


_model = None
_model_key = None


def transcribe(request: dict):
    global _model, _model_key

    model_path = Path(str(request["model_path"])).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Local Whisper model is missing at {model_path}. Run tools/setup_local_whisper.py first."
        )
    device = str(request.get("device", "cuda"))
    compute_type = str(request.get("compute_type", "int8_float16"))
    key = (str(model_path.resolve()), device, compute_type)
    if _model is None or key != _model_key:
        from faster_whisper import WhisperModel

        with contextlib.redirect_stdout(sys.stderr):
            _model = WhisperModel(
                str(model_path),
                device=device,
                compute_type=compute_type,
                local_files_only=True,
            )
        _model_key = key

    audio = base64.b64decode(request["audio_wav"], validate=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        handle.write(audio)
        audio_path = Path(handle.name)
    try:
        segments, _info = _model.transcribe(
            str(audio_path),
            language=str(request.get("language", "sa")),
            beam_size=int(request.get("beam_size", 5)),
            vad_filter=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return {"ok": True, "text": text}
    finally:
        audio_path.unlink(missing_ok=True)


def main() -> None:
    for line in sys.stdin:
        try:
            response = transcribe(json.loads(line))
        except Exception as err:
            response = {"ok": False, "error": str(err)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
