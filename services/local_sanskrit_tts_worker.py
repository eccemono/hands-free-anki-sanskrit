from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

_runtime_module = None
_sanskrit_engine = None
_runtime_worker_path = None


def _load_engine(runtime_worker: Path, runtime_root: Path):
    global _runtime_module, _sanskrit_engine, _runtime_worker_path
    if _sanskrit_engine is not None and _runtime_worker_path == runtime_worker:
        return _runtime_module, _sanskrit_engine

    if not runtime_worker.is_file():
        raise FileNotFoundError(f"Sanskrit TTS worker not found: {runtime_worker}")
    os.environ["HF_HOME"] = str(runtime_root / "models")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    spec = importlib.util.spec_from_file_location("sanskrit_tts_runtime_worker", runtime_worker)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Sanskrit TTS runtime: {runtime_worker}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
        engine = getattr(module, "SanskritEngine")()
    _runtime_module = module
    _sanskrit_engine = engine
    _runtime_worker_path = runtime_worker
    return module, engine


def synthesize(request: dict, cache_dir: Path, runtime_worker: Path, runtime_root: Path) -> dict:
    text = str(request.get("text", "")).strip()
    if not text:
        raise ValueError("Cannot synthesize blank Sanskrit text")
    module, engine = _load_engine(runtime_worker, runtime_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"sanskrit-{uuid.uuid4().hex}.wav"
    speed = float(request.get("speed", 1.0))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        source = Path(handle.name)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            engine.synthesize(text, source)
            convert = getattr(module, "_convert_to_wav")
            convert(source, output, speed)
    finally:
        source.unlink(missing_ok=True)
    return {"ok": True, "path": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--runtime-worker", required=True)
    args = parser.parse_args()
    runtime_root = Path(args.runtime_root).expanduser()
    runtime_worker = Path(args.runtime_worker).expanduser()
    cache_dir = runtime_root / "hands-free-anki-cache"

    for line in sys.stdin:
        try:
            response = synthesize(json.loads(line), cache_dir, runtime_worker, runtime_root)
        except Exception as err:
            response = {"ok": False, "error": str(err)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
