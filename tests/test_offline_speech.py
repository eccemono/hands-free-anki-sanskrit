import base64
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services"))
from language_policy import is_sanskrit_language, map_language_code, stt_provider_chain, tts_provider_chain


def load_worker(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfflineSpeechTests(unittest.TestCase):
    def test_sanskrit_locale_routes_only_to_local_recognition(self):
        self.assertTrue(is_sanskrit_language("sa-IN"))
        self.assertTrue(is_sanskrit_language("sa_IN"))
        self.assertEqual(map_language_code("sa"), "sa-IN")
        self.assertEqual(map_language_code("sa-IN"), "sa-IN")
        self.assertEqual(
            stt_provider_chain("whisper", "sa-IN", True, True),
            ["offline_whisper"],
        )

    def test_english_keeps_configured_provider_fallbacks(self):
        self.assertEqual(
            stt_provider_chain("whisper", "en-US", True, True),
            ["whisper", "google", "sphinx"],
        )

    def test_sanskrit_tts_has_no_non_sanskrit_fallback(self):
        self.assertEqual(
            tts_provider_chain(["elevenlabs", "gtts", "offline"], "sa-IN"),
            ["sanskrit_local"],
        )
        self.assertEqual(
            tts_provider_chain(["offline"], "en-US"),
            ["offline"],
        )
        self.assertEqual(
            stt_provider_chain("offline_whisper", "en-US", True, True),
            ["offline_whisper"],
        )

    def test_local_whisper_loads_local_model_with_language_hint(self):
        worker = load_worker("local_whisper_test_worker", ROOT / "services/local_whisper_worker.py")
        calls = {}

        class FakeModel:
            def transcribe(self, path, **kwargs):
                calls["transcribe"] = kwargs
                return iter([types.SimpleNamespace(text=" नमस्ते ")]), object()

        def fake_model(model_path, **kwargs):
            calls["load"] = (model_path, kwargs)
            return FakeModel()

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=fake_model)}
        ):
            model_path = Path(tmp) / "model"
            model_path.mkdir()
            worker._model = None
            result = worker.transcribe({
                "model_path": str(model_path),
                "audio_wav": base64.b64encode(b"local test audio").decode("ascii"),
                "language": "sa",
                "device": "cuda",
                "compute_type": "float16",
            })

        self.assertEqual(result["text"], "नमस्ते")
        self.assertEqual(calls["transcribe"]["language"], "sa")
        self.assertTrue(calls["load"][1]["local_files_only"])
        worker._model = None
        worker._model_key = None

    def test_sanskrit_tts_worker_uses_local_model_and_offline_mode(self):
        worker = load_worker("local_sanskrit_test_worker", ROOT / "services/local_sanskrit_tts_worker.py")
        worker._runtime_module = None
        worker._sanskrit_engine = None
        worker._runtime_worker_path = None
        fake_runtime = (
            "from pathlib import Path\n"
            "import shutil\n"
            "class SanskritEngine:\n"
            "    def synthesize(self, text, path):\n"
            "        Path(path).write_text(text, encoding='utf-8')\n"
            "def _convert_to_wav(source, output, speed):\n"
            "    shutil.copyfile(source, output)\n"
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            root = Path(tmp)
            runtime_worker = root / "runtime_worker.py"
            runtime_worker.write_text(fake_runtime, encoding="utf-8")
            response = worker.synthesize(
                {"text": "नमस्ते", "speed": 1.0},
                root / "cache",
                runtime_worker,
                root,
            )
            self.assertEqual(Path(response["path"]).read_text(encoding="utf-8"), "नमस्ते")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        worker._runtime_module = None
        worker._sanskrit_engine = None
        worker._runtime_worker_path = None


if __name__ == "__main__":
    unittest.main()
