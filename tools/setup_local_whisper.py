from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def setup(
    runtime_root: Path,
    python_executable: str | None = None,
    model_name: str = "medium",
    compute_type: str = "int8_float16",
) -> Path:
    runtime_root = runtime_root.expanduser().resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    venv_path = runtime_root / "stt-venv"
    if not venv_path.exists():
        interpreter = python_executable or shutil.which("python3.12") or sys.executable
        _run([interpreter, "-m", "venv", str(venv_path)])

    python = venv_path / "bin" / "python"
    if not python.is_file():
        python = venv_path / "Scripts" / "python.exe"
    if not python.is_file():
        raise FileNotFoundError(f"Could not find Python executable in {venv_path}")

    _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    _run([
        str(python), "-m", "pip", "install",
        "faster-whisper>=1.1,<2",
        "av>=11,<17",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12>=9,<10",
    ])
    model_path = runtime_root / "models" / f"whisper-{model_name}"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='Systran/faster-whisper-{model_name}', local_dir=r'{model_path}')"
    )
    _run([str(python), "-c", code])
    check = (
        "from faster_whisper import WhisperModel; "
        f"WhisperModel(r'{model_path}', device='cuda', compute_type='{compute_type}', local_files_only=True)"
    )
    env = os.environ.copy()
    cuda_library_dirs = []
    for site_packages in (venv_path / "lib").glob("python*/site-packages"):
        for library in ("cublas/lib", "cudnn/lib"):
            path = site_packages / "nvidia" / library
            if path.is_dir():
                cuda_library_dirs.append(str(path))
    if cuda_library_dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            cuda_library_dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
        )
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    _run([str(python), "-c", check], env=env)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Install local GPU Whisper for offline Anki reviews")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path.home() / ".local/share/Anki2/hands-free-anki-runtime",
    )
    parser.add_argument("--python", help="Python used to create the isolated runtime (prefer Python 3.12)")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--compute-type", default="int8_float16")
    args = parser.parse_args()
    model_path = setup(args.runtime_root, args.python, args.model, args.compute_type)
    print(f"Offline Whisper model installed at {model_path}")


if __name__ == "__main__":
    main()
