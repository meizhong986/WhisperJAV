"""
Faster Whisper XXL — self-contained subprocess runner.

This module calls faster-whisper-xxl as an external executable and reads
the resulting SRT file.  It has NO imports from WhisperJAV pipelines,
config resolution, or ensemble internals.

Integration patterns derived from pyvideotrans and Subtitle Edit research.
See: docs/research/XXL_BYOP_INTEGRATION_PATTERNS.md
"""

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def run_xxl(
    input_file: str,
    exe_path: str,
    model: str = "large-v3",
    language: str = "ja",
    output_dir: Optional[str] = None,
    extra_args: str = "",
    task: str = "transcribe",
) -> Path:
    """
    Call faster-whisper-xxl as a subprocess and return the path to the
    generated SRT file.

    Args:
        input_file: Path to input media file (video or audio).
        exe_path: Absolute path to faster-whisper-xxl executable.
        model: Whisper model name (e.g. "large-v3").
        language: Two-letter language code (e.g. "ja").
        output_dir: Directory where XXL writes the SRT.  If None, uses
                     a temporary directory under the exe's parent.
        extra_args: Whitespace-separated extra arguments passed verbatim
                    to XXL (e.g. "--standard_asia --ff_vocal_extract mdx_kim2").
        task: "transcribe" or "translate".

    Returns:
        Path to the generated SRT file.

    Raises:
        FileNotFoundError: If the XXL executable does not exist.
        RuntimeError: If XXL exits non-zero or no SRT file is found.
    """
    exe = Path(exe_path)
    if not exe.is_file():
        raise FileNotFoundError(f"XXL executable not found: {exe_path}")

    if output_dir is None:
        output_dir = str(exe.parent / "_whisperjav_xxl_output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Minimal integration args only — language, output location, format.
    # Model is NOT hardcoded here — it comes through extra_args so the user
    # can change it in the GUI (e.g. --model large-v3, --model large-v2).
    cmd = [
        str(exe),
        str(input_file),
        "--language", language,
        "--output_dir", str(output_dir),
        "--output_format", "srt",
    ]

    # Add --model only if not already specified in extra_args
    if "--model" not in (extra_args or ""):
        cmd.extend(["--model", model])

    if task == "translate":
        cmd.extend(["--task", "translate"])

    # Passthrough user-supplied extra args (shell-aware splitting)
    if extra_args and extra_args.strip():
        cmd.extend(shlex.split(extra_args.strip()))

    # pyvideotrans-style: load extra args from text file next to exe
    txt_file = exe.parent / "whisperjav_xxl.txt"
    if txt_file.is_file():
        file_args = txt_file.read_text(encoding="utf-8").strip()
        if file_args:
            cmd.extend(shlex.split(file_args))

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    start_time = time.monotonic()

    # Stream stdout to parent (visible in GUI console) while capturing
    # stderr separately for error diagnostics on failure.
    # errors="replace" prevents UnicodeDecodeError when the external process
    # emits non-UTF-8 bytes (e.g., Windows system codepage cp936/GBK for
    # Chinese filenames).  stderr is diagnostic-only, so lossy decoding is fine.
    # See: https://github.com/meizhong986/WhisperJAV/issues/244
    result = subprocess.run(
        cmd,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(exe.parent),
        env=env,
    )

    elapsed = time.monotonic() - start_time

    # Check for SRT output BEFORE checking exit code.
    # XXL uses ctranslate2 which crashes during C++ destructor shutdown
    # (STATUS_STACK_BUFFER_OVERRUN / 0xC0000409 on Windows) — the same
    # problem our own Nuclear Exit pattern solves.  The transcription
    # completes and the SRT is written, then the process crashes on exit.
    srt_path = _find_srt(output_dir, input_file)

    if srt_path is not None:
        if result.returncode != 0:
            print(
                f"[XXL] Warning: process crashed on exit (code {result.returncode}) "
                f"but SRT was produced successfully ({elapsed:.1f}s)",
                file=sys.stderr,
            )
        return srt_path

    # No SRT found — this is a real failure
    if result.returncode != 0:
        stderr_tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"XXL failed (exit {result.returncode}, {elapsed:.1f}s):\n{stderr_tail}"
        )

    raise RuntimeError(
        f"XXL completed ({elapsed:.1f}s) but no SRT found in {output_dir}"
    )


def _find_srt(output_dir: str, input_file: str) -> Optional[Path]:
    """Locate the SRT file produced by XXL in output_dir.

    Returns the Path if found and non-empty, None otherwise.
    """
    # XXL writes {input_stem}.srt in output_dir
    expected_srt = Path(output_dir) / f"{Path(input_file).stem}.srt"
    if expected_srt.is_file() and expected_srt.stat().st_size > 0:
        return expected_srt

    # Fallback: look for any .srt file in output_dir (XXL may use
    # a slightly different naming convention depending on version)
    srts = sorted(Path(output_dir).glob("*.srt"), key=os.path.getmtime, reverse=True)
    for srt in srts:
        if srt.stat().st_size > 0:
            return srt

    return None
