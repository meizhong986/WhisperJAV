"""
Faster Whisper XXL — self-contained subprocess runner.

This module calls faster-whisper-xxl as an external executable and reads
the resulting SRT file.  It has NO imports from WhisperJAV pipelines,
config resolution, or ensemble internals.

Integration patterns derived from pyvideotrans and Subtitle Edit research.
See: docs/research/XXL_BYOP_INTEGRATION_PATTERNS.md
"""

import os
import subprocess
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

    cmd = [
        str(exe),
        str(input_file),
        "--model", model,
        "--language", language,
        "--output_dir", str(output_dir),
        "--output_format", "srt",
        "--beep_off",
    ]

    if task == "translate":
        cmd.extend(["--task", "translate"])

    # Passthrough user-supplied extra args (split on whitespace)
    if extra_args and extra_args.strip():
        cmd.extend(extra_args.strip().split())

    # pyvideotrans-style: load extra args from text file next to exe
    txt_file = exe.parent / "whisperjav_xxl.txt"
    if txt_file.is_file():
        file_args = txt_file.read_text(encoding="utf-8").strip()
        if file_args:
            cmd.extend(file_args.split())

    import sys

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    start_time = time.monotonic()

    # Stream stdout to parent (visible in GUI console) while capturing
    # stderr separately for error diagnostics on failure.
    result = subprocess.run(
        cmd,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=str(exe.parent),
        env=env,
    )

    elapsed = time.monotonic() - start_time

    if result.returncode != 0:
        stderr_tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"XXL failed (exit {result.returncode}, {elapsed:.1f}s):\n{stderr_tail}"
        )

    # XXL writes {input_stem}.srt in output_dir
    expected_srt = Path(output_dir) / f"{Path(input_file).stem}.srt"
    if expected_srt.is_file():
        return expected_srt

    # Fallback: look for any .srt file in output_dir (XXL may use
    # a slightly different naming convention depending on version)
    srts = sorted(Path(output_dir).glob("*.srt"), key=os.path.getmtime, reverse=True)
    if srts:
        return srts[0]

    raise RuntimeError(
        f"XXL completed ({elapsed:.1f}s) but no SRT found in {output_dir}"
    )
