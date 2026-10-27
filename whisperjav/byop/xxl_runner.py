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
from typing import List, Optional, Tuple

# The model XXL runs when a pass names none (owner, W24 2: large-v2; was large-v3).
DEFAULT_XXL_MODEL = "large-v2"

# Extra arguments read from a text file next to the XXL program (pyvideotrans-style).
SIDE_FILE_NAME = "whisperjav_xxl.txt"


def strip_model_args(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """Remove every model named in XXL's extra arguments; return (kept, removed).

    Works on the arguments as XXL receives them (after shlex.split), so a quoted value is
    never touched. Removed: "--model X", "--model=X", "-m X", "-mX", and a bare "--model"
    or "-m" at the end. Kept: "--model_dir" and every other argument. (XXL has no other
    short option starting with -m; checked against r245.2's --help.)

    The pass's own model (--passN-model, the GUI's Model column) is the one XXL runs (#272);
    a model in the extra arguments would otherwise override it, because XXL takes the last.
    """
    kept: List[str] = []
    removed: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("--model", "-m"):
            if i + 1 < len(tokens):
                removed.append(f"{token} {tokens[i + 1]}")
                i += 2
            else:
                removed.append(token)
                i += 1
            continue
        if token.startswith("--model=") or (token.startswith("-m") and not token.startswith("--")):
            removed.append(token)
        else:
            kept.append(token)
        i += 1
    return kept, removed


def _split(args: str) -> List[str]:
    return shlex.split(args.strip()) if args and args.strip() else []


def _side_file_tokens(exe: Path) -> List[str]:
    side_file = exe.parent / SIDE_FILE_NAME
    if side_file.is_file():
        return _split(side_file.read_text(encoding="utf-8"))
    return []


def model_override_note(pass_number: int, model: str, extra_args: str, exe_path: str) -> Optional[str]:
    """The console warning for a model named in the extra arguments, or None (owner W271 4).

    Covers the Extra Args setting and the side file next to the program. Arguments that cannot
    be split (an unclosed quote) give None here; run_xxl then fails for each file with the reason.
    Text X1, approved by the owner as drafted (W275).
    """
    try:
        removed = strip_model_args(_split(extra_args))[1]
        removed += [f"{r} ({SIDE_FILE_NAME})" for r in strip_model_args(_side_file_tokens(Path(exe_path)))[1]]
    except (ValueError, OSError):
        return None
    if not removed:
        return None
    return (f'Pass {pass_number}: XXL extra arguments contain "{", ".join(removed)}", which is not used. '
            f"XXL runs {model}, the model chosen for pass {pass_number} (the Model column in the GUI, "
            f"--pass{pass_number}-model on the command line).")


def failure_reason(error: BaseException) -> str:
    """Why XXL failed, in one line: the error's first line (XXL's exit code) and, when XXL
    printed one, its last error line -- the part that says what went wrong."""
    lines = [line.strip() for line in str(error).strip().splitlines() if line.strip()]
    reason = (lines or [type(error).__name__])[0].rstrip(":. ")
    if len(lines) > 1:
        reason = f"{reason} — {lines[-1].rstrip('.')}"
    return reason


def failure_note(pass_number: int, basename: str, error: BaseException) -> str:
    """The plain console line for an XXL failure on one file (owner W92: never silent).

    A failed pass 1 fails the file (owner W271 1). Pass 2 runs even for a file whose pass 1
    failed, so its line does not promise pass 1's subtitles.
    Texts X2 (pass 1) and X3 (pass 2), approved by the owner as drafted (W275).
    """
    outcome = ("This file is marked failed." if pass_number == 1
               else "The file keeps pass 1's subtitles, if pass 1 made them.")
    return f"Pass {pass_number} (XXL Faster Whisper) failed for {basename}: {failure_reason(error)}. {outcome}"


def run_xxl(
    input_file: str,
    exe_path: str,
    model: str = DEFAULT_XXL_MODEL,
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
        model: Whisper model name (e.g. "large-v2").
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
    # Model comes from the dedicated model parameter (--pass1-model or
    # --pass2-model CLI flag, or the GUI's Model column).  Any --model in extra_args is stripped to
    # prevent conflict.  (#272)
    cmd = [
        str(exe),
        str(input_file),
        "--language", language,
        "--output_dir", str(output_dir),
        "--output_format", "srt",
        "--model", model,
    ]

    if task == "translate":
        cmd.extend(["--task", "translate"])

    # Passthrough user-supplied extra args (shell-aware splitting), then the
    # pyvideotrans-style side file next to the exe. A model named in either is
    # removed -- the explicit model parameter above is authoritative. Legacy GUI
    # defaults included --model in extra_args which silently overrode the
    # dedicated model field (#272). The pass worker says so in the console
    # (model_override_note, owner W271 4).
    cmd.extend(strip_model_args(_split(extra_args))[0])
    cmd.extend(strip_model_args(_side_file_tokens(exe))[0])

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
