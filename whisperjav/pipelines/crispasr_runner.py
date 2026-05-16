"""
CrispASR — self-contained subprocess runner.

This module calls the ``crispasr`` external binary and reads the resulting
SRT file.  It has **NO** imports from WhisperJAV pipelines, config resolution,
ensemble, or BYOP internals — a standalone provider call, the way SubtitleEdit
and Susurrus cleanly shell out to crispasr.  This is deliberately *not* a
mirror of ``whisperjav/byop/xxl_runner.py``; it only shares the same
defensive subprocess *pattern* (build argv → run → tolerate the C++-shutdown
crash by checking the SRT before the exit code).

Argv form verified vs CrispASR ``docs/cli.md`` (Phase-1 R-2 resolution,
``docs/plans/crispasr_v190/08`` §1/§3a)::

    crispasr -m <auto|auto:large-v2> --backend <whisper|parakeet|cohere>
             -f <input.wav> -l <lang> -osrt -of <outbase>
             [extra args]                              -> writes <outbase>.srt

    (v1.9.0 curated backends need no forced aligner; canary deferred.)

Design:  docs/plans/crispasr_v190/08_crispasr_simple_pipeline_design.md
Durable record:  memory/feedback_crispasr_simple_standalone_pipeline.md
"""

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# CL1 — backend curation (user decision 2026-05-15, revised 2026-05-16)
# ---------------------------------------------------------------------------
# v1.9.0 first release keeps the curated set minimal — all native
# word-timestamp, none needing a forced aligner:
#   * ``whispercpp`` — WhisperJAV-facing alias of crispasr's real ``whisper``
#                      backend (whisper.cpp). Defaults to large-v2 (the
#                      engine's bare ``whisper`` default is ggml-base — too
#                      weak), see BACKEND_MODEL_SPEC.
#   * ``parakeet``   — default backend.
#   * ``cohere``     — uses the OPEN ``cstr/cohere-transcribe-03-2026-GGUF``
#                      ggml asset (Apache-2.0); NOT the gated upstream
#                      CohereLabs repo, so no HF token is required
#                      (user clarification CL1(b), 2026-05-16).
# ``canary`` and ``kyutai-stt`` are intentionally REMOVED for the first
# release (CL1(d)/(e)): canary deferred to keep all forced-aligner /
# timestamp complexity out of v1.9.0; kyutai-stt deferred. Re-add later.
CURATED_BACKENDS = ("whispercpp", "parakeet", "cohere")
DEFAULT_BACKEND = "parakeet"

# WhisperJAV-facing name -> the actual crispasr ``--backend`` value.
# Only ``whispercpp`` is aliased (to crispasr's real backend id ``whisper``);
# the others pass through unchanged.
BACKEND_ALIAS = {"whispercpp": "whisper"}

# Per-backend ``-m`` model spec. ``DEFAULT_MODEL_SPEC`` ("auto") lets the
# engine self-download its registry-default GGUF (D5a — WhisperJAV writes no
# download code). ``whispercpp`` overrides to large-v2 because the engine's
# bare ``whisper`` default is ggml-base (CL1(c): "large-v2 default and auto").
# The ``auto[:variant]`` form follows docs/cli.md ``-m PATH|auto[:variant]``;
# the exact accepted variant token (``auto:large-v2``) is NOT runtime-verified
# in the dev env (manual-verify before release — doc 08 §5; ``--crispasr-args
# -m <...>`` is the override escape valve).
DEFAULT_MODEL_SPEC = "auto"
BACKEND_MODEL_SPEC = {"whispercpp": "auto:large-v2"}

# No curated backend needs a forced aligner in v1.9.0 (canary deferred).
# The aligner machinery is intentionally absent to keep the first release
# simple; re-introduce a per-backend aligner map when canary returns.


def run_crispasr(
    input_file: str,
    exe_path: str,
    backend: str = DEFAULT_BACKEND,
    language: str = "ja",
    output_dir: Optional[str] = None,
    extra_args: str = "",
) -> Path:
    """
    Call ``crispasr`` as a subprocess and return the path to the generated SRT.

    Args:
        input_file: Path to the (already-extracted) input WAV.
        exe_path: Absolute path to the ``crispasr`` executable.
        backend: One of :data:`CURATED_BACKENDS`.
        language: Source language code (e.g. ``"ja"``).
        output_dir: Directory where crispasr writes the SRT.  If ``None``,
            uses ``_whisperjav_crispasr_output`` next to the exe.
        extra_args: Whitespace-separated extra arguments passed verbatim to
            crispasr (shlex-split).  The single escape valve for all advanced
            CrispASR-internal control (doc 08 §3a Probe4).

    Returns:
        Path to the generated SRT file.

    Raises:
        FileNotFoundError: If the crispasr executable does not exist.
        ValueError: If ``backend`` is not curated for this release.
        RuntimeError: If crispasr produced no SRT.
    """
    exe = Path(exe_path)
    if not exe.is_file():
        raise FileNotFoundError(f"CrispASR executable not found: {exe_path}")

    if backend not in CURATED_BACKENDS:
        raise ValueError(
            f"CrispASR backend '{backend}' is not curated for this release. "
            f"Available: {', '.join(CURATED_BACKENDS)}"
        )

    if output_dir is None:
        output_dir = str(exe.parent / "_whisperjav_crispasr_output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # crispasr writes ``<outbase>.srt`` for the -osrt/-of output pair.
    outbase = str(Path(output_dir) / Path(input_file).stem)

    # Resolve the WhisperJAV-facing name to the real crispasr backend id and
    # pick the per-backend model spec (whispercpp -> large-v2; else auto).
    crispasr_backend = BACKEND_ALIAS.get(backend, backend)
    model_spec = BACKEND_MODEL_SPEC.get(backend, DEFAULT_MODEL_SPEC)

    cmd = [
        str(exe),
        "-m", model_spec,        # engine self-manages its model cache (D5a)
        "--backend", crispasr_backend,
        "-f", str(input_file),   # documented canonical input flag
        "-l", language,
        "-osrt",
        "-of", outbase,
    ]

    # No curated v1.9.0 backend needs a forced aligner (canary deferred).

    # Passthrough user-supplied extra args (shell-aware splitting).
    if extra_args and extra_args.strip():
        cmd.extend(shlex.split(extra_args.strip()))

    # WhisperJAV writes NO download/verify code (D5a).  The engine
    # self-manages models via ``-m auto``; we only redirect its cache into a
    # dedicated subdir under WhisperJAV's cache namespace without colliding
    # with the HF-hub blob layout, honoring I1's intent.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    env.setdefault(
        "CRISPASR_CACHE_DIR",
        str(
            Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
            / "whisperjav"
            / "crispasr_models"
        ),
    )

    start_time = time.monotonic()

    # Stream stdout to the parent (visible in the GUI console) while
    # capturing stderr separately for diagnostics on failure.
    # errors="replace" prevents UnicodeDecodeError when the external process
    # emits non-UTF-8 bytes (e.g. Windows system codepage for CJK filenames).
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

    # Check for the SRT BEFORE checking the exit code.  CrispASR is built on
    # the ggml / whisper.cpp stack which can crash in its C++ destructor on
    # process shutdown (the same class of issue WhisperJAV's own Nuclear Exit
    # solves) — the transcription completes and the SRT is written first.
    srt_path = _find_srt(output_dir, input_file, outbase)
    if srt_path is not None:
        if result.returncode != 0:
            print(
                f"[CrispASR] Warning: process exited non-zero "
                f"(code {result.returncode}) but SRT was produced "
                f"successfully ({elapsed:.1f}s)",
                file=sys.stderr,
            )
        return srt_path

    stderr_tail = (result.stderr or "(no stderr)")[-2000:]
    if result.returncode != 0:
        raise RuntimeError(
            f"CrispASR failed (exit {result.returncode}, {elapsed:.1f}s):\n{stderr_tail}"
        )
    raise RuntimeError(
        f"CrispASR completed ({elapsed:.1f}s) but no SRT found in "
        f"{output_dir}\n{stderr_tail}"
    )


def _find_srt(output_dir: str, input_file: str, outbase: str) -> Optional[Path]:
    """Locate the SRT produced by crispasr.  Returns Path if non-empty, else None."""
    # 1. The exact -of target.
    expected = Path(outbase + ".srt")
    if expected.is_file() and expected.stat().st_size > 0:
        return expected

    # 2. <input_stem>.srt in output_dir (defensive — naming may vary by version).
    expected2 = Path(output_dir) / f"{Path(input_file).stem}.srt"
    if expected2.is_file() and expected2.stat().st_size > 0:
        return expected2

    # 3. Newest non-empty .srt anywhere in output_dir.
    srts = sorted(
        Path(output_dir).glob("*.srt"), key=os.path.getmtime, reverse=True
    )
    for s in srts:
        if s.stat().st_size > 0:
            return s

    return None
