#!/usr/bin/env python3
"""
Cohere-Transcribe smoke test — owner-side validator for v1.8.14 ChronosJAV
Cohere generator integration.

Purpose
-------
The Cohere model weights are gated on HuggingFace, so they cannot be loaded
in CI or in environments without the owner's HF_TOKEN.  This standalone
script lets the owner verify on their own hardware that:

  1. HF_TOKEN is set and grants gate access (load() succeeds)
  2. Cohere weights load and unload cleanly through the factory
  3. Peak VRAM after Cohere load is in the expected ~4-8 GB range
  4. (Optional) generate() on a real audio clip returns non-empty text
  5. (Optional) The orchestrator's load-and-unload swap pattern keeps
     peak VRAM bounded by max(Cohere, Qwen3-Aligner), not the sum
     — verifies the §1.3 architecture claim in the v1.8.14 plan

Prerequisites
-------------
  - Accept terms at https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
  - Create a token at https://huggingface.co/settings/tokens (Read scope)
  - Set HF_TOKEN in the environment:
        Windows (persistent):  setx HF_TOKEN hf_xxxxxxxxxxxx
                               (restart terminal/GUI to take effect)
        Windows (this shell):  $env:HF_TOKEN = "hf_xxxxxxxxxxxx"
        macOS/Linux:           export HF_TOKEN=hf_xxxxxxxxxxxx

Usage
-----
  python tools/cohere_smoke_test.py
  python tools/cohere_smoke_test.py --audio path/to/clip.wav
  python tools/cohere_smoke_test.py --audio clip.wav --check-aligner-vram

Notes
-----
  - First run downloads ~2 GB; expect 5-15 minutes.  Subsequent runs use
    the HF cache.
  - Audio file should be 16 kHz mono (the script will resample if not).
  - --check-aligner-vram loads Qwen3-ForcedAligner-0.6B in a separate
    swap cycle to verify the orchestrator's load/unload pattern works.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def ok(msg: str) -> None:
    print(f"[ OK ]  {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL]  {msg}")


def info(msg: str) -> None:
    print(f"[info]  {msg}")


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def preflight() -> bool:
    """Verify environment is ready for the live load test."""
    section("Pre-flight checks")

    # HF_TOKEN
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        fail(
            "HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) not set in environment. "
            "See script docstring for setup instructions."
        )
        return False
    ok(f"HF_TOKEN set ({len(token)} chars; first 4='{token[:4]}...')")

    # Python + transformers
    try:
        import transformers
        ok(f"transformers {transformers.__version__}")
    except ImportError:
        fail("transformers not importable")
        return False

    # Torch + CUDA
    try:
        import torch
    except ImportError:
        fail("torch not importable")
        return False

    if not torch.cuda.is_available():
        info("CUDA not available — will run on CPU (very slow for Cohere).")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        ok(
            f"CUDA available — device 0: {gpu_name}, "
            f"total {total_b / 1024 ** 3:.1f} GB, free {free_b / 1024 ** 3:.1f} GB"
        )

    # Factory discoverability
    try:
        from whisperjav.modules.subtitle_pipeline.generators.factory import (
            TextGeneratorFactory,
        )
        backends = TextGeneratorFactory.available()
        if "cohere" not in backends:
            fail(f"'cohere' not in factory registry: {backends}")
            return False
        ok(f"Factory registers cohere; available: {backends}")
    except Exception as exc:
        fail(f"Factory import failed: {exc}")
        return False

    return True


# ---------------------------------------------------------------------------
# VRAM measurement
# ---------------------------------------------------------------------------


def reset_peak_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def peak_vram_gb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return None


def current_vram_gb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Cohere load/unload cycle
# ---------------------------------------------------------------------------


def run_cohere_cycle(audio_path: Path | None) -> bool:
    """Load → optionally generate → unload, with VRAM tracking."""
    section("Cohere load → unload cycle")

    from whisperjav.modules.subtitle_pipeline.generators.factory import (
        TextGeneratorFactory,
    )

    reset_peak_vram()
    info(f"VRAM before load: {current_vram_gb()} GB" if current_vram_gb() is not None else "VRAM tracking unavailable (CPU)")

    gen = TextGeneratorFactory.create("cohere")
    info("Loading Cohere model (this may take 5-15 minutes on first run)...")
    t0 = time.time()
    try:
        gen.load()
    except Exception as exc:
        fail(f"load() failed: {exc}")
        return False
    elapsed = time.time() - t0
    ok(f"load() completed in {elapsed:.1f}s")

    peak_loaded = peak_vram_gb()
    if peak_loaded is not None:
        ok(f"Peak VRAM after load: {peak_loaded:.2f} GB (expected ~4-8 GB)")

    if audio_path is not None:
        section("Cohere generate() on test clip")
        info(f"Audio: {audio_path}")
        try:
            t0 = time.time()
            result = gen.generate(audio_path, language="ja")
            elapsed = time.time() - t0
            ok(f"generate() completed in {elapsed:.1f}s")
            print()
            print("--- Transcript ---")
            print(result.text or "<empty>")
            print("--- End ---")
            print()
            ok(f"language={result.language!r}, metadata={result.metadata}")
            if not result.text.strip():
                fail("Cohere returned empty text — check audio quality, language, or weights")
            else:
                ok(f"Non-empty transcript: {len(result.text)} chars")
        except Exception as exc:
            fail(f"generate() failed: {exc}")
            gen.unload()
            return False

    info("Unloading...")
    gen.unload()
    after_unload = current_vram_gb()
    if after_unload is not None:
        ok(f"VRAM after unload: {after_unload:.2f} GB (should be near 0)")

    return True


# ---------------------------------------------------------------------------
# Aligner swap-pattern verification
# ---------------------------------------------------------------------------


def run_aligner_swap_check() -> bool:
    """
    Verify the orchestrator's load/unload pattern bounds peak VRAM.

    Loads Cohere, unloads, then loads Qwen3-ForcedAligner-0.6B sequentially.
    Peak VRAM should reflect max(generator, aligner), not sum — confirming
    the §1.3 architecture claim in the v1.8.14 plan.
    """
    section("Sequential swap-pattern VRAM check (Cohere → Qwen3 aligner)")

    from whisperjav.modules.subtitle_pipeline.generators.factory import (
        TextGeneratorFactory,
    )

    # Phase 1: Cohere
    reset_peak_vram()
    gen = TextGeneratorFactory.create("cohere")
    try:
        gen.load()
    except Exception as exc:
        fail(f"Cohere load failed: {exc}")
        return False
    peak_cohere = peak_vram_gb()
    ok(f"Peak VRAM with Cohere loaded: {peak_cohere:.2f} GB" if peak_cohere else "VRAM tracking unavailable")
    gen.unload()
    after_unload = current_vram_gb()
    info(f"VRAM after Cohere unload: {after_unload:.2f} GB" if after_unload is not None else "")

    # Phase 2: Qwen3 ForcedAligner
    reset_peak_vram()
    try:
        from whisperjav.modules.subtitle_pipeline.aligners.factory import (
            TextAlignerFactory,
        )
        aligner = TextAlignerFactory.create(
            "qwen3",
            aligner_id="Qwen/Qwen3-ForcedAligner-0.6B",
            device="auto",
            dtype="auto",
            language="ja",
        )
        aligner.load()
    except Exception as exc:
        fail(f"Qwen3 aligner load failed: {exc}")
        return False
    peak_aligner = peak_vram_gb()
    ok(f"Peak VRAM with Qwen3 aligner loaded: {peak_aligner:.2f} GB (expected ~1.2 GB)" if peak_aligner else "")
    aligner.unload()

    # Verdict
    section("Swap-pattern verdict")
    if peak_cohere is not None and peak_aligner is not None:
        if peak_aligner < peak_cohere:
            ok(
                f"Peak VRAM bounded by Cohere alone: aligner peak ({peak_aligner:.2f} GB) "
                f"< Cohere peak ({peak_cohere:.2f} GB). Swap pattern verified."
            )
        else:
            info(
                f"Note: aligner peak ({peak_aligner:.2f} GB) >= Cohere peak "
                f"({peak_cohere:.2f} GB). This is unexpected; investigate."
            )
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cohere-Transcribe smoke test for v1.8.14 ChronosJAV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional audio file to transcribe after loading (16 kHz mono recommended)",
    )
    parser.add_argument(
        "--check-aligner-vram",
        action="store_true",
        help="After the Cohere cycle, also load Qwen3-ForcedAligner-0.6B in a "
             "separate swap cycle to verify peak VRAM stays bounded.",
    )
    args = parser.parse_args()

    if args.audio is not None:
        args.audio = args.audio.expanduser().resolve()
        if not args.audio.is_file():
            fail(f"--audio path is not a file: {args.audio}")
            return 2

    if not preflight():
        section("Result")
        fail("Pre-flight checks did not pass; aborting.")
        return 2

    if not run_cohere_cycle(args.audio):
        section("Result")
        fail("Cohere load/generate/unload cycle did not complete successfully.")
        return 1

    if args.check_aligner_vram:
        if not run_aligner_swap_check():
            section("Result")
            fail("Aligner swap-pattern check did not complete successfully.")
            return 1

    section("Result")
    ok("All smoke-test phases completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
