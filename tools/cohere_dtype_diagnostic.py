#!/usr/bin/env python3
"""
Cohere-Transcribe dtype diagnostic — isolated, reference-driven.

Purpose
-------
The v1.8.14 GUI run on 2026-05-08 hit a runtime overflow inside
modeling_cohere_asr.py (`scores.masked_fill(expanded_mask, -1e9)` on a
half-precision tensor).  Before changing WhisperJAV's cohere.py, we need
empirical answers — NOT trace-driven hypotheses — to these questions:

  Q1. What dtype does the Cohere model card / config.json say to use?
  Q2. What does the canonical repka3 reference pattern actually load
      as (since it passes no dtype kwarg)?
  Q3. Which of {repka3-default, float32, bfloat16, float16} actually run
      generate() successfully on this hardware?
  Q4. What is the peak VRAM and end-to-end latency for each?

The script exercises Cohere DIRECTLY through transformers — no WhisperJAV
pipeline, no scene detection, no aligner.  Just AutoProcessor +
AutoModelForSpeechSeq2Seq + a tiny in-memory audio buffer.  Whatever
fails / succeeds in this script tells us the truth without the pipeline's
interference.

Prerequisites
-------------
  - Accept terms at https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
  - HF_TOKEN set in environment (or HUGGING_FACE_HUB_TOKEN)
  - Cohere weights already downloaded (re-uses HF cache)

Usage
-----
  python tools/cohere_dtype_diagnostic.py
  python tools/cohere_dtype_diagnostic.py --scenarios R FP32 BF16 FP16
  python tools/cohere_dtype_diagnostic.py --audio path/to/clip.wav
  python tools/cohere_dtype_diagnostic.py --skip-readme

Output
------
  - Pre-flight environment summary
  - Cohere README excerpt (official usage pattern, when fetchable)
  - config.json torch_dtype field
  - One block per scenario with: load result, model.dtype, peak VRAM,
    generate() result, transcript text or error
  - Final summary table

The user reads the summary table and chooses the dtype strategy for
cohere.py.  No code changes happen until empirical results are in hand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def subsection(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def info(msg: str) -> None:
    print(f"  {msg}")


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def preflight() -> dict:
    """Verify environment and report capabilities."""
    section("Pre-flight environment")

    info_dict: dict = {}

    # HF_TOKEN
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("  [FAIL] HF_TOKEN / HUGGING_FACE_HUB_TOKEN not set; aborting.")
        sys.exit(2)
    info(f"HF_TOKEN: set ({len(token)} chars, first 4='{token[:4]}...')")
    info_dict["token_set"] = True

    # Python / transformers / huggingface_hub / torch
    info(f"Python: {sys.version.split()[0]}")

    try:
        import transformers
        info(f"transformers: {transformers.__version__}")
        info_dict["transformers"] = transformers.__version__
    except ImportError:
        print("  [FAIL] transformers not installed.")
        sys.exit(2)

    try:
        import huggingface_hub
        info(f"huggingface_hub: {huggingface_hub.__version__}")
        info_dict["huggingface_hub"] = huggingface_hub.__version__
    except ImportError:
        info("huggingface_hub: (not directly importable)")

    try:
        import torch
        info(f"torch: {torch.__version__}")
        info_dict["torch"] = torch.__version__
        if torch.cuda.is_available():
            info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            info(f"  Compute capability: {torch.cuda.get_device_capability(0)}")
            free_b, total_b = torch.cuda.mem_get_info(0)
            info(f"  Total VRAM:  {total_b / 1024**3:.2f} GB")
            info(f"  Free VRAM:   {free_b / 1024**3:.2f} GB")
            try:
                info(f"  bf16 supported: {torch.cuda.is_bf16_supported()}")
                info_dict["bf16_supported"] = torch.cuda.is_bf16_supported()
            except Exception:
                info_dict["bf16_supported"] = False
            info_dict["cuda"] = True
        else:
            info("CUDA not available — running on CPU (very slow for Cohere).")
            info_dict["cuda"] = False
    except ImportError:
        print("  [FAIL] torch not installed.")
        sys.exit(2)

    return info_dict


# ---------------------------------------------------------------------------
# Reference-pattern fetches
# ---------------------------------------------------------------------------


def fetch_config_json() -> dict | None:
    """Fetch and return the model's config.json. Print torch_dtype field."""
    subsection("Cohere config.json (gated; fetched with HF_TOKEN)")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="config.json",
            token=token,
        )
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        # Print the most-relevant fields without dumping the whole config
        relevant_keys = [
            "model_type",
            "architectures",
            "torch_dtype",
            "_name_or_path",
            "transformers_version",
        ]
        for k in relevant_keys:
            if k in cfg:
                info(f"{k}: {cfg[k]!r}")
        if "torch_dtype" not in cfg:
            info("torch_dtype field: NOT PRESENT in config.json")
            info("  -> 'no dtype kwarg' loading falls back to torch's default (float32)")
        return cfg
    except Exception as exc:
        info(f"Could not fetch config.json: {exc}")
        return None


def fetch_readme_excerpt(skip: bool = False) -> str | None:
    """Fetch the model README and print the Python usage example block."""
    subsection("Cohere README — official usage pattern (Python example)")
    if skip:
        info("Skipped (--skip-readme).")
        return None
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="README.md",
            token=token,
        )
        text = Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        info(f"Could not fetch README.md: {exc}")
        return None

    # Extract the first ```python ... ``` block (the canonical usage example).
    in_block = False
    block_lines: list[str] = []
    for line in text.splitlines():
        if not in_block and line.strip().startswith("```python"):
            in_block = True
            continue
        if in_block and line.strip().startswith("```"):
            break
        if in_block:
            block_lines.append(line)
    if block_lines:
        print()
        for line in block_lines:
            print("  " + line)
        return "\n".join(block_lines)
    info("No ```python``` example block found in README.")
    return None


# ---------------------------------------------------------------------------
# Test audio
# ---------------------------------------------------------------------------


def make_test_audio(seconds: float = 3.0, sr: int = 16000) -> "np.ndarray":
    """Generate a short audio buffer for the diagnostic.

    Uses a low-amplitude tone (440 Hz) rather than pure silence so the
    encoder has measurable energy to process.  The dtype overflow we are
    investigating happens regardless of audio content (it is in the
    attention-mask code path), but a non-silent buffer also exercises the
    output decoder.
    """
    import numpy as np
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    # Tone at 440 Hz, very quiet (-40 dB).
    audio = 0.01 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    return audio


def load_audio_file(path: Path) -> "np.ndarray":
    """Load a user-provided audio file at 16 kHz mono float32."""
    import librosa
    a, _sr = librosa.load(str(path), sr=16000)
    return a.astype("float32")


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


def run_scenario(name: str, dtype_kw: dict, audio, env: dict, *, style: str = "repka3") -> dict:
    """Run one (load, generate) cycle in isolation and return a result dict.

    Args:
        name: Display name (e.g. 'R', 'FP16', 'OFFICIAL').
        dtype_kw: Either {"dtype": torch.<x>}, {"torch_dtype": torch.<x>},
            or {} (no dtype kwarg).
        audio: Pre-loaded numpy float32 mono 16 kHz audio array.
        env: Output of preflight().
        style: Calling-convention style.
            - "repka3": mirrors repka3's cohere_transcript.py — captures
              audio_chunk_index, conditional decode kwargs, .eval(),
              torch.inference_mode(), outputs.cpu(), do_sample/num_beams set.
            - "official": mirrors the Cohere Labs README's Japanese example
              byte-for-byte — device_map="auto" instead of .to(device),
              no punctuation in processor, no audio_chunk_index capture,
              no inference_mode, no .cpu() before decode, no greedy overrides
              in generate, minimal decode kwargs (skip_special_tokens only).
            - "dev3": mirrors a third community implementation —
              .to(device) + .eval(), inference_mode loop, no punctuation,
              UNCONDITIONAL audio_chunk_index+language in decode, [0] index
              on decode result, no greedy overrides, generate with
              max_new_tokens=256.  Pair with torch_dtype= (not dtype=) kwarg.
    """
    subsection(f"SCENARIO {name}: dtype_kw = {dtype_kw}, style = {style}")
    import torch

    result: dict[str, Any] = {
        "name": name,
        "dtype_kw": dtype_kw,
        "style": style,
        "load_ok": False,
        "model_dtype": None,
        "peak_vram_gb": None,
        "load_seconds": None,
        "generate_ok": False,
        "generate_seconds": None,
        "transcript": None,
        "error_class": None,
        "error_msg": None,
    }

    # Reset peak VRAM stats for this scenario
    if env.get("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        except Exception:
            pass

    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    try:
        info("Loading processor (trust_remote_code=True) ...")
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        if style == "official":
            info(f"Loading model {dtype_kw=} device_map='auto' (README pattern) ...")
            t0 = time.time()
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                device_map="auto",
                **dtype_kw,
            )
            # Note: NO explicit .to() and NO .eval() — README pattern.
            device = str(model.device)
        else:
            info(f"Loading model {dtype_kw=} (trust_remote_code=True) ...")
            t0 = time.time()
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                **dtype_kw,
            )
            device = "cuda:0" if env.get("cuda") else "cpu"
            model.to(device)
            model.eval()

        result["load_seconds"] = round(time.time() - t0, 1)
        result["load_ok"] = True
        result["model_dtype"] = str(model.dtype)
        info(f"Loaded in {result['load_seconds']}s; model.dtype = {model.dtype}; device = {device}")

        if env.get("cuda"):
            result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
            info(f"Peak VRAM after load: {result['peak_vram_gb']} GB")
    except Exception as exc:
        result["error_class"] = type(exc).__name__
        result["error_msg"] = str(exc)[:500]
        info(f"  [LOAD-FAIL] {type(exc).__name__}: {str(exc).splitlines()[0][:400]}")
        return result

    try:
        info(f"Running generate() on {len(audio) / 16000:.2f}s audio (style={style}) ...")

        if style == "official":
            # Mirror the README's Japanese example BYTE-FOR-BYTE:
            #   inputs = processor(audio, sampling_rate=sr, return_tensors="pt", language="ja")
            #   inputs.to(model.device, dtype=model.dtype)
            #   outputs = model.generate(**inputs, max_new_tokens=256)
            #   text = processor.decode(outputs, skip_special_tokens=True)
            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                language="ja",
            )
            inputs.to(model.device, dtype=model.dtype)

            t0 = time.time()
            outputs = model.generate(**inputs, max_new_tokens=256)
            result["generate_seconds"] = round(time.time() - t0, 2)

            transcript = processor.decode(outputs, skip_special_tokens=True)
        elif style == "dev3":
            # Mirror the third community implementation — same as repka3 but:
            #   - no punctuation kwarg in processor
            #   - UNCONDITIONAL audio_chunk_index + language in decode
            #   - [0] index on decode result (decode returns list)
            #   - no do_sample/num_beams in generate
            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                language="ja",
            )
            audio_chunk_index = inputs.get("audio_chunk_index")
            inputs = inputs.to(model.device, dtype=model.dtype)

            t0 = time.time()
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=256)
            result["generate_seconds"] = round(time.time() - t0, 2)

            transcript = processor.decode(
                outputs,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language="ja",
            )
            # Dev3 indexes [0] unconditionally; honor that
            if isinstance(transcript, list) and transcript:
                transcript = transcript[0]
        elif style == "seeded_mask":
            # Workaround for Cohere modeling line 906 bug: pre-seed
            # decoder_attention_mask so kwargs.pop() at line 896 returns a
            # real tensor, line 906 stores it, transformers' update path
            # can call .new_ones() on it without crashing.
            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                language="ja",
            )
            audio_chunk_index = inputs.get("audio_chunk_index")
            inputs = inputs.to(model.device, dtype=model.dtype)

            # Determine batch size from input_features.  Single-clip use
            # case → batch_size = 1.  Mask shape is (B, 1) initially —
            # transformers' generation loop extends along dim=-1 each step.
            if "input_features" in inputs:
                batch_size = inputs["input_features"].shape[0]
            else:
                batch_size = 1
            seed_mask = torch.ones(
                (batch_size, 1),
                dtype=torch.long,
                device=model.device,
            )
            info(f"  Seeding decoder_attention_mask shape={tuple(seed_mask.shape)} dtype={seed_mask.dtype}")

            t0 = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    decoder_attention_mask=seed_mask,
                    max_new_tokens=256,
                )
            result["generate_seconds"] = round(time.time() - t0, 2)

            transcript = processor.decode(
                outputs,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language="ja",
            )
            if isinstance(transcript, list) and transcript:
                transcript = transcript[0]
        elif style == "seeded_mask_no_static":
            # Same as seeded_mask but ALSO override Cohere's default
            # cache_implementation="static" (modeling line 922), since
            # transformers 4.57.6 falls in the version gap that keeps
            # static cache enabled.  Tests whether static-cache compat
            # is a SECOND bug after the line-906 mask fix.
            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                language="ja",
            )
            audio_chunk_index = inputs.get("audio_chunk_index")
            inputs = inputs.to(model.device, dtype=model.dtype)

            if "input_features" in inputs:
                batch_size = inputs["input_features"].shape[0]
            else:
                batch_size = 1
            seed_mask = torch.ones(
                (batch_size, 1),
                dtype=torch.long,
                device=model.device,
            )
            info(f"  Seeding decoder_attention_mask shape={tuple(seed_mask.shape)} + cache_implementation=None")

            t0 = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    decoder_attention_mask=seed_mask,
                    cache_implementation=None,  # override Cohere's static-cache default
                    max_new_tokens=256,
                )
            result["generate_seconds"] = round(time.time() - t0, 2)

            transcript = processor.decode(
                outputs,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language="ja",
            )
            if isinstance(transcript, list) and transcript:
                transcript = transcript[0]
        else:
            # repka3 style: punctuation=True, capture audio_chunk_index,
            # inference_mode, .cpu() before decode, conditional decode kwargs,
            # explicit greedy decoding.
            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                language="ja",
                punctuation=True,
            )
            audio_chunk_index = inputs.get("audio_chunk_index")
            inputs = inputs.to(device, dtype=model.dtype)

            t0 = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1,
                )
            result["generate_seconds"] = round(time.time() - t0, 2)
            outputs = outputs.cpu()

            decode_kwargs: dict = {"skip_special_tokens": True}
            if audio_chunk_index is not None:
                decode_kwargs["audio_chunk_index"] = audio_chunk_index
                decode_kwargs["language"] = "ja"
            transcript = processor.decode(outputs, **decode_kwargs)

        if isinstance(transcript, list):
            transcript = transcript[0] if len(transcript) == 1 else "\n".join(transcript)
        transcript = (transcript or "").strip()
        result["generate_ok"] = True
        result["transcript"] = transcript[:200]
        info(f"  [GEN-OK]  ({result['generate_seconds']}s) transcript: {transcript[:80]!r}")
    except Exception as exc:
        result["error_class"] = type(exc).__name__
        result["error_msg"] = str(exc)[:500]
        tb_lines = traceback.format_exc().splitlines()
        info(f"  [GEN-FAIL] {type(exc).__name__}: {str(exc).splitlines()[0][:400]}")
        for ln in tb_lines[-5:]:
            print(f"    {ln}")

    # Cleanup before next scenario
    try:
        del model
        del processor
    except Exception:
        pass
    if env.get("cuda"):
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


SCENARIO_DEFS = {
    "R": {
        "label": "repka3 verbatim (no dtype kwarg; loader uses config-default)",
        "dtype_kw_factory": lambda torch: {},
        "style": "repka3",
    },
    "FP32": {
        "label": "explicit float32, repka3-style call sequence",
        "dtype_kw_factory": lambda torch: {"dtype": torch.float32},
        "style": "repka3",
    },
    "BF16": {
        "label": "explicit bfloat16, repka3-style call sequence",
        "dtype_kw_factory": lambda torch: {"dtype": torch.bfloat16},
        "style": "repka3",
    },
    "FP16": {
        "label": "explicit float16, repka3-style call sequence (reproduces user error)",
        "dtype_kw_factory": lambda torch: {"dtype": torch.float16},
        "style": "repka3",
    },
    "OFFICIAL": {
        "label": "Cohere Labs README JA example, byte-for-byte (device_map=auto, no extra kwargs)",
        "dtype_kw_factory": lambda torch: {},
        "style": "official",
    },
    "OFFICIAL_BF16": {
        "label": "README pattern + explicit bfloat16",
        "dtype_kw_factory": lambda torch: {"dtype": torch.bfloat16},
        "style": "official",
    },
    # ── Third reference: community PyTorch backend ───────────────────────
    # Critical detail: uses torch_dtype= (canonical 4.57.6 kwarg) NOT dtype=.
    # If trust_remote_code's configuration_cohere_asr.py reads torch_dtype
    # from kwargs but ignores dtype, this kwarg-name distinction matters.
    "DEV3_FP16": {
        "label": "dev3 community pattern + torch_dtype=float16 (their reported working config)",
        "dtype_kw_factory": lambda torch: {"torch_dtype": torch.float16},
        "style": "dev3",
    },
    "DEV3_FP32": {
        "label": "dev3 community pattern + torch_dtype=float32",
        "dtype_kw_factory": lambda torch: {"torch_dtype": torch.float32},
        "style": "dev3",
    },
    # Sanity: same dev3 call style but with our dtype= kwarg name to isolate
    # whether the kwarg name itself is the load-time switch.
    "DEV3_FP16_DTYPE_KW": {
        "label": "dev3 call style + dtype=float16 (NOT torch_dtype=) — kwarg-name isolation",
        "dtype_kw_factory": lambda torch: {"dtype": torch.float16},
        "style": "dev3",
    },
    # ── Workaround scenarios for Cohere modeling line 906 bug ────────────
    # See modeling_cohere_asr.py:885-906 — Cohere's generate() puts
    # decoder_attention_mask=None into kwargs unconditionally when the
    # caller doesn't pass decoder_input_ids.  These scenarios pre-seed the
    # mask so the line-906 store carries our tensor (not None) downstream.
    "SEED_FP32": {
        "label": "FP32 + pre-seeded decoder_attention_mask (workaround for modeling line 906)",
        "dtype_kw_factory": lambda torch: {"dtype": torch.float32},
        "style": "seeded_mask",
    },
    "SEED_BF16": {
        "label": "BF16 + pre-seeded decoder_attention_mask",
        "dtype_kw_factory": lambda torch: {"dtype": torch.bfloat16},
        "style": "seeded_mask",
    },
    # If SEED_FP32/SEED_BF16 still fail, this isolates whether Cohere's
    # default static-cache (modeling line 922) is a second bug.
    "SEED_FP32_NOSTATIC": {
        "label": "FP32 + seeded mask + cache_implementation=None (override Cohere's static-cache default)",
        "dtype_kw_factory": lambda torch: {"dtype": torch.float32},
        "style": "seeded_mask_no_static",
    },
    "SEED_BF16_NOSTATIC": {
        "label": "BF16 + seeded mask + cache_implementation=None",
        "dtype_kw_factory": lambda torch: {"dtype": torch.bfloat16},
        "style": "seeded_mask_no_static",
    },
}


def print_summary_table(results: list[dict]) -> None:
    section("Summary")
    rows = []
    rows.append(("Scenario", "Load", "model.dtype", "Peak VRAM", "Generate", "Time", "Notes"))
    for r in results:
        load = "OK" if r["load_ok"] else "FAIL"
        gen = "OK" if r["generate_ok"] else ("FAIL" if r["load_ok"] else "-")
        vram = f"{r['peak_vram_gb']} GB" if r["peak_vram_gb"] is not None else "-"
        gen_t = f"{r['generate_seconds']}s" if r["generate_seconds"] else "-"
        notes = ""
        if not r["load_ok"]:
            notes = f"{r['error_class']}: {(r['error_msg'] or '').splitlines()[0][:60]}"
        elif not r["generate_ok"]:
            notes = f"{r['error_class']}: {(r['error_msg'] or '').splitlines()[0][:60]}"
        else:
            notes = "transcript: " + repr((r['transcript'] or '')[:40])
        rows.append((r["name"], load, str(r["model_dtype"]), vram, gen, gen_t, notes))

    # Compute column widths
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for ri, row in enumerate(rows):
        line = "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
        print(line)
        if ri == 0:
            print("  ".join("-" * w for w in widths))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Cohere-Transcribe isolated dtype diagnostic for v1.8.14",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=["OFFICIAL", "R", "FP16", "BF16"],
        choices=list(SCENARIO_DEFS.keys()),
        help="Which scenarios to run.  Default: OFFICIAL R FP16 BF16.",
    )
    p.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional audio file for the generate() step (default: in-memory 3s tone).",
    )
    p.add_argument(
        "--skip-readme",
        action="store_true",
        help="Skip the README fetch (saves a small download on repeat runs).",
    )
    args = p.parse_args()

    env = preflight()
    cfg = fetch_config_json()
    fetch_readme_excerpt(skip=args.skip_readme)

    if args.audio:
        audio = load_audio_file(args.audio.expanduser().resolve())
        info(f"Loaded user audio: {args.audio} ({len(audio) / 16000:.2f}s)")
    else:
        audio = make_test_audio(seconds=3.0)
        info(f"Using built-in 3s 440 Hz tone (no audio file given).")

    import torch
    results = []
    for s in args.scenarios:
        defn = SCENARIO_DEFS[s]
        section(f"Scenario {s}: {defn['label']}")
        kw = defn["dtype_kw_factory"](torch)
        results.append(run_scenario(s, kw, audio, env, style=defn.get("style", "repka3")))

    print_summary_table(results)

    # Surface a one-line diagnostic verdict
    section("Verdict")
    ok = [r for r in results if r["generate_ok"]]
    if ok:
        names = ", ".join(r["name"] for r in ok)
        print(f"  Generation succeeded under: {names}")
    else:
        print("  All scenarios failed generation. See errors above.")
    fail = [r for r in results if r["load_ok"] and not r["generate_ok"]]
    if fail:
        print(f"  Load OK but generate FAILED under: {', '.join(r['name'] for r in fail)}")
        for r in fail:
            print(f"    {r['name']}: {r['error_class']}: {(r['error_msg'] or '').splitlines()[0][:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
