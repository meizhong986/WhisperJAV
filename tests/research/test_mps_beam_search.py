"""
Isolated test: MPS Beam Search Crash Diagnosis (#198)

This script diagnoses the beam search crash on Apple Silicon MPS by:
1. Loading the model's generation_config.json to check default num_beams
2. Simulating what WhisperJAV's TransformersASR sends to generate()
3. Testing generate() with different beam settings on MPS (if available)

The crash: torch.AcceleratorError: index -4785543516806487491 is out of bounds
Location: transformers/generation/logits_process.py:2021 in __call__
           via transformers/generation/utils.py:3284 in _beam_search

Root cause hypothesis: The model's generation_config.json specifies num_beams > 1,
and even though WhisperJAV v1.8.7 stopped FORCING num_beams=5, the model's own
config still triggers beam search, which crashes on MPS.

Usage:
    # On any machine (checks config, no GPU needed):
    python tests/research/test_mps_beam_search.py

    # On Apple Silicon Mac (full test with actual inference):
    python tests/research/test_mps_beam_search.py --run-inference

    # With a specific model:
    python tests/research/test_mps_beam_search.py --model openai/whisper-large-v3
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Check the model's generation_config.json
# ---------------------------------------------------------------------------

def check_generation_config(model_id: str) -> dict:
    """Download and inspect the model's generation_config.json.

    This is the KEY diagnostic — if num_beams > 1 in the model's config,
    HuggingFace will use beam search by default even when WhisperJAV
    doesn't explicitly pass num_beams.
    """
    print(f"\n{'='*70}")
    print(f"CHECKING generation_config.json for: {model_id}")
    print(f"{'='*70}")

    result = {
        "model_id": model_id,
        "num_beams": None,
        "source": None,
        "beam_search_will_trigger": None,
    }

    # Method 1: Try loading from HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_id, "generation_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        result["source"] = "huggingface_hub"
        result["config"] = config
        result["num_beams"] = config.get("num_beams", 1)
        result["beam_search_will_trigger"] = result["num_beams"] > 1

        print(f"\n  Source: Downloaded from HuggingFace Hub")
        print(f"  Full generation_config.json:")
        for key, value in sorted(config.items()):
            marker = " <<<" if key == "num_beams" else ""
            print(f"    {key}: {value}{marker}")

    except ImportError:
        print("  huggingface_hub not installed — trying transformers...")

        try:
            from transformers import GenerationConfig
            gen_config = GenerationConfig.from_pretrained(model_id)
            config_dict = gen_config.to_dict()

            result["source"] = "transformers"
            result["config"] = config_dict
            result["num_beams"] = config_dict.get("num_beams", 1)
            result["beam_search_will_trigger"] = result["num_beams"] > 1

            print(f"\n  Source: transformers.GenerationConfig")
            print(f"  Key parameters:")
            for key in ["num_beams", "do_sample", "temperature", "top_p", "top_k",
                         "max_length", "max_new_tokens", "no_repeat_ngram_size",
                         "begin_suppress_tokens", "forced_decoder_ids"]:
                val = config_dict.get(key, "<not set>")
                marker = " <<<" if key == "num_beams" else ""
                print(f"    {key}: {val}{marker}")

        except Exception as e:
            print(f"  ERROR: Cannot load generation config: {e}")
            result["source"] = "error"
            result["error"] = str(e)

    except Exception as e:
        print(f"  ERROR: {e}")
        result["source"] = "error"
        result["error"] = str(e)

    # Diagnosis
    print(f"\n  DIAGNOSIS:")
    if result["num_beams"] is not None:
        if result["beam_search_will_trigger"]:
            print(f"  >>> num_beams = {result['num_beams']} — BEAM SEARCH WILL BE USED BY DEFAULT")
            print(f"  >>> This WILL crash on MPS (Apple Silicon)")
            print(f"  >>> Fix: Force num_beams=1 when device is MPS")
        else:
            print(f"  >>> num_beams = {result['num_beams']} — Greedy decoding (safe on MPS)")
            print(f"  >>> The v1.8.7 fix should work for this model")
    else:
        print(f"  >>> Could not determine num_beams — manual check needed")

    return result


# ---------------------------------------------------------------------------
# 2. Check what WhisperJAV actually passes to generate()
# ---------------------------------------------------------------------------

def check_whisperjav_generate_kwargs(model_id: str) -> None:
    """Show what generate_kwargs WhisperJAV constructs, pre and post v1.8.7."""
    print(f"\n{'='*70}")
    print(f"WHISPERJAV generate_kwargs COMPARISON")
    print(f"{'='*70}")

    print("\n  BEFORE v1.8.7 (forced defaults):")
    before = {
        "language": "ja",
        "task": "transcribe",
        "num_beams": 5,           # FORCED — overrides model config
        "temperature": 0.0,       # FORCED
        "compression_ratio_threshold": 2.4,  # FORCED
        "logprob_threshold": -1.0,           # FORCED
        "no_speech_threshold": 0.6,          # FORCED
        "condition_on_prev_tokens": True,    # FORCED
    }
    for k, v in before.items():
        print(f"    {k}: {v}")

    print("\n  AFTER v1.8.7 (deferred to model defaults):")
    after = {
        "language": "ja",
        "task": "transcribe",
        # Everything else: NOT passed → model's generation_config.json defaults used
    }
    for k, v in after.items():
        print(f"    {k}: {v}")
    print(f"    (all other params: deferred to model's generation_config.json)")

    print(f"\n  THE PROBLEM:")
    print(f"    Before v1.8.7: num_beams=5 was FORCED -> beam search crash on MPS")
    print(f"    After v1.8.7:  num_beams is NOT passed -> model's default is used")
    print(f"    If model's default num_beams > 1 -> STILL crashes on MPS!")
    print(f"    But: all Whisper models default to num_beams=1 (greedy).")
    print(f"    HOWEVER: Whisper's generate_with_fallback() may use beam search")
    print(f"    as a quality fallback when greedy decoding fails quality checks.")
    print(f"    This is the likely crash path on MPS.")


# ---------------------------------------------------------------------------
# 3. Test inference on MPS (if available)
# ---------------------------------------------------------------------------

def test_mps_inference(model_id: str) -> None:
    """Actually test inference on MPS with different beam settings.

    This test REQUIRES:
    - Apple Silicon Mac (M1/M2/M3/M4)
    - PyTorch with MPS support
    - transformers library
    - The model downloaded (~3GB for kotoba-whisper)
    - A short audio file (generated synthetically)
    """
    print(f"\n{'='*70}")
    print(f"MPS INFERENCE TEST")
    print(f"{'='*70}")

    import torch

    # Check MPS availability
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print(f"\n  MPS not available on this system.")
        print(f"  This test must be run on an Apple Silicon Mac.")
        print(f"  Skipping inference test.")
        return

    print(f"\n  MPS is available! Running inference tests...")

    import numpy as np
    from transformers import pipeline, GenerationConfig

    # Load the model on MPS
    print(f"  Loading {model_id} on MPS (this may take a minute)...")
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch.float16,
            device="mps",
        )
        print(f"  Model loaded successfully on MPS")
    except Exception as e:
        print(f"  ERROR loading model on MPS: {e}")
        return

    # Check the model's actual generation config
    gen_config = pipe.model.generation_config
    print(f"  Model's generation_config.num_beams = {gen_config.num_beams}")

    # Generate a short synthetic audio (1 second of silence with a beep)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Simple sine wave (440 Hz) to give the model something to "transcribe"
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Test 1: Default settings (what v1.8.7 does — defer to model)
    print(f"\n  Test 1: Default generate_kwargs (v1.8.7 behavior)")
    print(f"    generate_kwargs = {{'language': 'ja', 'task': 'transcribe'}}")
    try:
        result = pipe(
            audio,
            chunk_length_s=30,
            return_timestamps=True,
            generate_kwargs={"language": "ja", "task": "transcribe"},
        )
        print(f"    RESULT: Success! Text: '{result.get('text', '')[:50]}...'")
    except Exception as e:
        print(f"    RESULT: CRASH! {type(e).__name__}: {e}")
        if "beam" in str(e).lower() or "out of bounds" in str(e).lower():
            print(f"    >>> This confirms the beam search crash on MPS!")

    # Test 2: Force num_beams=1 (the proposed fix)
    print(f"\n  Test 2: Force num_beams=1 (proposed fix)")
    print(f"    generate_kwargs = {{'language': 'ja', 'task': 'transcribe', 'num_beams': 1}}")
    try:
        result = pipe(
            audio,
            chunk_length_s=30,
            return_timestamps=True,
            generate_kwargs={"language": "ja", "task": "transcribe", "num_beams": 1},
        )
        print(f"    RESULT: Success! Text: '{result.get('text', '')[:50]}...'")
    except Exception as e:
        print(f"    RESULT: CRASH! {type(e).__name__}: {e}")

    # Test 3: Force num_beams=5 (the old v1.8.6 behavior)
    print(f"\n  Test 3: Force num_beams=5 (old v1.8.6 behavior — should crash)")
    print(f"    generate_kwargs = {{'language': 'ja', 'task': 'transcribe', 'num_beams': 5}}")
    try:
        result = pipe(
            audio,
            chunk_length_s=30,
            return_timestamps=True,
            generate_kwargs={"language": "ja", "task": "transcribe", "num_beams": 5},
        )
        print(f"    RESULT: Success! Text: '{result.get('text', '')[:50]}...'")
        print(f"    >>> Unexpected: beam search worked on MPS. May be model/version specific.")
    except Exception as e:
        print(f"    RESULT: CRASH! {type(e).__name__}: {e}")
        if "beam" in str(e).lower() or "out of bounds" in str(e).lower():
            print(f"    >>> Confirmed: beam search crashes on MPS as expected")


# ---------------------------------------------------------------------------
# 4. Proposed fix
# ---------------------------------------------------------------------------

def print_proposed_fix() -> None:
    """Print the proposed code fix."""
    print(f"\n{'='*70}")
    print(f"PROPOSED FIX")
    print(f"{'='*70}")
    print("""
  File: whisperjav/modules/transformers_asr.py
  Location: transcribe() method, where generate_kwargs is assembled

  The fix: When device is MPS, ALWAYS force num_beams=1 regardless of
  model defaults or user settings. Beam search on MPS is broken in
  HuggingFace Transformers as of 2026-03 (uninitialized GPU memory in
  beam index tensors).

  Code change:

    generate_kwargs = {
        "language": self.language,
        "task": self.task,
    }
    if self.beam_size is not None:
        generate_kwargs["num_beams"] = self.beam_size
    # ... other conditional params ...

    # FIX: Force greedy decoding on MPS — beam search crashes with
    # uninitialized memory in beam_indices tensor.
    # See: https://github.com/huggingface/transformers/issues/XXXXX
    if self._device == "mps" and generate_kwargs.get("num_beams", 1) != 1:
        logger.warning(
            "Beam search is not supported on MPS (Apple Silicon). "
            "Forcing num_beams=1 (greedy decoding)."
        )
        generate_kwargs["num_beams"] = 1

    # ALSO: Force num_beams=1 even if not explicitly set, because the
    # model's generation_config.json may specify num_beams > 1
    if self._device == "mps" and "num_beams" not in generate_kwargs:
        generate_kwargs["num_beams"] = 1

  This is a DEVICE-SPECIFIC override, not a general default change.
  CUDA and CPU continue to use whatever the model or user specifies.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MPS Beam Search Crash Diagnosis")
    parser.add_argument("--model", type=str,
                        default="kotoba-tech/kotoba-whisper-bilingual-v1.0",
                        help="Model to test (default: kotoba-whisper)")
    parser.add_argument("--run-inference", action="store_true",
                        help="Actually run inference on MPS (requires Mac + model)")
    parser.add_argument("--check-models", type=str, default=None,
                        help="Comma-separated list of models to check configs for")
    args = parser.parse_args()

    models_to_check = [args.model]
    if args.check_models:
        models_to_check = [m.strip() for m in args.check_models.split(",")]

    # Check generation configs
    results = {}
    for model_id in models_to_check:
        results[model_id] = check_generation_config(model_id)

    # Show WhisperJAV's generate_kwargs
    check_whisperjav_generate_kwargs(args.model)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for model_id, result in results.items():
        beams = result.get("num_beams", "?")
        will_crash = result.get("beam_search_will_trigger", None)
        status = "WILL CRASH on MPS" if will_crash else "Safe on MPS" if will_crash is False else "Unknown"
        print(f"  {model_id}: num_beams={beams} → {status}")

    # Run inference test if requested
    if args.run_inference:
        test_mps_inference(args.model)

    # Print proposed fix
    print_proposed_fix()


if __name__ == "__main__":
    main()
