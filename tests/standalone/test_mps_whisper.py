#!/usr/bin/env python3
"""
Standalone MPS + Whisper test for Apple Silicon Macs.

Run this inside your WhisperJAV Python environment to verify that:
  1. PyTorch detects MPS (Metal Performance Shaders)
  2. MPS actually accelerates inference (not silently falling back to CPU)
  3. HuggingFace Transformers Whisper pipeline runs on MPS without errors

Usage:
    python test_mps_whisper.py
    python test_mps_whisper.py --model openai/whisper-large-v3-turbo
    python test_mps_whisper.py --audio /path/to/short_clip.wav
"""
import argparse
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Test MPS + Whisper on Apple Silicon")
    parser.add_argument("--model", default="openai/whisper-large-v3-turbo",
                        help="HuggingFace model ID (default: whisper-large-v3-turbo)")
    parser.add_argument("--audio", default=None,
                        help="Path to a short audio file (WAV/MP3). If omitted, uses synthetic audio.")
    args = parser.parse_args()

    print("=" * 60)
    print("MPS + Whisper Standalone Test")
    print("=" * 60)

    # --- Step 1: PyTorch MPS check ---
    print("\n[1/5] Checking PyTorch MPS support...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  MPS available:   {torch.backends.mps.is_available()}")
        print(f"  MPS built:       {torch.backends.mps.is_built()}")
    except ImportError:
        print("  ERROR: PyTorch not installed.")
        sys.exit(1)

    if not torch.backends.mps.is_available():
        print("\n  MPS is NOT available. This test requires Apple Silicon (M1/M2/M3/M4).")
        print("  If you are on Apple Silicon, check your PyTorch installation.")
        sys.exit(1)

    # --- Step 2: MPS tensor operations benchmark ---
    print("\n[2/5] Benchmarking MPS vs CPU (matrix multiply)...")
    size = 2048
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)

    # CPU
    t0 = time.perf_counter()
    for _ in range(5):
        _ = a_cpu @ b_cpu
    cpu_time = (time.perf_counter() - t0) / 5

    # MPS
    a_mps = a_cpu.to("mps")
    b_mps = b_cpu.to("mps")
    # Warmup
    _ = a_mps @ b_mps
    torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(5):
        _ = a_mps @ b_mps
        torch.mps.synchronize()
    mps_time = (time.perf_counter() - t0) / 5

    speedup = cpu_time / mps_time if mps_time > 0 else 0
    print(f"  CPU matmul ({size}x{size}): {cpu_time*1000:.1f} ms")
    print(f"  MPS matmul ({size}x{size}): {mps_time*1000:.1f} ms")
    print(f"  Speedup: {speedup:.1f}x")

    if speedup < 1.5:
        print("  WARNING: MPS is not significantly faster than CPU.")
        print("  This may indicate MPS is not working correctly.")
    else:
        print(f"  OK — MPS is {speedup:.1f}x faster than CPU.")

    # --- Step 3: Load Whisper model on MPS ---
    print(f"\n[3/5] Loading model: {args.model} on MPS...")
    try:
        import transformers
        print(f"  Transformers version: {transformers.__version__}")
    except ImportError:
        print("  ERROR: transformers not installed.")
        sys.exit(1)

    t0 = time.perf_counter()
    try:
        pipe = transformers.pipeline(
            "automatic-speech-recognition",
            model=args.model,
            device="mps",
            torch_dtype=torch.float16,
        )
        load_time = time.perf_counter() - t0
        print(f"  Model loaded in {load_time:.1f}s")
        print(f"  Device: {pipe.device}")
        print(f"  Dtype:  {pipe.model.dtype}")
    except Exception as e:
        print(f"  ERROR loading model on MPS: {e}")
        print("  Try: pip install --upgrade transformers torch")
        sys.exit(1)

    # --- Step 4: Prepare audio ---
    print("\n[4/5] Preparing audio...")
    if args.audio:
        import numpy as np
        try:
            import librosa
            audio, sr = librosa.load(args.audio, sr=16000, duration=30)
            duration = len(audio) / sr
            print(f"  Loaded: {args.audio} ({duration:.1f}s at 16kHz)")
        except ImportError:
            print("  librosa not available, trying soundfile...")
            import soundfile as sf
            audio, sr = sf.read(args.audio)
            if sr != 16000:
                print(f"  WARNING: Sample rate is {sr}, not 16000. Results may vary.")
            duration = len(audio) / sr
            audio = audio[:int(16000 * 30)]  # Limit to 30s
            print(f"  Loaded: {args.audio} ({min(duration, 30):.1f}s)")
    else:
        import numpy as np
        print("  No audio file provided — generating 15s of synthetic audio (silence + tone).")
        sr = 16000
        duration = 15.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        # Mix of silence and a tone so Whisper has something to process
        audio = np.sin(2 * np.pi * 440 * t) * 0.1
        audio[:sr * 5] = 0  # 5s silence at start

    # --- Step 5: Transcribe on MPS vs CPU ---
    print(f"\n[5/5] Transcribing {duration:.0f}s audio...")

    # MPS transcription
    print("\n  --- MPS ---")
    t0 = time.perf_counter()
    try:
        result_mps = pipe(
            audio.copy(),
            generate_kwargs={"language": "ja", "task": "transcribe", "num_beams": 1},
            chunk_length_s=15,
            stride_length_s=2,
            return_timestamps=True,
        )
        mps_transcribe_time = time.perf_counter() - t0
        n_chunks = len(result_mps.get("chunks", []))
        text_preview = (result_mps.get("text", "")[:100] + "...") if result_mps.get("text") else "(empty)"
        print(f"  Time:     {mps_transcribe_time:.1f}s")
        print(f"  Segments: {n_chunks}")
        print(f"  Text:     {text_preview}")
    except Exception as e:
        mps_transcribe_time = None
        print(f"  ERROR on MPS: {e}")
        print(f"  Error type: {type(e).__name__}")

    # CPU transcription for comparison
    print("\n  --- CPU (comparison) ---")
    pipe_cpu = transformers.pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device="cpu",
        torch_dtype=torch.float32,
    )
    t0 = time.perf_counter()
    try:
        result_cpu = pipe_cpu(
            audio.copy(),
            generate_kwargs={"language": "ja", "task": "transcribe", "num_beams": 1},
            chunk_length_s=15,
            stride_length_s=2,
            return_timestamps=True,
        )
        cpu_transcribe_time = time.perf_counter() - t0
        print(f"  Time:     {cpu_transcribe_time:.1f}s")
    except Exception as e:
        cpu_transcribe_time = None
        print(f"  ERROR on CPU: {e}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  PyTorch:          {torch.__version__}")
    print(f"  Transformers:     {transformers.__version__}")
    print(f"  Model:            {args.model}")
    print(f"  MPS available:    YES")
    print(f"  MPS matmul:       {speedup:.1f}x faster than CPU")

    if mps_transcribe_time and cpu_transcribe_time:
        whisper_speedup = cpu_transcribe_time / mps_transcribe_time
        print(f"  MPS transcribe:   {mps_transcribe_time:.1f}s")
        print(f"  CPU transcribe:   {cpu_transcribe_time:.1f}s")
        print(f"  Whisper speedup:  {whisper_speedup:.1f}x")

        if whisper_speedup < 1.2:
            print("\n  VERDICT: MPS is NOT providing meaningful acceleration for Whisper.")
            print("  Possible causes:")
            print("    - Memory pressure (16GB shared between CPU and GPU)")
            print("    - Model too large for efficient MPS inference")
            print("    - PyTorch MPS backend limitations for this workload")
            print("  Recommendation: try a smaller model (e.g., openai/whisper-large-v3-turbo)")
        else:
            print(f"\n  VERDICT: MPS is working and provides {whisper_speedup:.1f}x speedup.")
    elif mps_transcribe_time:
        print(f"  MPS transcribe:   {mps_transcribe_time:.1f}s (CPU comparison failed)")
    else:
        print("  MPS transcribe:   FAILED")
        print("  This confirms MPS has issues with Whisper on your system.")

    # Cleanup
    del pipe, pipe_cpu
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

if __name__ == "__main__":
    main()
