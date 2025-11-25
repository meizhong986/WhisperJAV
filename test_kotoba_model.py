#!/usr/bin/env python3
"""
Test script for faster-whisper model loading and transcription.
Default: Tests kotoba-tech/kotoba-whisper-v2.0-faster model
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional

def print_step(step: str, status: str = "INFO"):
    """Print formatted step message."""
    colors = {
        "INFO": "\033[94m",  # Blue
        "SUCCESS": "\033[92m",  # Green
        "ERROR": "\033[91m",  # Red
        "WARNING": "\033[93m"  # Yellow
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}{status}: {step}{reset}")


def check_cuda():
    """Check CUDA availability."""
    print_step("Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_step(f"CUDA available: {device_name} ({memory_total:.1f}GB)", "SUCCESS")

            # Show current memory usage
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print_step(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved", "INFO")
            return True
        else:
            print_step("CUDA not available, will use CPU", "WARNING")
            return False
    except ImportError:
        print_step("PyTorch not found, cannot check CUDA", "WARNING")
        return False


def check_faster_whisper():
    """Check if faster-whisper is installed."""
    print_step("Checking faster-whisper installation...")
    try:
        import faster_whisper
        version = faster_whisper.__version__ if hasattr(faster_whisper, '__version__') else "unknown"
        print_step(f"faster-whisper installed: version {version}", "SUCCESS")
        return True
    except ImportError:
        print_step("faster-whisper not installed", "ERROR")
        print_step("Install with: pip install faster-whisper", "INFO")
        return False


def check_model_cache(model_id: str) -> bool:
    """Check if model is already cached."""
    print_step(f"Checking if model is cached: {model_id}...")

    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()

        for repo in cache_info.repos:
            if model_id in repo.repo_id:
                size_gb = repo.size_on_disk / 1e9
                print_step(f"Model found in cache: {repo.repo_path} ({size_gb:.2f}GB)", "SUCCESS")
                return True

        print_step("Model not in cache, will download from HuggingFace", "INFO")
        return False
    except Exception as e:
        print_step(f"Could not check cache: {e}", "WARNING")
        return False


def load_model(model_id: str, device: str = "cuda", compute_type: str = "float16"):
    """Load faster-whisper model with monitoring."""
    from faster_whisper import WhisperModel

    print_step(f"Loading model: {model_id}")
    print_step(f"Device: {device}, Compute type: {compute_type}")

    start_time = time.time()

    try:
        model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type if device == "cuda" else "int8",
            download_root=None  # Use default HF cache
        )

        load_time = time.time() - start_time
        print_step(f"Model loaded successfully in {load_time:.2f}s", "SUCCESS")
        return model

    except Exception as e:
        print_step(f"Model loading failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None


def transcribe_audio(model, audio_path: Path, language: str = "ja",
                     vad_filter: bool = True, vad_threshold: float = 0.35,
                     min_speech_duration_ms: int = 250, max_speech_duration_s: float = 30.0,
                     temperature: list = None, beam_size: int = 5, best_of: int = 5,
                     no_speech_threshold: float = 0.6, logprob_threshold: float = -1.0,
                     compression_ratio_threshold: float = 2.4):
    """Transcribe audio file with detailed parameters."""
    print_step(f"Transcribing audio: {audio_path.name}")
    print_step(f"Language: {language}")

    # Default temperature if not provided
    if temperature is None:
        temperature = [0.0]

    # Report transcription parameters
    print_step(f"VAD Filter: {vad_filter}, Threshold: {vad_threshold}", "INFO")
    print_step(f"Temperature: {temperature}, Beam Size: {beam_size}, Best Of: {best_of}", "INFO")
    print_step(f"No Speech Threshold: {no_speech_threshold}, LogProb Threshold: {logprob_threshold}", "INFO")

    if not audio_path.exists():
        print_step(f"Audio file not found: {audio_path}", "ERROR")
        return None

    start_time = time.time()

    try:
        # Prepare VAD parameters if enabled
        vad_params = None
        if vad_filter:
            vad_params = {
                "threshold": vad_threshold,
                "min_speech_duration_ms": min_speech_duration_ms,
                "max_speech_duration_s": max_speech_duration_s,
                "min_silence_duration_ms": 150,
                "speech_pad_ms": 400
            }
            print_step(f"VAD Parameters: {vad_params}", "INFO")

        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=logprob_threshold,
            condition_on_previous_text=True,
            initial_prompt=None,
            word_timestamps=False,
            vad_filter=vad_filter,
            vad_parameters=vad_params
        )

        print_step(f"Detected language: {info.language} (probability: {info.language_probability:.2f})", "INFO")
        print_step(f"Duration: {info.duration:.2f}s", "INFO")

        # Collect segments
        results = []
        for segment in segments:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        transcribe_time = time.time() - start_time
        print_step(f"Transcription completed in {transcribe_time:.2f}s", "SUCCESS")
        print_step(f"Found {len(results)} segments", "INFO")

        return results

    except Exception as e:
        print_step(f"Transcription failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None


def cleanup_model(model):
    """Clean up model and free GPU memory."""
    print_step("Cleaning up model...")
    try:
        del model

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            print_step(f"GPU Memory after cleanup: {memory_allocated:.2f}GB", "INFO")

        print_step("Cleanup complete", "SUCCESS")
    except Exception as e:
        print_step(f"Cleanup warning: {e}", "WARNING")


def main():
    parser = argparse.ArgumentParser(
        description="Test faster-whisper model loading and transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test default kotoba model with defaults
  python test_kotoba_model.py audio.wav

  # Test with aggressive VAD for more detail capture
  python test_kotoba_model.py audio.wav --vad-threshold 0.25 --min-speech-duration 100

  # Test with higher beam size for better accuracy
  python test_kotoba_model.py audio.wav --beam-size 8 --best-of 8

  # Test with lower logprob threshold for more segments
  python test_kotoba_model.py audio.wav --logprob-threshold -0.5

  # Test with temperature fallback for difficult audio
  python test_kotoba_model.py audio.wav --temperature 0.0,0.2,0.4,0.6,0.8,1.0

  # Test without VAD filtering
  python test_kotoba_model.py audio.wav --no-vad-filter

  # Test with different model
  python test_kotoba_model.py audio.wav --model large-v2

  # Test on CPU
  python test_kotoba_model.py audio.wav --device cpu

  # Test with English audio
  python test_kotoba_model.py audio.wav --model large-v3 --language en

  # Maximum detail capture (aggressive settings)
  python test_kotoba_model.py audio.wav --vad-threshold 0.2 --beam-size 10 \
         --logprob-threshold -0.8 --no-speech-threshold 0.4
        """
    )

    parser.add_argument(
        "audio_file",
        type=Path,
        help="Input audio/video file to transcribe"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="kotoba-tech/kotoba-whisper-v2.0-faster",
        help="Model ID or path (default: kotoba-tech/kotoba-whisper-v2.0-faster)"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)"
    )

    parser.add_argument(
        "--compute-type",
        type=str,
        choices=["float16", "int8", "float32"],
        default="float16",
        help="Compute type for inference (default: float16)"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="Audio language code (default: ja)"
    )

    # Transcription parameters
    vad_group = parser.add_mutually_exclusive_group(required=False)
    vad_group.add_argument(
        "--vad-filter",
        dest="vad_filter",
        action="store_true",
        help="Enable VAD filtering (default: True)"
    )
    vad_group.add_argument(
        "--no-vad-filter",
        dest="vad_filter",
        action="store_false",
        help="Disable VAD filtering"
    )
    parser.set_defaults(vad_filter=True)

    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.01,
        help="VAD threshold (0.0-1.0, default: 0.05, lower=more aggressive)"
    )

    parser.add_argument(
        "--min-speech-duration",
        type=int,
        default=90,
        help="Minimum speech duration in ms (default: 90)"
    )

    parser.add_argument(
        "--max-speech-duration",
        type=float,
        default=28.0,
        help="Maximum speech duration in seconds (default: 28.0)"
    )

    parser.add_argument(
        "--temperature",
        type=str,
        default="0.0,0.3",
        help="Temperature for sampling (default: 0.0,0.3). Can be single value (0.0) or comma-separated list (0.0,0.2,0.4) for fallback"
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=3,
        help="Beam size for beam search (default: 3, higher=more accurate but slower)"
    )

    parser.add_argument(
        "--best-of",
        type=int,
        default=3,
        help="Number of candidates when sampling (default: 3)"
    )

    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.34,
        help="No speech threshold (0.0-1.0, default: 0.34, higher=more filtering)"
    )

    parser.add_argument(
        "--logprob-threshold",
        type=float,
        default=-1.5,
        help="LogProb threshold (default: -1.5, higher=more filtering)"
    )

    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=2.4,
        help="Compression ratio threshold (default: 2.4)"
    )

    args = parser.parse_args()

    # Parse temperature into list of floats
    try:
        if ',' in args.temperature:
            # Multiple temperatures: "0.0,0.2,0.4" -> [0.0, 0.2, 0.4]
            temperature_list = [float(t.strip()) for t in args.temperature.split(',')]
        else:
            # Single temperature: "0.0" -> [0.0]
            temperature_list = [float(args.temperature)]
    except ValueError:
        print_step(f"Invalid temperature value: {args.temperature}", "ERROR")
        print_step("Temperature must be a number or comma-separated numbers (e.g., 0.0 or 0.0,0.2,0.4)", "ERROR")
        return 1

    args.temperature = temperature_list

    print("=" * 70)
    print("FASTER-WHISPER MODEL TEST")
    print("=" * 70)
    print()

    # Step 1: Check CUDA
    cuda_available = check_cuda()
    if args.device == "cuda" and not cuda_available:
        print_step("CUDA requested but not available, switching to CPU", "WARNING")
        args.device = "cpu"
    print()

    # Step 2: Check faster-whisper installation
    if not check_faster_whisper():
        return 1
    print()

    # Step 3: Check model cache
    check_model_cache(args.model)
    print()

    # Step 4: Load model
    print_step("STEP 1: Loading Model", "INFO")
    print("-" * 70)
    model = load_model(args.model, args.device, args.compute_type)
    if not model:
        return 1
    print()

    # Step 5: Transcribe audio
    print_step("STEP 2: Transcribing Audio", "INFO")
    print("-" * 70)
    results = transcribe_audio(
        model,
        args.audio_file,
        language=args.language,
        vad_filter=args.vad_filter,
        vad_threshold=args.vad_threshold,
        min_speech_duration_ms=args.min_speech_duration,
        max_speech_duration_s=args.max_speech_duration,
        temperature=args.temperature,
        beam_size=args.beam_size,
        best_of=args.best_of,
        no_speech_threshold=args.no_speech_threshold,
        logprob_threshold=args.logprob_threshold,
        compression_ratio_threshold=args.compression_ratio_threshold
    )
    if results is None:
        cleanup_model(model)
        return 1
    print()

    # Step 6: Cleanup
    print_step("STEP 3: Cleanup", "INFO")
    print("-" * 70)
    cleanup_model(model)
    print()

    # Summary
    print("=" * 70)
    print_step("TEST COMPLETED SUCCESSFULLY", "SUCCESS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Audio: {args.audio_file}")
    print(f"Segments: {len(results) if results else 0}")
    print()
    print("Transcription Parameters:")
    print(f"  VAD Filter: {args.vad_filter}")
    if args.vad_filter:
        print(f"  VAD Threshold: {args.vad_threshold}")
        print(f"  Min Speech Duration: {args.min_speech_duration}ms")
        print(f"  Max Speech Duration: {args.max_speech_duration}s")
    print(f"  Temperature: {args.temperature}")
    print(f"  Beam Size: {args.beam_size}")
    print(f"  Best Of: {args.best_of}")
    print(f"  No Speech Threshold: {args.no_speech_threshold}")
    print(f"  LogProb Threshold: {args.logprob_threshold}")
    print(f"  Compression Ratio Threshold: {args.compression_ratio_threshold}")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print_step("Test interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        print()
        print_step(f"Unexpected error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
