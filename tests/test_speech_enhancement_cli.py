#!/usr/bin/env python
"""
Speech Enhancement Test Suite (CLI)

Black-box test suite for the speech enhancement module.
Tests the public factory interface and contract.

Usage:
    # Test with default settings (clearvoice, MossFormer2_SE_48K)
    python tests/test_speech_enhancement_cli.py video.mp4

    # Test with specific backend and model
    python tests/test_speech_enhancement_cli.py video.mp4 --backend clearvoice --model FRCRN_SE_16K

    # List available backends and models
    python tests/test_speech_enhancement_cli.py --list-backends
    python tests/test_speech_enhancement_cli.py --list-models clearvoice

    # Specify output file
    python tests/test_speech_enhancement_cli.py video.mp4 --output enhanced.wav

    # Extract specific segment (seconds)
    python tests/test_speech_enhancement_cli.py video.mp4 --start 30 --duration 60

Author: WhisperJAV Team
Version: 1.7.3
"""

import argparse
import sys
import time
import os
from pathlib import Path
from typing import Optional, Tuple
import subprocess
import tempfile

# Ensure whisperjav is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_stat(label: str, value: str, indent: int = 2) -> None:
    """Print a formatted statistic line."""
    padding = " " * indent
    print(f"{padding}{label:<28} {value}")


def format_size(bytes_size: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def list_backends() -> None:
    """List all available backends with their status."""
    from whisperjav.modules.speech_enhancement import SpeechEnhancerFactory

    print_header("Available Speech Enhancement Backends")

    backends = SpeechEnhancerFactory.get_available_backends()

    for backend in backends:
        status = "AVAILABLE" if backend["available"] else "NOT INSTALLED"
        status_color = "\033[92m" if backend["available"] else "\033[91m"
        reset_color = "\033[0m"

        print(f"\n  [{status_color}{status}{reset_color}] {backend['name']}")
        print(f"      Display Name: {backend['display_name']}")
        print(f"      Description:  {backend['description']}")
        if backend.get("default_model"):
            print(f"      Default Model: {backend['default_model']}")
        if not backend["available"] and backend["install_hint"]:
            print(f"      Install: {backend['install_hint']}")

    print()


def list_models(backend: str) -> None:
    """List available models for a specific backend."""
    from whisperjav.modules.speech_enhancement import SpeechEnhancerFactory

    print_header(f"Available Models for '{backend}'")

    # Check if backend is available
    available, hint = SpeechEnhancerFactory.is_backend_available(backend)
    if not available:
        print(f"\n  Backend '{backend}' is not available.")
        if hint:
            print(f"  Install: {hint}")
        print()
        return

    models = SpeechEnhancerFactory.get_backend_models(backend)

    if not models:
        print(f"\n  No models found for backend '{backend}'")
    else:
        print(f"\n  Found {len(models)} model(s):")
        for model in models:
            print(f"    - {model}")

    print()


def extract_audio(
    media_path: Path,
    output_path: Path,
    sample_rate: int,
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Extract audio from media file using FFmpeg.

    Returns:
        Tuple of (success, message)
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(media_path),
    ]

    if start_sec is not None:
        cmd.extend(["-ss", str(start_sec)])

    if duration_sec is not None:
        cmd.extend(["-t", str(duration_sec)])

    cmd.extend([
        "-vn",  # No video
        "-ac", "1",  # Mono
        "-ar", str(sample_rate),  # Sample rate
        "-acodec", "pcm_f32le",  # 32-bit float WAV
        str(output_path),
    ])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, f"FFmpeg error: {result.stderr}"
        return True, "OK"
    except FileNotFoundError:
        return False, "FFmpeg not found. Please install FFmpeg."
    except Exception as e:
        return False, f"Error: {e}"


def get_audio_info(audio_path: Path, sample_rate: int) -> dict:
    """Get information about an audio file."""
    import numpy as np
    import soundfile as sf

    try:
        data, sr = sf.read(str(audio_path), dtype='float32')
        duration = len(data) / sr

        # Calculate basic stats
        rms = np.sqrt(np.mean(data ** 2))
        peak = np.max(np.abs(data))
        db_rms = 20 * np.log10(rms + 1e-10)
        db_peak = 20 * np.log10(peak + 1e-10)

        return {
            "duration_sec": duration,
            "samples": len(data),
            "sample_rate": sr,
            "channels": 1 if data.ndim == 1 else data.shape[1],
            "rms_db": db_rms,
            "peak_db": db_peak,
            "file_size": audio_path.stat().st_size,
        }
    except Exception as e:
        return {"error": str(e)}


def run_enhancement_test(
    media_path: Path,
    backend: str,
    model: Optional[str],
    output_path: Optional[Path],
    start_sec: Optional[float],
    duration_sec: Optional[float],
) -> int:
    """
    Run the speech enhancement test.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    import numpy as np
    import soundfile as sf
    from whisperjav.modules.speech_enhancement import (
        SpeechEnhancerFactory,
        EnhancementResult,
    )

    print_header("Speech Enhancement Test Suite")
    print_stat("Input File", str(media_path))
    print_stat("Backend", backend)
    print_stat("Model", model or "(default)")

    # Check backend availability
    print_header("Step 1: Backend Validation")
    available, hint = SpeechEnhancerFactory.is_backend_available(backend)
    if not available:
        print(f"\n  ERROR: Backend '{backend}' is not available.")
        if hint:
            print(f"  Install: {hint}")
        return 1
    print_stat("Status", "Backend is available")

    # Create enhancer through factory (black-box test)
    print_header("Step 2: Create Enhancer (Factory)")

    create_start = time.time()
    try:
        kwargs = {}
        if model:
            kwargs["model"] = model

        enhancer = SpeechEnhancerFactory.create(backend, **kwargs)
        create_time = time.time() - create_start

        print_stat("Factory Create Time", f"{create_time:.3f}s")
        print_stat("Enhancer Name", enhancer.name)
        print_stat("Display Name", enhancer.display_name)
        print_stat("Preferred Sample Rate", f"{enhancer.get_preferred_sample_rate()} Hz")
        print_stat("Output Sample Rate", f"{enhancer.get_output_sample_rate()} Hz")
        print_stat("Supported Models", ", ".join(enhancer.get_supported_models()))

    except Exception as e:
        print(f"\n  ERROR: Failed to create enhancer: {e}")
        return 1

    # Get preferred sample rate for extraction
    preferred_sr = enhancer.get_preferred_sample_rate()

    # Extract audio from media file
    print_header("Step 3: Audio Extraction")

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        extracted_audio_path = Path(tmp.name)

    try:
        print_stat("Extracting to", str(extracted_audio_path))
        print_stat("Target Sample Rate", f"{preferred_sr} Hz")
        if start_sec is not None:
            print_stat("Start Time", f"{start_sec}s")
        if duration_sec is not None:
            print_stat("Duration", f"{duration_sec}s")

        extract_start = time.time()
        success, msg = extract_audio(
            media_path, extracted_audio_path, preferred_sr,
            start_sec, duration_sec
        )
        extract_time = time.time() - extract_start

        if not success:
            print(f"\n  ERROR: {msg}")
            enhancer.cleanup()
            return 1

        print_stat("Extraction Time", f"{extract_time:.3f}s")

        # Get input audio info
        input_info = get_audio_info(extracted_audio_path, preferred_sr)
        if "error" in input_info:
            print(f"\n  ERROR: {input_info['error']}")
            enhancer.cleanup()
            return 1

        print_stat("Input Duration", format_duration(input_info["duration_sec"]))
        print_stat("Input Samples", f"{input_info['samples']:,}")
        print_stat("Input File Size", format_size(input_info["file_size"]))
        print_stat("Input RMS Level", f"{input_info['rms_db']:.1f} dB")
        print_stat("Input Peak Level", f"{input_info['peak_db']:.1f} dB")

        # Load audio for enhancement
        print_header("Step 4: Speech Enhancement")

        audio_data, sr = sf.read(str(extracted_audio_path), dtype='float32')
        print_stat("Loaded Audio Shape", str(audio_data.shape))
        print_stat("Loaded Audio dtype", str(audio_data.dtype))
        print_stat("Sample Rate", f"{sr} Hz")

        # Run enhancement (the core test)
        print("\n  Running enhancement...")
        enhance_start = time.time()

        try:
            result: EnhancementResult = enhancer.enhance(audio_data, sr)
            enhance_time = time.time() - enhance_start

        except Exception as e:
            print(f"\n  ERROR during enhancement: {e}")
            import traceback
            traceback.print_exc()
            enhancer.cleanup()
            return 1

        # Report enhancement results
        print_header("Step 5: Enhancement Results")

        print_stat("Success", "YES" if result.success else "NO")
        if not result.success:
            print_stat("Error Message", result.error_message or "Unknown error")

        print_stat("Method", result.method)
        print_stat("Processing Time", f"{result.processing_time_sec:.3f}s")
        print_stat("Total Enhance Time", f"{enhance_time:.3f}s")
        print_stat("Output Duration", format_duration(result.duration_sec))
        print_stat("Output Samples", f"{result.num_samples:,}")
        print_stat("Output Sample Rate", f"{result.sample_rate} Hz")

        if result.parameters:
            print_stat("Parameters", "")
            for k, v in result.parameters.items():
                print_stat(f"  {k}", str(v))

        if result.metadata:
            print_stat("Metadata", "")
            for k, v in result.metadata.items():
                print_stat(f"  {k}", str(v))

        # Calculate output audio stats
        output_rms = np.sqrt(np.mean(result.audio ** 2))
        output_peak = np.max(np.abs(result.audio))
        output_db_rms = 20 * np.log10(output_rms + 1e-10)
        output_db_peak = 20 * np.log10(output_peak + 1e-10)

        print_stat("Output RMS Level", f"{output_db_rms:.1f} dB")
        print_stat("Output Peak Level", f"{output_db_peak:.1f} dB")

        # Compare input vs output
        print_header("Step 6: Input vs Output Comparison")
        print_stat("Duration Change", f"{result.duration_sec - input_info['duration_sec']:.3f}s")
        print_stat("RMS Change", f"{output_db_rms - input_info['rms_db']:+.1f} dB")
        print_stat("Peak Change", f"{output_db_peak - input_info['peak_db']:+.1f} dB")

        # Calculate processing speed
        realtime_factor = input_info["duration_sec"] / enhance_time if enhance_time > 0 else 0
        print_stat("Realtime Factor", f"{realtime_factor:.2f}x")

        # Save output file
        print_header("Step 7: Save Output")

        if output_path is None:
            # Auto-generate output path
            stem = media_path.stem
            output_path = media_path.parent / f"{stem}_enhanced_{backend}.wav"

        print_stat("Output File", str(output_path))

        try:
            sf.write(str(output_path), result.audio, result.sample_rate)
            output_size = output_path.stat().st_size
            print_stat("Output File Size", format_size(output_size))
            print_stat("Status", "SAVED SUCCESSFULLY")
        except Exception as e:
            print(f"\n  ERROR saving output: {e}")
            enhancer.cleanup()
            return 1

        # Cleanup
        print_header("Step 8: Cleanup")
        cleanup_start = time.time()
        enhancer.cleanup()
        cleanup_time = time.time() - cleanup_start
        print_stat("Cleanup Time", f"{cleanup_time:.3f}s")
        print_stat("Status", "Resources released")

    finally:
        # Remove temp extracted audio
        try:
            extracted_audio_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Final summary
    print_header("Test Summary")
    total_time = extract_time + enhance_time + cleanup_time
    print_stat("Backend", backend)
    print_stat("Model", model or "(default)")
    print_stat("Enhancement Success", "YES" if result.success else "NO")
    print_stat("Total Test Time", format_duration(total_time))
    print_stat("Output File", str(output_path))
    print("\n  Listen to the output file to verify enhancement quality.\n")

    return 0 if result.success else 1


def main():
    parser = argparse.ArgumentParser(
        description="Speech Enhancement Test Suite - Black-box testing for the speech enhancer module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings (clearvoice, MossFormer2_SE_48K)
  python tests/test_speech_enhancement_cli.py video.mp4

  # Test with specific model
  python tests/test_speech_enhancement_cli.py video.mp4 --model FRCRN_SE_16K

  # Test BS-RoFormer backend
  python tests/test_speech_enhancement_cli.py video.mp4 --backend bs-roformer

  # Extract and test specific segment
  python tests/test_speech_enhancement_cli.py video.mp4 --start 60 --duration 30

  # List available backends
  python tests/test_speech_enhancement_cli.py --list-backends

  # List models for a backend
  python tests/test_speech_enhancement_cli.py --list-models clearvoice
        """
    )

    # Positional argument (optional when using --list-*)
    parser.add_argument(
        "media_file",
        nargs="?",
        help="Path to media file (video or audio)"
    )

    # Backend selection
    parser.add_argument(
        "--backend", "-b",
        default="clearvoice",
        help="Speech enhancement backend (default: clearvoice)"
    )

    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model to use (default: backend's default model)"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output WAV file path (default: auto-generate)"
    )

    # Segment extraction
    parser.add_argument(
        "--start", "-s",
        type=float,
        default=None,
        help="Start time in seconds (default: from beginning)"
    )

    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=None,
        help="Duration in seconds (default: entire file)"
    )

    # Discovery options
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available backends and exit"
    )

    parser.add_argument(
        "--list-models",
        metavar="BACKEND",
        help="List available models for a backend and exit"
    )

    args = parser.parse_args()

    # Handle discovery commands
    if args.list_backends:
        list_backends()
        return 0

    if args.list_models:
        list_models(args.list_models)
        return 0

    # Require media file for enhancement test
    if not args.media_file:
        parser.error("media_file is required (or use --list-backends / --list-models)")

    media_path = Path(args.media_file)
    if not media_path.exists():
        print(f"ERROR: File not found: {media_path}")
        return 1

    output_path = Path(args.output) if args.output else None

    return run_enhancement_test(
        media_path=media_path,
        backend=args.backend,
        model=args.model,
        output_path=output_path,
        start_sec=args.start,
        duration_sec=args.duration,
    )


if __name__ == "__main__":
    sys.exit(main())
