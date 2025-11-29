#!/usr/bin/env python3
"""
HuggingFace Transformers Chunked Long-Form ASR Reference Implementation

Uses kotoba-tech/kotoba-whisper-v2.0 for Japanese audio transcription.
Produces SRT subtitle files from audio input.

Based on: https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0#chunked-long-form

Usage:
    python reference_implementation_kotoba_transformers.py audio.wav
    python reference_implementation_kotoba_transformers.py video.mp4 --output subtitles.srt
    python reference_implementation_kotoba_transformers.py audio.mp3 --timestamps word
"""

import argparse
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
from transformers import pipeline


# ============== Constants ==============

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus', '.aac'}

DEFAULT_MODEL_ID = "kotoba-tech/kotoba-whisper-v2.0"
DEFAULT_CHUNK_LENGTH = 15  # Optimal for distil-large-v3
DEFAULT_STRIDE = None  # None = use default (chunk_length_s / 6)
DEFAULT_BATCH_SIZE = 16
DEFAULT_LANGUAGE = "ja"
DEFAULT_TASK = "transcribe"


# ============== Audio Extraction ==============

def is_video_file(path: Path) -> bool:
    """Check if file is a video format."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_audio_file(path: Path) -> bool:
    """Check if file is an audio format."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """
    Extract audio from video using ffmpeg.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted audio

    Returns:
        Path to extracted WAV file
    """
    output_path = output_dir / f"{video_path.stem}_extracted.wav"

    print(f"  Extracting audio from video...")
    print(f"  Input:  {video_path}")
    print(f"  Output: {output_path}")

    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vn',                    # No video
        '-acodec', 'pcm_s16le',   # PCM 16-bit
        '-ar', '16000',           # 16kHz sample rate
        '-ac', '1',               # Mono
        '-y',                     # Overwrite
        str(output_path)
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            text=True
        )
        print(f"  Audio extracted successfully!")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: ffmpeg failed: {e.stderr}", file=sys.stderr)
        raise RuntimeError(f"Failed to extract audio: {e.stderr}")
    except FileNotFoundError:
        print("  ERROR: ffmpeg not found. Please install ffmpeg.", file=sys.stderr)
        raise RuntimeError("ffmpeg not found in PATH")


# ============== SRT Formatting ==============

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds (float)

    Returns:
        SRT formatted timestamp string
    """
    if seconds is None:
        seconds = 0.0

    # Handle negative values
    seconds = max(0.0, seconds)

    # Calculate components
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def chunks_to_srt(chunks: list) -> str:
    """
    Convert pipeline chunks to SRT format string.

    Args:
        chunks: List of dicts with 'text' and 'timestamp' keys

    Returns:
        SRT formatted string
    """
    srt_lines = []
    subtitle_index = 1

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        timestamp = chunk.get("timestamp")

        # Skip empty segments
        if not text:
            continue

        # Handle timestamp tuple
        if timestamp and isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
            start, end = timestamp
            # Handle None values
            if start is None:
                start = 0.0
            if end is None:
                # Estimate end time if missing
                end = start + 2.0
        else:
            # Skip segments without valid timestamps
            continue

        # Add SRT entry
        srt_lines.append(str(subtitle_index))
        srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        srt_lines.append(text)
        srt_lines.append("")  # Blank line between entries

        subtitle_index += 1

    return "\n".join(srt_lines)


# ============== Device & Dtype Detection ==============

def detect_device(requested: str) -> str:
    """
    Detect best available device.

    Args:
        requested: 'auto', 'cuda', or 'cpu'

    Returns:
        Device string for PyTorch
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"
    elif requested == "cuda":
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    else:
        return "cpu"


def detect_dtype(requested: str, device: str) -> torch.dtype:
    """
    Detect best dtype for device.

    Args:
        requested: 'auto', 'float16', 'bfloat16', or 'float32'
        device: Target device string

    Returns:
        PyTorch dtype
    """
    if requested == "auto":
        if "cuda" in device:
            # bfloat16 is generally better for modern GPUs
            return torch.bfloat16
        else:
            return torch.float32
    elif requested == "float16":
        return torch.float16
    elif requested == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float32


# ============== Pipeline Creation ==============

def create_pipeline(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    attn_implementation: str,
    batch_size: int
):
    """
    Create and configure the ASR pipeline.

    Args:
        model_id: HuggingFace model ID
        device: Target device
        torch_dtype: Data type for model
        attn_implementation: Attention implementation (sdpa, flash_attention_2, eager)
        batch_size: Batch size for chunk processing

    Returns:
        Configured pipeline object
    """
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    print(f"  Model:    {model_id}")
    print(f"  Device:   {device}")
    print(f"  Dtype:    {torch_dtype}")
    print(f"  Attention: {attn_implementation}")
    print(f"  Batch:    {batch_size}")

    # Build model_kwargs
    model_kwargs = {}
    if "cuda" in device and attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    start_time = time.time()

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs if model_kwargs else None,
        batch_size=batch_size
    )

    load_time = time.time() - start_time
    print(f"  Loaded in {load_time:.1f}s")

    return pipe


# ============== Transcription ==============

def transcribe(
    audio_path: Path,
    pipe,
    chunk_length_s: int,
    stride_length_s: float,
    language: str,
    task: str,
    timestamp_type: str,
    beam_size: int = 5,
    temperature: float = 0.0,
    compression_ratio_threshold: float = 2.4,
    logprob_threshold: float = -1.0,
    no_speech_threshold: float = 0.6,
    condition_on_previous: bool = True
) -> dict:
    """
    Run transcription with chunked long-form algorithm.

    Args:
        audio_path: Path to audio file
        pipe: ASR pipeline
        chunk_length_s: Chunk length in seconds
        stride_length_s: Overlap between chunks (handles speech at boundaries)
        language: Language code (e.g., 'ja')
        task: 'transcribe' or 'translate'
        timestamp_type: 'segment' or 'word'
        beam_size: Beam size for decoding (higher = more accurate)
        temperature: Sampling temperature (0 = deterministic)
        compression_ratio_threshold: Filter high compression segments
        logprob_threshold: Filter low confidence segments
        no_speech_threshold: Threshold for non-speech detection
        condition_on_previous: Condition on previous text for coherence

    Returns:
        Pipeline result dict with 'text' and 'chunks'
    """
    # Calculate effective stride
    if stride_length_s is None:
        stride_length_s = chunk_length_s / 6  # Default: 1/6 of chunk length

    print(f"\n{'='*60}")
    print("TRANSCRIPTION")
    print(f"{'='*60}")
    print(f"  Audio:     {audio_path.name}")
    print(f"  Chunk:     {chunk_length_s}s")
    print(f"  Stride:    {stride_length_s:.1f}s (overlap at boundaries)")
    print(f"  Language:  {language}")
    print(f"  Task:      {task}")
    print(f"  Timestamps: {timestamp_type}")
    print(f"  Beam size: {beam_size}")
    print(f"  Temperature: {temperature}")

    # Get audio duration for progress estimation
    duration = get_audio_duration(audio_path)
    if duration > 0:
        estimated_chunks = int(duration / chunk_length_s) + 1
        print(f"  Duration:  {duration:.1f}s (~{estimated_chunks} chunks)")

    # Configure generate_kwargs with accuracy parameters
    generate_kwargs = {
        "language": language,
        "task": task,
        "num_beams": beam_size,
        "temperature": temperature,
        "compression_ratio_threshold": compression_ratio_threshold,
        "logprob_threshold": logprob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "condition_on_prev_tokens": condition_on_previous,
    }

    # Configure return_timestamps
    if timestamp_type == "word":
        return_timestamps = "word"
    else:
        return_timestamps = True  # Segment-level

    print(f"\n  Processing...")
    start_time = time.time()

    result = pipe(
        str(audio_path),
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        return_timestamps=return_timestamps,
        generate_kwargs=generate_kwargs
    )

    process_time = time.time() - start_time

    # Print stats
    print(f"\n  Completed in {process_time:.1f}s")
    if duration > 0:
        rtf = process_time / duration
        print(f"  Real-time factor: {rtf:.2f}x")

    num_chunks = len(result.get("chunks", []))
    print(f"  Segments: {num_chunks}")

    return result


# ============== Output ==============

def print_config_summary(args, device: str, torch_dtype: torch.dtype):
    """Print configuration summary."""
    stride_display = f"{args.stride}s" if args.stride else f"auto ({args.chunk_length/6:.1f}s)"

    print(f"\n{'='*60}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Input:      {args.input}")
    print(f"  Output:     {args.output}")
    print(f"  Model:      {args.model_id}")
    print(f"  Device:     {device}")
    print(f"  Dtype:      {torch_dtype}")
    print(f"  Chunk:      {args.chunk_length}s")
    print(f"  Stride:     {stride_display}")
    print(f"  Batch:      {args.batch_size}")
    print(f"  Language:   {args.language}")
    print(f"  Task:       {args.task}")
    print(f"  Timestamps: {args.timestamps}")
    if args.attn:
        print(f"  Attention:  {args.attn}")


def print_final_stats(total_time: float, input_path: Path, output_path: Path, num_segments: int):
    """Print final statistics."""
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Segments:   {num_segments}")
    print(f"  Output:     {output_path}")

    # Show output file size
    if output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        print(f"  Size:       {size_kb:.1f} KB")


# ============== Main ==============

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HuggingFace Transformers Chunked Long-Form ASR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.wav
  %(prog)s video.mp4 --output subtitles.srt
  %(prog)s audio.mp3 --model-id openai/whisper-large-v3
  %(prog)s long_audio.wav --timestamps word --batch-size 1
        """
    )

    # Positional
    parser.add_argument(
        "input",
        type=Path,
        help="Audio or video file to transcribe"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output SRT file path (default: <input>.srt)"
    )

    # Model
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})"
    )

    # Processing
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=DEFAULT_CHUNK_LENGTH,
        help=f"Chunk length in seconds (default: {DEFAULT_CHUNK_LENGTH})"
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=DEFAULT_STRIDE,
        help="Stride/overlap in seconds (default: chunk_length/6). Handles speech at chunk boundaries."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for parallel processing (default: {DEFAULT_BATCH_SIZE})"
    )

    # Language
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE})"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["transcribe", "translate"],
        default=DEFAULT_TASK,
        help=f"Task type (default: {DEFAULT_TASK})"
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Data type (default: auto)"
    )
    parser.add_argument(
        "--attn",
        type=str,
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
        help="Attention implementation (default: sdpa)"
    )

    # Timestamps
    parser.add_argument(
        "--timestamps",
        type=str,
        choices=["segment", "word"],
        default="segment",
        help="Timestamp granularity (default: segment)"
    )

    # Accuracy tuning
    accuracy_group = parser.add_argument_group("Accuracy Tuning")
    accuracy_group.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (higher = more accurate, slower). Default: 5"
    )
    accuracy_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0 = greedy/deterministic). Default: 0.0"
    )
    accuracy_group.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=2.4,
        help="Threshold for filtering high compression ratio segments. Default: 2.4"
    )
    accuracy_group.add_argument(
        "--logprob-threshold",
        type=float,
        default=-1.0,
        help="Log probability threshold for filtering low confidence. Default: -1.0"
    )
    accuracy_group.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.6,
        help="Threshold for detecting non-speech segments. Default: 0.6"
    )
    accuracy_group.add_argument(
        "--condition-on-previous",
        action="store_true",
        default=True,
        help="Condition on previous text (improves coherence). Default: True"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    total_start = time.time()

    # Validate input file
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Set default output path
    if args.output is None:
        args.output = args.input.with_suffix(".srt")

    # Word-level timestamps require batch_size=1
    if args.timestamps == "word" and args.batch_size > 1:
        print(f"NOTE: Word-level timestamps require batch_size=1. Adjusting from {args.batch_size} to 1.")
        args.batch_size = 1

    # Detect device and dtype
    device = detect_device(args.device)
    torch_dtype = detect_dtype(args.dtype, device)

    # Print configuration
    print_config_summary(args, device, torch_dtype)

    # Handle video files
    audio_path = args.input
    temp_dir = None

    if is_video_file(args.input):
        print(f"\n{'='*60}")
        print("VIDEO PROCESSING")
        print(f"{'='*60}")
        temp_dir = Path(tempfile.mkdtemp(prefix="kotoba_asr_"))
        try:
            audio_path = extract_audio(args.input, temp_dir)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    elif not is_audio_file(args.input):
        print(f"WARNING: Unknown file type: {args.input.suffix}")
        print("  Attempting to process as audio...")

    try:
        # Create pipeline
        pipe = create_pipeline(
            model_id=args.model_id,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn,
            batch_size=args.batch_size
        )

        # Run transcription
        result = transcribe(
            audio_path=audio_path,
            pipe=pipe,
            chunk_length_s=args.chunk_length,
            stride_length_s=args.stride,
            language=args.language,
            task=args.task,
            timestamp_type=args.timestamps,
            beam_size=args.beam_size,
            temperature=args.temperature,
            compression_ratio_threshold=args.compression_ratio_threshold,
            logprob_threshold=args.logprob_threshold,
            no_speech_threshold=args.no_speech_threshold,
            condition_on_previous=args.condition_on_previous
        )

        # Convert to SRT
        chunks = result.get("chunks", [])
        srt_content = chunks_to_srt(chunks)

        # Write output
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(srt_content, encoding="utf-8")

        # Print final stats
        total_time = time.time() - total_start
        print_final_stats(total_time, args.input, args.output, len(chunks))

    finally:
        # Cleanup temp files
        if temp_dir and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
