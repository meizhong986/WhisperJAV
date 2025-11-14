"""
Media Extractor Module
======================

Extract media chunks for missing/partial segments using FFmpeg.
"""

import subprocess
from pathlib import Path
from typing import Optional
from .srt_parser import Segment


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as FFmpeg timestamp (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds

    Returns:
        FFmpeg-compatible timestamp string

    Examples:
        >>> format_timestamp(90.5)
        '00:01:30.500'
        >>> format_timestamp(3661.123)
        '01:01:01.123'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def extract_media_chunk(
    media_path: str,
    segment: Segment,
    output_path: str,
    padding_seconds: float = 1.0,
    status: str = "missing"
) -> Optional[str]:
    """
    Extract a media chunk for a given segment using FFmpeg.

    Args:
        media_path: Path to source media file
        segment: Segment to extract
        output_path: Directory where chunk should be saved
        padding_seconds: Padding before and after segment (default 1.0s)
        status: Status suffix for filename ("missing" or "partial")

    Returns:
        Path to extracted chunk, or None if extraction failed

    FFmpeg command:
        ffmpeg -i input.mp4 -ss START -t DURATION -c copy output.mp4

    Using -c copy for fast extraction without re-encoding.
    """
    media_file = Path(media_path)
    output_dir = Path(output_path)

    if not media_file.exists():
        print(f"Error: Media file not found: {media_path}")
        return None

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate padded start and duration
    start_time = max(0.0, segment.start - padding_seconds)
    end_time = segment.end + padding_seconds
    duration = end_time - start_time

    # Format timestamps
    start_timestamp = format_timestamp(start_time)

    # Generate output filename
    # Format: seg_{index:04d}_{MM-SS-MS}_{status}.{ext}
    minutes = int(segment.start // 60)
    seconds = int(segment.start % 60)
    milliseconds = int((segment.start % 1) * 1000)
    time_str = f"{minutes:02d}-{seconds:02d}-{milliseconds:03d}"

    # Preserve original file extension
    extension = media_file.suffix
    output_filename = f"seg_{segment.index:04d}_{time_str}_{status}{extension}"
    output_filepath = output_dir / output_filename

    # Build FFmpeg command
    # -ss: start time
    # -t: duration
    # -i: input file
    # -c copy: copy streams without re-encoding (fast)
    # -avoid_negative_ts make_zero: handle timestamp edge cases
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if exists
        '-ss', start_timestamp,
        '-i', str(media_file),
        '-t', str(duration),
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        str(output_filepath)
    ]

    try:
        # Run FFmpeg with suppressed output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60  # 1 minute timeout
        )

        if result.returncode == 0:
            # Verify output file exists and has content
            if output_filepath.exists() and output_filepath.stat().st_size > 0:
                return str(output_filepath)
            else:
                print(f"Error: FFmpeg completed but output file is empty: {output_filepath}")
                return None
        else:
            print(f"Error: FFmpeg failed for segment {segment.index}")
            print(f"  Command: {' '.join(cmd)}")
            print(f"  Error: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print(f"Error: FFmpeg timeout for segment {segment.index}")
        return None
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in PATH.")
        return None
    except Exception as e:
        print(f"Error extracting segment {segment.index}: {e}")
        return None


def extract_multiple_chunks(
    media_path: str,
    segments: list,
    coverage_results: list,
    output_path: str,
    padding_seconds: float = 1.0
) -> dict:
    """
    Extract media chunks for multiple segments.

    Args:
        media_path: Path to source media file
        segments: List of Segment objects to extract
        coverage_results: List of CoverageResult objects (for status)
        output_path: Directory where chunks should be saved
        padding_seconds: Padding before and after segments

    Returns:
        Dictionary with extraction statistics and file paths
    """
    # Create mapping from segment index to coverage result
    coverage_map = {r.ref_segment.index: r for r in coverage_results}

    extracted = {
        'total': len(segments),
        'successful': 0,
        'failed': 0,
        'missing_files': [],
        'partial_files': [],
        'failed_indices': [],
    }

    for segment in segments:
        # Determine status from coverage result
        coverage_result = coverage_map.get(segment.index)
        if coverage_result:
            status = coverage_result.status.lower()
        else:
            status = "unknown"

        # Extract chunk
        output_file = extract_media_chunk(
            media_path=media_path,
            segment=segment,
            output_path=output_path,
            padding_seconds=padding_seconds,
            status=status
        )

        if output_file:
            extracted['successful'] += 1
            if status == "missing":
                extracted['missing_files'].append(output_file)
            elif status == "partial":
                extracted['partial_files'].append(output_file)
        else:
            extracted['failed'] += 1
            extracted['failed_indices'].append(segment.index)

    return extracted


if __name__ == "__main__":
    # Test module
    import sys
    from .srt_parser import parse_srt_file

    if len(sys.argv) < 4:
        print("Usage: python media_extractor.py <media_file> <srt_file> <output_dir>")
        sys.exit(1)

    media_path = sys.argv[1]
    srt_path = sys.argv[2]
    output_dir = sys.argv[3]

    print(f"Loading SRT: {srt_path}")
    segments = parse_srt_file(srt_path)
    print(f"  Loaded {len(segments)} segments")

    if segments:
        # Extract first segment as test
        print(f"\nExtracting first segment as test:")
        print(f"  {segments[0]}")

        output_file = extract_media_chunk(
            media_path=media_path,
            segment=segments[0],
            output_path=output_dir,
            padding_seconds=1.0,
            status="test"
        )

        if output_file:
            print(f"\n✓ Successfully extracted to: {output_file}")
        else:
            print(f"\n✗ Extraction failed")
