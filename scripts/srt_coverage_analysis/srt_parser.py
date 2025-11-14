"""
SRT Parser Module
=================

Parse SRT subtitle files into structured Segment objects for analysis.
"""

import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Segment:
    """Represents a single subtitle segment."""
    index: int
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str

    @property
    def duration(self) -> float:
        """Calculate segment duration in seconds."""
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        """Calculate temporal midpoint of segment."""
        return (self.start + self.end) / 2.0

    def __repr__(self):
        return f"Segment({self.index}, {self.start:.2f}s-{self.end:.2f}s, '{self.text[:30]}...')"


def parse_srt_timestamp(timestamp: str) -> float:
    """
    Parse SRT timestamp format to seconds.

    Args:
        timestamp: SRT timestamp string (e.g., "00:01:23,456")

    Returns:
        Time in seconds as float

    Examples:
        >>> parse_srt_timestamp("00:00:05,000")
        5.0
        >>> parse_srt_timestamp("00:01:30,500")
        90.5
    """
    # Format: HH:MM:SS,mmm
    pattern = r'(\d{2}):(\d{2}):(\d{2}),(\d{3})'
    match = re.match(pattern, timestamp.strip())

    if not match:
        raise ValueError(f"Invalid SRT timestamp format: {timestamp}")

    hours, minutes, seconds, milliseconds = map(int, match.groups())

    total_seconds = (
        hours * 3600 +
        minutes * 60 +
        seconds +
        milliseconds / 1000.0
    )

    return total_seconds


def parse_srt_file(srt_path: str) -> List[Segment]:
    """
    Parse an SRT file into a list of Segment objects.

    Args:
        srt_path: Path to the SRT file

    Returns:
        List of Segment objects, sorted by start time

    Raises:
        FileNotFoundError: If SRT file doesn't exist
        ValueError: If SRT format is invalid
    """
    srt_file = Path(srt_path)

    if not srt_file.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    # Read file with UTF-8 encoding (standard for SRT files)
    try:
        content = srt_file.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback to UTF-16 if UTF-8 fails
        try:
            content = srt_file.read_text(encoding='utf-16')
        except UnicodeDecodeError:
            # Last resort: latin-1 (never fails)
            content = srt_file.read_text(encoding='latin-1')

    # Split into subtitle blocks (separated by blank lines)
    blocks = re.split(r'\n\s*\n', content.strip())

    segments = []

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')

        if len(lines) < 3:
            # Invalid block (needs: index, timestamp, text)
            continue

        try:
            # Parse index
            index = int(lines[0].strip())

            # Parse timestamp line: "00:00:05,000 --> 00:00:10,000"
            timestamp_line = lines[1].strip()
            timestamp_match = re.match(r'(.+?)\s*-->\s*(.+)', timestamp_line)

            if not timestamp_match:
                continue

            start_str, end_str = timestamp_match.groups()
            start = parse_srt_timestamp(start_str)
            end = parse_srt_timestamp(end_str)

            # Parse text (may span multiple lines)
            text = '\n'.join(lines[2:]).strip()

            # Create segment
            segment = Segment(
                index=index,
                start=start,
                end=end,
                text=text
            )

            segments.append(segment)

        except (ValueError, IndexError) as e:
            # Skip malformed blocks
            print(f"Warning: Skipping malformed SRT block: {e}")
            continue

    # Sort by start time (in case SRT file is not chronological)
    segments.sort(key=lambda s: s.start)

    return segments


def calculate_srt_statistics(segments: List[Segment]) -> dict:
    """
    Calculate basic statistics for an SRT file.

    Args:
        segments: List of parsed segments

    Returns:
        Dictionary with statistics:
            - total_lines: Number of subtitle lines
            - total_duration: Sum of all segment durations (seconds)
            - total_duration_formatted: Formatted as HH:MM:SS.mmm
            - average_duration: Average segment duration (seconds)
            - min_duration: Shortest segment duration (seconds)
            - max_duration: Longest segment duration (seconds)
            - timeline_start: First segment start time (seconds)
            - timeline_end: Last segment end time (seconds)
            - timeline_span: Total timeline coverage (seconds)
    """
    if not segments:
        return {
            'total_lines': 0,
            'total_duration': 0.0,
            'total_duration_formatted': '00:00:00.000',
            'average_duration': 0.0,
            'min_duration': 0.0,
            'max_duration': 0.0,
            'timeline_start': 0.0,
            'timeline_end': 0.0,
            'timeline_span': 0.0,
        }

    total_lines = len(segments)
    total_duration = sum(seg.duration for seg in segments)
    durations = [seg.duration for seg in segments]

    # Format total duration as HH:MM:SS.mmm
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = total_duration % 60
    duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    # Timeline coverage
    timeline_start = segments[0].start
    timeline_end = segments[-1].end
    timeline_span = timeline_end - timeline_start

    return {
        'total_lines': total_lines,
        'total_duration': total_duration,
        'total_duration_formatted': duration_formatted,
        'average_duration': total_duration / total_lines if total_lines > 0 else 0.0,
        'min_duration': min(durations) if durations else 0.0,
        'max_duration': max(durations) if durations else 0.0,
        'timeline_start': timeline_start,
        'timeline_end': timeline_end,
        'timeline_span': timeline_span,
    }


def validate_srt_segments(segments: List[Segment]) -> dict:
    """
    Validate parsed SRT segments and return statistics.

    Args:
        segments: List of parsed segments

    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'total_segments': len(segments),
        'valid_segments': 0,
        'zero_duration': 0,
        'negative_duration': 0,
        'overlapping_segments': 0,
        'gaps': 0,
        'total_duration': 0.0,
    }

    for i, seg in enumerate(segments):
        # Check duration
        if seg.duration == 0:
            stats['zero_duration'] += 1
        elif seg.duration < 0:
            stats['negative_duration'] += 1
        else:
            stats['valid_segments'] += 1
            stats['total_duration'] += seg.duration

        # Check for overlaps with next segment
        if i < len(segments) - 1:
            next_seg = segments[i + 1]
            if seg.end > next_seg.start:
                stats['overlapping_segments'] += 1
            elif next_seg.start - seg.end > 1.0:
                stats['gaps'] += 1

    return stats


if __name__ == "__main__":
    # Test module
    import sys

    if len(sys.argv) < 2:
        print("Usage: python srt_parser.py <srt_file>")
        sys.exit(1)

    srt_path = sys.argv[1]
    segments = parse_srt_file(srt_path)
    stats = validate_srt_segments(segments)

    print(f"\nParsed {len(segments)} segments from {srt_path}")
    print(f"\nValidation Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if segments:
        print(f"\nFirst segment: {segments[0]}")
        print(f"Last segment: {segments[-1]}")
