#!/usr/bin/env python3
"""
SRT Comparison Utility - Scientific comparison of subtitle files.

Compares two or more SRT files across multiple dimensions:
- Time coverage and density
- Character and word counts
- Duration statistics
- Overlap analysis

Outputs:
- Tally table with all metrics
- Gantt-chart style timeline visualization

Usage:
    python srt_compare.py file1.srt file2.srt [file3.srt ...]
    python srt_compare.py --help
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Dict, Tuple, Optional
import shutil


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SRTEntry:
    """Represents a single subtitle entry."""
    index: int
    start_ms: int  # Start time in milliseconds
    end_ms: int    # End time in milliseconds
    text: str      # Subtitle text (may contain newlines)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0

    @property
    def char_count(self) -> int:
        return len(self.text.replace('\n', ''))

    @property
    def word_count(self) -> int:
        # Handle both Japanese (no spaces) and English (spaces)
        text = self.text.replace('\n', ' ')
        # Count Japanese characters as individual "words" for CJK
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or
                                            '\u3040' <= c <= '\u309f' or
                                            '\u30a0' <= c <= '\u30ff')
        # Count space-separated words for non-CJK
        non_cjk = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', ' ', text)
        space_words = len([w for w in non_cjk.split() if w.strip()])
        return cjk_count + space_words


@dataclass
class SRTFile:
    """Represents a parsed SRT file with computed statistics."""
    path: Path
    entries: List[SRTEntry] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def short_name(self) -> str:
        """Shortened name for display (max 20 chars)."""
        name = self.path.stem
        if len(name) > 20:
            return name[:17] + "..."
        return name

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def total_chars(self) -> int:
        return sum(e.char_count for e in self.entries)

    @property
    def total_words(self) -> int:
        return sum(e.word_count for e in self.entries)

    @property
    def total_duration_ms(self) -> int:
        """Sum of all subtitle durations."""
        return sum(e.duration_ms for e in self.entries)

    @property
    def timeline_span_ms(self) -> int:
        """Time from first subtitle start to last subtitle end."""
        if not self.entries:
            return 0
        return self.entries[-1].end_ms - self.entries[0].start_ms

    @property
    def first_timestamp_ms(self) -> int:
        return self.entries[0].start_ms if self.entries else 0

    @property
    def last_timestamp_ms(self) -> int:
        return self.entries[-1].end_ms if self.entries else 0

    @property
    def coverage_percent(self) -> float:
        """Percentage of timeline covered by subtitles."""
        if self.timeline_span_ms == 0:
            return 0.0
        return (self.total_duration_ms / self.timeline_span_ms) * 100

    @property
    def avg_chars_per_entry(self) -> float:
        if not self.entries:
            return 0.0
        return self.total_chars / len(self.entries)

    @property
    def avg_words_per_entry(self) -> float:
        if not self.entries:
            return 0.0
        return self.total_words / len(self.entries)

    @property
    def avg_duration_ms(self) -> float:
        if not self.entries:
            return 0.0
        return mean(e.duration_ms for e in self.entries)

    @property
    def median_duration_ms(self) -> float:
        if not self.entries:
            return 0.0
        return median(e.duration_ms for e in self.entries)

    @property
    def min_duration_ms(self) -> int:
        if not self.entries:
            return 0
        return min(e.duration_ms for e in self.entries)

    @property
    def max_duration_ms(self) -> int:
        if not self.entries:
            return 0
        return max(e.duration_ms for e in self.entries)

    @property
    def stdev_duration_ms(self) -> float:
        if len(self.entries) < 2:
            return 0.0
        return stdev(e.duration_ms for e in self.entries)

    @property
    def entries_per_minute(self) -> float:
        """Subtitle density: entries per minute of timeline."""
        if self.timeline_span_ms == 0:
            return 0.0
        minutes = self.timeline_span_ms / 60000.0
        return len(self.entries) / minutes

    @property
    def avg_gap_ms(self) -> float:
        """Average gap between consecutive subtitles."""
        if len(self.entries) < 2:
            return 0.0
        gaps = []
        for i in range(1, len(self.entries)):
            gap = self.entries[i].start_ms - self.entries[i-1].end_ms
            gaps.append(max(0, gap))  # Ignore overlaps for gap calculation
        return mean(gaps) if gaps else 0.0

    def get_coverage_at(self, time_ms: int) -> bool:
        """Check if any subtitle covers the given time."""
        for entry in self.entries:
            if entry.start_ms <= time_ms <= entry.end_ms:
                return True
            if entry.start_ms > time_ms:
                break  # Entries are sorted, no need to continue
        return False


# =============================================================================
# Parser
# =============================================================================

def parse_timestamp(ts: str) -> int:
    """Parse SRT timestamp to milliseconds."""
    # Format: HH:MM:SS,mmm or HH:MM:SS.mmm
    ts = ts.replace('.', ',')
    match = re.match(r'(\d+):(\d+):(\d+),(\d+)', ts.strip())
    if not match:
        raise ValueError(f"Invalid timestamp format: {ts}")
    h, m, s, ms = map(int, match.groups())
    return h * 3600000 + m * 60000 + s * 1000 + ms


def format_timestamp(ms: int) -> str:
    """Format milliseconds as HH:MM:SS,mmm."""
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def format_duration(ms: int) -> str:
    """Format milliseconds as human-readable duration."""
    if ms < 1000:
        return f"{ms}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        m = ms // 60000
        s = (ms % 60000) / 1000
        return f"{m}m {s:.0f}s"


def parse_srt_file(path: Path) -> SRTFile:
    """Parse an SRT file and return SRTFile object."""
    entries = []

    with open(path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # Split by double newline (entry separator)
    # Handle both \n and \r\n
    content = content.replace('\r\n', '\n')
    blocks = re.split(r'\n\n+', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue

        # First line: index
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Second line: timestamps
        ts_match = re.match(r'(.+?)\s*-->\s*(.+)', lines[1])
        if not ts_match:
            continue

        try:
            start_ms = parse_timestamp(ts_match.group(1))
            end_ms = parse_timestamp(ts_match.group(2))
        except ValueError:
            continue

        # Remaining lines: text
        text = '\n'.join(lines[2:]) if len(lines) > 2 else ""

        entries.append(SRTEntry(
            index=index,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text
        ))

    # Sort by start time
    entries.sort(key=lambda e: e.start_ms)

    return SRTFile(path=path, entries=entries)


# =============================================================================
# Comparator
# =============================================================================

@dataclass
class OverlapAnalysis:
    """Results of overlap analysis between two SRT files."""
    file_a: str
    file_b: str
    overlap_time_ms: int       # Time covered by both
    unique_a_time_ms: int      # Time covered only by A
    unique_b_time_ms: int      # Time covered only by B
    overlap_entries_a: int     # Entries in A that overlap with B
    overlap_entries_b: int     # Entries in B that overlap with A
    unique_entries_a: int      # Entries in A with no overlap
    unique_entries_b: int      # Entries in B with no overlap


def compute_overlap(srt_a: SRTFile, srt_b: SRTFile, resolution_ms: int = 100) -> OverlapAnalysis:
    """
    Compute overlap between two SRT files.
    Uses time-sampling at given resolution for accuracy.
    """
    if not srt_a.entries or not srt_b.entries:
        return OverlapAnalysis(
            file_a=srt_a.short_name,
            file_b=srt_b.short_name,
            overlap_time_ms=0,
            unique_a_time_ms=srt_a.total_duration_ms,
            unique_b_time_ms=srt_b.total_duration_ms,
            overlap_entries_a=0,
            overlap_entries_b=0,
            unique_entries_a=len(srt_a.entries),
            unique_entries_b=len(srt_b.entries),
        )

    # Find timeline bounds
    start = min(srt_a.first_timestamp_ms, srt_b.first_timestamp_ms)
    end = max(srt_a.last_timestamp_ms, srt_b.last_timestamp_ms)

    # Sample timeline
    overlap_samples = 0
    unique_a_samples = 0
    unique_b_samples = 0

    for t in range(start, end, resolution_ms):
        a_covers = srt_a.get_coverage_at(t)
        b_covers = srt_b.get_coverage_at(t)

        if a_covers and b_covers:
            overlap_samples += 1
        elif a_covers:
            unique_a_samples += 1
        elif b_covers:
            unique_b_samples += 1

    # Convert samples to time
    overlap_time = overlap_samples * resolution_ms
    unique_a_time = unique_a_samples * resolution_ms
    unique_b_time = unique_b_samples * resolution_ms

    # Count overlapping entries
    def entry_overlaps_file(entry: SRTEntry, srt: SRTFile) -> bool:
        for other in srt.entries:
            if entry.start_ms < other.end_ms and entry.end_ms > other.start_ms:
                return True
        return False

    overlap_entries_a = sum(1 for e in srt_a.entries if entry_overlaps_file(e, srt_b))
    overlap_entries_b = sum(1 for e in srt_b.entries if entry_overlaps_file(e, srt_a))

    return OverlapAnalysis(
        file_a=srt_a.short_name,
        file_b=srt_b.short_name,
        overlap_time_ms=overlap_time,
        unique_a_time_ms=unique_a_time,
        unique_b_time_ms=unique_b_time,
        overlap_entries_a=overlap_entries_a,
        overlap_entries_b=overlap_entries_b,
        unique_entries_a=len(srt_a.entries) - overlap_entries_a,
        unique_entries_b=len(srt_b.entries) - overlap_entries_b,
    )


# =============================================================================
# Renderers
# =============================================================================

def render_tally_table(srt_files: List[SRTFile], overlaps: List[OverlapAnalysis]) -> str:
    """Render comparison tally table."""
    # Get terminal width
    term_width = shutil.get_terminal_size().columns

    # Calculate column widths
    metric_col_width = 24
    file_col_width = max(15, min(25, (term_width - metric_col_width - 10) // (len(srt_files) + 1)))

    # Build header
    lines = []

    # Title
    lines.append("")
    lines.append("=" * min(term_width, 100))
    lines.append("  SRT COMPARISON REPORT")
    lines.append("=" * min(term_width, 100))
    lines.append("")

    # File info
    lines.append("Files analyzed:")
    for i, srt in enumerate(srt_files):
        lines.append(f"  [{i+1}] {srt.name}")
    lines.append("")

    # Helper to format numbers
    def fmt_num(n, decimals=0):
        if isinstance(n, float):
            if decimals == 0:
                return f"{n:,.0f}"
            return f"{n:,.{decimals}f}"
        return f"{n:,}"

    def fmt_diff(a, b, decimals=0, invert=False):
        """Format difference with arrow. invert=True means lower is better."""
        diff = b - a
        if decimals == 0:
            diff_str = f"{diff:+,.0f}" if isinstance(diff, float) else f"{diff:+,}"
        else:
            diff_str = f"{diff:+,.{decimals}f}"

        if diff > 0:
            arrow = "+" if not invert else "-"
        elif diff < 0:
            arrow = "-" if not invert else "+"
        else:
            arrow = "="
        return f"({arrow}) {diff_str}"

    # Table header
    header_parts = [f"{'Metric':<{metric_col_width}}"]
    for srt in srt_files:
        header_parts.append(f"{srt.short_name:>{file_col_width}}")
    if len(srt_files) == 2:
        header_parts.append(f"{'Difference':>{file_col_width}}")

    separator = "-" * len("  ".join(header_parts))

    lines.append(separator)
    lines.append("  ".join(header_parts))
    lines.append(separator)

    # Metrics
    metrics = [
        # (name, getter, decimals, invert_for_diff)
        ("Entries", lambda s: s.entry_count, 0, False),
        ("Total Characters", lambda s: s.total_chars, 0, False),
        ("Total Words", lambda s: s.total_words, 0, False),
        ("", None, 0, False),  # Separator
        ("Timeline Span", lambda s: format_duration(s.timeline_span_ms), None, False),
        ("Total Sub Duration", lambda s: format_duration(s.total_duration_ms), None, False),
        ("Coverage %", lambda s: s.coverage_percent, 1, False),
        ("", None, 0, False),  # Separator
        ("Avg Chars/Entry", lambda s: s.avg_chars_per_entry, 1, False),
        ("Avg Words/Entry", lambda s: s.avg_words_per_entry, 1, False),
        ("Entries/Minute", lambda s: s.entries_per_minute, 2, False),
        ("", None, 0, False),  # Separator
        ("Avg Duration", lambda s: format_duration(int(s.avg_duration_ms)), None, False),
        ("Median Duration", lambda s: format_duration(int(s.median_duration_ms)), None, False),
        ("Shortest Sub", lambda s: format_duration(s.min_duration_ms), None, False),
        ("Longest Sub", lambda s: format_duration(s.max_duration_ms), None, False),
        ("Std Dev Duration", lambda s: format_duration(int(s.stdev_duration_ms)), None, False),
        ("", None, 0, False),  # Separator
        ("Avg Gap Between", lambda s: format_duration(int(s.avg_gap_ms)), None, True),
    ]

    for name, getter, decimals, invert in metrics:
        if getter is None:
            lines.append("")
            continue

        row_parts = [f"{name:<{metric_col_width}}"]
        values = []

        for srt in srt_files:
            val = getter(srt)
            values.append(val)
            if decimals is None:
                row_parts.append(f"{val:>{file_col_width}}")
            else:
                row_parts.append(f"{fmt_num(val, decimals):>{file_col_width}}")

        # Add difference column for 2-file comparison
        if len(srt_files) == 2 and decimals is not None:
            diff_str = fmt_diff(values[0], values[1], decimals, invert)
            row_parts.append(f"{diff_str:>{file_col_width}}")
        elif len(srt_files) == 2:
            row_parts.append(f"{'--':>{file_col_width}}")

        lines.append("  ".join(row_parts))

    lines.append(separator)

    # Overlap analysis (for 2-file comparison)
    if len(srt_files) == 2 and overlaps:
        overlap = overlaps[0]
        lines.append("")
        lines.append("OVERLAP ANALYSIS")
        lines.append("-" * 50)
        lines.append(f"  Time covered by both:     {format_duration(overlap.overlap_time_ms)}")
        lines.append(f"  Time unique to [{srt_files[0].short_name}]: {format_duration(overlap.unique_a_time_ms)}")
        lines.append(f"  Time unique to [{srt_files[1].short_name}]: {format_duration(overlap.unique_b_time_ms)}")
        lines.append("")
        lines.append(f"  Overlapping entries in [{srt_files[0].short_name}]: {overlap.overlap_entries_a}/{srt_files[0].entry_count}")
        lines.append(f"  Overlapping entries in [{srt_files[1].short_name}]: {overlap.overlap_entries_b}/{srt_files[1].entry_count}")
        lines.append(f"  Unique entries in [{srt_files[0].short_name}]:      {overlap.unique_entries_a}")
        lines.append(f"  Unique entries in [{srt_files[1].short_name}]:      {overlap.unique_entries_b}")

    lines.append("")

    return "\n".join(lines)


def render_gantt_chart(srt_files: List[SRTFile], width: int = 80) -> str:
    """Render ASCII Gantt-chart style timeline visualization."""
    if not srt_files:
        return ""

    # Find global timeline bounds
    all_entries = [e for srt in srt_files for e in srt.entries]
    if not all_entries:
        return "No subtitle entries to visualize.\n"

    global_start = min(e.start_ms for e in all_entries)
    global_end = max(e.end_ms for e in all_entries)
    global_span = global_end - global_start

    if global_span == 0:
        return "Timeline has zero duration.\n"

    # Calculate chart width (leave room for labels)
    label_width = max(len(srt.short_name) for srt in srt_files) + 2
    chart_width = width - label_width - 4

    if chart_width < 20:
        chart_width = 20

    lines = []
    lines.append("")
    lines.append("TIMELINE COVERAGE")
    lines.append("-" * width)

    # Time axis
    def time_to_pos(ms: int) -> int:
        return int((ms - global_start) / global_span * chart_width)

    # Render time labels
    time_labels = "  " + " " * label_width
    for i in range(5):
        pos = int(chart_width * i / 4)
        time_ms = global_start + int(global_span * i / 4)
        time_str = format_timestamp(time_ms)[:8]  # HH:MM:SS
        # Center the label at position
        if i == 0:
            time_labels += time_str
        elif i == 4:
            time_labels = time_labels.rstrip() + " " + time_str
        else:
            current_len = len(time_labels) - label_width - 2
            padding = pos - current_len - len(time_str) // 2
            if padding > 0:
                time_labels += " " * padding + time_str

    lines.append(time_labels)

    # Render axis line
    axis_line = "  " + " " * label_width + "|" + "-" * (chart_width - 2) + "|"
    lines.append(axis_line)

    # Render each file's coverage
    chars = {
        'full': '#',
        'partial': '+',
        'empty': '-',
        'overlap': '=',
    }

    for srt in srt_files:
        # Build coverage array
        coverage = [False] * chart_width
        for entry in srt.entries:
            start_pos = time_to_pos(entry.start_ms)
            end_pos = time_to_pos(entry.end_ms)
            for p in range(max(0, start_pos), min(chart_width, end_pos + 1)):
                coverage[p] = True

        # Build bar
        bar = "".join(chars['full'] if c else chars['empty'] for c in coverage)

        # Calculate coverage percentage
        covered = sum(coverage)
        pct = (covered / chart_width) * 100

        label = f"{srt.short_name:>{label_width}}"
        lines.append(f"  {label} |{bar}| {pct:.0f}%")

    # Render overlap bar (for 2 files)
    if len(srt_files) == 2:
        coverage_a = [False] * chart_width
        coverage_b = [False] * chart_width

        for entry in srt_files[0].entries:
            start_pos = time_to_pos(entry.start_ms)
            end_pos = time_to_pos(entry.end_ms)
            for p in range(max(0, start_pos), min(chart_width, end_pos + 1)):
                coverage_a[p] = True

        for entry in srt_files[1].entries:
            start_pos = time_to_pos(entry.start_ms)
            end_pos = time_to_pos(entry.end_ms)
            for p in range(max(0, start_pos), min(chart_width, end_pos + 1)):
                coverage_b[p] = True

        # Build overlap bar
        overlap_bar = []
        for a, b in zip(coverage_a, coverage_b):
            if a and b:
                overlap_bar.append(chars['full'])
            elif a or b:
                overlap_bar.append(chars['partial'])
            else:
                overlap_bar.append(chars['empty'])

        overlap_str = "".join(overlap_bar)
        overlap_count = sum(1 for a, b in zip(coverage_a, coverage_b) if a and b)
        overlap_pct = (overlap_count / chart_width) * 100

        label = f"{'[Overlap]':>{label_width}}"
        lines.append(f"  {label} |{overlap_str}| {overlap_pct:.0f}%")

    lines.append("")

    # Legend
    lines.append("  Legend: # = covered  - = no subtitle  + = partial (one file only)")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scientific comparison of SRT subtitle files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python srt_compare.py file1.srt file2.srt
  python srt_compare.py v1.7.1.srt v1.7.3.srt --width 120
  python srt_compare.py *.srt --no-chart
        """
    )

    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help='SRT files to compare (2 or more)'
    )

    parser.add_argument(
        '--width', '-w',
        type=int,
        default=100,
        help='Output width for charts (default: 100)'
    )

    parser.add_argument(
        '--no-chart',
        action='store_true',
        help='Skip Gantt chart visualization'
    )

    parser.add_argument(
        '--no-table',
        action='store_true',
        help='Skip tally table'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Write output to file instead of stdout'
    )

    args = parser.parse_args()

    # Validate files
    if len(args.files) < 2:
        parser.error("At least 2 SRT files are required for comparison")

    for f in args.files:
        if not f.exists():
            parser.error(f"File not found: {f}")
        if not f.suffix.lower() == '.srt':
            parser.error(f"Not an SRT file: {f}")

    # Parse files
    srt_files = []
    for f in args.files:
        try:
            srt = parse_srt_file(f)
            srt_files.append(srt)
            print(f"Parsed: {f.name} ({srt.entry_count} entries)", file=sys.stderr)
        except Exception as e:
            parser.error(f"Failed to parse {f}: {e}")

    # Compute overlaps (pairwise)
    overlaps = []
    if len(srt_files) == 2:
        overlap = compute_overlap(srt_files[0], srt_files[1])
        overlaps.append(overlap)

    # Generate output
    output_parts = []

    if not args.no_table:
        output_parts.append(render_tally_table(srt_files, overlaps))

    if not args.no_chart:
        output_parts.append(render_gantt_chart(srt_files, args.width))

    output = "\n".join(output_parts)

    # Write output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Report written to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
