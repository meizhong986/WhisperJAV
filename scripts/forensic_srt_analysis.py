"""
Forensic SRT Analysis: v1.7.1 vs v1.7.3
Identifies patterns in missing subtitles to narrow down root cause hypotheses.
"""

import re
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import defaultdict
import statistics

# Force UTF-8 encoding for Windows console output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry"""
    index: int
    start_time: float  # in seconds
    end_time: float
    duration: float
    text: str

    @property
    def is_short_utterance(self) -> bool:
        """Check if this is a short interjection"""
        short_patterns = ['うん', 'ああ', 'はい', 'ええ', 'そう', 'うーん', 'あー', 'ん']
        return any(pattern in self.text for pattern in short_patterns) and len(self.text) < 10

    @property
    def char_count(self) -> int:
        """Count characters in text"""
        return len(self.text)


def parse_timestamp(timestamp: str) -> float:
    """Convert SRT timestamp to seconds"""
    # Format: 00:00:00,000
    time_part, ms_part = timestamp.split(',')
    h, m, s = map(int, time_part.split(':'))
    ms = int(ms_part)
    return h * 3600 + m * 60 + s + ms / 1000.0


def parse_srt_file(filepath: str) -> List[SubtitleEntry]:
    """Parse an SRT file into SubtitleEntry objects"""
    entries = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to get individual subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0].strip())
            timestamp_line = lines[1].strip()
            text = '\n'.join(lines[2:]).strip()

            # Parse timestamps
            start_str, end_str = timestamp_line.split(' --> ')
            start_time = parse_timestamp(start_str.strip())
            end_time = parse_timestamp(end_str.strip())
            duration = end_time - start_time

            entries.append(SubtitleEntry(index, start_time, end_time, duration, text))
        except Exception as e:
            print(f"Warning: Failed to parse block starting with '{lines[0][:50]}': {e}")
            continue

    return entries


def find_matching_subtitle(ref_entry: SubtitleEntry, result_entries: List[SubtitleEntry],
                           tolerance: float = 2.0) -> Optional[SubtitleEntry]:
    """Find a matching subtitle in result within tolerance window"""
    for result in result_entries:
        # Check if time ranges overlap or are within tolerance
        if (abs(ref_entry.start_time - result.start_time) <= tolerance or
            abs(ref_entry.end_time - result.end_time) <= tolerance or
            (result.start_time <= ref_entry.start_time <= result.end_time) or
            (ref_entry.start_time <= result.start_time <= ref_entry.end_time)):
            return result
    return None


def analyze_gaps(ref_entries: List[SubtitleEntry], result_entries: List[SubtitleEntry]) -> dict:
    """Comprehensive gap analysis"""

    # Find missing subtitles
    missing = []
    matched = []

    for ref in ref_entries:
        match = find_matching_subtitle(ref, result_entries)
        if match:
            matched.append((ref, match))
        else:
            missing.append(ref)

    # Calculate total video duration
    video_duration = max(entry.end_time for entry in ref_entries)

    # Analysis 1: Time Gap Identification
    print("\n" + "="*80)
    print("ANALYSIS 1: TIME GAP IDENTIFICATION")
    print("="*80)
    print(f"\nTotal v1.7.1 entries: {len(ref_entries)}")
    print(f"Total v1.7.3 entries: {len(result_entries)}")
    print(f"Matched entries: {len(matched)}")
    print(f"Missing entries: {len(missing)}")
    print(f"Missing percentage: {len(missing)/len(ref_entries)*100:.1f}%")

    # Analysis 2: Gap Duration Patterns
    print("\n" + "="*80)
    print("ANALYSIS 2: GAP DURATION PATTERNS")
    print("="*80)

    # Sort missing by time
    missing_sorted = sorted(missing, key=lambda x: x.start_time)

    # Find continuous gaps
    continuous_gaps = []
    if missing_sorted:
        gap_start = missing_sorted[0].start_time
        gap_end = missing_sorted[0].end_time
        gap_entries = [missing_sorted[0]]

        for i in range(1, len(missing_sorted)):
            curr = missing_sorted[i]
            # If this entry is within 5 seconds of previous gap end, extend the gap
            if curr.start_time - gap_end <= 5.0:
                gap_end = max(gap_end, curr.end_time)
                gap_entries.append(curr)
            else:
                # Save previous gap and start new one
                continuous_gaps.append((gap_start, gap_end, len(gap_entries), gap_entries))
                gap_start = curr.start_time
                gap_end = curr.end_time
                gap_entries = [curr]

        # Don't forget last gap
        continuous_gaps.append((gap_start, gap_end, len(gap_entries), gap_entries))

    # Sort gaps by duration
    continuous_gaps.sort(key=lambda x: x[1] - x[0], reverse=True)

    print(f"\nFound {len(continuous_gaps)} continuous gap regions")
    print("\nTop 20 largest gaps:")
    print(f"{'Start':<12} {'End':<12} {'Duration':<10} {'Entries':<8} {'Sample Text'}")
    print("-" * 100)

    for gap_start, gap_end, count, entries in continuous_gaps[:20]:
        duration = gap_end - gap_start
        sample_text = entries[0].text[:40] + "..." if len(entries[0].text) > 40 else entries[0].text
        print(f"{gap_start:>10.1f}s  {gap_end:>10.1f}s  {duration:>8.1f}s  {count:<8}  {sample_text}")

    # Gap size distribution
    gap_durations = [gap[1] - gap[0] for gap in continuous_gaps]
    large_gaps = [d for d in gap_durations if d >= 30]
    medium_gaps = [d for d in gap_durations if 10 <= d < 30]
    small_gaps = [d for d in gap_durations if d < 10]

    print(f"\nGap size distribution:")
    print(f"  Large gaps (≥30s): {len(large_gaps)}")
    print(f"  Medium gaps (10-30s): {len(medium_gaps)}")
    print(f"  Small gaps (<10s): {len(small_gaps)}")

    # Analysis 3: Segment Duration Correlation
    print("\n" + "="*80)
    print("ANALYSIS 3: SEGMENT DURATION CORRELATION")
    print("="*80)

    missing_durations = [entry.duration for entry in missing]
    matched_ref_durations = [ref.duration for ref, _ in matched]

    # Categorize by duration
    def categorize_durations(durations):
        very_short = [d for d in durations if d < 1.0]
        short = [d for d in durations if 1.0 <= d < 2.0]
        medium = [d for d in durations if 2.0 <= d < 5.0]
        long = [d for d in durations if d >= 5.0]
        return very_short, short, medium, long

    missing_cats = categorize_durations(missing_durations)
    matched_cats = categorize_durations(matched_ref_durations)

    print("\nDuration distribution:")
    print(f"{'Category':<15} {'Missing':<12} {'Matched':<12} {'Missing %':<12}")
    print("-" * 55)

    categories = [
        ("Very Short (<1s)", missing_cats[0], matched_cats[0]),
        ("Short (1-2s)", missing_cats[1], matched_cats[1]),
        ("Medium (2-5s)", missing_cats[2], matched_cats[2]),
        ("Long (≥5s)", missing_cats[3], matched_cats[3]),
    ]

    for cat_name, miss_list, match_list in categories:
        miss_count = len(miss_list)
        match_count = len(match_list)
        total = miss_count + match_count
        miss_pct = (miss_count / total * 100) if total > 0 else 0
        print(f"{cat_name:<15} {miss_count:<12} {match_count:<12} {miss_pct:.1f}%")

    if missing_durations:
        print(f"\nMissing subtitles statistics:")
        print(f"  Mean duration: {statistics.mean(missing_durations):.2f}s")
        print(f"  Median duration: {statistics.median(missing_durations):.2f}s")
        print(f"  Min duration: {min(missing_durations):.2f}s")
        print(f"  Max duration: {max(missing_durations):.2f}s")

    if matched_ref_durations:
        print(f"\nMatched subtitles statistics:")
        print(f"  Mean duration: {statistics.mean(matched_ref_durations):.2f}s")
        print(f"  Median duration: {statistics.median(matched_ref_durations):.2f}s")
        print(f"  Min duration: {min(matched_ref_durations):.2f}s")
        print(f"  Max duration: {max(matched_ref_durations):.2f}s")

    # Analysis 4: Position Analysis
    print("\n" + "="*80)
    print("ANALYSIS 4: POSITION ANALYSIS (VIDEO SEGMENTS)")
    print("="*80)

    segment_count = 10
    segment_duration = video_duration / segment_count

    # Count missing per segment
    segment_missing = defaultdict(list)
    segment_total = defaultdict(int)

    for entry in ref_entries:
        segment = int(entry.start_time // segment_duration)
        segment_total[segment] += 1

    for entry in missing:
        segment = int(entry.start_time // segment_duration)
        segment_missing[segment].append(entry)

    print(f"\nVideo divided into {segment_count} segments of {segment_duration:.1f}s each")
    print(f"\n{'Segment':<10} {'Time Range':<25} {'Missing':<10} {'Total':<10} {'Missing %':<10}")
    print("-" * 70)

    for seg in range(segment_count):
        seg_start = seg * segment_duration
        seg_end = (seg + 1) * segment_duration
        miss_count = len(segment_missing[seg])
        total = segment_total[seg]
        miss_pct = (miss_count / total * 100) if total > 0 else 0

        time_range = f"{seg_start:>6.0f}s - {seg_end:>6.0f}s"
        print(f"{seg+1:<10} {time_range:<25} {miss_count:<10} {total:<10} {miss_pct:.1f}%")

    # Analysis 5: Content Analysis
    print("\n" + "="*80)
    print("ANALYSIS 5: CONTENT ANALYSIS")
    print("="*80)

    # Character count analysis
    missing_char_counts = [entry.char_count for entry in missing]
    matched_char_counts = [ref.char_count for ref, _ in matched]

    print(f"\nCharacter count statistics:")
    if missing_char_counts:
        print(f"  Missing - Mean: {statistics.mean(missing_char_counts):.1f}, "
              f"Median: {statistics.median(missing_char_counts):.1f}")
    if matched_char_counts:
        print(f"  Matched - Mean: {statistics.mean(matched_char_counts):.1f}, "
              f"Median: {statistics.median(matched_char_counts):.1f}")

    # Short utterance analysis
    missing_short = [e for e in missing if e.is_short_utterance]
    matched_short = [ref for ref, _ in matched if ref.is_short_utterance]

    print(f"\nShort utterance patterns (うん, ああ, はい, etc.):")
    print(f"  Missing short utterances: {len(missing_short)}")
    print(f"  Matched short utterances: {len(matched_short)}")
    if len(missing_short) + len(matched_short) > 0:
        print(f"  Short utterance missing rate: "
              f"{len(missing_short)/(len(missing_short)+len(matched_short))*100:.1f}%")

    # Sample missing texts
    print(f"\nSample of missing subtitle texts (first 20):")
    for i, entry in enumerate(missing[:20], 1):
        text_preview = entry.text[:50] + "..." if len(entry.text) > 50 else entry.text
        print(f"  {i}. [{entry.start_time:.1f}s] {text_preview}")

    # Executive Summary and Recommendations
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)

    summary_points = []

    # Key finding 1: Overall loss
    loss_pct = len(missing)/len(ref_entries)*100
    summary_points.append(f"1. LOSS MAGNITUDE: {len(missing)} subtitles missing ({loss_pct:.1f}% reduction)")

    # Key finding 2: Gap clustering
    large_gap_total = sum(gap[1] - gap[0] for gap in continuous_gaps if gap[1] - gap[0] >= 30)
    if large_gaps:
        summary_points.append(
            f"2. GAP PATTERN: {len(large_gaps)} large gaps (≥30s) account for {large_gap_total:.1f}s of missing content"
        )
    else:
        summary_points.append("2. GAP PATTERN: Missing subtitles are distributed (no large continuous gaps)")

    # Key finding 3: Duration correlation
    very_short_missing_pct = len(missing_cats[0])/(len(missing_cats[0])+len(matched_cats[0]))*100 if (len(missing_cats[0])+len(matched_cats[0])) > 0 else 0
    long_missing_pct = len(missing_cats[3])/(len(missing_cats[3])+len(matched_cats[3]))*100 if (len(missing_cats[3])+len(matched_cats[3])) > 0 else 0

    if very_short_missing_pct > long_missing_pct * 1.5:
        summary_points.append(
            f"3. DURATION BIAS: Very short segments (<1s) disproportionately missing "
            f"({very_short_missing_pct:.1f}% vs {long_missing_pct:.1f}% for long segments)"
        )
    elif long_missing_pct > very_short_missing_pct * 1.5:
        summary_points.append(
            f"3. DURATION BIAS: Long segments (≥5s) disproportionately missing "
            f"({long_missing_pct:.1f}% vs {very_short_missing_pct:.1f}% for short segments)"
        )
    else:
        summary_points.append("3. DURATION BIAS: Missing rate roughly uniform across duration categories")

    # Key finding 4: Position bias
    segment_miss_rates = {}
    for seg in range(segment_count):
        total = segment_total[seg]
        if total > 0:
            segment_miss_rates[seg] = len(segment_missing[seg]) / total * 100

    if segment_miss_rates:
        max_seg = max(segment_miss_rates, key=segment_miss_rates.get)
        min_seg = min(segment_miss_rates, key=segment_miss_rates.get)
        if segment_miss_rates[max_seg] > segment_miss_rates[min_seg] * 1.5:
            summary_points.append(
                f"4. POSITION BIAS: Segment {max_seg+1} has highest loss rate "
                f"({segment_miss_rates[max_seg]:.1f}% vs {segment_miss_rates[min_seg]:.1f}% in segment {min_seg+1})"
            )
        else:
            summary_points.append("4. POSITION BIAS: Loss rate fairly uniform across video timeline")

    # Key finding 5: Content patterns
    short_missing_rate = len(missing_short)/(len(missing_short)+len(matched_short))*100 if (len(missing_short)+len(matched_short)) > 0 else 0
    if short_missing_rate > loss_pct * 1.3:
        summary_points.append(
            f"5. CONTENT PATTERN: Short utterances (うん, ああ, etc.) have elevated loss rate "
            f"({short_missing_rate:.1f}% vs {loss_pct:.1f}% overall)"
        )
    else:
        summary_points.append("5. CONTENT PATTERN: No clear bias against short utterances")

    print()
    for point in summary_points:
        print(point)

    # Recommended test subset
    print("\n" + "="*80)
    print("RECOMMENDED TEST SUBSET (25 minutes)")
    print("="*80)

    # Find a 25-minute window with high missing rate
    test_window = 25 * 60  # 25 minutes in seconds
    best_window = None
    best_missing_count = 0

    # Slide window across video
    for start in range(0, int(video_duration - test_window), 60):
        end = start + test_window
        window_missing = [e for e in missing if start <= e.start_time <= end]
        if len(window_missing) > best_missing_count:
            best_missing_count = len(window_missing)
            best_window = (start, end)

    if best_window:
        start, end = best_window
        window_total = len([e for e in ref_entries if start <= e.start_time <= end])
        print(f"\nRecommended time range: {start//60}:{start%60:02d} - {end//60}:{end%60:02d}")
        print(f"  Contains {best_missing_count} missing subtitles out of {window_total} total")
        print(f"  Missing rate in window: {best_missing_count/window_total*100:.1f}%")
        print(f"  Rationale: Highest concentration of missing subtitles for efficient testing")

    # Hypothesis implications
    print("\n" + "="*80)
    print("HYPOTHESIS IMPLICATIONS")
    print("="*80)

    print("\nBased on the patterns found:")
    print()

    # Analyze which hypothesis is supported
    if len(large_gaps) > 5:
        print("✓ STRONG SUPPORT for VAD over-merging hypothesis:")
        print("  - Large continuous gaps suggest entire segments being skipped")
        print("  - VAD likely merging speech into fewer, longer segments")
    else:
        print("✗ WEAK SUPPORT for VAD over-merging hypothesis:")
        print("  - Missing subtitles are scattered, not in large continuous gaps")

    print()

    if very_short_missing_pct > loss_pct * 1.5:
        print("✓ STRONG SUPPORT for ASR filtering hypothesis:")
        print("  - Very short segments (<1s) disproportionately missing")
        print("  - Suggests post-ASR filtering removing short segments")
    else:
        print("✗ WEAK SUPPORT for ASR filtering hypothesis:")
        print("  - No clear bias against short segments")

    print()

    if short_missing_rate > loss_pct * 1.5:
        print("✓ MODERATE SUPPORT for hallucination removal hypothesis:")
        print("  - Short utterances (うん, ああ) have elevated loss rate")
        print("  - May be incorrectly flagged as hallucinations")
    else:
        print("✗ WEAK SUPPORT for hallucination removal hypothesis:")
        print("  - No clear bias against short utterances")

    print()

    # Check if loss is uniform
    duration_variance = max(
        very_short_missing_pct, long_missing_pct
    ) - min(very_short_missing_pct, long_missing_pct)

    position_variance = max(segment_miss_rates.values()) - min(segment_miss_rates.values()) if segment_miss_rates else 0

    if duration_variance < 10 and position_variance < 10:
        print("✓ MODERATE SUPPORT for scene detection change hypothesis:")
        print("  - Loss is fairly uniform across durations and positions")
        print("  - Suggests systematic reduction in detected speech regions")

    print("\n" + "="*80)

    return {
        'missing': missing,
        'matched': matched,
        'continuous_gaps': continuous_gaps,
        'segment_missing': segment_missing,
    }


def main():
    ref_file = r"C:\BIN\git\WhisperJav_V1_Minami_Edition\test_results\1_7_1-vs-1_7_3\HODV-22019.1_7_1.srt"
    result_file = r"C:\BIN\git\WhisperJav_V1_Minami_Edition\test_results\1_7_1-vs-1_7_3\1.7.3-f\HODV-22019.ja.pass1.srt"

    print("FORENSIC SRT ANALYSIS: v1.7.1 vs v1.7.3")
    print("="*80)
    print(f"\nReference file (v1.7.1): {ref_file}")
    print(f"Result file (v1.7.3): {result_file}")

    print("\nParsing SRT files...")
    ref_entries = parse_srt_file(ref_file)
    result_entries = parse_srt_file(result_file)

    print(f"Parsed {len(ref_entries)} entries from v1.7.1")
    print(f"Parsed {len(result_entries)} entries from v1.7.3")

    analyze_gaps(ref_entries, result_entries)


if __name__ == '__main__':
    main()
