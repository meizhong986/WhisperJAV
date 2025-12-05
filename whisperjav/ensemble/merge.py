"""Merge Engine for combining two-pass SRT results."""

import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from whisperjav.utils.logger import logger


class MergeStrategy(Enum):
    """Available merge strategies."""
    FULL_MERGE = 'full_merge'
    PASS1_PRIMARY = 'pass1_primary'
    PASS2_PRIMARY = 'pass2_primary'
    PASS1_OVERLAP = 'pass1_overlap'
    PASS2_OVERLAP = 'pass2_overlap'
    SMART_MERGE = 'smart_merge'


@dataclass
class Subtitle:
    """Represents a single subtitle entry."""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class MergeEngine:
    """Merges SRT files from two-pass processing."""

    # Overlap threshold: 30% of base subtitle duration
    OVERLAP_THRESHOLD = 0.30

    def __init__(self):
        """Initialize the merge engine."""
        self.strategies = {
            'full_merge': self._merge_full,
            'pass1_primary': self._merge_pass1_primary,
            'pass2_primary': self._merge_pass2_primary,
            'pass1_overlap': self._merge_pass1_overlap,
            'pass2_overlap': self._merge_pass2_overlap,
            'smart_merge': self._merge_smart,
        }

    def merge(
        self,
        srt1_path: Path,
        srt2_path: Path,
        output_path: Path,
        strategy: str = 'smart_merge'
    ) -> Dict[str, Any]:
        """
        Merge two SRT files using the specified strategy.

        Args:
            srt1_path: Path to first SRT (pass 1)
            srt2_path: Path to second SRT (pass 2)
            output_path: Path for merged output
            strategy: Merge strategy name

        Returns:
            Dictionary with merge statistics
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown merge strategy: {strategy}. "
                           f"Available: {list(self.strategies.keys())}")

        # Parse both SRT files
        subs1 = self._parse_srt(srt1_path)
        subs2 = self._parse_srt(srt2_path)

        logger.info(f"Merging {len(subs1)} subtitles from pass 1 with {len(subs2)} from pass 2")

        # Apply merge strategy
        merge_func = self.strategies[strategy]
        merged_subs = merge_func(subs1, subs2)

        # Re-index subtitles
        for i, sub in enumerate(merged_subs, 1):
            sub.index = i

        # Write output
        self._write_srt(merged_subs, output_path)

        stats = {
            'pass1_count': len(subs1),
            'pass2_count': len(subs2),
            'merged_count': len(merged_subs),
            'strategy': strategy
        }

        logger.info(f"Merge complete: {len(merged_subs)} subtitles in final output")
        return stats

    def _parse_srt(self, path: Path) -> List[Subtitle]:
        """Parse an SRT file into Subtitle objects."""
        if not path.exists():
            logger.warning(f"SRT file not found: {path}")
            return []

        subtitles = []
        content = path.read_text(encoding='utf-8')

        # Split by double newline to get blocks
        blocks = re.split(r'\n\s*\n', content.strip())

        for block in blocks:
            if not block.strip():
                continue

            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            try:
                # Parse index
                index = int(lines[0].strip())

                # Parse timestamp line
                timestamp_match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                    lines[1].strip()
                )
                if not timestamp_match:
                    continue

                start_time = self._timestamp_to_seconds(timestamp_match.groups()[:4])
                end_time = self._timestamp_to_seconds(timestamp_match.groups()[4:])

                # Join remaining lines as text
                text = '\n'.join(lines[2:]).strip()

                subtitles.append(Subtitle(
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    text=text
                ))
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse subtitle block: {e}")
                continue

        return subtitles

    def _write_srt(self, subtitles: List[Subtitle], path: Path):
        """Write subtitles to SRT file."""
        lines = []
        for sub in subtitles:
            start_ts = self._seconds_to_timestamp(sub.start_time)
            end_ts = self._seconds_to_timestamp(sub.end_time)

            lines.append(str(sub.index))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(sub.text)
            lines.append('')

        path.write_text('\n'.join(lines), encoding='utf-8')

    def _timestamp_to_seconds(self, parts: Tuple) -> float:
        """Convert timestamp parts (h, m, s, ms) to seconds."""
        h, m, s, ms = [int(p) for p in parts]
        return h * 3600 + m * 60 + s + ms / 1000

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _overlap_duration(self, sub1: Subtitle, sub2: Subtitle) -> float:
        """Return absolute overlap duration between two subtitles in seconds."""
        overlap_start = max(sub1.start_time, sub2.start_time)
        overlap_end = min(sub1.end_time, sub2.end_time)
        if overlap_end <= overlap_start:
            return 0.0
        return overlap_end - overlap_start

    def _calculate_overlap(self, sub1: Subtitle, sub2: Subtitle) -> float:
        """Calculate temporal overlap as a ratio of the shorter subtitle duration."""
        overlap_duration = self._overlap_duration(sub1, sub2)
        if overlap_duration <= 0:
            return 0.0

        min_duration = min(sub1.duration, sub2.duration)
        if min_duration <= 0:
            return 0.0

        return overlap_duration / min_duration

    def _coverage_ratio(self, base: Subtitle, overlap_duration: float) -> float:
        """Return how much of *base* is covered by `overlap_duration`."""
        if base.duration <= 0:
            return 0.0
        return min(1.0, max(0.0, overlap_duration / base.duration))

    def _choose_by_timing(self, sub1: Subtitle, sub2: Subtitle) -> Subtitle:
        """Select the subtitle whose timing best matches the overlap window."""
        overlap_duration = self._overlap_duration(sub1, sub2)
        coverage1 = self._coverage_ratio(sub1, overlap_duration)
        coverage2 = self._coverage_ratio(sub2, overlap_duration)

        coverage_delta = coverage1 - coverage2
        if abs(coverage_delta) > 0.05:
            return sub1 if coverage_delta >= 0 else sub2

        # Coverage is effectively identical; prefer the subtitle with shorter duration
        # to avoid inflating cue lengths. Fall back to earliest start time for stability.
        if sub1.duration != sub2.duration:
            return sub1 if sub1.duration <= sub2.duration else sub2
        return sub1 if sub1.start_time <= sub2.start_time else sub2

    # Merge Strategies

    def _has_overlap(self, base_sub: Subtitle, other_sub: Subtitle, allow_threshold: bool = False) -> bool:
        """
        Check if two subtitles overlap.

        Args:
            base_sub: The base subtitle to check against
            other_sub: The other subtitle
            allow_threshold: If True, allow up to OVERLAP_THRESHOLD overlap

        Returns:
            True if they overlap (considering threshold)
        """
        overlap_start = max(base_sub.start_time, other_sub.start_time)
        overlap_end = min(base_sub.end_time, other_sub.end_time)

        if overlap_end <= overlap_start:
            return False  # No overlap

        overlap_duration = overlap_end - overlap_start

        if allow_threshold:
            # Allow overlap up to threshold of base duration
            allowed_overlap = base_sub.duration * self.OVERLAP_THRESHOLD
            return overlap_duration > allowed_overlap
        else:
            return True  # Any overlap counts

    def _merge_full(
        self,
        subs1: List[Subtitle],
        subs2: List[Subtitle]
    ) -> List[Subtitle]:
        """
        Full Merge: include every subtitle line from both passes.

        No filtering - combines all subtitles and sorts by time.
        """
        merged = []

        # Add all from both passes
        for sub in subs1:
            merged.append(Subtitle(0, sub.start_time, sub.end_time, sub.text))
        for sub in subs2:
            merged.append(Subtitle(0, sub.start_time, sub.end_time, sub.text))

        # Sort by start time
        merged.sort(key=lambda s: s.start_time)
        return merged

    def _merge_pass1_primary(
        self,
        subs1: List[Subtitle],
        subs2: List[Subtitle]
    ) -> List[Subtitle]:
        """
        Pass 1 + Fill From Pass 2: keeps Pass 1 as primary, fills gaps from Pass 2.

        Only adds from Pass 2 if there's no overlap with any Pass 1 subtitle.
        """
        return self._merge_primary_fill(subs1, subs2, allow_threshold=False)

    def _merge_pass2_primary(
        self,
        subs1: List[Subtitle],
        subs2: List[Subtitle]
    ) -> List[Subtitle]:
        """
        Pass 2 + Fill From Pass 1: keeps Pass 2 as primary, fills gaps from Pass 1.

        Only adds from Pass 1 if there's no overlap with any Pass 2 subtitle.
        """
        return self._merge_primary_fill(subs2, subs1, allow_threshold=False)

    def _merge_pass1_overlap(
        self,
        subs1: List[Subtitle],
        subs2: List[Subtitle]
    ) -> List[Subtitle]:
        """
        Pass 1 + Fill with Overlap Threshold: Pass 1 primary with 30% overlap tolerance.

        Adds from Pass 2 unless overlap exceeds 30% of Pass 1 subtitle duration.
        """
        return self._merge_primary_fill(subs1, subs2, allow_threshold=True)

    def _merge_pass2_overlap(
        self,
        subs1: List[Subtitle],
        subs2: List[Subtitle]
    ) -> List[Subtitle]:
        """
        Pass 2 + Fill with Overlap Threshold: Pass 2 primary with 30% overlap tolerance.

        Adds from Pass 1 unless overlap exceeds 30% of Pass 2 subtitle duration.
        """
        return self._merge_primary_fill(subs2, subs1, allow_threshold=True)

    def _merge_primary_fill(
        self,
        primary: List[Subtitle],
        secondary: List[Subtitle],
        allow_threshold: bool
    ) -> List[Subtitle]:
        """
        Core logic for primary + fill strategies.

        Args:
            primary: Primary subtitle list (kept as-is)
            secondary: Secondary list to fill gaps from
            allow_threshold: Whether to allow partial overlap
        """
        merged = []

        # Add all primary subtitles
        for sub in primary:
            merged.append(Subtitle(0, sub.start_time, sub.end_time, sub.text))

        # Add secondary subtitles that don't overlap with primary
        for sec_sub in secondary:
            has_conflict = False
            for pri_sub in primary:
                if self._has_overlap(pri_sub, sec_sub, allow_threshold):
                    has_conflict = True
                    break

            if not has_conflict:
                merged.append(Subtitle(0, sec_sub.start_time, sec_sub.end_time, sec_sub.text))

        # Sort by start time
        merged.sort(key=lambda s: s.start_time)
        return merged

    def _merge_smart(
        self,
        subs1: List[Subtitle],
        subs2: List[Subtitle]
    ) -> List[Subtitle]:
        """
        Smart Merge: timing-driven selection for overlapping subtitles.

        When segments overlap, prefer the subtitle whose timestamps best match the
        shared speech window (highest coverage, then shortest duration). Non-overlapping
        segments are included from both passes.
        """
        if not subs1:
            return [Subtitle(0, s.start_time, s.end_time, s.text) for s in subs2]
        if not subs2:
            return [Subtitle(0, s.start_time, s.end_time, s.text) for s in subs1]

        merged = []
        used_from_2 = set()

        for sub1 in subs1:
            best_match = None
            best_overlap = 0.0

            # Find best overlapping subtitle from subs2
            for i, sub2 in enumerate(subs2):
                if i in used_from_2:
                    continue

                overlap = self._calculate_overlap(sub1, sub2)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (i, sub2)

            if best_match and best_overlap >= self.OVERLAP_THRESHOLD:
                i, sub2 = best_match
                used_from_2.add(i)
                chosen = self._choose_by_timing(sub1, sub2)
                merged.append(Subtitle(
                    index=0,
                    start_time=chosen.start_time,
                    end_time=chosen.end_time,
                    text=chosen.text
                ))
            else:
                # No good match, keep sub1
                merged.append(Subtitle(
                    index=0,
                    start_time=sub1.start_time,
                    end_time=sub1.end_time,
                    text=sub1.text
                ))

        # Add unmatched subtitles from subs2
        for i, sub2 in enumerate(subs2):
            if i not in used_from_2:
                merged.append(Subtitle(
                    index=0,
                    start_time=sub2.start_time,
                    end_time=sub2.end_time,
                    text=sub2.text
                ))

        # Sort by start time
        merged.sort(key=lambda s: s.start_time)
        return merged


def get_available_strategies() -> List[Dict[str, str]]:
    """Return list of available merge strategies with descriptions."""
    return [
        {
            'name': 'smart_merge',
            'label': 'Smart Merge (Best Quality)',
            'description': 'Automatically picks the better subtitle line from each pass'
        },
        {
            'name': 'full_merge',
            'label': 'Full Merge (All Lines)',
            'description': 'Includes every subtitle line from both passes'
        },
        {
            'name': 'pass1_primary',
            'label': 'Pass 1 + Fill From Pass 2',
            'description': 'Keeps Pass 1 as primary and fills missing parts from Pass 2'
        },
        {
            'name': 'pass2_primary',
            'label': 'Pass 2 + Fill From Pass 1',
            'description': 'Keeps Pass 2 as primary and fills missing parts from Pass 1'
        },
        {
            'name': 'pass1_overlap',
            'label': 'Pass 1 + Fill (30% Overlap)',
            'description': 'Pass 1 primary, allows partial overlap when filling from Pass 2'
        },
        {
            'name': 'pass2_overlap',
            'label': 'Pass 2 + Fill (30% Overlap)',
            'description': 'Pass 2 primary, allows partial overlap when filling from Pass 1'
        }
    ]
