"""Merge Engine for combining two-pass SRT results."""

import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from whisperjav.utils.logger import logger


class MergeStrategy(Enum):
    """Available merge strategies."""
    CONFIDENCE = 'confidence'
    UNION = 'union'
    INTERSECTION = 'intersection'
    TIMING = 'timing'


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

    def __init__(self):
        """Initialize the merge engine."""
        self.strategies = {
            'confidence': self._merge_confidence,
            'union': self._merge_union,
            'intersection': self._merge_intersection,
            'timing': self._merge_timing,
        }

    def merge(
        self,
        srt1_path: Path,
        srt2_path: Path,
        output_path: Path,
        strategy: str = 'confidence'
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

    def _calculate_overlap(self, sub1: Subtitle, sub2: Subtitle) -> float:
        """Calculate temporal overlap between two subtitles as a ratio."""
        overlap_start = max(sub1.start_time, sub2.start_time)
        overlap_end = min(sub1.end_time, sub2.end_time)

        if overlap_end <= overlap_start:
            return 0.0

        overlap_duration = overlap_end - overlap_start
        min_duration = min(sub1.duration, sub2.duration)

        if min_duration <= 0:
            return 0.0

        return overlap_duration / min_duration

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity ratio."""
        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        if not t1 or not t2:
            return 0.0

        if t1 == t2:
            return 1.0

        # Simple character-based similarity
        chars1 = set(t1)
        chars2 = set(t2)
        intersection = len(chars1 & chars2)
        union = len(chars1 | chars2)

        return intersection / union if union > 0 else 0.0

    # Merge Strategies

    def _merge_confidence(
        self,
        subs1: List[Subtitle],
        subs2: List[Subtitle]
    ) -> List[Subtitle]:
        """
        Confidence-based merge: prefer longer, more detailed subtitles.

        For overlapping segments, choose the one with more text content
        as it likely captured more detail.
        """
        if not subs1:
            return subs2.copy()
        if not subs2:
            return subs1.copy()

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

            if best_match and best_overlap > 0.5:
                i, sub2 = best_match
                used_from_2.add(i)

                # Choose subtitle with more content
                if len(sub2.text) > len(sub1.text):
                    # Use sub2's text with averaged timing
                    merged.append(Subtitle(
                        index=0,
                        start_time=min(sub1.start_time, sub2.start_time),
                        end_time=max(sub1.end_time, sub2.end_time),
                        text=sub2.text
                    ))
                else:
                    merged.append(Subtitle(
                        index=0,
                        start_time=min(sub1.start_time, sub2.start_time),
                        end_time=max(sub1.end_time, sub2.end_time),
                        text=sub1.text
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

    def _merge_union(
        self,
        subs1: List[Subtitle],
        subs2: List[Subtitle]
    ) -> List[Subtitle]:
        """
        Union merge: include all unique subtitles from both passes.

        Remove duplicates based on timing and text similarity.
        """
        if not subs1:
            return subs2.copy()
        if not subs2:
            return subs1.copy()

        # Start with all from subs1
        merged = [Subtitle(0, s.start_time, s.end_time, s.text) for s in subs1]

        # Add from subs2 only if not duplicate
        for sub2 in subs2:
            is_duplicate = False
            for sub1 in merged:
                overlap = self._calculate_overlap(sub1, sub2)
                text_sim = self._text_similarity(sub1.text, sub2.text)

                # Consider duplicate if high overlap and similar text
                if overlap > 0.7 and text_sim > 0.5:
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(Subtitle(
                    index=0,
                    start_time=sub2.start_time,
                    end_time=sub2.end_time,
                    text=sub2.text
                ))

        # Sort by start time
        merged.sort(key=lambda s: s.start_time)
        return merged

    def _merge_intersection(
        self,
        subs1: List[Subtitle],
        subs2: List[Subtitle]
    ) -> List[Subtitle]:
        """
        Intersection merge: only include subtitles present in both passes.

        High precision, lower recall. Good for confident output.
        """
        if not subs1 or not subs2:
            return []

        merged = []
        used_from_2 = set()

        for sub1 in subs1:
            best_match = None
            best_score = 0.0

            for i, sub2 in enumerate(subs2):
                if i in used_from_2:
                    continue

                overlap = self._calculate_overlap(sub1, sub2)
                text_sim = self._text_similarity(sub1.text, sub2.text)

                # Combined score
                score = overlap * 0.6 + text_sim * 0.4

                if score > best_score:
                    best_score = score
                    best_match = (i, sub2)

            # Only include if both passes agree (score > 0.5)
            if best_match and best_score > 0.5:
                i, sub2 = best_match
                used_from_2.add(i)

                # Merge timing and choose longer text
                merged.append(Subtitle(
                    index=0,
                    start_time=(sub1.start_time + sub2.start_time) / 2,
                    end_time=(sub1.end_time + sub2.end_time) / 2,
                    text=sub1.text if len(sub1.text) >= len(sub2.text) else sub2.text
                ))

        # Sort by start time
        merged.sort(key=lambda s: s.start_time)
        return merged

    def _merge_timing(
        self,
        subs1: List[Subtitle],
        subs2: List[Subtitle]
    ) -> List[Subtitle]:
        """
        Timing-based merge: prefer subtitles with tighter timing.

        Shorter duration often indicates more precise timing boundaries.
        """
        if not subs1:
            return subs2.copy()
        if not subs2:
            return subs1.copy()

        merged = []
        used_from_2 = set()

        for sub1 in subs1:
            best_match = None
            best_overlap = 0.0

            for i, sub2 in enumerate(subs2):
                if i in used_from_2:
                    continue

                overlap = self._calculate_overlap(sub1, sub2)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (i, sub2)

            if best_match and best_overlap > 0.5:
                i, sub2 = best_match
                used_from_2.add(i)

                # Choose subtitle with tighter timing (shorter duration)
                # but keep text from longer one
                if sub1.duration <= sub2.duration:
                    timing_source = sub1
                    text_source = sub1 if len(sub1.text) >= len(sub2.text) else sub2
                else:
                    timing_source = sub2
                    text_source = sub1 if len(sub1.text) >= len(sub2.text) else sub2

                merged.append(Subtitle(
                    index=0,
                    start_time=timing_source.start_time,
                    end_time=timing_source.end_time,
                    text=text_source.text
                ))
            else:
                merged.append(Subtitle(
                    index=0,
                    start_time=sub1.start_time,
                    end_time=sub1.end_time,
                    text=sub1.text
                ))

        # Add unmatched from subs2
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
            'name': 'confidence',
            'label': 'Confidence-based',
            'description': 'Prefer subtitles with more content detail'
        },
        {
            'name': 'union',
            'label': 'Union',
            'description': 'Include all unique subtitles from both passes'
        },
        {
            'name': 'intersection',
            'label': 'Intersection',
            'description': 'Only include subtitles confirmed by both passes'
        },
        {
            'name': 'timing',
            'label': 'Timing-based',
            'description': 'Prefer subtitles with tighter timing boundaries'
        }
    ]
