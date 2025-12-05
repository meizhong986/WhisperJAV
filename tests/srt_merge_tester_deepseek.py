#!/usr/bin/env python3
"""
SRT Merger Test Utility - TEXT-AGNOSTIC VERSION with better error reporting
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import timedelta, time, datetime
from typing import List, Tuple, Dict, Set, Optional, Any
import logging
from dataclasses import dataclass

# Try to use pysrt first, fall back to srt if not available
try:
    import pysrt
    SRT_PACKAGE = 'pysrt'
    Subtitle = pysrt.SubRipItem
    SubtitleFile = pysrt.SubRipFile
except ImportError:
    try:
        import srt
        SRT_PACKAGE = 'srt'
        Subtitle = srt.Subtitle
        SubtitleFile = List[srt.Subtitle]
    except ImportError:
        print("ERROR: No SRT package found. Please install either 'pysrt' or 'srt'")
        print("Install with: pip install pysrt")
        print("Or: pip install srt")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Merge option constants
MERGE_OPTIONS = {
    '1': 'first_base_fill_gaps',
    '2': 'second_base_fill_gaps', 
    '3': 'combine_both',
    '4': 'first_base_30_overlap',
    '5': 'second_base_30_overlap'
}

def time_to_timedelta(t: time) -> timedelta:
    """Convert datetime.time to timedelta."""
    return timedelta(
        hours=t.hour,
        minutes=t.minute,
        seconds=t.second,
        microseconds=t.microsecond
    )

@dataclass
class SubtitleSlot:
    """Represents a time slot for a subtitle, agnostic of text content."""
    start: timedelta
    end: timedelta
    duration: timedelta
    index: int = 0  # Original subtitle index (1-based)
    source_file: str = "unknown"  # Source file name
    is_merged: bool = False  # Whether this is from merged file
    
    @classmethod
    def from_subtitle(cls, sub: Subtitle, index: int = 0, source_file: str = "unknown", 
                      is_merged: bool = False):
        """Create a SubtitleSlot from a subtitle object."""
        if SRT_PACKAGE == 'pysrt':
            # pysrt returns datetime.time objects from to_time()
            start_time = sub.start.to_time()
            end_time = sub.end.to_time()
            start = time_to_timedelta(start_time)
            end = time_to_timedelta(end_time)
        else:  # srt package
            start = sub.start
            end = sub.end
        
        return cls(
            start=start, 
            end=end, 
            duration=end - start,
            index=index,
            source_file=source_file,
            is_merged=is_merged
        )
    
    def __hash__(self):
        return hash((self.start.total_seconds(), self.end.total_seconds()))
    
    def __eq__(self, other):
        if not isinstance(other, SubtitleSlot):
            return False
        return (self.start == other.start and self.end == other.end)
    
    def overlaps_with(self, other: 'SubtitleSlot', tolerance: timedelta) -> Tuple[bool, Optional[timedelta]]:
        """Check if this slot overlaps with another slot."""
        latest_start = max(self.start, other.start)
        earliest_end = min(self.end, other.end)
        
        if latest_start < earliest_end:
            overlap = earliest_end - latest_start
            return True, overlap
        elif abs(latest_start - earliest_end) <= tolerance:
            # Consider as touching (not overlapping) within tolerance
            return False, None
        return False, None
    
    def fits_in_gap(self, gap_start: timedelta, gap_end: timedelta, tolerance: timedelta) -> bool:
        """Check if this slot fits entirely within a gap."""
        return (self.start >= gap_start - tolerance and 
                self.end <= gap_end + tolerance)
    
    def contains(self, other: 'SubtitleSlot', tolerance: timedelta) -> bool:
        """Check if this slot contains another slot."""
        return (self.start <= other.start + tolerance and 
                self.end >= other.end - tolerance)
    
    def __str__(self):
        """String representation for debugging."""
        start_str = str(self.start).split('.')[0]
        end_str = str(self.end).split('.')[0]
        source = self.source_file
        if self.is_merged:
            source = f"merged[{self.index}]"
        else:
            source = f"{source}[{self.index}]"
        return f"{source}: {start_str} --> {end_str}"
    
    def detailed_str(self):
        """Detailed string representation with all info."""
        start_str = str(self.start).split('.')[0]
        end_str = str(self.end).split('.')[0]
        duration_str = str(self.duration).split('.')[0]
        return (f"Subtitle {self.index} from {self.source_file}: "
                f"{start_str} --> {end_str} (duration: {duration_str})")

class SRTTestValidator:
    """Main validator class for testing SRT merge operations - TEXT AGNOSTIC."""
    
    def __init__(self, tolerance_ms: int = 50):
        """
        Initialize the validator.
        
        Args:
            tolerance_ms: Time tolerance in milliseconds for time comparisons
        """
        self.tolerance_ms = tolerance_ms
        self.tolerance_td = timedelta(milliseconds=tolerance_ms)
        
    def load_srt_file(self, file_path: Path, is_merged: bool = False) -> List[SubtitleSlot]:
        """
        Load SRT file and convert to SubtitleSlots (ignoring text).
        
        Args:
            file_path: Path to SRT file
            is_merged: Whether this is the merged output file
            
        Returns:
            List of SubtitleSlots
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If SRT file is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                
            if SRT_PACKAGE == 'pysrt':
                subs = pysrt.from_string(content)
            else:  # srt package
                subs = list(srt.parse(content))
            
            # Convert to SubtitleSlots (ignoring text)
            slots = []
            for i, sub in enumerate(subs, 1):  # Start from 1 for subtitle index
                try:
                    slot = SubtitleSlot.from_subtitle(
                        sub, 
                        index=i, 
                        source_file=file_path.name,
                        is_merged=is_merged
                    )
                    slots.append(slot)
                except Exception as e:
                    logger.warning(f"Failed to convert subtitle {i} in {file_path.name}: {e}")
                    continue
            
            logger.info(f"Successfully converted {len(slots)} subtitles to time slots from {file_path.name}")
            return slots
                
        except Exception as e:
            raise ValueError(f"Failed to parse SRT file {file_path}: {e}")
    
    def time_equal(self, t1: timedelta, t2: timedelta) -> bool:
        """Compare two times within tolerance."""
        return abs(t1 - t2) <= self.tolerance_td
    
    def find_overlap(self, slot1: SubtitleSlot, slot2: SubtitleSlot) -> Optional[timedelta]:
        """Calculate overlap duration between two time slots."""
        latest_start = max(slot1.start, slot2.start)
        earliest_end = min(slot1.end, slot2.end)
        
        if latest_start < earliest_end:
            return earliest_end - latest_start
        
        return None
    
    def find_overlap_percentage(self, slot1: SubtitleSlot, slot2: SubtitleSlot) -> Optional[float]:
        """Calculate overlap percentage relative to shorter slot."""
        overlap = self.find_overlap(slot1, slot2)
        if overlap is None:
            return None
        
        shorter_duration = min(slot1.duration, slot2.duration)
        if shorter_duration.total_seconds() <= 0:
            return None
            
        return (overlap.total_seconds() / shorter_duration.total_seconds()) * 100
    
    def validate_slot_ordering(self, slots: List[SubtitleSlot]) -> Tuple[bool, List[str]]:
        """Validate that time slots are in chronological order."""
        errors = []
        last_end = timedelta(seconds=0)
        
        for i, slot in enumerate(slots):
            # Check start time is before end time
            if slot.start >= slot.end:
                error_msg = f"{slot.detailed_str()}: Start time >= End time"
                errors.append(error_msg)
            
            # Check chronological order (allow small overlaps)
            if slot.start < last_end - self.tolerance_td:
                warning_msg = f"{slot.detailed_str()}: Starts before previous subtitle ends ({last_end})"
                errors.append(warning_msg)
            
            last_end = max(last_end, slot.end)
        
        return len(errors) == 0, errors
    
    def find_gaps(self, slots: List[SubtitleSlot], 
                  start_time: timedelta = None, 
                  end_time: timedelta = None) -> List[Tuple[timedelta, timedelta, List[SubtitleSlot]]]:
        """
        Find gaps in timeline with context about surrounding subtitles.
        
        Args:
            slots: List of time slots
            start_time: Optional start time boundary
            end_time: Optional end time boundary
            
        Returns:
            List of (gap_start, gap_end, context_slots) tuples
        """
        if not slots:
            return [(start_time or timedelta(seconds=0), end_time or timedelta(hours=10), [])]
        
        gaps = []
        sorted_slots = sorted(slots, key=lambda x: x.start)
        
        # Check before first subtitle
        first_start = sorted_slots[0].start
        if start_time is not None and first_start > start_time + self.tolerance_td:
            gaps.append((start_time, first_start, []))
        
        # Check between subtitles
        for i in range(len(sorted_slots) - 1):
            end1 = sorted_slots[i].end
            start2 = sorted_slots[i + 1].start
            
            if start2 > end1 + self.tolerance_td:
                context = [sorted_slots[i], sorted_slots[i + 1]]
                gaps.append((end1, start2, context))
        
        # Check after last subtitle
        if end_time is not None:
            last_end = sorted_slots[-1].end
            if last_end < end_time - self.tolerance_td:
                gaps.append((last_end, end_time, [sorted_slots[-1]]))
        
        return gaps
    
    def slot_in_gap(self, slot: SubtitleSlot, gap: Tuple[timedelta, timedelta, List[SubtitleSlot]]) -> bool:
        """Check if time slot fits entirely within a gap."""
        gap_start, gap_end, _ = gap
        return slot.fits_in_gap(gap_start, gap_end, self.tolerance_td)
    
    def match_slot(self, slot: SubtitleSlot, slot_list: List[SubtitleSlot]) -> Optional[SubtitleSlot]:
        """Find a matching slot in the list (within tolerance)."""
        for candidate in slot_list:
            if (self.time_equal(slot.start, candidate.start) and 
                self.time_equal(slot.end, candidate.end)):
                return candidate
        return None
    
    def find_best_match(self, slot: SubtitleSlot, slot_list: List[SubtitleSlot]) -> Tuple[Optional[SubtitleSlot], float]:
        """Find the best matching slot and similarity score (0-1)."""
        best_match = None
        best_score = 0
        
        for candidate in slot_list:
            # Calculate time similarity
            start_diff = abs((slot.start - candidate.start).total_seconds())
            end_diff = abs((slot.end - candidate.end).total_seconds())
            
            # Normalize to similarity score (1 - normalized difference)
            max_diff = max(start_diff, end_diff, 0.001)  # Avoid division by zero
            similarity = 1.0 - (min(max_diff, 2.0) / 2.0)  # Cap at 2 seconds difference
            
            if similarity > best_score:
                best_score = similarity
                best_match = candidate
        
        return best_match, best_score
    
    def test_first_base_fill_gaps(self, srt1_slots: List[SubtitleSlot], 
                                  srt2_slots: List[SubtitleSlot], 
                                  merged_slots: List[SubtitleSlot]) -> Dict[str, Any]:
        """Test option 1: First SRT as base, fill gaps from second SRT."""
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Find all base slots in merged
        base_in_merged = []
        remaining_merged = merged_slots.copy()
        
        for slot1 in srt1_slots:
            matched = self.match_slot(slot1, remaining_merged)
            if matched:
                base_in_merged.append(matched)
                remaining_merged.remove(matched)
            else:
                results['passed'] = False
                results['errors'].append(
                    f"Base subtitle {slot1.index} from {slot1.source_file} missing in merged: "
                    f"{slot1.detailed_str()}"
                )
        
        # Find gaps in base timeline
        gaps = self.find_gaps(srt1_slots)
        
        # Check remaining merged slots (should be fill slots from SRT2)
        fill_in_merged = remaining_merged
        
        for fill_slot in fill_in_merged:
            # Check if this slot fits in any gap
            fits = False
            gap_context = ""
            for gap_start, gap_end, context in gaps:
                if self.slot_in_gap(fill_slot, (gap_start, gap_end, context)):
                    fits = True
                    # Create gap context string
                    if context:
                        gap_context = f" (gap between {context[0].index} and {context[1].index} in base)"
                    else:
                        gap_context = " (gap at start/end of base)"
                    break
            
            if not fits:
                results['passed'] = False
                # Find which gap it might be trying to fit into
                best_gap = None
                min_distance = float('inf')
                for gap_start, gap_end, context in gaps:
                    # Calculate how far outside the gap this slot is
                    distance_before = max(0, (gap_start - fill_slot.start).total_seconds())
                    distance_after = max(0, (fill_slot.end - gap_end).total_seconds())
                    total_distance = distance_before + distance_after
                    
                    if total_distance < min_distance:
                        min_distance = total_distance
                        best_gap = (gap_start, gap_end, context)
                
                error_msg = (f"{fill_slot.detailed_str()} doesn't fit in any gap of base file "
                           f"(closest gap: {best_gap[0]} to {best_gap[1] if best_gap else 'N/A'})")
                results['errors'].append(error_msg)
            
            # Check if this slot could come from SRT2 (approximate match)
            best_match, similarity = self.find_best_match(fill_slot, srt2_slots)
            
            if best_match:
                if similarity < 0.9:  # Less than 90% similarity
                    results['warnings'].append(
                        f"{fill_slot.detailed_str()} differs from source "
                        f"{best_match.detailed_str()} (similarity: {similarity:.1%})"
                    )
            else:
                results['warnings'].append(
                    f"{fill_slot.detailed_str()} not found in SRT2 (may be modified or from different source)"
                )
        
        # Check for overlaps between base and fill slots
        for base_slot in base_in_merged:
            for fill_slot in fill_in_merged:
                overlap = self.find_overlap(base_slot, fill_slot)
                if overlap and overlap > self.tolerance_td:
                    results['passed'] = False
                    overlap_percent = self.find_overlap_percentage(base_slot, fill_slot)
                    overlap_str = f"{overlap.total_seconds():.3f}s"
                    if overlap_percent:
                        overlap_str += f" ({overlap_percent:.1f}%)"
                    results['errors'].append(
                        f"Overlap between {base_slot.detailed_str()} and "
                        f"{fill_slot.detailed_str()}: {overlap_str}"
                    )
        
        # Check fill slots don't overlap each other
        sorted_fill = sorted(fill_in_merged, key=lambda x: x.start)
        for i in range(len(sorted_fill) - 1):
            overlap = self.find_overlap(sorted_fill[i], sorted_fill[i + 1])
            if overlap and overlap > self.tolerance_td:
                results['passed'] = False
                results['errors'].append(
                    f"Fill slots overlap each other: {sorted_fill[i].detailed_str()} with "
                    f"{sorted_fill[i + 1].detailed_str()} ({overlap.total_seconds():.3f}s)"
                )
        
        results['stats'] = {
            'base_slots': len(srt1_slots),
            'base_found': len(base_in_merged),
            'fill_slots_found': len(fill_in_merged),
            'gaps_found': len(gaps),
            'merged_total': len(merged_slots),
            'missing_base': len(srt1_slots) - len(base_in_merged),
            'extra_fill': len(fill_in_merged) - len([s for s in fill_in_merged if self.find_best_match(s, srt2_slots)[0]])
        }
        
        return results
    
    def test_combine_both(self, srt1_slots: List[SubtitleSlot], 
                          srt2_slots: List[SubtitleSlot], 
                          merged_slots: List[SubtitleSlot]) -> Dict[str, Any]:
        """Test option 3: Combine both together regardless of overlap."""
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check all SRT1 slots are in merged
        srt1_missing = []
        for slot1 in srt1_slots:
            if not self.match_slot(slot1, merged_slots):
                srt1_missing.append(slot1)
                results['passed'] = False
        
        # Check all SRT2 slots are in merged
        srt2_missing = []
        for slot2 in srt2_slots:
            if not self.match_slot(slot2, merged_slots):
                srt2_missing.append(slot2)
                results['passed'] = False
        
        if srt1_missing:
            results['errors'].append(f"Missing {len(srt1_missing)} slots from first SRT:")
            for slot in srt1_missing[:5]:  # Show first 5 missing
                results['errors'].append(f"  - {slot.detailed_str()}")
            if len(srt1_missing) > 5:
                results['errors'].append(f"  ... and {len(srt1_missing) - 5} more")
        
        if srt2_missing:
            results['errors'].append(f"Missing {len(srt2_missing)} slots from second SRT:")
            for slot in srt2_missing[:5]:  # Show first 5 missing
                results['errors'].append(f"  - {slot.detailed_str()}")
            if len(srt2_missing) > 5:
                results['errors'].append(f"  ... and {len(srt2_missing) - 5} more")
        
        # Check for duplicates (same time slot appears multiple times)
        merged_set = set()
        for slot in merged_slots:
            slot_key = (slot.start.total_seconds(), slot.end.total_seconds())
            if slot_key in merged_set:
                results['warnings'].append(f"Duplicate time slot: {slot.detailed_str()}")
            merged_set.add(slot_key)
        
        # Check chronological order
        is_ordered, ordering_errors = self.validate_slot_ordering(merged_slots)
        if not is_ordered:
            results['warnings'].append(f"Merged file has {len(ordering_errors)} ordering issues:")
            for error in ordering_errors[:3]:  # Show first 3 ordering errors
                results['warnings'].append(f"  - {error}")
            if len(ordering_errors) > 3:
                results['warnings'].append(f"  ... and {len(ordering_errors) - 3} more")
        
        # Check for overlaps (since option 3 doesn't prevent overlaps)
        sorted_merged = sorted(merged_slots, key=lambda x: x.start)
        overlap_count = 0
        for i in range(len(sorted_merged) - 1):
            overlap = self.find_overlap(sorted_merged[i], sorted_merged[i + 1])
            if overlap and overlap > self.tolerance_td:
                overlap_count += 1
                if overlap_count <= 3:  # Show first 3 overlaps
                    overlap_percent = self.find_overlap_percentage(sorted_merged[i], sorted_merged[i + 1])
                    overlap_str = f"{overlap.total_seconds():.3f}s"
                    if overlap_percent:
                        overlap_str += f" ({overlap_percent:.1f}%)"
                    results['warnings'].append(
                        f"Overlap in merged: {sorted_merged[i].detailed_str()} with "
                        f"{sorted_merged[i + 1].detailed_str()}: {overlap_str}"
                    )
        
        if overlap_count > 3:
            results['warnings'].append(f"  ... and {overlap_count - 3} more overlaps")
        
        results['stats'] = {
            'srt1_slots': len(srt1_slots),
            'srt2_slots': len(srt2_slots),
            'merged_total': len(merged_slots),
            'expected_total': len(srt1_slots) + len(srt2_slots),
            'srt1_missing': len(srt1_missing),
            'srt2_missing': len(srt2_missing),
            'duplicates': len(merged_slots) - len(merged_set),
            'ordering_issues': len(ordering_errors),
            'overlaps': overlap_count
        }
        
        return results
    
    def test_with_overlap(self, srt1_slots: List[SubtitleSlot], 
                          srt2_slots: List[SubtitleSlot], 
                          merged_slots: List[SubtitleSlot], 
                          base_is_first: bool = True) -> Dict[str, Any]:
        """
        Test options 4/5: Base SRT with fill and 30% overlap allowance.
        
        Args:
            base_is_first: True for option 4, False for option 5
        """
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Determine which is base and which is fill
        base_name = "first" if base_is_first else "second"
        fill_name = "second" if base_is_first else "first"
        base_slots = srt1_slots if base_is_first else srt2_slots
        fill_slots_source = srt2_slots if base_is_first else srt1_slots
        
        # Find all base slots in merged
        base_in_merged = []
        remaining_merged = merged_slots.copy()
        
        for base_slot in base_slots:
            matched = self.match_slot(base_slot, remaining_merged)
            if matched:
                base_in_merged.append(matched)
                remaining_merged.remove(matched)
            else:
                results['passed'] = False
                results['errors'].append(
                    f"Base subtitle {base_slot.index} from {base_slot.source_file} missing in merged: "
                    f"{base_slot.detailed_str()}"
                )
        
        # Remaining merged slots are fill slots
        fill_in_merged = remaining_merged
        
        # Check fill slots for overlap conditions
        for fill_slot in fill_in_merged:
            max_overlap_percent = 0
            overlapping_base = None
            
            for base_slot in base_in_merged:
                overlap_percent = self.find_overlap_percentage(fill_slot, base_slot)
                if overlap_percent is not None and overlap_percent > max_overlap_percent:
                    max_overlap_percent = overlap_percent
                    overlapping_base = base_slot
            
            if max_overlap_percent > 30:
                results['passed'] = False
                results['errors'].append(
                    f"Fill slot {fill_slot.detailed_str()} overlaps {max_overlap_percent:.1f}% "
                    f"with base slot {overlapping_base.detailed_str()} (exceeds 30% limit)"
                )
            elif max_overlap_percent > 0:
                results['warnings'].append(
                    f"Fill slot {fill_slot.detailed_str()} overlaps {max_overlap_percent:.1f}% "
                    f"with base slot {overlapping_base.detailed_str()}"
                )
        
        # Check fill slots don't overlap each other beyond tolerance
        sorted_fill = sorted(fill_in_merged, key=lambda x: x.start)
        for i in range(len(sorted_fill) - 1):
            overlap = self.find_overlap(sorted_fill[i], sorted_fill[i + 1])
            if overlap and overlap > self.tolerance_td:
                results['warnings'].append(
                    f"Fill slots overlap each other: {sorted_fill[i].detailed_str()} with "
                    f"{sorted_fill[i + 1].detailed_str()} ({overlap.total_seconds():.3f}s)"
                )
        
        # Try to match fill slots with their source
        for fill_slot in fill_in_merged:
            best_match, similarity = self.find_best_match(fill_slot, fill_slots_source)
            if best_match and similarity < 0.9:
                results['warnings'].append(
                    f"Fill slot {fill_slot.detailed_str()} differs from source "
                    f"{best_match.detailed_str()} in {fill_name} SRT (similarity: {similarity:.1%})"
                )
        
        results['stats'] = {
            'base_slots': len(base_slots),
            'base_found': len(base_in_merged),
            'fill_slots_found': len(fill_in_merged),
            'merged_total': len(merged_slots),
            'missing_base': len(base_slots) - len(base_in_merged),
            'overlap_violations': len([f for f in fill_in_merged if 
                                      any(self.find_overlap_percentage(f, b) and 
                                          self.find_overlap_percentage(f, b) > 30 
                                          for b in base_in_merged)])
        }
        
        return results
    
    def run_test(self, srt1_path: Path, srt2_path: Path, merged_path: Path, 
                 option: str) -> Dict[str, Any]:
        """
        Run the specified test.
        
        Args:
            srt1_path: Path to first SRT file
            srt2_path: Path to second SRT file
            merged_path: Path to merged SRT file
            option: Merge option (1-5)
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running test for option {option}: {MERGE_OPTIONS[option]}")
        logger.info("TEXT-AGNOSTIC MODE: Only checking timestamps, gaps, and durations")
        logger.info(f"Time tolerance: {self.tolerance_ms}ms")
        logger.info(f"Loading files...")
        
        try:
            srt1_slots = self.load_srt_file(srt1_path, is_merged=False)
            srt2_slots = self.load_srt_file(srt2_path, is_merged=False)
            merged_slots = self.load_srt_file(merged_path, is_merged=True)
            
            logger.info(f"Loaded {len(srt1_slots)} slots from {srt1_path.name}")
            logger.info(f"Loaded {len(srt2_slots)} slots from {srt2_path.name}")
            logger.info(f"Loaded {len(merged_slots)} slots from {merged_path.name}")
            
            # Validate individual SRT files
            logger.info("Validating input SRT files...")
            valid1, errors1 = self.validate_slot_ordering(srt1_slots)
            if not valid1:
                logger.warning(f"First SRT file {srt1_path.name} has {len(errors1)} ordering issues")
                for error in errors1[:2]:
                    logger.warning(f"  - {error}")
            
            valid2, errors2 = self.validate_slot_ordering(srt2_slots)
            if not valid2:
                logger.warning(f"Second SRT file {srt2_path.name} has {len(errors2)} ordering issues")
                for error in errors2[:2]:
                    logger.warning(f"  - {error}")
            
            valid_m, errors_m = self.validate_slot_ordering(merged_slots)
            if not valid_m:
                logger.warning(f"Merged SRT file {merged_path.name} has {len(errors_m)} ordering issues")
                for error in errors_m[:2]:
                    logger.warning(f"  - {error}")
            
            # Run specific test based on option
            if option == '1':
                results = self.test_first_base_fill_gaps(srt1_slots, srt2_slots, merged_slots)
            elif option == '2':
                results = self.test_first_base_fill_gaps(srt2_slots, srt1_slots, merged_slots)
            elif option == '3':
                results = self.test_combine_both(srt1_slots, srt2_slots, merged_slots)
            elif option == '4':
                results = self.test_with_overlap(srt1_slots, srt2_slots, merged_slots, base_is_first=True)
            elif option == '5':
                results = self.test_with_overlap(srt1_slots, srt2_slots, merged_slots, base_is_first=False)
            else:
                raise ValueError(f"Invalid option: {option}")
            
            # Add general validation
            if len(merged_slots) == 0:
                results['errors'].append("Merged file is empty")
                results['passed'] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}", exc_info=True)
            return {
                'passed': False,
                'errors': [f"Test execution failed: {str(e)}"],
                'warnings': [],
                'stats': {}
            }

def print_results(results: Dict[str, Any], option: str):
    """Print test results in a readable format."""
    print("\n" + "="*80)
    print(f"TEST RESULTS - Option {option}: {MERGE_OPTIONS[option]}")
    print("TEXT-AGNOSTIC VALIDATION (timestamps, gaps, durations only)")
    print("="*80)
    
    if results['passed']:
        print("✅ TEST PASSED")
    else:
        print("❌ TEST FAILED")
    
    print("\n📊 STATISTICS:")
    for key, value in results['stats'].items():
        formatted_key = key.replace('_', ' ').title()
        print(f"  {formatted_key}: {value}")
    
    if results['errors']:
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for i, error in enumerate(results['errors'][:15], 1):  # Show first 15 errors
            print(f"  {i:2d}. {error}")
        if len(results['errors']) > 15:
            print(f"  ... and {len(results['errors']) - 15} more errors")
    
    if results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for i, warning in enumerate(results['warnings'][:10], 1):  # Show first 10 warnings
            print(f"  {i:2d}. {warning}")
        if len(results['warnings']) > 10:
            print(f"  ... and {len(results['warnings']) - 10} more warnings")
    
    print("\n" + "="*80)
    print("Note: All validations are text-agnostic - only timestamps, gaps, and durations are checked")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='Test SRT merge operations for correctness (TEXT-AGNOSTIC)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TEXT-AGNOSTIC MODE: This validator only checks timestamps, gaps, and durations.
                    Subtitle text content is completely ignored.
                    
Merge Options:
  1 - First SRT as base, fill gaps from second SRT
  2 - Second SRT as base, fill gaps from first SRT
  3 - Combine both together regardless of overlap
  4 - First SRT as base, fill gaps from second SRT, allow 30% overlap
  5 - Second SRT as base, fill gaps from first SRT, allow 30% overlap
  
Examples:
  %(prog)s input1.srt input2.srt merged.srt 1
  %(prog)s file1.srt file2.srt output.srt 3 --verbose --tolerance 100
        """
    )
    
    parser.add_argument('srt1', type=Path, help='First input SRT file')
    parser.add_argument('srt2', type=Path, help='Second input SRT file')
    parser.add_argument('merged', type=Path, help='Merged SRT file to test')
    parser.add_argument('option', choices=['1', '2', '3', '4', '5'], 
                       help='Merge option used (1-5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--tolerance', '-t', type=int, default=50,
                       help='Time tolerance in milliseconds (default: 50)')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug mode with extra output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose or args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Validate file existence
    for file_path in [args.srt1, args.srt2, args.merged]:
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
    
    print(f"Using SRT package: {SRT_PACKAGE}")
    print(f"TEXT-AGNOSTIC MODE: Ignoring all subtitle text content")
    print(f"Time tolerance: {args.tolerance}ms")
    
    # Create validator and run test
    validator = SRTTestValidator(tolerance_ms=args.tolerance)
    results = validator.run_test(args.srt1, args.srt2, args.merged, args.option)
    
    # Print results
    print_results(results, args.option)
    
    # Exit with appropriate code
    sys.exit(0 if results['passed'] else 1)

if __name__ == '__main__':
    main()