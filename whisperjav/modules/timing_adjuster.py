# whisperjav/modules/timing_adjuster.py
#V12

import pysrt
from datetime import timedelta
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from whisperjav.utils.logger import logger
from whisperjav.config.sanitization_constants import TimingConstants, CrossSubtitleConstants

HALLUCINATION_THRESHOLD = 12.0  # seconds


@dataclass
class TimingAdjustmentInfo:
    """Information about timing adjustments made"""
    subtitle_index: int
    original_text: str
    modified_text: str
    original_start: pysrt.SubRipTime
    original_end: pysrt.SubRipTime
    new_start: pysrt.SubRipTime
    new_end: pysrt.SubRipTime
    content_length_change: int
    reason: str
    adjustment_type: str


class TimingAdjuster:
    """STRICT timing adjuster - only conditions (a) and (b) with enhanced validation"""
    
    def __init__(self, timing_constants: TimingConstants, 
                 cross_subtitle_constants: CrossSubtitleConstants,
                 primary_language: Optional[str] = None):
        self.timing_constants = timing_constants
        self.cross_subtitle_constants = cross_subtitle_constants
        self.primary_language = primary_language or 'default'
        
        # STRICT: Hallucination threshold (condition b)
        self.HALLUCINATION_THRESHOLD = 12.0  # seconds
        
        
    def adjust_timings_content_aware(self,
                                   original_subtitles: List[pysrt.SubRipItem],
                                   modified_subtitles: List[pysrt.SubRipItem]) -> Tuple[List[pysrt.SubRipItem], List[Dict]]:
        """
        FIXED: Corrected the method for getting duration in seconds from a SubRipTime object.
        """
        if not modified_subtitles:
            return modified_subtitles, []

        modifications = []
        result_subtitles = []

        original_by_index = {sub.index: sub for sub in original_subtitles}

        # Get constants for the audit at the start
        timing_consts = self.timing_constants
        min_cps = timing_consts.MIN_SAFE_CPS
        min_len_for_check = timing_consts.MIN_TEXT_LENGTH_FOR_CPS_CHECK

        # Counters for logging
        condition_a_count, condition_b_count, condition_c_count, condition_d_count = 0, 0, 0, 0
        untouched_count = 0

        for modified_sub in modified_subtitles:
            original_sub = original_by_index.get(modified_sub.index)

            should_adjust = False
            adjustment_reason = ""

            if original_sub:
                # Condition (a): Substantial content change (text difference)
                original_length = len(original_sub.text.strip())
                modified_length = len(modified_sub.text.strip())

                if original_length > 0:
                    change_ratio = abs(original_length - modified_length) / original_length
                    if change_ratio > 0.3:
                        should_adjust = True
                        adjustment_reason = "substantial_content_change"
                        condition_a_count += 1

                # Condition (c): Detect merges by checking duration change
                if not should_adjust:
                    original_duration = original_sub.duration.ordinal
                    modified_duration = modified_sub.duration.ordinal
                    if abs(original_duration - modified_duration) > 250 and modified_length < 20:
                        should_adjust = True
                        adjustment_reason = "merged_line_duration_change"
                        condition_c_count += 1
                
                # Condition (b): Original duration was a hallucination (> 12s)
                if not should_adjust:
                    duration_s = (modified_sub.end.ordinal - modified_sub.start.ordinal) / 1000.0
                    if duration_s > self.HALLUCINATION_THRESHOLD:
                        should_adjust = True
                        adjustment_reason = "duration_hallucination"
                        condition_b_count += 1
                
                # Condition (d): Abnormally Slow CPS (Case Y)
                if not should_adjust:
                    text_len = len(modified_sub.text_without_tags.strip())
                    
                    # --- THIS IS THE CORRECTED LINE ---
                    duration_s = modified_sub.duration.ordinal / 1000.0
                    # ------------------------------------

                    if text_len >= min_len_for_check and duration_s > 0:
                        actual_cps = text_len / duration_s
                        if actual_cps < min_cps:
                            should_adjust = True
                            adjustment_reason = "abnormally_slow_cps"
                            condition_d_count += 1

            if should_adjust:
                adjusted_sub, timing_mods = self._apply_timing_adjustment(
                    original_sub, modified_sub, adjustment_reason
                )
                result_subtitles.append(adjusted_sub)
                if timing_mods:
                    modifications.extend(timing_mods)
            else:
                result_subtitles.append(modified_sub)
                untouched_count += 1

        # Renumber the final list of subtitles
        for i, sub in enumerate(result_subtitles, 1):
            sub.index = i

        logger.debug(f"STRICT timing adjustment complete:")
        logger.debug(f"  Condition (a) - Substantial content changed: {condition_a_count} subtitles")
        logger.debug(f"  Condition (b) - Duration > {self.HALLUCINATION_THRESHOLD}s: {condition_b_count} subtitles")
        logger.debug(f"  Condition (c) - Merged lines detected: {condition_c_count} subtitles")
        logger.debug(f"  Condition (d) - Abnormally Slow CPS (< {min_cps} CPS): {condition_d_count} subtitles")
        logger.debug(f"  Untouched (as required): {untouched_count} subtitles")

        return result_subtitles, modifications
        

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity to determine if content was substantially changed"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.strip(), text2.strip()).ratio()
        

    def _apply_timing_adjustment(self, 
                               original_sub: pysrt.SubRipItem,
                               modified_sub: pysrt.SubRipItem,
                               reason: str) -> Tuple[pysrt.SubRipItem, List[Dict]]:
        """Apply timing adjustment and return a comprehensive modification dictionary."""
        
        chars_per_second = self.cross_subtitle_constants.CHARS_PER_SECOND.get(
            self.primary_language,
            self.cross_subtitle_constants.CHARS_PER_SECOND['default']
        )
        
        text_length = len(modified_sub.text.strip())
        ideal_duration_s = max(
            text_length / chars_per_second,
            self.timing_constants.MIN_SUBTITLE_DURATION
        )
        ideal_duration_s = min(ideal_duration_s, self.timing_constants.MAX_SUBTITLE_DURATION)
        
        fixed_end = modified_sub.end
        ideal_start_ordinal = fixed_end.ordinal - int(ideal_duration_s * 1000)
        new_start = pysrt.SubRipTime(milliseconds=max(0, ideal_start_ordinal))
        
        adjusted_sub = pysrt.SubRipItem(
            index=modified_sub.index,
            start=new_start,
            end=fixed_end,
            text=modified_sub.text
        )
        
        original_duration = (modified_sub.end.ordinal - modified_sub.start.ordinal) / 1000.0
        new_duration = (adjusted_sub.end.ordinal - adjusted_sub.start.ordinal) / 1000.0
        
        # --- FIX: Added 'modified_text' to the returned dictionary ---
        modifications = [{
            'type': f'timing_adjustment_{reason}',
            'subtitle_index': modified_sub.index,
            'original_start': modified_sub.start,
            'new_start': new_start,
            'end_timestamp': fixed_end,
            'original_duration': original_duration,
            'new_duration': new_duration,
            'text_length': text_length,
            'reason': reason,
            'original_text': original_sub.text if original_sub else 'N/A',
            'modified_text': modified_sub.text
        }]
        
        logger.debug(f"Sub {modified_sub.index}: {reason} timing: {original_duration:.1f}s -> {new_duration:.1f}s")
        
        return adjusted_sub, modifications

        
    def _has_duration_threshold_violation(self, subtitle: pysrt.SubRipItem) -> bool:
        """
        Check condition (b): Does this subtitle have duration threshold violations?
        This catches cases where the model hallucinated timing, regardless of content changes.
        """
        duration_ms = subtitle.end.ordinal - subtitle.start.ordinal
        duration_s = duration_ms / 1000.0

        
        if duration_s > HALLUCINATION_THRESHOLD:
            logger.debug(f"Sub {subtitle.index}: MAX Duration violation {duration_s:.1f}s (outside {self.timing_constants.MIN_SUBTITLE_DURATION}-{self.timing_constants.MAX_SUBTITLE_DURATION}s range)")

            return True  # Likely timing hallucination
    
            
        # Check for extremely short content with long duration (likely hallucination)
        text_length = len(subtitle.text.strip())
        if text_length <= 3 and duration_s > 3.0:
            logger.debug(f"Sub {subtitle.index}: Likely timing hallucination - {text_length} chars with {duration_s:.1f}s duration")
            return True
            
        # Check for extremely long content with very short duration
        if text_length > 50 and duration_s < 1.0:
            logger.debug(f"Sub {subtitle.index}: Likely timing error - {text_length} chars with {duration_s:.1f}s duration")
            return True
            
        return False
        
    def _adjust_single_subtitle_content_aware(self, 
                                            original_sub: pysrt.SubRipItem,
                                            modified_sub: pysrt.SubRipItem,
                                            previous_adjusted_subs: List[pysrt.SubRipItem],
                                            adjustment_reason: str) -> Tuple[pysrt.SubRipItem, List[Dict]]:
        """
        Apply user's strategy: adjust timing only for subtitles meeting conditions (a) or (b)
        Keep end fixed, adjust start based on new content length
        """
        modifications = []
        
        if adjustment_reason == "substantial_content_change":
            # Condition (a): Content changed substantially, apply content-based timing
            return self._apply_content_based_timing_adjustment(
                original_sub, modified_sub, previous_adjusted_subs
            )
        elif adjustment_reason == "duration_threshold_violation":
            # Condition (b): Duration violation, apply threshold-based timing
            return self._apply_threshold_based_timing_adjustment(
                modified_sub, previous_adjusted_subs
            )
        else:
            # Should not reach here
            return modified_sub, modifications
            
    def _apply_content_based_timing_adjustment(self,
                                            original_sub: pysrt.SubRipItem,
                                            modified_sub: pysrt.SubRipItem,
                                            previous_adjusted_subs: List[pysrt.SubRipItem]) -> Tuple[pysrt.SubRipItem, List[Dict]]:
        """Apply timing adjustment based on content length change"""
        
        # Calculate content length change
        original_length = len(original_sub.text)
        modified_length = len(modified_sub.text)
        
        # Calculate ideal duration for modified content
        chars_per_second = self.cross_subtitle_constants.CHARS_PER_SECOND.get(
            self.primary_language,
            self.cross_subtitle_constants.CHARS_PER_SECOND['default']
        )
        
        ideal_duration_s = max(
            modified_length / chars_per_second,
            self.timing_constants.MIN_SUBTITLE_DURATION
        )
        
        # Apply maximum duration constraint
        ideal_duration_s = min(ideal_duration_s, self.timing_constants.MAX_SUBTITLE_DURATION)
        
        # Keep end fixed, calculate new start
        fixed_end = modified_sub.end
        ideal_start_ordinal = fixed_end.ordinal - int(ideal_duration_s * 1000)
        new_start = pysrt.SubRipTime(milliseconds=max(0, ideal_start_ordinal))
        
        # Prevent overlap with previous subtitle
        if previous_adjusted_subs:
            prev_sub = previous_adjusted_subs[-1]
            min_gap_ms = 50
            earliest_allowed_start = prev_sub.end.ordinal + min_gap_ms
            
            if new_start.ordinal < earliest_allowed_start:
                new_start = pysrt.SubRipTime(milliseconds=earliest_allowed_start)
                logger.debug(f"Sub {modified_sub.index}: Adjusted start to prevent overlap")
                
        # Create adjusted subtitle
        adjusted_sub = pysrt.SubRipItem(
            index=modified_sub.index,
            start=new_start,
            end=fixed_end,
            text=modified_sub.text
        )
        
        # Record the modification
        original_duration = (modified_sub.end.ordinal - modified_sub.start.ordinal) / 1000.0
        new_duration = (adjusted_sub.end.ordinal - adjusted_sub.start.ordinal) / 1000.0
        
        modifications = [{
            'type': 'content_based_timing_adjustment',
            'subtitle_index': modified_sub.index,
            'original_start': modified_sub.start,
            'new_start': new_start,
            'end_timestamp': fixed_end,
            'original_duration': original_duration,
            'new_duration': new_duration,
            'original_text_length': len(original_sub.text),
            'modified_text_length': len(modified_sub.text),
            'reason': 'content_changed_end_fixed_adjustment',
            'strategy': 'user_condition_a'
        }]
        
        logger.debug(f"Sub {modified_sub.index}: Content-based timing: {original_duration:.1f}s -> {new_duration:.1f}s")
        return adjusted_sub, modifications
        
    def _apply_threshold_based_timing_adjustment(self,
                                               modified_sub: pysrt.SubRipItem,
                                               previous_adjusted_subs: List[pysrt.SubRipItem]) -> Tuple[pysrt.SubRipItem, List[Dict]]:
        """Apply timing adjustment for duration threshold violations"""
        
        current_duration_s = (modified_sub.end.ordinal - modified_sub.start.ordinal) / 1000.0
        text_length = len(modified_sub.text.strip())
        
        # Calculate reasonable duration for content
        chars_per_second = self.cross_subtitle_constants.CHARS_PER_SECOND.get(
            self.primary_language,
            self.cross_subtitle_constants.CHARS_PER_SECOND['default']
        )
        
        reasonable_duration_s = max(
            text_length / chars_per_second,
            self.timing_constants.MIN_SUBTITLE_DURATION
        )
        reasonable_duration_s = min(reasonable_duration_s, self.timing_constants.MAX_SUBTITLE_DURATION)
        
        # Keep end fixed, adjust start
        fixed_end = modified_sub.end
        new_start_ordinal = fixed_end.ordinal - int(reasonable_duration_s * 1000)
        new_start = pysrt.SubRipTime(milliseconds=max(0, new_start_ordinal))
        
        # Prevent overlap with previous subtitle
        if previous_adjusted_subs:
            prev_sub = previous_adjusted_subs[-1]
            min_gap_ms = 50
            earliest_allowed_start = prev_sub.end.ordinal + min_gap_ms
            
            if new_start.ordinal < earliest_allowed_start:
                new_start = pysrt.SubRipTime(milliseconds=earliest_allowed_start)
                
        adjusted_sub = pysrt.SubRipItem(
            index=modified_sub.index,
            start=new_start,
            end=fixed_end,
            text=modified_sub.text
        )
        
        new_duration = (adjusted_sub.end.ordinal - adjusted_sub.start.ordinal) / 1000.0
        
        modifications = [{
            'type': 'threshold_violation_timing_adjustment',
            'subtitle_index': modified_sub.index,
            'original_start': modified_sub.start,
            'new_start': new_start,
            'end_timestamp': fixed_end,
            'original_duration': current_duration_s,
            'new_duration': new_duration,
            'text_length': text_length,
            'reason': 'duration_threshold_violation',
            'strategy': 'user_condition_b'
        }]
        
        logger.debug(f"Sub {modified_sub.index}: Threshold-based timing: {current_duration_s:.1f}s -> {new_duration:.1f}s")
        return adjusted_sub, modifications