#!/usr/bin/env python3
"""SRT post-processing module for hallucination and repetition removal."""

import pysrt
import unicodedata
import regex as re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import requests
from whisperjav.utils.logger import logger 


class SRTPostProcessor:
    """Post-process SRT files to remove hallucinations and repetitions."""
    
    # Default hallucination patterns for Japanese
    DEFAULT_HALLUCINATIONS = [
        "ご視聴ありがとうございました",
        "字幕作成者",
        "♪",
        "[音楽]",
        "(音楽)",
        "続きは次回",
        "お楽しみに",
        "提供",
        "スポンサー"
    ]
    
    def __init__(self,
                 patterns_url: Optional[str] = None,
                 remove_hallucinations: bool = True,
                 hallucination_patterns: Optional[List[str]] = None,
                 remove_repetitions: bool = True,
                 repetition_threshold: int = 2,
                 min_subtitle_duration: float = 0.5,
                 max_subtitle_duration: float = 7.0):
        self.remove_hallucinations = remove_hallucinations
        self.hallucination_patterns = hallucination_patterns or self.DEFAULT_HALLUCINATIONS
        self.remove_repetitions = remove_repetitions
        self.repetition_threshold = repetition_threshold
        self.min_subtitle_duration = min_subtitle_duration
        self.max_subtitle_duration = max_subtitle_duration
        
        # Load patterns from URL if provided
        self.patterns = []
        if patterns_url:
            self.patterns = self._load_patterns_from_url(patterns_url)
            
    def _load_patterns_from_url(self, url: str) -> List[Dict]:
        """Load regex patterns from URL."""
        try:
            logger.info(f"Loading patterns from: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('patterns', [])
        except Exception as e:
            logger.warning(f"Failed to load patterns from URL: {e}")
            return []
            
    def process(self, srt_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Tuple[Path, Dict]:
        """Process SRT file and return path to processed file and statistics."""
        srt_path = Path(srt_path)
        if output_path is None:
            output_path = srt_path.parent / f"{srt_path.stem}_processed.srt"
        else:
            output_path = Path(output_path)
            
        logger.info(f"Processing SRT: {srt_path}")
        
        # Load subtitles
        try:
            subs = pysrt.open(str(srt_path), encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to load SRT file: {e}")
            raise
            
        stats = {
            'total_subtitles': len(subs),
            'removed_hallucinations': 0,
            'removed_repetitions': 0,
            'duration_adjustments': 0,
            'empty_removed': 0
        }
        
        # Process subtitles
        processed_subs = []
        
        for sub in subs:
            original_text = sub.text
            
            # Skip empty subtitles
            if not original_text or not original_text.strip():
                stats['empty_removed'] += 1
                continue
                
            # Remove hallucinations
            if self.remove_hallucinations:
                text, hallucination_removed = self._remove_hallucinations(original_text)
                if hallucination_removed:
                    stats['removed_hallucinations'] += 1
                    sub.text = text
                    
            # Remove repetitions
            if self.remove_repetitions and sub.text:
                text, repetition_removed = self._remove_repetitions(sub.text)
                if repetition_removed:
                    stats['removed_repetitions'] += 1
                    sub.text = text
                    
            # Skip if text was completely removed
            if not sub.text or not sub.text.strip():
                stats['empty_removed'] += 1
                continue
                
            # Adjust duration if needed
            duration = self._get_duration_seconds(sub)
            if duration < self.min_subtitle_duration or duration > self.max_subtitle_duration:
                self._adjust_duration(sub)
                stats['duration_adjustments'] += 1
                
            processed_subs.append(sub)
            
        # Renumber subtitles
        for i, sub in enumerate(processed_subs, 1):
            sub.index = i
            
        # Save processed subtitles
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_file = pysrt.SubRipFile(processed_subs)
        processed_file.save(str(output_path), encoding='utf-8')
        
        logger.info(f"Processed SRT saved to: {output_path}")
        logger.info(f"Statistics: {stats}")
        
        return output_path, stats
        
    def _remove_hallucinations(self, text: str) -> Tuple[str, bool]:
        """Remove known hallucination patterns from text."""
        original_text = text
        
        # Check against simple patterns
        for pattern in self.hallucination_patterns:
            if pattern in text:
                text = text.replace(pattern, "").strip()
                
        # Check against regex patterns if loaded
        for pattern_info in self.patterns:
            pattern = pattern_info.get('pattern', '')
            category = pattern_info.get('category', '')
            
            if category in ['meta_reference', 'media_reference', 'closing_phrase', 'nonsensical']:
                try:
                    text = re.sub(pattern, '', text).strip()
                except Exception:
                    pass
                    
        return text, (text != original_text)
        
    def _remove_repetitions(self, text: str) -> Tuple[str, bool]:
        """Remove excessive repetitions from text."""
        original_text = text
        
        # Handle character repetitions (あああああ -> ああ)
        pattern = r'(([あアぁァいイぃィうウぅゥえエぇェおオぉォはハぱパんンっッ])\2{' + str(self.repetition_threshold) + ',})'
        
        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):
            full_match = match.group(1)
            char = match.group(2)
            replacement = char * self.repetition_threshold
            start, end = match.span(1)
            text = text[:start] + replacement + text[end:]
            
        # Handle word repetitions
        words = text.split()
        if len(words) > 1:
            # Count consecutive repetitions
            new_words = []
            i = 0
            while i < len(words):
                word = words[i]
                count = 1
                
                # Count consecutive same words
                while i + count < len(words) and words[i + count] == word:
                    count += 1
                    
                # Add limited repetitions
                new_words.extend([word] * min(count, self.repetition_threshold))
                i += count
                
            text = ' '.join(new_words)
            
        return text.strip(), (text != original_text)
        
    def _get_duration_seconds(self, sub: pysrt.SubRipItem) -> float:
        """Get subtitle duration in seconds."""
        duration_ms = ((sub.end.hours - sub.start.hours) * 3600000 +
                      (sub.end.minutes - sub.start.minutes) * 60000 +
                      (sub.end.seconds - sub.start.seconds) * 1000 +
                      (sub.end.milliseconds - sub.start.milliseconds))
        return duration_ms / 1000.0
        
    def _adjust_duration(self, sub: pysrt.SubRipItem):
        """Adjust subtitle duration to be within acceptable range."""
        current_duration = self._get_duration_seconds(sub)
        
        if current_duration < self.min_subtitle_duration:
            # Extend to minimum duration
            target_duration_ms = int(self.min_subtitle_duration * 1000)
        elif current_duration > self.max_subtitle_duration:
            # Reduce to maximum duration
            target_duration_ms = int(self.max_subtitle_duration * 1000)
        else:
            return
            
        # Calculate new end time
        new_end_ms = (sub.start.hours * 3600000 +
                     sub.start.minutes * 60000 +
                     sub.start.seconds * 1000 +
                     sub.start.milliseconds + target_duration_ms)
                     
        # Convert to time components
        new_end_hours = new_end_ms // 3600000
        new_end_ms %= 3600000
        new_end_minutes = new_end_ms // 60000
        new_end_ms %= 60000
        new_end_seconds = new_end_ms // 1000
        new_end_milliseconds = new_end_ms % 1000
        
        sub.end.hours = new_end_hours
        sub.end.minutes = new_end_minutes
        sub.end.seconds = new_end_seconds
        sub.end.milliseconds = new_end_milliseconds