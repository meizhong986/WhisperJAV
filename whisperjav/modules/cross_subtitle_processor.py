# whisperjav/modules/cross_subtitle_processor.py

import pysrt
from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Optional
from collections import Counter
from datetime import timedelta
from whisperjav.utils.logger import logger
from whisperjav.config.sanitization_constants import CrossSubtitleConstants

class CrossSubtitleProcessor:
    """Inter-subtitle analysis and merging"""
    
    def __init__(self, constants: CrossSubtitleConstants, primary_language: Optional[str] = None):
        self.constants = constants
        self.primary_language = primary_language or 'default'
        
    def process_cross_subtitle(self, subtitles: List[pysrt.SubRipItem]) -> Tuple[List[pysrt.SubRipItem], List[Dict]]:
        """Process cross-subtitle repetitions and merging"""
        if len(subtitles) <= 1:
            return subtitles, []
            
        modifications = []
        
        # Step 1: Merge consecutive repetitions
        merged_subs, merge_mods = self._merge_consecutive_repetitions(subtitles)
        modifications.extend(merge_mods)
        
        # Step 2: Detect and handle high-density repetitions across subtitles
        processed_subs, density_mods = self._process_high_density_cross_subtitle(merged_subs)
        modifications.extend(density_mods)
        
        return processed_subs, modifications
        
    def _merge_consecutive_repetitions(self, subtitles: List[pysrt.SubRipItem]) -> Tuple[List[pysrt.SubRipItem], List[Dict]]:
        """Merge consecutive subtitles with similar content"""
        if len(subtitles) < self.constants.DEDUP_THRESHOLD:
            return subtitles, []
            
        new_subs = []
        modifications = []
        i = 0
        
        while i < len(subtitles):
            current_sub = subtitles[i]
            
            # Collect consecutive similar subtitles
            similar_group = [current_sub]
            j = i + 1
            
            while j < len(subtitles):
                next_sub = subtitles[j]
                
                # Calculate time gap
                gap_ms = self._calculate_gap_ms(similar_group[-1], next_sub)
                
                if gap_ms > self.constants.MAX_GAP_MS:
                    break
                    
                # Calculate similarity
                similarity = self._calculate_similarity(current_sub.text, next_sub.text)
                
                if similarity >= self.constants.MERGE_SIMILARITY_THRESHOLD:
                    similar_group.append(next_sub)
                    j += 1
                else:
                    break
                    
            # Process the group
            if len(similar_group) >= self.constants.DEDUP_THRESHOLD:
                # Merge the group
                merged_sub = self._merge_subtitle_group(similar_group)
                new_subs.append(merged_sub)
                
                modifications.append({
                    'type': 'subtitle_merge',
                    'subtitles_merged': len(similar_group),
                    'original_indices': [s.index for s in similar_group],
                    'original_texts': [s.text for s in similar_group],
                    'merged_text': merged_sub.text,
                    'confidence': 0.9
                })
                
                i = j
            else:
                # Keep original subtitle
                new_subs.append(current_sub)
                i += 1
                
        return new_subs, modifications
        
    def _process_high_density_cross_subtitle(self, subtitles: List[pysrt.SubRipItem]) -> Tuple[List[pysrt.SubRipItem], List[Dict]]:
        """Detect and process high-density repetitions across multiple subtitles"""
        if len(subtitles) < 3:
            return subtitles, []
            
        modifications = []
        
        # Analyze phrase frequency across all subtitles
        all_text = ' '.join(sub.text for sub in subtitles)
        if len(all_text) < 50:
            return subtitles, []
            
        # Find most common phrases
        phrase_counts = self._count_phrases_across_subtitles(subtitles)
        if not phrase_counts:
            return subtitles, []
            
        # Process high-frequency phrases
        for phrase, count in phrase_counts.most_common(3):
            if count < 5 or len(phrase) < 2:
                continue
                
            # Calculate density
            total_words = sum(len(sub.text.split()) for sub in subtitles)
            density = count / max(total_words, 1)
            
            if density > 0.2:  # If phrase appears in >20% of words
                # Mark subtitles containing this phrase
                affected_indices = []
                for i, sub in enumerate(subtitles):
                    if phrase in sub.text:
                        affected_indices.append(i)
                        
                if len(affected_indices) > 3:
                    modifications.append({
                        'type': 'high_density_cross_subtitle',
                        'phrase': phrase,
                        'occurrences': count,
                        'density': density,
                        'affected_subtitle_indices': affected_indices,
                        'confidence': 0.8
                    })
                    
        return subtitles, modifications
        

    def _calculate_gap_ms(self, sub1: pysrt.SubRipItem, sub2: pysrt.SubRipItem) -> int:
        """Calculate gap between two subtitles in milliseconds"""
        # Handles overlaps gracefully
        if sub2.start < sub1.end:
            return 0

        # This part only runs for positive gaps, which is safe
        gap = sub2.start - sub1.end
        #return int(gap)
        
        return gap.milliseconds + gap.seconds * 1000 + gap.minutes * 60000 + gap.hours * 3600000
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.strip(), text2.strip()).ratio()
        

    def _merge_subtitle_group(self, group: List[pysrt.SubRipItem]) -> pysrt.SubRipItem:
        """Merge a group of similar subtitles"""
        # Combine unique texts
        texts = []
        seen = set()
        for sub in group:
            if sub.text not in seen:
                texts.append(sub.text)
                seen.add(sub.text)
                
        merged_text = ' '.join(texts)
        
        # Calculate appropriate duration
        chars_per_second = self.constants.CHARS_PER_SECOND.get(
            self.primary_language, 
            self.constants.CHARS_PER_SECOND['default']
        )
        ideal_duration = len(merged_text) / chars_per_second
        
        # Use first subtitle's start and adjust end time
        start_time = group[0].start
        end_time = group[-1].end
        
        # Calculate duration in seconds using ordinal property (milliseconds)
        duration_ms = end_time.ordinal - start_time.ordinal
        duration_seconds = duration_ms / 1000.0
        
        # Ensure minimum duration
        min_duration = max(ideal_duration, 1.0)
        if duration_seconds < min_duration:
            end_time = start_time + pysrt.SubRipTime(milliseconds=int(min_duration * 1000))
            
        return pysrt.SubRipItem(
            index=group[0].index,
            start=start_time,
            end=end_time,
            text=merged_text
        )
        
    def _count_phrases_across_subtitles(self, subtitles: List[pysrt.SubRipItem]) -> Counter:
        """Count phrase occurrences across all subtitles"""
        from collections import Counter
        import regex
        
        phrase_counts = Counter()
        
        for sub in subtitles:
            # Extract words and phrases
            words = regex.findall(r'(\p{Hiragana}+|\p{Katakana}+|[一-龯々ヶ]+)', sub.text)
            
            # Count individual words
            for word in words:
                if len(word) >= 2:
                    phrase_counts[word] += 1
                    
            # Count 2-word phrases
            for i in range(len(words) - 1):
                phrase = words[i] + words[i + 1]
                if len(phrase) >= 3:
                    phrase_counts[phrase] += 1
                    
        return phrase_counts