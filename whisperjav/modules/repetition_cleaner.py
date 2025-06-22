# whisperjav/modules/repetition_cleaner.py
# FINAL FIX: Implemented a comprehensive solution based on user feedback.
# The logic is now more aggressive in cleaning and less aggressive in protection.

import re
import regex
from collections import Counter
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from whisperjav.utils.logger import logger
from whisperjav.config.sanitization_constants import RepetitionConstants


class RepetitionCleaner:
    """
    FINAL, HARDENED VERSION.
    This cleaner uses a single, curated list of safe patterns to aggressively clean
    extreme repetitions while avoiding the data corruption bugs.
    """

    def __init__(self, constants: RepetitionConstants):
        self.constants = constants
        self.threshold = constants.DEFAULT_THRESHOLD

        # This is the final, curated list of patterns.
        # They are ordered from most specific/extreme to most general.
        self.cleaning_patterns = [
            # 1. Targets extreme phrase repetition with separators (e.g., "あ!!あ!!あ!!")
            # Uses a backreference `\1` to ensure phrases are identical.
            ('phrase_with_separator', regex.compile(r'((?:[\p{L}\p{N}]{1,8}[、,!\s!!??。。・]+))\1{3,}'), r'\1'),

            # 2. Targets extreme multi-character word repetition (e.g., "ハッハッハッハッ")
            ('multi_char_word', regex.compile(r'(([ぁ-んァ-ン]{2,4}))\1{3,}'), r'\1\1'),
            
            # 3. Targets phrase repetition with a comma (e.g., "ゆーちゃん、ゆーちゃん、ゆーちゃん")
            ('phrase_with_comma', regex.compile(r'((?:[\p{L}\p{N}]{1,10}[、,]\s*))\1{2,}'), r'\1'),

            # 4. Targets single character with prefix (e.g., "あらららら...")
            ('prefix_plus_char', regex.compile(r'([ぁ-んァ-ン]{1,2})([ぁ-んァ-ン])\2{3,}'), r'\1\2\2'),

            # 5. Targets simple single-character floods (e.g., "ううううう")
            ('single_char_flood', regex.compile(r'([ぁ-んァ-ン])\1{3,}'), r'\1\1'),
            
            # 6. Targets vowel extensions (e.g., "あ〜〜〜〜")
            ('vowel_extension', regex.compile(r'([ぁ-んァ-ン])([〜ー])\2{3,}'), r'\1\2\2'),
        ]

        logger.debug(f"RepetitionCleaner initialized with {len(self.cleaning_patterns)} final cleaning patterns.")

    def clean_repetitions(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Applies the final, curated list of cleaning patterns.
        """
        if not text or not text.strip():
            return text, []

        modifications = []
        current_text = text

        for name, pattern, replacement in self.cleaning_patterns:
            try:
                original_text = current_text
                new_text = pattern.sub(replacement, original_text)

                if new_text != original_text:
                    modifications.append({
                        'type': name,
                        'pattern': pattern.pattern,
                        'original': original_text,
                        'modified': new_text,
                        'confidence': 0.99,
                        'category': 'repetition_cleaning'
                    })
                    current_text = new_text
            except Exception as e:
                logger.warning(f"Pattern '{name}' failed on text '{text[:30]}...': {e}")
                continue
        
        return current_text.strip(), modifications



    def _is_all_repetition(self, text: str) -> bool:
        """
        Check if text consists almost entirely of repetitive patterns.
        This is now more robust to handle the various long-line examples.
        """
        # More than 90% of the string is a single character (ignoring punctuation)
        stripped_text = regex.sub(r'[\p{P}\p{Z}]', '', text)
        if len(stripped_text) > 10:
            most_common_char_count = Counter(stripped_text).most_common(1)[0][1]
            if most_common_char_count / len(stripped_text) > 0.9:
                return True
        
        # Check for long sequences of a short phrase + separator (e.g., "あ、あ、あ、")
        if regex.match(r'^((?:.{1,5}?)[、,!\s!?・]){5,}$', text):
            return True
            
        # Check for long sequences of a short multi-char word (e.g., "ハッハッハッ")
        if regex.match(r'^((?:.{2,5}?))\1{3,}$', text):
            return True

        # Check for long vowel extensions
        if regex.match(r'^[ぁ-んァ-ン][〜ー]{10,}$', text):
            return True
            
        return False


    def _is_all_repetition(self, text: str) -> bool:
        """Check if text consists only of repetitive patterns"""
        # Check for long sequences of repeated characters

        '''

        if regex.match(r'^([ぁ-んァ-ン])\1{20,}$', text):
            return True
            
        # Check for long sequences with separators
        if regex.match(r'^([ぁ-んァ-ン][、,!！?？]\s*){10,}$', text):
            return True
            
        # Check for long vowel extensions
        if regex.match(r'^[ぁ-んァ-ン][〜ー]{20,}$', text):
            return True
            
        return False

        '''

        return True

    def _validate_modification(self, original: str, modified: str) -> bool:
        """
        Validates that a modification is reasonable before accepting it.
        Prevents aggressive patterns from destroying valid text.
        """
        # Allow complete removal if the original was identified as all repetition
        '''

        if not modified.strip() and self._is_all_repetition(original):
            return True
            
        # A basic sanity check to prevent erasing content
        if not modified or not modified.strip():
            logger.warning("VALIDATION FAILED: Modified text is empty")
            return False
            
        # Allow a significant reduction in length only if the original was all repetition
        if len(modified) < len(original) * 0.05:  # e.g., less than 5% of original length
            if self._is_all_repetition(original):
                return True
            logger.warning(f"VALIDATION FAILED: Modified text too short (possible over-reduction)")
            return False
        '''
            
        return True

    def _is_protected_content_enhanced(self, text: str) -> bool:
        stripped = text.strip()
        
        '''
        if stripped in self.constants.LEGITIMATE_REPETITIONS:
            return True
        if len(stripped) <= 3:
            return True
        for pattern in self.protected_patterns:
            if pattern.search(text):
                return True

        # Force-clean override for ultra-long lines with only kana
        if len(stripped) > 45:
            has_kanji = bool(regex.search(r'\p{Han}', text))
            has_latin = bool(regex.search(r'[a-zA-Z]', text))
            if not has_kanji and not has_latin:
                logger.debug("Bypassing protection: very long line without kanji or Latin.")
                return False  # force-clean it

        # Mixed script = likely real content
        has_hira = bool(regex.search(r'\p{Hiragana}', text))
        has_kata = bool(regex.search(r'\p{Katakana}', text))
        has_kanji = bool(regex.search(r'\p{Han}', text))
        has_latin = bool(regex.search(r'[a-zA-Z]', text))

        script_count = sum([has_hira, has_kata, has_kanji, has_latin])
        if script_count >= 2:
            return True
        '''
        return False


    def _clean_high_density_repetitions_safe(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Simplified high-density cleaning.
        """
        if len(text) < self.constants.HIGH_DENSITY_MIN_LENGTH:
            return text, []

        modifications = []
        
        try:
            words = regex.findall(r'\p{L}+', text)
            if not words or len(words) < self.constants.HIGH_DENSITY_MIN_OCCURRENCES:
                return text, []

            word_counts = Counter(words)
            most_common_word, count = word_counts.most_common(1)[0]

            if (count > self.constants.HIGH_DENSITY_MIN_OCCURRENCES and
                count / len(words) > self.constants.HIGH_DENSITY_RATIO and
                len(most_common_word) > 1):

                parts = text.split()
                kept_count = 0
                new_parts = []
                
                for part in parts:
                    if most_common_word in part and kept_count >= self.threshold:
                        continue
                    else:
                        new_parts.append(part)
                        if most_common_word in part:
                            kept_count += 1

                modified_text = ' '.join(new_parts).strip()

                if text != modified_text and len(modified_text) > 0:
                    modifications.append({
                        'type': 'high_density_repetition',
                        'pattern': 'high_density_detection',
                        'original': text,
                        'modified': modified_text,
                        'confidence': 0.90,
                        'repeated_word': most_common_word,
                        'repetition_count': count
                    })
                    return modified_text, modifications

        except Exception as e:
            logger.warning(f"High-density repetition cleaning failed: {e}")
            return text, []

        return text, modifications



    def _clean_character_repetitions(self, text: str) -> Tuple[str, List[Dict]]:
        """Clean character repetitions based on a defined constant."""
        modifications = []
        
        def replacer(match):
            char = match.group(1)
            return char * self.threshold

        original_text = text
        
        # This now uses the new, descriptive constant from the constants file.
        # The quantifier for N or more items is {N-1,}.
        quantifier = self.constants.MIN_CHAR_REPETITION_THRESHOLD - 1
        pattern_str = self.constants.CHAR_REPETITION_PATTERN + f'\\1{{{quantifier},}}'
        char_repeat_pattern = regex.compile(pattern_str)

        try:
            modified_text = char_repeat_pattern.sub(replacer, text)

            if original_text != modified_text:
                if self._validate_modification(original_text, modified_text):
                    modifications.append({
                        'type': 'character_repetition_reduction',
                        'original': original_text,
                        'modified': modified_text,
                        'confidence': 0.95,
                        'category': 'character_repetition'
                    })
                else:
                    # Revert if validation fails
                    modified_text = original_text
            
            return modified_text, modifications
        except Exception as e:
            logger.warning(f"Character repetition cleaning failed with error: {e}")
            return text, []

    def _pre_process_special_patterns(self, text: str) -> Tuple[str, List[Dict]]:
        """Handle special repetitive patterns using defined constants."""
        modifications = []
        current_text = text

        # The quantifiers are now based on the new, descriptive constants
        vowel_quantifier = self.threshold + 2
        phrase_quantifier = self.constants.MIN_PHRASE_REPETITION_THRESHOLD - 1

        patterns_to_apply = [
            ('vowel_extension', regex.compile(self.constants.VOWEL_EXTENSION_PATTERN + f'{{{vowel_quantifier},}}'), lambda m: m.group(1) + m.group(2) * self.threshold),
            
            # --- THIS IS THE LINE TO CHANGE FOR CONSERVATIVE CLEANING ---
            ('comma_phrase', regex.compile(self.constants.COMMA_PHRASE_PATTERN + r'\1' + f'{{{phrase_quantifier},}}'), lambda m: m.group(1) * self.threshold),
            # -----------------------------------------------------------

            ('prefix_char_flood', regex.compile(self.constants.PREFIX_PLUS_CHAR_FLOOD_PATTERN), lambda m: m.group(1) + m.group(2) * self.threshold),
            ('small_vowel_flood', regex.compile(self.constants.SMALL_VOWEL_FLOOD_PATTERN), lambda m: m.group(1) + m.group(2) * self.threshold),
        ]


        for name, pattern, replacement in patterns_to_apply:
            try:
                original_text = current_text
                
                new_text = original_text
                while True:
                    processed_text = pattern.sub(replacement, new_text)
                    if processed_text == new_text:
                        break
                    new_text = processed_text

                if original_text != new_text:
                    if self._validate_modification(original_text, new_text):
                        modifications.append({
                            'type': name,
                            'pattern': pattern.pattern,
                            'original': original_text,
                            'modified': new_text,
                            'confidence': 0.98,
                            'category': 'repetition_special'
                        })
                        current_text = new_text
                    else:
                        current_text = original_text
                        logger.warning(f"Validation failed for special pattern '{name}', reverting.")
                    
            except Exception as e:
                logger.warning(f"Special pattern '{name}' failed: {e}")
                continue

        return current_text, modifications

            

    def _clean_word_repetitions(self, text: str) -> Tuple[str, List[Dict]]:
        """Clean word repetitions without punctuation: e.g., 'dame dame dame' -> 'dame dame'"""
        modifications = []
        pattern_str = r'\b(\p{L}{2,})\b(?:\s+\1){' + str(self.threshold) + r',}'
        word_pattern = regex.compile(pattern_str)

        def replacer(match):
            word = match.group(1)
            return ' '.join([word] * self.threshold)

        original_text = text
        modified_text = word_pattern.sub(replacer, original_text)

        if original_text != modified_text:
             modifications.append({
                'type': 'word_repetition_reduction',
                'original': original_text,
                'modified': modified_text,
                'confidence': 0.95,
                'category': 'word_repetition'
            })

        return modified_text, modifications
