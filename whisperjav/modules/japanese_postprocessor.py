#!/usr/bin/env python3
"""
Japanese-specific post-processing for ASR results.

This module provides Japanese dialogue post-processing that can be used by
multiple ASR backends (StableTSASR, QwenASR, etc.). It uses stable-ts methods
to regroup word timestamps into natural Japanese conversational segments.

Features:
- Removes fillers and aizuchi (backchanneling)
- Anchors on sentence endings, polite forms, and quote boundaries
- Merges by punctuation and gaps
- Splits by length and duration for readability
- Supports presets: "default", "high_moan", "narrative"

Usage:
    from whisperjav.modules.japanese_postprocessor import JapanesePostProcessor

    processor = JapanesePostProcessor()
    result = processor.process(whisper_result, preset="default")
"""

import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import stable_whisper

from whisperjav.utils.logger import logger


@dataclass
class JapaneseLinguisticSets:
    """
    Japanese linguistic patterns for dialogue segmentation.

    These patterns are used to identify natural sentence boundaries in
    Japanese conversational speech. They cover:
    - Sentence-final particles (ね, よ, わ, etc.)
    - Polite verb endings (です, ます, etc.)
    - Dialectal variations (Kansai-ben, feminine/masculine speech)
    - Backchanneling (aizuchi) and fillers
    - Emotional interjections
    """

    # Base sentence-final particles
    base_endings: List[str] = field(default_factory=lambda: [
        'ね', 'よ', 'わ', 'の', 'ぞ', 'ぜ', 'さ', 'か', 'かな', 'な'
    ])

    # Modern casual endings
    modern_endings: List[str] = field(default_factory=lambda: [
        'じゃん', 'っしょ', 'んだ', 'わけ', 'かも', 'だろう'
    ])

    # Kansai dialect endings
    kansai_endings: List[str] = field(default_factory=lambda: [
        'わ', 'で', 'ねん', 'な', 'や'
    ])

    # Feminine speech patterns
    feminine_endings: List[str] = field(default_factory=lambda: [
        'かしら', 'こと', 'わね', 'のよ'
    ])

    # Masculine speech patterns
    masculine_endings: List[str] = field(default_factory=lambda: [
        'ぜ', 'ぞ', 'だい', 'かい'
    ])

    # Polite verb forms
    polite_forms: List[str] = field(default_factory=lambda: [
        'です', 'ます', 'でした', 'ましょう', 'ませんか'
    ])

    # Casual contractions that form natural boundaries
    casual_contractions: List[str] = field(default_factory=lambda: [
        'ちゃ', 'じゃ', 'きゃ', 'にゃ', 'ひゃ', 'みゃ', 'りゃ'
    ])

    # Question particles
    question_particles: List[str] = field(default_factory=lambda: ['の', 'か'])

    # Conversational verbal endings (contracted forms)
    conversational_verbal_endings: List[str] = field(default_factory=lambda: [
        'てる',   # from -te iru
        'でる',   # from -de iru
        'ちゃう', # from -te shimau
        'じゃう', # from -de shimau
        'とく',   # from -te oku
        'どく',   # from -de oku
        'んない'  # from -ranai
    ])

    # Backchanneling and fillers (aizuchi)
    aizuchi_fillers: List[str] = field(default_factory=lambda: [
        'あの', 'あのー', 'ええと', 'えっと', 'まあ', 'なんか', 'こう',
        'うん', 'はい', 'ええ', 'そう', 'えっ', 'あっ',
    ])

    # Emotional/expressive interjections
    expressive_emotions: List[str] = field(default_factory=lambda: [
        'ああ', 'うう', 'ええ', 'おお', 'はあ', 'ふう', 'あっ',
        'うっ', 'はっ', 'ふっ', 'んっ'
    ])

    def get_all_final_endings(self) -> List[str]:
        """Get combined list of all sentence-final patterns."""
        return list(set(
            self.base_endings +
            self.modern_endings +
            self.kansai_endings +
            self.feminine_endings +
            self.masculine_endings
        ))


@dataclass
class PresetParameters:
    """
    Preset parameters for different content types.

    Presets:
    - default: General conversational dialogue
    - high_moan: Adult content with frequent short vocalizations
    - narrative: Longer narrative passages, documentaries
    """
    gap_threshold: float  # Minimum gap (seconds) before splitting
    segment_length: int   # Maximum characters per segment


class JapanesePostProcessor:
    """
    Japanese-specific post-processor for stable-ts WhisperResult objects.

    This processor applies a multi-pass regrouping strategy to convert
    word-level timestamps into natural Japanese conversational segments.

    Usage:
        processor = JapanesePostProcessor()
        result = processor.process(whisper_result, preset="default")
    """

    # Preset configurations
    PRESETS = {
        "default": PresetParameters(gap_threshold=0.3, segment_length=35),
        "high_moan": PresetParameters(gap_threshold=0.1, segment_length=25),
        "narrative": PresetParameters(gap_threshold=0.4, segment_length=45),
    }

    def __init__(self):
        """Initialize with default linguistic patterns."""
        self.ling = JapaneseLinguisticSets()

    def process(
        self,
        result: stable_whisper.WhisperResult,
        preset: str = "default",
        language: Optional[str] = None,
        skip_if_not_japanese: bool = True
    ) -> stable_whisper.WhisperResult:
        """
        Apply Japanese-specific post-processing to ASR result.

        This method modifies the result in-place and also returns it for
        method chaining.

        Args:
            result: WhisperResult from stable-ts (must have word timestamps)
            preset: Processing preset ("default", "high_moan", "narrative")
            language: Detected language code (e.g., "ja", "Japanese")
            skip_if_not_japanese: If True, skip processing for non-Japanese

        Returns:
            The modified WhisperResult (same object, modified in-place)

        Raises:
            ValueError: If preset is invalid
        """
        # Validate preset
        if preset not in self.PRESETS:
            logger.warning(f"Unknown preset '{preset}', using 'default'")
            preset = "default"

        params = self.PRESETS[preset]

        # Check language if requested
        if skip_if_not_japanese and language:
            lang_lower = language.lower()
            # Accept "ja", "japanese", "jp" as Japanese
            if lang_lower not in ('ja', 'japanese', 'jp', 'jpn'):
                logger.debug(f"Skipping Japanese post-processing for language: {language}")
                return result

        # Check for segments
        if not result.segments:
            logger.debug("No segments found in result. Skipping post-processing.")
            return result

        logger.debug(f"Applying Japanese post-processing with '{preset}' preset")
        logger.debug(f"  gap_threshold={params.gap_threshold}s, segment_length={params.segment_length} chars")

        try:
            self._apply_processing_passes(result, params)
            logger.debug("Japanese post-processing complete")
        except Exception as e:
            self._handle_processing_error(e)

        return result

    def _apply_processing_passes(
        self,
        result: stable_whisper.WhisperResult,
        params: PresetParameters
    ) -> None:
        """
        Apply the multi-pass processing pipeline.

        Pass 1: Sanitization (remove fillers/aizuchi)
        Pass 2: Structural anchoring (lock sentence boundaries)
        Pass 3: Merging (combine by punctuation and gaps)
        Pass 4: Formatting (split for readability)
        """
        # --- Pass 1: Remove Fillers and Aizuchi ---
        logger.debug("Phase 1: Sanitization (Fillers and Aizuchi Removal)")
        result.remove_words_by_str(
            self.ling.aizuchi_fillers,
            case_sensitive=False,
            strip=True,
            verbose=False
        )

        # --- Pass 2: Structural Anchoring ---
        logger.debug("Phase 2: Structural Anchoring")

        # 2a: Split by strong punctuation marks (full-width and half-width)
        result.regroup("sp= 。 / ？ / ！ / … / ． /. /? /!+1")

        # 2b: Lock known structural boundaries
        # Quote boundaries (「」『』)
        result.lock(startswith=['「', '『'], left=True, right=False, strip=False)
        result.lock(endswith=['」', '』'], right=True, left=False, strip=False)

        # Sentence-final particles and patterns
        result.lock(
            endswith=self.ling.get_all_final_endings(),
            right=True,
            left=False,
            strip=True
        )

        # Polite forms
        result.lock(endswith=self.ling.polite_forms, right=True, strip=True)

        # Question particles
        result.lock(endswith=self.ling.question_particles, right=True, strip=True)

        # Conversational verbal endings
        for ending in self.ling.conversational_verbal_endings:
            result.custom_operation('word', 'end', ending, 'lockright', word_level=True)

        # 2c: Lock expressive/emotive interjections
        result.lock(
            startswith=self.ling.expressive_emotions,
            endswith=self.ling.expressive_emotions,
            left=True,
            right=True
        )

        # --- Pass 3: Merge & Heuristic Refinement ---
        logger.debug("Phase 3: Heuristic-Based Merging & Refinement")

        # Merge by comma-like punctuation
        result.merge_by_punctuation(
            punctuation=['、', '，', ','],
            max_chars=40,
            max_words=15
        )

        # Merge by time gap
        result.merge_by_gap(
            min_gap=params.gap_threshold,
            max_chars=40,
            max_words=15,
            is_sum_max=True
        )

        # Split by long pause (conversation break)
        result.split_by_gap(max_gap=1.2)

        # --- Pass 4: Final Cleanup & Formatting ---
        logger.debug("Phase 4: Final Cleanup & Formatting")

        # Split by character length for readability
        result.split_by_length(
            max_chars=params.segment_length,
            max_words=15,
            even_split=False
        )

        # Safety: split very long segments by duration
        result.split_by_duration(max_dur=8.5, even_split=False, lock=True)

        # Reassign segment IDs
        result.reassign_ids()

    def _handle_processing_error(self, error: Exception) -> None:
        """
        Handle errors during processing gracefully.

        Word timestamp errors are logged at DEBUG level since they're common
        when using ASR without forced alignment. Other errors are logged as
        ERROR.
        """
        error_msg = str(error).lower()

        # Check for word timestamp related errors (common and expected)
        word_ts_keywords = [
            'word timestamp', 'word_level', 'word-level',
            'missing word', 'no word', 'cannot clamp'
        ]

        if any(keyword in error_msg for keyword in word_ts_keywords):
            logger.debug(f"Japanese post-processing skipped - no word timestamps: {error}")
        else:
            # Unexpected error - log at ERROR level
            logger.error(f"Error during Japanese post-processing: {error}")
            logger.debug(traceback.format_exc())


def create_japanese_postprocessor() -> JapanesePostProcessor:
    """
    Factory function to create a JapanesePostProcessor instance.

    Returns:
        Configured JapanesePostProcessor instance
    """
    return JapanesePostProcessor()
