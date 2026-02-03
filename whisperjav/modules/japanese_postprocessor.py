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

    # === Hierarchical Splitting Patterns (v1.8.5+) ===
    # These are used for unpunctuated text segmentation

    # Level 1: Definite sentence-final verb endings (ALWAYS split after these)
    # These ALWAYS end sentences in Japanese - unconditional split
    definite_endings: List[str] = field(default_factory=lambda: [
        'です',      # Copula polite
        'ます',      # Verb polite
        'でした',    # Copula polite past
        'ました',    # Verb polite past
        'ません',    # Verb polite negative
        'ましょう',  # Verb polite volitional
        'でしょう',  # Copula polite presumptive
        'ください',  # Please (request)
        'ございます', # Honorific copula
    ])

    # Level 2: Strong sentence-final particles (split with small gap)
    # These are emphatic and almost always sentence-final
    strong_particles: List[str] = field(default_factory=lambda: [
        'よ',   # Emphatic assertion
        'ぞ',   # Masculine emphatic
        'ぜ',   # Masculine emphatic (casual)
        'わよ', # Feminine emphatic
        'のよ', # Feminine emphatic explanation
        'ぜよ', # Dialectal emphatic (Kochi)
    ])

    # Level 3: Soft sentence-final particles (split with larger gap)
    # These CAN be sentence-final but also appear mid-sentence
    soft_particles: List[str] = field(default_factory=lambda: [
        'ね',   # Confirmation seeking / softening
        'な',   # Masculine soft ending / self-directed
        'わ',   # Feminine soft ending
        'の',   # Explanation / question (context-dependent)
        'かな', # Wondering / self-question
        'っけ', # Trying to remember
        'さ',   # Casual assertion
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

    # Gap thresholds for hierarchical splitting (v1.8.5+)
    # These are used for particle + gap condition splitting
    strong_particle_gap: float = 0.25  # よ, ぞ, ぜ - emphatic particles
    soft_particle_gap: float = 0.4     # ね, な, わ, の - softer particles
    pure_gap_threshold: float = 0.6    # Fallback: any word followed by this gap


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
        "default": PresetParameters(
            gap_threshold=0.3,
            segment_length=35,
            strong_particle_gap=0.25,
            soft_particle_gap=0.4,
            pure_gap_threshold=0.6
        ),
        "high_moan": PresetParameters(
            gap_threshold=0.1,
            segment_length=25,
            strong_particle_gap=0.15,  # More aggressive for short utterances
            soft_particle_gap=0.25,
            pure_gap_threshold=0.4
        ),
        "narrative": PresetParameters(
            gap_threshold=0.4,
            segment_length=45,
            strong_particle_gap=0.35,  # More conservative for longer phrases
            soft_particle_gap=0.5,
            pure_gap_threshold=0.7
        ),
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

        For UNPUNCTUATED text (like Qwen-ASR output), this uses hierarchical
        linguistic splitting. For punctuated text, standard regrouping is used.

        Pass 1: Sanitization (remove fillers/aizuchi)
        Pass 2: Hierarchical linguistic splitting (for unpunctuated text)
        Pass 3: Structural anchoring (lock remaining boundaries)
        Pass 4: Merging (combine by punctuation and gaps)
        Pass 5: Formatting (split for readability)
        """
        # --- Pass 1: Remove Fillers and Aizuchi ---
        logger.debug("Phase 1: Sanitization (Fillers and Aizuchi Removal)")
        result.remove_words_by_str(
            self.ling.aizuchi_fillers,
            case_sensitive=False,
            strip=True,
            verbose=False
        )

        # --- Pass 2: Hierarchical Linguistic Splitting (v1.8.5+) ---
        # This is the key change for handling unpunctuated Qwen-ASR output
        logger.debug("Phase 2: Hierarchical Linguistic Splitting")
        self._apply_hierarchical_splitting(result, params)

        # --- Pass 3: Structural Anchoring ---
        logger.debug("Phase 3: Structural Anchoring")

        # Lock known structural boundaries (don't split across these)
        # Quote boundaries (「」『』)
        result.lock(startswith=['「', '『'], left=True, right=False, strip=False)
        result.lock(endswith=['」', '』'], right=True, left=False, strip=False)

        # Lock sentence-final particles (already split, now protect them)
        result.lock(
            endswith=self.ling.get_all_final_endings(),
            right=True,
            left=False,
            strip=True
        )

        # Lock polite forms
        result.lock(endswith=self.ling.polite_forms, right=True, strip=True)

        # Lock question particles
        result.lock(endswith=self.ling.question_particles, right=True, strip=True)

        # Lock conversational verbal endings
        for ending in self.ling.conversational_verbal_endings:
            result.custom_operation('word', 'end', ending, 'lockright', word_level=True)

        # Lock expressive/emotive interjections
        result.lock(
            startswith=self.ling.expressive_emotions,
            endswith=self.ling.expressive_emotions,
            left=True,
            right=True
        )

        # --- Pass 4: Merge & Heuristic Refinement ---
        logger.debug("Phase 4: Heuristic-Based Merging & Refinement")

        # Merge by comma-like punctuation
        result.merge_by_punctuation(
            punctuation=['、', '，', ','],
            max_chars=40,
            max_words=15
        )

        # Merge by time gap (combine fragments that are close together)
        result.merge_by_gap(
            min_gap=params.gap_threshold,
            max_chars=40,
            max_words=15,
            is_sum_max=True
        )

        # --- Pass 5: Final Cleanup & Formatting ---
        logger.debug("Phase 5: Final Cleanup & Formatting")

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

    def _apply_hierarchical_splitting(
        self,
        result: stable_whisper.WhisperResult,
        params: PresetParameters
    ) -> None:
        """
        Apply hierarchical linguistic splitting for unpunctuated Japanese text.

        This method handles the core challenge of Qwen-ASR output: continuous
        text without punctuation. It uses a cascade of rules from most certain
        to least certain:

        Level 0 (TOP): Punctuation - if 。？！ exist, split by them FIRST
        Level 1: Definite verb endings - です, ます always end sentences
        Level 2: Strong particles + gap - よ, ぞ, ぜ with gap >= 0.25s
        Level 3: Soft particles + gap - ね, な, わ with gap >= 0.4s
        Level 4: Pure gap - any word followed by gap >= 0.6s
        Level 5: Long pause - split at gaps >= 1.2s (conversation break)

        The key insight is that higher levels split first, then lower levels
        handle remaining long segments. Within punctuated segments, linguistic
        markers still apply to catch multi-sentence fragments.

        Args:
            result: WhisperResult with word-level timestamps
            params: Preset parameters including gap thresholds
        """
        if not result.has_words:
            logger.debug("No word timestamps, skipping hierarchical splitting")
            return

        # === Level 0: Punctuation Split (TOP RULE) ===
        # If punctuation exists, split by it first - this is explicit intent
        # Includes both Japanese and Western punctuation for robustness
        # Also includes SPACE - in Japanese, space indicates explicit sentence boundary
        #
        # Japanese punctuation: 。？！…．「」
        # Western punctuation:  . ? !
        # Space as delimiter:   (space) - significant in Japanese, indicates boundary
        #
        # Note: Commas (、,) are NOT included as they don't end sentences
        logger.debug("  Level 0: Punctuation split (JP + Western + space)")
        punctuation_pattern = 'any=。,？,！,…,．,.,?,!,」,』, '
        result.custom_operation(
            'word', 'end', punctuation_pattern, 'splitright', word_level=True
        )

        # === Level 1: Definite Endings (Unconditional) ===
        # These ALWAYS end sentences in Japanese - no gap check needed
        logger.debug("  Level 1: Definite endings split (unconditional)")
        definite_pattern = ','.join(self.ling.definite_endings)
        result.custom_operation(
            'word', 'end', f'any={definite_pattern}', 'splitright', word_level=True
        )

        # === Level 2: Strong Particles with Gap Check ===
        # よ, ぞ, ぜ are emphatic - split if followed by sufficient pause
        logger.debug(f"  Level 2: Strong particles split (gap >= {params.strong_particle_gap}s)")
        strong_split_method = self._make_gap_aware_split_method(params.strong_particle_gap)
        strong_pattern = ','.join(self.ling.strong_particles)
        result.custom_operation(
            'word', 'end', f'any={strong_pattern}', strong_split_method, word_level=True
        )

        # === Level 3: Soft Particles with Larger Gap Check ===
        # ね, な, わ can be mid-sentence - need larger gap to confirm boundary
        logger.debug(f"  Level 3: Soft particles split (gap >= {params.soft_particle_gap}s)")
        soft_split_method = self._make_gap_aware_split_method(params.soft_particle_gap)
        soft_pattern = ','.join(self.ling.soft_particles)
        result.custom_operation(
            'word', 'end', f'any={soft_pattern}', soft_split_method, word_level=True
        )

        # === Level 4: Pure Gap Split (Fallback) ===
        # Any word followed by significant pause indicates likely boundary
        logger.debug(f"  Level 4: Pure gap split (>= {params.pure_gap_threshold}s)")
        result.split_by_gap(max_gap=params.pure_gap_threshold)

        # === Level 5: Long Pause Split (Conversation Break) ===
        # Very long pauses (> 1.2s) definitely indicate new utterance
        logger.debug("  Level 5: Long pause split (>= 1.2s)")
        result.split_by_gap(max_gap=1.2)

    def _make_gap_aware_split_method(self, min_gap: float):
        """
        Create a custom split method that only splits if gap >= min_gap.

        This is used with custom_operation to combine linguistic patterns
        (word endings) with temporal patterns (pause duration).

        The method checks:
        1. If this is NOT the last word in the segment (can't split after last word)
        2. If the gap to the next word is >= min_gap
        3. Only then performs the split

        Args:
            min_gap: Minimum gap in seconds required to trigger split

        Returns:
            A method function compatible with custom_operation
        """
        def split_if_sufficient_gap(
            result: stable_whisper.WhisperResult,
            seg_idx: int,
            word_idx: int
        ) -> None:
            """
            Split at word_idx only if gap to next word >= min_gap.

            This method is called by custom_operation when a word matches
            the ending pattern. It adds the gap condition before splitting.
            """
            seg = result[seg_idx]

            # If this is the last word in the segment, don't split
            # (there's nothing after it to form a new segment)
            if word_idx >= len(seg.words) - 1:
                return

            # Get current and next word
            current_word = seg.words[word_idx]
            next_word = seg.words[word_idx + 1]

            # Calculate gap between words
            gap = next_word.start - current_word.end

            # Only split if gap meets threshold
            if gap >= min_gap:
                result.split_segment_by_index(seg, word_idx, reassign_ids=False)

        return split_if_sufficient_gap

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
