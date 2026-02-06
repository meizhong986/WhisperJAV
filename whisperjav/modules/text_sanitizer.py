"""
Text-only sanitizer adapter for the Decoupled Assembly Line flow.

Cleans raw ASR text BETWEEN generation and alignment to prevent
hallucinated text from causing ForcedAligner NULL timestamp failures.

This adapter composes existing sanitization tools (HallucinationRemover,
RepetitionCleaner) but operates on raw text strings — no SRT/timestamp
dependency. It is used exclusively in the assembly input mode.

Architecture:
    ASR text-only → TextSanitizer.clean() → ForcedAligner
"""

import re
from typing import Dict, List, Optional, Tuple

from whisperjav.utils.logger import logger
from whisperjav.config.sanitization_constants import (
    HallucinationConstants,
    RepetitionConstants,
)
from whisperjav.modules.hallucination_remover import HallucinationRemover
from whisperjav.modules.repetition_cleaner import RepetitionCleaner


class TextSanitizer:
    """
    String adapter that cleans raw ASR transcription text.

    Designed for the assembly pipeline where text must be sanitized
    before being passed to the standalone ForcedAligner. Hallucinated
    text causes the aligner to fail (NULL timestamps) because it
    cannot match ghost words to audio features.

    Cleaning stages:
        1. Repetition cleaning (extreme character/phrase repetitions)
        2. Hallucination removal (known hallucination phrases)
        3. Whitespace normalization
    """

    def __init__(self, language: Optional[str] = None):
        """
        Initialize the text sanitizer.

        Args:
            language: Primary language code (e.g., 'ja') for
                      language-specific hallucination detection.
        """
        self.language = language or "ja"

        # Reuse existing sanitization components
        self._hallucination_remover = HallucinationRemover(
            constants=HallucinationConstants(),
            primary_language=self.language,
        )
        self._repetition_cleaner = RepetitionCleaner(
            constants=RepetitionConstants(),
        )

        logger.debug(
            "TextSanitizer initialized (language=%s)", self.language,
        )

    def clean(self, text: str) -> Tuple[str, Dict]:
        """
        Clean raw ASR text for safe alignment.

        Args:
            text: Raw transcription text from ASR (may contain
                  hallucinations, extreme repetitions, artifacts).

        Returns:
            Tuple of (cleaned_text, stats_dict).
            stats_dict contains counts of modifications made.
        """
        if not text or not text.strip():
            return "", {"empty_input": True}

        original = text
        stats = {
            "original_length": len(text),
            "repetitions_cleaned": 0,
            "hallucinations_removed": 0,
            "lines_removed": 0,
        }

        # Stage 1: Repetition cleaning (operates on full text)
        # This catches extreme character floods (ああああああ...) and
        # phrase repetitions (すごい、すごい、すごい...) that are
        # common hallucination artifacts in LALM output.
        text, rep_mods = self._repetition_cleaner.clean_repetitions(text)
        stats["repetitions_cleaned"] = len(rep_mods)

        # Stage 2: Per-line hallucination removal
        # The hallucination remover does exact-match on full lines.
        # Split by common sentence boundaries, clean each, rejoin.
        cleaned_lines = []
        lines = self._split_into_lines(text)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            cleaned, hal_mods = self._hallucination_remover.remove_hallucinations(
                line, language=self.language,
            )
            if hal_mods:
                stats["hallucinations_removed"] += len(hal_mods)

            if cleaned and cleaned.strip():
                cleaned_lines.append(cleaned.strip())
            else:
                stats["lines_removed"] += 1

        text = "".join(cleaned_lines)

        # Stage 3: Whitespace normalization
        # Collapse runs of whitespace, strip edges
        text = re.sub(r'\s+', ' ', text).strip()

        stats["cleaned_length"] = len(text)
        stats["chars_removed"] = stats["original_length"] - stats["cleaned_length"]

        if stats["chars_removed"] > 0:
            logger.debug(
                "TextSanitizer: %d→%d chars (-%d), "
                "%d repetitions, %d hallucinations, %d lines removed",
                stats["original_length"], stats["cleaned_length"],
                stats["chars_removed"], stats["repetitions_cleaned"],
                stats["hallucinations_removed"], stats["lines_removed"],
            )

        return text, stats

    def clean_batch(self, texts: List[str]) -> Tuple[List[str], List[Dict]]:
        """
        Clean a batch of raw ASR texts.

        Args:
            texts: List of raw transcription text strings.

        Returns:
            Tuple of (cleaned_texts, stats_list).
        """
        cleaned_texts = []
        all_stats = []
        total_removed = 0

        for i, text in enumerate(texts):
            cleaned, stats = self.clean(text)
            cleaned_texts.append(cleaned)
            all_stats.append(stats)
            total_removed += stats.get("chars_removed", 0)

        if total_removed > 0:
            logger.info(
                "TextSanitizer batch: %d scenes, %d total chars removed",
                len(texts), total_removed,
            )

        return cleaned_texts, all_stats

    @staticmethod
    def _split_into_lines(text: str) -> List[str]:
        """
        Split text into logical lines for per-line hallucination checking.

        Japanese text doesn't use newlines consistently, so we split on
        sentence-ending punctuation (。！？) while preserving the punctuation
        on the preceding segment.
        """
        # Split on sentence-ending punctuation, keeping the delimiter
        # attached to the preceding text
        parts = re.split(r'(?<=[。！？!?])', text)
        return [p for p in parts if p.strip()]
