"""
anime-whisper TextCleaner — model-specific text cleaning.

Handles known output artifacts from litagin/anime-whisper:
    1. Missing sentence-final 。 (almost always omitted by model)
    2. Phrase repetition (when no_repeat_ngram_size is insufficient)
    3. Whitespace normalization

Intentionally preserved (model behavior, not artifacts):
    - Circle censoring ○ (intentional profanity masking)
    - Half-width ! ? numbers (common in Japanese SRT)
    - Ellipsis normalization (model already handles …… → …)
"""

import re
from typing import Any

from whisperjav.utils.logger import logger

# Punctuation characters that count as valid sentence endings.
# If text already ends with one of these, no 。 is appended.
_SENTENCE_FINAL_PUNCT = frozenset("。、!?…！？♪～")


class AnimeWhisperCleaner:
    """TextCleaner for anime-whisper output artifacts."""

    def clean(self, text: str, **kwargs: Any) -> str:
        """
        Clean a single transcription string.

        Applies:
            1. Whitespace strip + empty check
            2. Phrase repetition removal (3+ consecutive repeats)
            3. Sentence-final 。 insertion if missing

        Returns:
            Cleaned text, or empty string for blank input.
        """
        if not text or not text.strip():
            return ""

        text = text.strip()
        text = self._remove_repetition(text)
        text = self._ensure_sentence_ending(text)
        return text

    def clean_batch(self, texts: list[str], **kwargs: Any) -> list[str]:
        """Clean a batch of transcription strings."""
        cleaned = [self.clean(t) for t in texts]

        # Batch summary logging
        n_items = len(texts)
        raw_chars = sum(len(t) for t in texts)
        clean_chars = sum(len(t) for t in cleaned)
        n_modified = sum(1 for r, c in zip(texts, cleaned) if r != c)
        if n_modified > 0:
            logger.info(
                "[AnimeWhisperCleaner] Cleaned %d/%d items — %d → %d chars (-%d)",
                n_modified, n_items, raw_chars, clean_chars, raw_chars - clean_chars,
            )
        else:
            logger.debug(
                "[AnimeWhisperCleaner] Batch of %d items — no changes needed (%d chars)",
                n_items, raw_chars,
            )

        return cleaned

    @staticmethod
    def _ensure_sentence_ending(text: str) -> str:
        """
        Add 。 if text doesn't end with sentence-final punctuation.

        anime-whisper almost always omits sentence-final 。 per the model card.
        This restores it for natural Japanese subtitle formatting.
        """
        if text and text[-1] not in _SENTENCE_FINAL_PUNCT:
            return text + "。"
        return text

    @staticmethod
    def _remove_repetition(text: str) -> str:
        """
        Remove phrases repeated 3+ times consecutively.

        This catches the model's known failure mode when no_repeat_ngram_size
        is insufficient for a particular input — the same 2-20 character
        phrase loops.  Collapses to a single occurrence.
        """
        # Match any 2-20 character substring repeated 3+ consecutive times.
        # Non-greedy (.{2,20}?) ensures we match the shortest repeating unit.
        return re.sub(r"(.{2,20}?)\1{2,}", r"\1", text)
