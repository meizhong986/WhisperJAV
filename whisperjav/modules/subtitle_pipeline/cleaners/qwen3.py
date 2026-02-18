"""
Qwen3-ASR TextCleaner adapter.

Thin wrapper around AssemblyTextCleaner (the existing mid-pipeline cleaner
for Qwen3 assembly mode).  Does NOT reimplement — delegates.

The AssemblyTextCleaner returns Tuple[str, Dict] but the TextCleaner protocol
expects just str.  This adapter unwraps the tuple and stores stats in metadata
for optional diagnostic access.
"""

from typing import Any, Optional

from whisperjav.utils.logger import logger


class Qwen3TextCleaner:
    """
    TextCleaner backed by AssemblyTextCleaner.

    Cleans raw Qwen3-ASR transcription text before forced alignment.
    Handles Qwen3-specific artifacts: phrase repetition, character flooding,
    hallucinated phrases, and whitespace normalization.
    """

    def __init__(self, config: Optional[Any] = None, language: str = "ja"):
        """
        Initialize with optional AssemblyCleanerConfig.

        Args:
            config: Optional AssemblyCleanerConfig.  If None, defaults are used.
            language: Language code for language-specific cleaning rules.
        """
        from whisperjav.modules.assembly_text_cleaner import (
            AssemblyCleanerConfig,
            AssemblyTextCleaner,
        )

        if config is None:
            config = AssemblyCleanerConfig()
        elif not isinstance(config, AssemblyCleanerConfig):
            logger.warning(
                "[Qwen3TextCleaner] Expected AssemblyCleanerConfig, got %s. Using defaults.",
                type(config).__name__,
            )
            config = AssemblyCleanerConfig()

        self._cleaner = AssemblyTextCleaner(config=config, language=language)
        self._last_stats: list[dict] = []

    @property
    def last_stats(self) -> list[dict]:
        """Stats from the most recent clean/clean_batch call."""
        return self._last_stats

    def clean(self, text: str, **kwargs: Any) -> str:
        """
        Clean a single transcription string.

        Returns only the cleaned text (protocol contract).
        Stats are accessible via self.last_stats.
        """
        cleaned, stats = self._cleaner.clean(text)
        self._last_stats = [stats]
        return cleaned

    def clean_batch(self, texts: list[str], **kwargs: Any) -> list[str]:
        """
        Clean a batch of transcription strings.

        Returns only the cleaned texts (protocol contract).
        Stats are accessible via self.last_stats.
        """
        cleaned_texts, all_stats = self._cleaner.clean_batch(texts)
        self._last_stats = all_stats

        # Batch summary logging
        n_items = len(texts)
        raw_chars = sum(len(t) for t in texts)
        clean_chars = sum(len(t) for t in cleaned_texts)
        n_modified = sum(1 for r, c in zip(texts, cleaned_texts) if r != c)
        if n_modified > 0:
            logger.info(
                "[Qwen3TextCleaner] Cleaned %d/%d items — %d → %d chars (-%d)",
                n_modified, n_items, raw_chars, clean_chars, raw_chars - clean_chars,
            )
        else:
            logger.debug(
                "[Qwen3TextCleaner] Batch of %d items — no changes needed (%d chars)",
                n_items, raw_chars,
            )

        return cleaned_texts
