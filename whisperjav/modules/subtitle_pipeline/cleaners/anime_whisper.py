"""
anime-whisper TextCleaner — model-specific text cleaning.

Handles known output artifacts from litagin/anime-whisper:
    1. Missing sentence-final 。 (almost always omitted by model)
    2. Phrase repetition (when no_repeat_ngram_size is insufficient)
    3. Whitespace normalization
    4. Ellipsis-only artifacts (v1.8.12.post2): drop segments / SRT entries
       that are only "…" (with optional closing punct/quote/whitespace).
       These are produced for short non-speech regions (breathing, ambient).

Intentionally preserved (model behavior, not artifacts):
    - Circle censoring ○ (intentional profanity masking)
    - Half-width ! ? numbers (common in Japanese SRT)
    - Ellipsis normalization (model already handles …… → …)
"""

import re
from pathlib import Path
from typing import Any, Dict, Union

import pysrt

from whisperjav.utils.logger import logger

# Punctuation characters that count as valid sentence endings.
# If text already ends with one of these, no 。 is appended.
_SENTENCE_FINAL_PUNCT = frozenset("。、!?…！？♪～")

# v1.8.12.post2 — ellipsis-only artifact detection.
# Drops segments whose stripped text consists of:
#   ellipsis-like chars  + optional closing punct/quote + whitespace
#   AND contains at least one ellipsis/dot character.
# Covers (drop): "…", "‥", "...", "……", "…?", "…!", "…」", "…？", "…！", "…)", etc.
# Preserves (keep): "?" alone, "!" alone, "」" alone, "あ…", "…あ", any text with letters.
_ELLIPSIS_NOISE_CHARS = (
    r'…'   # … horizontal ellipsis
    r'‥'   # ‥ two-dot leader
    r'\.'       # . ASCII full stop (covers "...")
    r'．'   # ． fullwidth full stop
    r'?!'       # ASCII question / exclamation
    r'？'   # ？ fullwidth question
    r'！'   # ！ fullwidth exclamation
    r'」'   # 」 right corner bracket
    r'』'   # 』 right white corner bracket
    r'\)'       # ) ASCII closing paren
    r'）'   # ） fullwidth closing paren
    r'\]'       # ] ASCII closing bracket
    r'\s'       # any whitespace
)
_ELLIPSIS_NOISE_RE = re.compile(rf'^[{_ELLIPSIS_NOISE_CHARS}]+$')
_HAS_DOT_RE = re.compile(r'[…‥\.．]')


class AnimeWhisperCleaner:
    """TextCleaner for anime-whisper output artifacts."""

    def clean(self, text: str, **kwargs: Any) -> str:
        """
        Clean a single transcription string.

        Applies:
            1. Whitespace strip + empty check
            2. Ellipsis-only drop (v1.8.12.post2): returns "" for ellipsis
               artifacts like "…", "…?", "…!", "…」", etc.
            3. Phrase repetition removal (3+ consecutive repeats)
            4. Sentence-final 。 insertion if missing

        Returns:
            Cleaned text, or empty string for blank/ellipsis-only input.
        """
        if not text or not text.strip():
            return ""

        text = text.strip()

        # v1.8.12.post2: drop ellipsis-only artifacts before further processing.
        # These survived prior cleaner stages because "…" is in
        # _SENTENCE_FINAL_PUNCT so _ensure_sentence_ending leaves them alone.
        if self._is_ellipsis_only(text):
            return ""

        text = self._remove_repetition(text)
        text = self._ensure_sentence_ending(text)
        return text

    def clean_batch(self, texts: list[str], **kwargs: Any) -> list[str]:
        """Clean a batch of transcription strings."""
        cleaned: list[str] = []
        n_ellipsis_dropped = 0
        for t in texts:
            stripped = t.strip() if t else ""
            c = self.clean(t)
            # Track ellipsis-only drops separately (input non-empty, output empty,
            # input matched the ellipsis pattern).
            if not c and stripped and self._is_ellipsis_only(stripped):
                n_ellipsis_dropped += 1
            cleaned.append(c)

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
        if n_ellipsis_dropped > 0:
            logger.info(
                "[AnimeWhisperCleaner] Dropped %d ellipsis-only segment(s)",
                n_ellipsis_dropped,
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

    @staticmethod
    def _is_ellipsis_only(text: str) -> bool:
        """
        Return True if text is an ellipsis-only artifact (no dialogue content).

        Matches text consisting solely of:
            ellipsis chars (… ‥) + ASCII/fullwidth dots + closing punct/quotes
            + whitespace
        AND containing at least one ellipsis/dot character.

        Drops:  "…", "‥", "...", "…?", "…!", "…」", "…？", "…！", "……", "…)" etc.
        Keeps:  "?" alone, "!" alone, "」" alone, "あ…", "…あ", any text with
                actual letters (hira/kata/kanji/alnum).
        """
        if not text:
            return False
        if not _HAS_DOT_RE.search(text):
            return False
        return bool(_ELLIPSIS_NOISE_RE.match(text))

    def filter_srt_file(self, srt_path: Union[str, Path]) -> Dict[str, int]:
        """
        Read an SRT file, drop entries with empty text or ellipsis-only
        artifacts, renumber remaining entries, and write back in place.

        Defense-in-depth complement to per-segment ``clean()``: catches any
        ellipsis-only artifact that slipped through (e.g., from edge-case
        alignment behavior or from outside the orchestrator path).

        Args:
            srt_path: Path to the SRT file (modified in place).

        Returns:
            Statistics dict with keys:
                - original_count: subs in the file before filtering
                - dropped_ellipsis: subs dropped for being ellipsis-only
                - dropped_empty: subs dropped for being empty/whitespace-only
                - final_count: subs in the file after filtering and renumbering
        """
        path = Path(srt_path)
        stats = {
            "original_count": 0,
            "dropped_ellipsis": 0,
            "dropped_empty": 0,
            "final_count": 0,
        }
        if not path.exists() or path.stat().st_size == 0:
            return stats

        try:
            subs = pysrt.open(str(path), encoding='utf-8')
        except Exception as e:
            logger.warning(
                "[AnimeWhisperCleaner] filter_srt_file failed to parse %s: %s",
                path, e,
            )
            return stats

        stats["original_count"] = len(subs)
        kept: list = []
        for sub in subs:
            text = (sub.text or "").strip()
            if not text:
                stats["dropped_empty"] += 1
                continue
            if self._is_ellipsis_only(text):
                stats["dropped_ellipsis"] += 1
                continue
            kept.append(sub)

        # Renumber remaining entries (1, 2, 3, ...).
        for new_idx, sub in enumerate(kept, start=1):
            sub.index = new_idx

        # Write back. Use SubRipFile to preserve formatting.
        out = pysrt.SubRipFile(items=kept)
        out.save(str(path), encoding='utf-8')

        stats["final_count"] = len(kept)
        if stats["dropped_ellipsis"] or stats["dropped_empty"]:
            logger.info(
                "[AnimeWhisperCleaner] SRT filter %s: %d → %d entries "
                "(-%d ellipsis-only, -%d empty)",
                path.name,
                stats["original_count"], stats["final_count"],
                stats["dropped_ellipsis"], stats["dropped_empty"],
            )
        return stats
