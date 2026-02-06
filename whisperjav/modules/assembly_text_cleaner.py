"""
Assembly Text Cleaner for the Qwen3-ASR Decoupled Assembly Line.

Purpose-built text cleaner that operates on raw ASR output BEFORE
forced alignment. Designed for Qwen3-ASR hallucination and repetition
characteristics, which differ fundamentally from Whisper-era ASR.

Key differences from Whisper-era SubtitleSanitizer:
    - Operates on raw text (no timestamps, no SRT format)
    - Uses full Unicode character classes (catches kanji+kana phrases)
    - Preserves punctuation that helps the ForcedAligner's nagisa tokenizer
    - Modular stages that can be individually enabled/disabled/configured
    - No CPS-based filtering (timestamps don't exist yet)

Architecture:
    ASR text-only -> AssemblyTextCleaner.clean() -> ForcedAligner

Stages (executed in order):
    1. Phrase Repetition Reducer  - consecutive + sentence-level dedup
    2. Character Flood Reducer    - single-char floods (full Unicode)
    3. Hallucination Phrase Filter - known phantom phrases (exact match)
    4. Whitespace Normalizer       - collapse whitespace, preserve structure

Each stage:
    - Is a pure function: (text, config) -> (text, stage_stats)
    - Can be disabled independently via config
    - Logs what it changed
    - Preserves sentence-ending punctuation (。！？) for nagisa tokenization
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import regex

from whisperjav.utils.logger import logger


# ---------------------------------------------------------------------------
# Stage Configurations
# ---------------------------------------------------------------------------

@dataclass
class PhraseRepetitionConfig:
    """Config for Stage 1: Phrase Repetition Reducer.

    Controls detection and reduction of repeated multi-character phrases
    using full Unicode letter/number classes (catches kanji+kana combos
    like 行け, 早く, 気持ちいい that kana-only patterns miss).
    """
    enabled: bool = True

    # Consecutive identical phrases: "行け行け行け" -> "行け行け"
    max_consecutive: int = 2  # Keep at most N consecutive identical phrases

    # Sentence-level dedup: if the exact same sentence appears >N times, reduce
    max_sentence_repeats: int = 2  # Keep at most N identical sentences


@dataclass
class CharFloodConfig:
    """Config for Stage 2: Character Flood Reducer.

    Reduces single-character floods (ああああ, 行行行行) and vowel
    extensions (あ〜〜〜〜) using full Unicode letter class.
    """
    enabled: bool = True

    # "ああああ" -> "ああ"
    max_consecutive_chars: int = 2

    # "あ〜〜〜〜" -> "あ〜〜"
    max_consecutive_extension: int = 2


@dataclass
class HallucinationFilterConfig:
    """Config for Stage 3: Hallucination Phrase Filter.

    Removes known phantom phrases that ASR models commonly hallucinate.
    Reuses the existing hallucination phrase lists but operates on raw
    text lines (not SRT subtitles).
    """
    enabled: bool = True

    # Only use exact full-line matching (safe for pre-alignment)
    # Regex patterns from the Whisper-era list may be too aggressive
    use_exact_match: bool = True
    use_regex_match: bool = False


@dataclass
class WhitespaceConfig:
    """Config for Stage 4: Whitespace Normalizer.

    Minimal cleanup that preserves semantic structure for the aligner.
    """
    enabled: bool = True
    collapse_spaces: bool = True
    strip_edges: bool = True


@dataclass
class AssemblyCleanerConfig:
    """Top-level config for the Assembly Text Cleaner.

    Designed for incremental testing:
        enabled=False              -> bypass entirely (raw text -> aligner)
        phrase_repetition.enabled  -> toggle phrase repetition reduction
        char_flood.enabled         -> toggle character flood reduction
        hallucination_filter.enabled -> toggle hallucination removal
        whitespace.enabled         -> toggle whitespace normalization
    """
    enabled: bool = True

    phrase_repetition: PhraseRepetitionConfig = field(
        default_factory=PhraseRepetitionConfig,
    )
    char_flood: CharFloodConfig = field(
        default_factory=CharFloodConfig,
    )
    hallucination_filter: HallucinationFilterConfig = field(
        default_factory=HallucinationFilterConfig,
    )
    whitespace: WhitespaceConfig = field(
        default_factory=WhitespaceConfig,
    )


# ---------------------------------------------------------------------------
# Assembly Text Cleaner
# ---------------------------------------------------------------------------

class AssemblyTextCleaner:
    """
    Modular text cleaner for the Qwen3-ASR assembly pipeline.

    Cleans raw ASR transcription text BEFORE forced alignment.
    Each cleaning stage is independently configurable and logs its actions.

    Usage:
        cleaner = AssemblyTextCleaner()  # defaults
        cleaned, stats = cleaner.clean("行け行け行け行け進め。")

        # Disable all cleaning (passthrough):
        cleaner = AssemblyTextCleaner(config=AssemblyCleanerConfig(enabled=False))

        # Only phrase repetition:
        cfg = AssemblyCleanerConfig(
            char_flood=CharFloodConfig(enabled=False),
            hallucination_filter=HallucinationFilterConfig(enabled=False),
            whitespace=WhitespaceConfig(enabled=False),
        )
        cleaner = AssemblyTextCleaner(config=cfg)
    """

    def __init__(
        self,
        config: Optional[AssemblyCleanerConfig] = None,
        language: str = "ja",
    ):
        self.config = config or AssemblyCleanerConfig()
        self.language = language

        # Build the ordered stage pipeline
        self._stages: List[Tuple[str, Callable]] = self._build_stages()

        # Lazy-loaded hallucination data (only if stage 3 is enabled)
        self._hallucination_set: Optional[set] = None

        logger.debug(
            "AssemblyTextCleaner initialized (language=%s, enabled=%s, stages=%s)",
            self.language,
            self.config.enabled,
            [name for name, _ in self._stages],
        )

    def _build_stages(self) -> List[Tuple[str, Callable]]:
        """Build the ordered list of active cleaning stages."""
        stages = []
        if self.config.phrase_repetition.enabled:
            stages.append(("phrase_repetition", self._stage_phrase_repetition))
        if self.config.char_flood.enabled:
            stages.append(("char_flood", self._stage_char_flood))
        if self.config.hallucination_filter.enabled:
            stages.append(("hallucination_filter", self._stage_hallucination_filter))
        if self.config.whitespace.enabled:
            stages.append(("whitespace", self._stage_whitespace))
        return stages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> Tuple[str, Dict]:
        """
        Clean a single text string through all enabled stages.

        Args:
            text: Raw transcription text from ASR.

        Returns:
            Tuple of (cleaned_text, stats_dict).
            stats_dict contains per-stage modification counts and
            overall before/after metrics.
        """
        if not text or not text.strip():
            return "", {"empty_input": True, "stages": {}}

        if not self.config.enabled:
            return text, {"bypassed": True, "stages": {}}

        original = text
        stats = {
            "original_length": len(text),
            "stages": {},
        }

        for stage_name, stage_fn in self._stages:
            text, stage_stats = stage_fn(text)
            stats["stages"][stage_name] = stage_stats

        stats["cleaned_length"] = len(text)
        stats["chars_removed"] = stats["original_length"] - stats["cleaned_length"]

        if stats["chars_removed"] > 0:
            logger.debug(
                "AssemblyTextCleaner: %d->%d chars (-%d) | %s",
                stats["original_length"],
                stats["cleaned_length"],
                stats["chars_removed"],
                ", ".join(
                    f"{k}: {v.get('modifications', 0)}"
                    for k, v in stats["stages"].items()
                    if v.get("modifications", 0) > 0
                ),
            )

        return text, stats

    def clean_batch(self, texts: List[str]) -> Tuple[List[str], List[Dict]]:
        """
        Clean a batch of text strings.

        Args:
            texts: List of raw transcription text strings (one per scene).

        Returns:
            Tuple of (cleaned_texts, stats_list).
        """
        cleaned_texts = []
        all_stats = []
        total_removed = 0

        for text in texts:
            cleaned, stats = self.clean(text)
            cleaned_texts.append(cleaned)
            all_stats.append(stats)
            total_removed += stats.get("chars_removed", 0)

        if total_removed > 0:
            logger.info(
                "AssemblyTextCleaner batch: %d scenes, %d total chars removed",
                len(texts), total_removed,
            )
        elif self.config.enabled:
            logger.info(
                "AssemblyTextCleaner batch: %d scenes, no modifications",
                len(texts),
            )

        return cleaned_texts, all_stats

    # ------------------------------------------------------------------
    # Stage 1: Phrase Repetition Reducer
    # ------------------------------------------------------------------

    def _stage_phrase_repetition(self, text: str) -> Tuple[str, Dict]:
        """
        Reduce repeated multi-character phrases and deduplicate sentences.

        Sub-stages:
            1a. Consecutive phrase reduction (regex):
                "行け行け行け行け" -> "行け行け"
                Uses [\\p{L}\\p{N}] (full Unicode) instead of kana-only.

            1b. Sentence-level dedup:
                If the exact same sentence appears >N times, keep only N.
                Operates on sentences split by 。！？!?

        Preserves punctuation between phrases for nagisa tokenization.
        """
        cfg = self.config.phrase_repetition
        stats = {"modifications": 0, "consecutive_reductions": 0, "sentence_dedup": 0}

        original = text

        # --- 1a: Consecutive identical phrase reduction ---
        # Match 2-8 character phrases repeated 3+ times consecutively.
        # The {2,} means the backreference matches 2+ additional times,
        # so total occurrences are 3+.
        max_keep = cfg.max_consecutive
        # Quantifier: we want to match phrases repeated (max_keep+1) or more
        # times. The backreference count is (total - 1), so threshold = max_keep.
        if max_keep < 1:
            max_keep = 1

        pattern_consecutive = regex.compile(
            r'([\p{L}\p{N}]{2,8})\1{' + str(max_keep) + r',}'
        )

        def _replace_consecutive(m):
            return m.group(1) * max_keep

        new_text = pattern_consecutive.sub(_replace_consecutive, text)
        if new_text != text:
            stats["consecutive_reductions"] += 1
            stats["modifications"] += 1
            text = new_text

        # --- 1b: Sentence-level dedup ---
        # Split on sentence-ending punctuation, count exact duplicates,
        # keep at most max_sentence_repeats of each.
        max_repeats = cfg.max_sentence_repeats
        sentences = self._split_sentences(text)

        if len(sentences) > 1:
            seen: Counter = Counter()
            kept = []
            dropped = 0

            for sentence in sentences:
                key = sentence.strip()
                if not key:
                    continue
                seen[key] += 1
                if seen[key] <= max_repeats:
                    kept.append(sentence)
                else:
                    dropped += 1

            if dropped > 0:
                text = "".join(kept)
                stats["sentence_dedup"] = dropped
                stats["modifications"] += 1

        if text != original:
            logger.debug(
                "Stage 1 (phrase_repetition): %d->%d chars, "
                "%d consecutive reductions, %d sentences deduped",
                len(original), len(text),
                stats["consecutive_reductions"], stats["sentence_dedup"],
            )

        return text, stats

    # ------------------------------------------------------------------
    # Stage 2: Character Flood Reducer
    # ------------------------------------------------------------------

    def _stage_char_flood(self, text: str) -> Tuple[str, Dict]:
        """
        Reduce single-character floods and vowel extensions.

        Uses \\p{L} (full Unicode letter class) instead of kana-only [ぁ-んァ-ン],
        so it catches kanji floods (行行行行) as well as kana floods (ああああ).

        Also handles extension markers: あ〜〜〜〜 -> あ〜〜
        """
        cfg = self.config.char_flood
        stats = {"modifications": 0, "char_reductions": 0, "extension_reductions": 0}

        original = text
        max_chars = cfg.max_consecutive_chars
        max_ext = cfg.max_consecutive_extension

        if max_chars < 1:
            max_chars = 1
        if max_ext < 1:
            max_ext = 1

        # Single-character floods: any Unicode letter repeated 4+ times
        # (threshold is max_chars+1 additional repetitions)
        threshold = max_chars  # backreference count = max_chars means total = max_chars+1
        pattern_flood = regex.compile(
            r'([\p{L}])\1{' + str(threshold) + r',}'
        )

        def _replace_flood(m):
            return m.group(1) * max_chars

        new_text = pattern_flood.sub(_replace_flood, text)
        if new_text != text:
            stats["char_reductions"] += 1
            stats["modifications"] += 1
            text = new_text

        # Vowel extension floods: X〜〜〜〜 or Xーーーー
        ext_threshold = max_ext
        pattern_ext = regex.compile(
            r'([\p{L}])([〜ー])\2{' + str(ext_threshold) + r',}'
        )

        def _replace_ext(m):
            return m.group(1) + m.group(2) * max_ext

        new_text = pattern_ext.sub(_replace_ext, text)
        if new_text != text:
            stats["extension_reductions"] += 1
            stats["modifications"] += 1
            text = new_text

        if text != original:
            logger.debug(
                "Stage 2 (char_flood): %d->%d chars, "
                "%d char reductions, %d extension reductions",
                len(original), len(text),
                stats["char_reductions"], stats["extension_reductions"],
            )

        return text, stats

    # ------------------------------------------------------------------
    # Stage 3: Hallucination Phrase Filter
    # ------------------------------------------------------------------

    def _stage_hallucination_filter(self, text: str) -> Tuple[str, Dict]:
        """
        Remove known hallucination phrases via exact full-line matching.

        Splits text into sentences (on 。！？!?), checks each against the
        hallucination set, and removes matching lines.

        The hallucination set is loaded lazily from the existing
        HallucinationRemover infrastructure (shared filter lists).
        """
        cfg = self.config.hallucination_filter
        stats = {"modifications": 0, "lines_removed": 0}

        if not cfg.use_exact_match:
            return text, stats

        # Lazy-load hallucination set
        if self._hallucination_set is None:
            self._hallucination_set = self._load_hallucination_set()

        if not self._hallucination_set:
            return text, stats

        original = text
        sentences = self._split_sentences(text)
        kept = []

        for sentence in sentences:
            stripped = sentence.strip()
            if not stripped:
                continue

            # Check exact match (case-insensitive, stripped)
            normalized = stripped.lower()
            if normalized in self._hallucination_set:
                stats["lines_removed"] += 1
                stats["modifications"] += 1
                logger.debug(
                    "Stage 3 (hallucination): removed '%s'",
                    stripped[:40],
                )
                continue

            kept.append(sentence)

        text = "".join(kept)

        if text != original:
            logger.debug(
                "Stage 3 (hallucination): %d->%d chars, %d lines removed",
                len(original), len(text), stats["lines_removed"],
            )

        return text, stats

    def _load_hallucination_set(self) -> set:
        """
        Load hallucination phrases from the existing HallucinationRemover.

        Extracts the exact-match phrase set for the configured language.
        Falls back to an empty set on any error (graceful degradation).
        """
        try:
            from whisperjav.config.sanitization_constants import HallucinationConstants
            from whisperjav.modules.hallucination_remover import HallucinationRemover

            remover = HallucinationRemover(
                constants=HallucinationConstants(),
                primary_language=self.language,
            )

            # Access the internal exact lists
            exact_lists = getattr(remover, '_exact_lists', None)
            if not exact_lists:
                return set()

            # Normalize language key for lookup
            lang_key = self.language.lower() if self.language else "japanese"

            # Map common codes to list keys
            code_map = {
                'ja': 'japanese', 'jp': 'japanese', 'japanese': 'japanese',
                'en': 'english', 'english': 'english',
                'ko': 'korean', 'korean': 'korean',
                'zh': 'chinese', 'chinese': 'chinese',
            }
            mapped = code_map.get(lang_key, lang_key)

            phrases = exact_lists.get(mapped, set())
            if not phrases:
                # Try original key
                phrases = exact_lists.get(lang_key, set())
            if not phrases:
                # Final fallback to Japanese
                phrases = exact_lists.get('japanese', set())

            # Normalize all phrases to lowercase for matching
            normalized = {p.lower().strip() for p in phrases if p and p.strip()}

            logger.debug(
                "AssemblyTextCleaner: loaded %d hallucination phrases for '%s'",
                len(normalized), mapped,
            )
            return normalized

        except Exception as e:
            logger.warning(
                "AssemblyTextCleaner: failed to load hallucination set: %s", e,
            )
            return set()

    # ------------------------------------------------------------------
    # Stage 4: Whitespace Normalizer
    # ------------------------------------------------------------------

    def _stage_whitespace(self, text: str) -> Tuple[str, Dict]:
        """
        Normalize whitespace while preserving sentence structure.

        Collapses multiple spaces to single space and strips edges.
        Does NOT remove or modify punctuation.
        """
        cfg = self.config.whitespace
        stats = {"modifications": 0}

        original = text

        if cfg.collapse_spaces:
            text = re.sub(r'[ \t]+', ' ', text)  # Collapse horizontal whitespace
            text = re.sub(r'\n\s*\n', '\n', text)  # Collapse blank lines

        if cfg.strip_edges:
            text = text.strip()

        if text != original:
            stats["modifications"] = 1

        return text, stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Split text into sentences on Japanese/general sentence-ending punctuation.

        Keeps the punctuation attached to the preceding sentence.
        This preserves structure for the ForcedAligner's nagisa tokenizer.

        Returns:
            List of sentence strings (punctuation attached).
            Example: "行け。早く。" -> ["行け。", "早く。"]
        """
        # Split on sentence-ending punctuation, keeping delimiter
        # attached to the preceding text
        parts = regex.split(r'(?<=[。！？!?])', text)
        return [p for p in parts if p.strip()]
