#!/usr/bin/env python3
"""Shared helpers for post-ASR segment filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SegmentFilterConfig:
    """Configuration bundle for reusable logprob/nonverbal filtering."""

    logprob_threshold: Optional[float] = None
    logprob_margin: float = 0.0
    drop_nonverbal_vocals: bool = False
    short_segment_window: float = 1.6


class SegmentFilterHelper:
    """Apply consistent logprob and optional nonverbal filters to segments."""

    NONVERBAL_KEYWORDS = {
        "music",
        "applause",
        "laugh",
        "laughs",
        "laughter",
        "sfx",
        "fx",
        "noise",
        "silence",
        "ambient",
        "moan",
        "moans",
        "moaning",
        "groan",
        "groans",
        "sigh",
        "sighs",
        "breath",
        "breathing",
        "喘",
        "喘ぎ",
        "喘ぎ声",
        "うめき",
        "うめき声",
    }
    NOTE_CHARACTERS = set("♪♫")
    SIMPLE_VOCAL_CHARSET = set(
        "ahmnou"  # romanised moans like "ah" or "mmm"
        "ぁあァアんンっッふフぅゥうウおオえエはハほホ"  # common JP kana used for vocalisations
    )
    SIMPLE_VOCAL_IGNORES = set("!！?？。、,.・~〜～ー… 　")
    SIMPLE_VOCAL_MAX_LENGTH = 6

    def __init__(self, config: SegmentFilterConfig):
        self.threshold = config.logprob_threshold
        self.margin = max(0.0, config.logprob_margin or 0.0)
        self.drop_nonverbal = bool(config.drop_nonverbal_vocals)
        self.short_window = max(0.4, float(config.short_segment_window or 1.6))

    def should_filter(self, avg_logprob: float, duration: float, text: str) -> Tuple[bool, Optional[str], Optional[float]]:
        """Determine whether a segment should be filtered.

        Returns (should_filter, reason, effective_threshold).
        Reason is 'logprob' or 'nonverbal'.
        """
        effective_threshold: Optional[float] = self.threshold

        if effective_threshold is not None and self.margin > 0 and duration <= self.short_window:
            effective_threshold = effective_threshold - self.margin

        if effective_threshold is not None and avg_logprob < effective_threshold:
            return True, "logprob", effective_threshold

        if self.drop_nonverbal and self._looks_nonverbal(text):
            return True, "nonverbal", effective_threshold

        return False, None, effective_threshold

    @classmethod
    def _looks_nonverbal(cls, text: str) -> bool:
        if not text:
            return False

        stripped = text.strip()
        if not stripped:
            return False

        if all(ch in cls.NOTE_CHARACTERS or ch in cls.SIMPLE_VOCAL_IGNORES for ch in stripped):
            return True

        lowered = stripped.lower()
        collapsed = cls._collapse_descriptor(lowered)

        if not collapsed:
            return False

        if cls._contains_keyword(collapsed):
            return True

        simplified = cls._simplify(collapsed)
        if simplified and len(simplified) <= cls.SIMPLE_VOCAL_MAX_LENGTH and cls._is_simple_vocal(simplified):
            return True

        return False

    @classmethod
    def _collapse_descriptor(cls, text: str) -> str:
        collapsed = text.strip()
        while collapsed and collapsed[0] in "[](){}<>":
            collapsed = collapsed[1:]
        while collapsed and collapsed[-1] in "[](){}<>":
            collapsed = collapsed[:-1]
        return collapsed.strip()

    @classmethod
    def _contains_keyword(cls, text: str) -> bool:
        for keyword in cls.NONVERBAL_KEYWORDS:
            if keyword and keyword in text:
                return True
        return False

    @classmethod
    def _simplify(cls, text: str) -> str:
        return "".join(ch for ch in text if ch not in cls.SIMPLE_VOCAL_IGNORES)

    @classmethod
    def _is_simple_vocal(cls, text: str) -> bool:
        if not text:
            return False
        return all(ch in cls.SIMPLE_VOCAL_CHARSET for ch in text)
