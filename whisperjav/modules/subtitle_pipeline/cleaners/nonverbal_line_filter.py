"""Nonverbal single-token line filter for Qwen-family SRT output (v1.9.0).

Qwen3-ASR is run in ChronosJAV with deliberately sensitive VAD / decoding to
capture maximum detail. A side effect is that many moan onsets, breaths and
ASR truncations surface as standalone one-token subtitle lines (e.g. "あ。",
"は。", "え。", "切。"). Too many of these clutter the subtitle for the viewer.

This filter drops SRT entries whose ENTIRE (stripped) text is a single
owner-curated nonverbal / truncation token, optionally followed by one "。".
It is deliberately conservative: whole-line exact match means multi-token
lines, sentences, and meaningful backchannel ("はい。", "うん。") are NEVER
touched — only lone artifact tokens are removed.

Owner-curated token set (2026-07-03), validated against a real qwen3 pass2
SRT (MIMK-276): matched 302 / 2414 lines (12.5%), all high-confidence junk,
with zero real-dialogue hits. Broadening to other trailing punctuation was
tested and added nothing, so only "。" is honoured.

Intentionally NOT dropped here (higher false-positive risk / a different
problem — see pipelines/AGENTS.md):
    - multi-token stutters ("あ、あ。"), laughter ("はは。", "あはは。"),
      long-vowel onsets ("あー。"), agreement backchannel ("うん。", "はい。").
"""

import re
from pathlib import Path
from typing import Dict, Union

import pysrt

from whisperjav.utils.logger import logger

# Owner-curated whole-line nonverbal / truncation tokens (v1.9.0).
NONVERBAL_TOKENS = ("つ", "ふ", "ふっ", "切", "は", "え", "あ", "ん")

# Whole-line match: optional surrounding whitespace, exactly one token, and an
# optional single "。". Longer tokens are placed first in the alternation so
# "ふっ" wins over "ふ" (anchoring already makes this safe, but keep it explicit).
_NONVERBAL_RE = re.compile(
    r"^(?:"
    + "|".join(sorted((re.escape(t) for t in NONVERBAL_TOKENS), key=len, reverse=True))
    + r")。?$"
)


class NonverbalLineFilter:
    """Drops whole-line single-token nonverbal artifacts from an SRT file."""

    @staticmethod
    def is_nonverbal_line(text: str) -> bool:
        """Return True iff the whole (stripped) line is one curated nonverbal token.

        Examples (drop):  "あ。", "あ", "は。", "え。", "ん。", "つ。", "ふ。",
                          "ふっ。", "切。", "  は。 " (surrounding whitespace)
        Examples (keep):  "はい。", "うん。", "あー。", "あ、あ。", "はは。",
                          "気持ちいい。", "", any multi-token / sentence line.
        """
        if not text:
            return False
        return bool(_NONVERBAL_RE.match(text.strip()))

    def filter_srt_file(self, srt_path: Union[str, Path]) -> Dict[str, int]:
        """Drop empty + nonverbal-token entries in place, renumber, write back.

        Mirrors ``AnimeWhisperCleaner.filter_srt_file`` (parse → drop → renumber
        → write) so the two Phase-8 filters compose cleanly when both run.

        Returns:
            Stats dict: original_count, dropped_nonverbal, dropped_empty,
            final_count.
        """
        path = Path(srt_path)
        stats = {
            "original_count": 0,
            "dropped_nonverbal": 0,
            "dropped_empty": 0,
            "final_count": 0,
        }
        if not path.exists() or path.stat().st_size == 0:
            return stats

        try:
            subs = pysrt.open(str(path), encoding="utf-8")
        except Exception as e:  # pragma: no cover - defensive parse guard
            logger.warning(
                "[NonverbalLineFilter] filter_srt_file failed to parse %s: %s", path, e
            )
            return stats

        stats["original_count"] = len(subs)
        kept: list = []
        for sub in subs:
            text = (sub.text or "").strip()
            if not text:
                stats["dropped_empty"] += 1
                continue
            if self.is_nonverbal_line(text):
                stats["dropped_nonverbal"] += 1
                continue
            kept.append(sub)

        # Renumber surviving entries (1, 2, 3, ...).
        for new_idx, sub in enumerate(kept, start=1):
            sub.index = new_idx

        pysrt.SubRipFile(items=kept).save(str(path), encoding="utf-8")

        stats["final_count"] = len(kept)
        if stats["dropped_nonverbal"] or stats["dropped_empty"]:
            logger.info(
                "[NonverbalLineFilter] SRT filter %s: %d -> %d entries "
                "(-%d nonverbal, -%d empty)",
                path.name,
                stats["original_count"],
                stats["final_count"],
                stats["dropped_nonverbal"],
                stats["dropped_empty"],
            )
        return stats
