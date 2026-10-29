"""Nonverbal lone-line filter for Qwen-family SRT output (v1.9.0; extended v1.9.2).

Qwen3-ASR is run in ChronosJAV with deliberately sensitive VAD / decoding to
capture maximum detail. A side effect is that many moan onsets, breaths and
ASR truncations surface as standalone subtitle lines (e.g. "あ。", "は。",
"え。", "切。"). Too many of these clutter the subtitle for the viewer.

This filter drops SRT entries whose ENTIRE text, with all punctuation and
whitespace removed, is exactly one owner-curated token. Whole-line exact match
means multi-token lines and sentences are never touched.

Owner-curated token set:
    - v1.9.0 (2026-07-03): single-character artifacts, validated against a real
      qwen3 pass2 SRT (MIMK-276): matched 302 / 2414 lines (12.5%), all
      high-confidence junk, zero real-dialogue hits. Broadening to other
      trailing punctuation was tested on Qwen3-ASR output and added nothing, so
      only "。" was honoured until v1.10.0.
    - v1.10.0 (2026-10-29, owner W339-W342): the token is judged with ALL
      punctuation removed (PUNCTUATION_CHARS of the non-linguistic filter), so
      "あ!", "うん!", "うん......", "え?" are dropped like "あ。". Parakeet-CrispASR
      and Cohere-CrispASR write "!", "?", "!?", "……", "......", "、" after the token.
    - v1.9.2 (2026-09-05, owner CFF5): lone "はい。" and "うん。" added. In the
      owner's JAV tests roughly 90% of these lone lines are moans during
      intimate scenes, not agreement (evidence: two SRT screenshots, dozens of
      lone cues of 0.07-2.9 s). Only the exact lone token is dropped; "はいはい。",
      "うんうん。", "ううん。", "あ、うん。" and any sentence containing them are kept.

Intentionally NOT dropped here (different problem — see pipelines/AGENTS.md):
    - multi-token stutters ("あ、あ。"), laughter ("はは。", "あはは。"),
      long-vowel onsets ("あー。"), repeated backchannel ("はいはい。").
"""

import re
from pathlib import Path
from typing import Dict, Union

import pysrt

from whisperjav.utils.logger import logger

# Owner-curated whole-line nonverbal / truncation tokens (v1.9.0), plus the lone
# backchannel tokens the owner asked to drop in v1.9.2 (CFF5: ~90% moans in JAV).
NONVERBAL_TOKENS = ("つ", "ふ", "ふっ", "切", "は", "え", "あ", "ん", "はい", "うん")

# Whole-line match on the text with all punctuation and whitespace removed:
# exactly one token. Longer tokens are placed first in the alternation so
# "ふっ" wins over "ふ" (anchoring already makes this safe, but keep it explicit).
_NONVERBAL_RE = re.compile(
    r"^(?:"
    + "|".join(sorted((re.escape(t) for t in NONVERBAL_TOKENS), key=len, reverse=True))
    + r")$"
)


def _without_punctuation(text: str) -> str:
    """The text with every PUNCTUATION_CHARS character, CR and LF removed (v1.10.0, owner W339-W342)."""
    from whisperjav.modules.subtitle_pipeline.cleaners.nonlinguistic_utterance_filter import (
        PUNCTUATION_CHARS,
    )
    return "".join(ch for ch in text if ch not in PUNCTUATION_CHARS and ch not in "\r\n")


def _normalizes_to_nothing(text: str) -> bool:
    """True when nothing is left once whitespace and punctuation are removed.

    Owner rule (2026-09-06, #413, i2): an entry that is only punctuation — a lone
    「。」 or 「、」, an ellipsis on its own, 「！？」 — carries no words and is
    removed whole. Inline punctuation inside text (anime-whisper's ellipses, owner
    i1) is untouched because the text remains. Lives in this Qwen-only filter on
    purpose: the shared NonlinguisticUtteranceFilter also runs on the legacy
    pipelines, whose sanitizer already handles symbol-only residue.
    """
    from whisperjav.modules.subtitle_pipeline.cleaners.nonlinguistic_utterance_filter import (
        PUNCTUATION_CHARS,
    )
    return all(ch in PUNCTUATION_CHARS or ch in "\r\n" for ch in text)


class NonverbalLineFilter:
    """Drops whole-line single-token nonverbal artifacts from an SRT file."""

    @staticmethod
    def is_nonverbal_line(text: str) -> bool:
        """Return True iff the line, with all punctuation removed, is one curated nonverbal token.

        Examples (drop):  "あ。", "あ", "は。", "え。", "ん。", "つ。", "ふ。",
                          "ふっ。", "切。", "はい。", "うん。", "  は。 " (whitespace),
                          "あ!", "あ！", "うん!", "うん......", "え?", "あ、" (v1.10.0)
        Examples (keep):  "はいはい。", "うんうん。", "ううん。", "あ、うん。", "あー。",
                          "あ、あ。", "はは。", "あっ!", "気持ちいい。", "", any sentence line.
        """
        if not text:
            return False
        return bool(_NONVERBAL_RE.match(_without_punctuation(text)))

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
            if not text or _normalizes_to_nothing(text):
                # empty, or nothing left after whitespace and punctuation (#413)
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
