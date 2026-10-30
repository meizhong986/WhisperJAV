"""Text normalisation before comparison. One named profile per rule set, so results name the rules they used.

Profile "ja-v1" (the default), applied to each subtitle line in this order:
  1. remove every bracketed note: an opening "(" or "（", the shortest run of characters that contains neither ")"
     nor "）", and a closing ")" or "）" (speaker names and sound notes such as （村西） or (笑));
  2. Unicode NFKC normalisation (full-width and half-width forms become one form);
  3. keep only characters whose Unicode general category is a letter (L*) or a number (N*): punctuation, symbols,
     spaces and line breaks are removed.
This is the normalisation used by the WhisperJAV 1.9.4 measurement scripts (score_segmenters.py), so the plain
character error rate of the two agrees exactly.
"""
from __future__ import annotations

import re
import unicodedata

_NOTE = re.compile(r"[（(][^）)]*[）)]")


def _ja_v1(text: str) -> str:
    text = _NOTE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


PROFILES = {"ja-v1": _ja_v1}


def normalizer(profile: str = "ja-v1"):
    try:
        return PROFILES[profile]
    except KeyError:
        raise ValueError(f"unknown normalisation profile {profile!r}; known: {sorted(PROFILES)}") from None


def unbalanced_brackets(text: str) -> bool:
    """True when a line has an opening or closing bracket without its partner: a note that may be damaged (its text
    is then compared as speech)."""
    depth = 0
    for ch in text:
        if ch in "(（":
            depth += 1
        elif ch in ")）":
            depth -= 1
            if depth < 0:
                return True
    return depth != 0
