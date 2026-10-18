"""The line-length rule of the CrispASR pipelines, the last step of their line building (1.10.0).

CrispASR master plan 3.8 item 3 (owner W129 5 (i), W135 3, W138 3). Built, and switched off until the
phase D scores pick its two values (W138 3: "built but switched OFF"): OFF is every pipeline's default.

- A line is split only after a punctuation mark that closes or separates (、。！？…」 and the like:
  Unicode categories Po, Pe and Pf). Not after an opening bracket or quote, which belongs to the text
  after it; not after a dash or connector (Pd, Pc: 〜 - _); and not after a mark that sits inside words
  (・ in ジョン・スミス, the apostrophe in CAN'T, the straight quotes " ', which open as often as they
  close). Both engines give pieces of words, not words, so a split between other pieces could cut a
  word in two.
- A line whose time on screen (end minus start) is over ``max_s`` seconds, or whose text is over
  ``max_chars`` characters (spaces not counted), is split at the mark nearest its middle in time, and
  again, until every part is within both limits or has no mark left. A line with no mark inside stays
  whole.
- Each part runs from its first piece's start to its last piece's end. A piece with a mark inside it
  (a segment the program gave no times inside, plan 3.8 item 2 (iii)) is first cut after each such
  mark, its time shared out by characters.

Pure: no ASR import, no stable-ts.
"""

import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# (text, start, end), in seconds.
Piece = Tuple[str, float, float]


@dataclass(frozen=True)
class LineLimits:
    """The rule's two limits; a limit of None is not applied, and with both None the rule is off."""
    max_s: Optional[float] = None
    max_chars: Optional[int] = None

    @property
    def active(self) -> bool:
        return self.max_s is not None or self.max_chars is not None

    def as_dict(self) -> dict:
        return {"max_s": self.max_s, "max_chars": self.max_chars}


OFF = LineLimits()

_CLOSING_CATEGORIES = frozenset({"Po", "Pe", "Pf"})
# Marks of those categories that sit inside words, or open as often as they close.
_INSIDE_WORDS = frozenset("・･'\"＇＂’")


def is_closing_mark(ch: str) -> bool:
    """A punctuation mark a line may end with: category Po, Pe or Pf, except the marks that sit inside words."""
    return unicodedata.category(ch) in _CLOSING_CATEGORIES and ch not in _INSIDE_WORDS


def is_closing_marks_only(text: str) -> bool:
    """True for text made only of such marks (spaces aside), e.g. "、" or "!?"."""
    stripped = "".join(text.split())
    return bool(stripped) and all(is_closing_mark(ch) for ch in stripped)


def _chars(text: str) -> int:
    return len("".join(text.split()))


def _ends_with_mark(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and is_closing_mark(stripped[-1])


def _within(limits: LineLimits, text: str, start: float, end: float) -> bool:
    if limits.max_s is not None and end - start > limits.max_s + 1e-9:
        return False
    if limits.max_chars is not None and _chars(text) > limits.max_chars:
        return False
    return True


def _cut_inside_marks(piece: Piece) -> List[Piece]:
    """Cut a piece after each run of marks that has text after it; the time is shared out by characters."""
    text, start, end = piece
    parts: List[str] = []
    current = ""
    for i, ch in enumerate(text):
        current += ch
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if is_closing_mark(ch) and nxt and not is_closing_mark(nxt) and text[i + 1:].strip():
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    if len(parts) < 2:
        return [piece]
    weights = [max(1, _chars(p)) for p in parts]
    total = float(sum(weights))
    out: List[Piece] = []
    done = 0
    for part, weight in zip(parts, weights):
        s = start + (end - start) * done / total
        done += weight
        out.append((part, s, start + (end - start) * done / total))
    return out


def _split(pieces: List[Piece], limits: LineLimits) -> List[List[Piece]]:
    text = "".join(p[0] for p in pieces)
    start, end = pieces[0][1], max(pieces[-1][2], pieces[0][1])
    if _within(limits, text, start, end):
        return [pieces]
    candidates = [i for i in range(len(pieces) - 1) if _ends_with_mark(pieces[i][0])]
    if not candidates:
        return [pieces]
    middle = (start + end) / 2.0
    i = min(candidates, key=lambda k: (abs(pieces[k][2] - middle), k))
    return _split(pieces[:i + 1], limits) + _split(pieces[i + 1:], limits)


def split_line(text: str, start: float, end: float, pieces: Sequence[Piece],
               limits: LineLimits) -> List[Tuple[str, float, float]]:
    """One line -> its parts, as (text, start, end).

    ``text``, ``start`` and ``end`` are the line as built; ``pieces`` its pieces in order, with their
    times. A line within the limits, or with nothing to split it at, comes back as it was given.
    """
    if not limits.active or _within(limits, text, start, end) or not pieces:
        return [(text, start, end)]
    expanded: List[Piece] = []
    for piece in pieces:
        expanded.extend(_cut_inside_marks(piece))
    groups = _split(expanded, limits)
    if len(groups) < 2:
        return [(text, start, end)]
    out = []
    for group in groups:
        part_text = "".join(p[0] for p in group).strip()
        if part_text:
            out.append((part_text, group[0][1], max(group[-1][2], group[0][1])))
    return out


def lines_from_segments(segments, limits: LineLimits) -> List[Tuple[str, float, float]]:
    """The lines of a stable-ts result's segments (each with .text, .start, .end and .words, whose items
    have .word, .start and .end), each put through split_line. Lines with no text are left out."""
    lines: List[Tuple[str, float, float]] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        pieces = [(w.word, float(w.start), float(w.end)) for w in (seg.words or [])]
        lines.extend(split_line(text, float(seg.start), float(seg.end), pieces, limits))
    return lines
