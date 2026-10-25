"""Taking one VAD group's CrispASR result apart and building its subtitles (route B design 4.9, 7).

Two line rules, chosen by the pipeline's row of the per-model table:

  "tokens" (Parakeet-CrispASR, option P2, owner W175): the group's tokens become one or more subtitles.
      A subtitle is cut only at natural cues (owner W203: "no brute force allowed. Only scientific
      approaches."; W204: "pauses count as a natural cue"), never by a number of characters or a
      duration:
        1. a sentence-end mark (。？！… ．.?!) lies between two spoken tokens;
        2. the silence between two spoken tokens is at least 0.6 s;
        3. a 、 (or another closing mark) or a space between Japanese characters lies between them,
           and the silence is at least 0.4 s (W208: "agreed as a paude of 0.4 after space or 、should cut").
      The two silences are defaults of CueSettings, which a pipeline may change (W208).
      About 600 ms is the silence at a strong clause boundary in spontaneous Japanese, about 900 ms at
      a sentence boundary and about 400 ms at a weak clause boundary (Watanabe, Kashiwagi & Maekawa
      2015, Corpus of Spontaneous Japanese); 0.6 s is also JapanesePostProcessor's pure_gap_threshold.
      Every time comes from the tokens, where each punctuation mark has its own time: a subtitle
      starts at its first spoken token and ends where its last spoken token ends, not where the mark
      after it ends (a mark's time is the frames after the last syllable).
  "group" (Cohere-CrispASR, option C1, owner W175): the group's text becomes one subtitle with the
      VAD group's start and end, which is where the speech segmenter found the speech.

Pure: no ASR import, no stable-ts.
"""

import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from whisperjav.crispasr.engine import CrispSegment

# (text, start, end) in media-file time, seconds.
Subtitle = Tuple[str, float, float]

LINE_RULES = ("tokens", "group")

# JapanesePostProcessor's Level 0 sentence-end marks (japanese_postprocessor.py:727), the space left out.
SENTENCE_END_MARKS = frozenset("。？！…．.?!")
# Cue 2: a cut at a silence of at least this (W204; Watanabe et al. 2015; japanese_postprocessor.py:220, 247).
PAUSE_CUT_S = 0.6
# Cue 3: a cut at a weak mark followed by a silence of at least this (W203, W208; Watanabe et al. 2015).
WEAK_MARK_PAUSE_S = 0.4


@dataclass(frozen=True)
class CueSettings:
    """The two pause values of the "tokens" rule, adjustable (owner W208: "ideally the values are not
    hardcoded and can be adjusted as parameters"); the defaults are cue 2's and cue 3's above."""
    pause_cut_s: float = PAUSE_CUT_S
    weak_mark_pause_s: float = WEAK_MARK_PAUSE_S

    def __post_init__(self):
        for name in ("pause_cut_s", "weak_mark_pause_s"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be more than 0 s, not {getattr(self, name)}")

    def as_dict(self) -> dict:
        return {"pause_cut_s": self.pause_cut_s, "weak_mark_pause_s": self.weak_mark_pause_s}


DEFAULT_CUES = CueSettings()
# A time moved further than this into the group counts as clamped (metadata only).
CLAMP_TOLERANCE_S = 0.05
# The shortest subtitle when a stretch's end is not after its start: a display repair, never a cut.
MIN_DISPLAY_S = 0.1

_CLOSING_CATEGORIES = frozenset({"Po", "Pe", "Pf"})
# Marks of those categories that sit inside words, or open as often as they close.
_INSIDE_WORDS = frozenset("・･'\"＇＂’")


def is_closing_mark(ch: str) -> bool:
    """A punctuation mark a subtitle may end with: category Po, Pe or Pf, except the marks that sit
    inside words (・ in ジョン・スミス, the apostrophe in CAN'T, straight quotes)."""
    return unicodedata.category(ch) in _CLOSING_CATEGORIES and ch not in _INSIDE_WORDS


def _is_opening(ch: str) -> bool:
    return unicodedata.category(ch) in ("Ps", "Pi")


def is_japanese_char(ch: str) -> bool:
    """Hiragana, katakana (full and half width), kanji, the iteration mark and the long-vowel mark."""
    o = ord(ch)
    return (0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF
            or 0xF900 <= o <= 0xFAFF or 0xFF66 <= o <= 0xFF9F or ch in "々〆ヶ")


# Kinds of unit
SPOKEN = "spoken"
END_MARK = "end"     # a sentence-end mark, with any closing marks beside it
WEAK_MARK = "weak"   # another closing mark (、), or a space between Japanese characters
OPEN_MARK = "open"   # an opening bracket or quote: text that belongs to what follows
SPACE = "space"      # a space that is part of the text (between Latin words), not a cue


@dataclass
class _Unit:
    text: str
    start: float
    end: float
    kind: str


@dataclass
class LineStats:
    clamped_times: int = 0
    units: int = 0
    fallback_segments: int = 0  # segments read from words or as a whole, because a token time was missing


def _is_mark_char(ch: str) -> bool:
    """A character design 7.2 Step A.3 treats as a mark: a sentence-end mark, a mark a subtitle may end with
    (is_closing_mark: 、，」 and the like), or an opening bracket or quote. Anything else -- ・ in ジョン・スミス,
    〜, a hyphen, a straight quote -- is part of the spoken text and never a cue."""
    return ch in SENTENCE_END_MARKS or is_closing_mark(ch) or _is_opening(ch)


def _mark_kind(text: str) -> str:
    chars = [c for c in text if not c.isspace()]
    if any(c in SENTENCE_END_MARKS for c in chars):
        return END_MARK
    if chars and all(_is_opening(c) for c in chars):
        return OPEN_MARK
    return WEAK_MARK


def _split_marks(text: str, start: float, end: float) -> List[_Unit]:
    """One timed piece of text as units: leading opening marks, the spoken part with the piece's times,
    trailing marks as zero-length units at the piece's end. A piece that is only marks is one mark unit."""
    core = text.strip()
    if not core:
        return []
    if all(_is_mark_char(c) or c.isspace() for c in core):
        return [_Unit(core, start, end, _mark_kind(core))]
    head = 0
    while head < len(core) and _is_opening(core[head]):
        head += 1
    tail = len(core)
    while tail > head and _is_mark_char(core[tail - 1]) and not _is_opening(core[tail - 1]):
        tail -= 1
    units = []
    if head:
        units.append(_Unit(core[:head], start, start, OPEN_MARK))
    units.append(_Unit(core[head:tail], start, end, SPOKEN))
    if tail < len(core):
        units.append(_Unit(core[tail:], end, end, _mark_kind(core[tail:])))
    return units


def _last_char(units: List[_Unit]) -> str:
    for u in reversed(units):
        if u.text:
            return u.text[-1]
    return ""


def _is_latin_or_digit(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _space_unit(units: List[_Unit], space: str, at: float, next_text: str) -> Optional[_Unit]:
    """The unit for a space in the text before ``next_text``: a cue between two Japanese characters
    (cue 3), part of the text next to a Latin letter or a digit, nothing at the start of the group."""
    if not units or not next_text:
        return None
    prev, nxt = _last_char(units), next_text[0]
    if is_japanese_char(prev) and is_japanese_char(nxt):
        return _Unit(space, at, at, WEAK_MARK)
    if _is_latin_or_digit(prev) or _is_latin_or_digit(nxt):
        return _Unit(" ", at, at, SPACE)
    return None


def _segment_units(seg: CrispSegment, stats: LineStats) -> List[_Unit]:
    """Step A.1-3: the units of one segment, in the model's order, with their times."""
    timed = lambda xs: bool(xs) and all(x.start is not None and x.end is not None for x in xs)  # noqa: E731
    units: List[_Unit] = []
    if timed(seg.tokens):
        # Where the segment text has a space is decided position by position: a token's leading space is a
        # space of the text only where the segment text has one before that token's text; otherwise it is
        # the model's word-start marker (" そう", whose segment text "そうですよね監督。" has none). A token
        # that is only a space is a space of the text, except the blank token that starts most groups.
        text, pos = seg.text, 0
        pending: Optional[Tuple[str, float]] = None  # a space of the text, waiting for the text after it
        for tok in seg.tokens:
            raw = tok.text
            body = raw.strip()
            if not body:
                if units and pending is None:
                    pending = ("　" if "　" in raw else " ", float(tok.start))
                continue
            found = text.find(body, pos) if text else -1
            if found >= 0:
                text_has_space = found > pos and not text[pos:found].strip()
                pos = found + len(body)
            else:  # the text does not show it: only a space between Latin words counts
                text_has_space = raw[:1].isspace() and _is_latin_or_digit(_last_char(units) or "x")
            if pending is None and text_has_space:
                pending = ("　" if "　" in raw[:len(raw) - len(raw.lstrip())] else " ", float(tok.start))
            if pending is not None:
                space = _space_unit(units, pending[0], pending[1], body)
                if space is not None:
                    units.append(space)
                pending = None
            units.extend(_split_marks(body, float(tok.start), float(tok.end)))
            if raw[-1:].isspace():
                pending = (" ", float(tok.end))
        return units
    stats.fallback_segments += 1
    if timed(seg.words):
        for w in seg.words:
            body = w.text.strip()
            if w.text[:1].isspace():
                space = _space_unit(units, " ", float(w.start), body)
                if space is not None and space.kind == SPACE:  # words carry no reliable cue spaces
                    units.append(space)
            units.extend(_split_marks(body, float(w.start), float(w.end)))
        return units
    # No times inside the segment: its whole text is one unit with the segment's own times. No cut by
    # characters (W203): split_frame_to_words, which cuts at 15 and 20 characters, is not used.
    text = seg.text.strip()
    if text:
        units.append(_Unit(text, float(seg.start), float(seg.end), SPOKEN))
    return units


def _clean_times(units: List[_Unit], duration: float, stats: LineStats) -> None:
    """Step A.4: clamp to the group; spoken starts non-decreasing; no spoken unit ends after the next starts."""
    for u in units:
        for attr in ("start", "end"):
            t = getattr(u, attr)
            c = min(max(t, 0.0), duration)
            if abs(c - t) > CLAMP_TOLERANCE_S:
                stats.clamped_times += 1
            setattr(u, attr, c)
    spoken = [u for u in units if u.kind == SPOKEN]
    previous = 0.0
    for u in spoken:
        u.start = max(u.start, previous)
        u.end = max(u.end, u.start)
        previous = u.start
    for a, b in zip(spoken, spoken[1:]):
        a.end = min(a.end, b.start)


def _is_cut(between: List[_Unit], gap: float, cues: CueSettings) -> bool:
    """Step B: the three cues."""
    if any(u.kind == END_MARK for u in between):
        return True
    if gap >= cues.pause_cut_s:
        return True
    return gap >= cues.weak_mark_pause_s and any(u.kind == WEAK_MARK for u in between)


def _pieces(units: List[_Unit], cues: CueSettings) -> List[List[_Unit]]:
    """Step B-C: the units split into stretches at the cuts. Marks before a cut close the stretch before
    it; opening marks go with the stretch after it."""
    spoken_idx = [i for i, u in enumerate(units) if u.kind == SPOKEN]
    if not spoken_idx:
        return []
    pieces: List[List[_Unit]] = []
    current: List[_Unit] = list(units[:spoken_idx[0] + 1])
    for i, j in zip(spoken_idx, spoken_idx[1:]):
        between = units[i + 1:j]
        if _is_cut(between, units[j].start - units[i].end, cues):
            closing = [u for u in between if u.kind != OPEN_MARK]
            opening = [u for u in between if u.kind == OPEN_MARK]
            pieces.append(current + closing)
            current = opening + [units[j]]
        else:
            current.extend(between + [units[j]])
    pieces.append(current + list(units[spoken_idx[-1] + 1:]))
    return pieces


def _join(piece: List[_Unit]) -> str:
    return "".join(u.text for u in piece).strip()


def tokens_rule(group_start: float, group_end: float, segments: Sequence[CrispSegment],
                stats: Optional[LineStats] = None, cues: CueSettings = DEFAULT_CUES) -> List[Subtitle]:
    """Parakeet option P2: one group's tokens → one or more subtitles, in media-file time."""
    stats = stats if stats is not None else LineStats()
    duration = max(0.0, group_end - group_start)
    units: List[_Unit] = []
    for seg in segments:
        units.extend(_segment_units(seg, stats))
    stats.units += len(units)
    _clean_times(units, duration, stats)
    pieces = _pieces(units, cues)
    if pieces:
        # Closing marks before the group's first spoken word close nothing in this group.
        first = pieces[0]
        while first and first[0].kind in (END_MARK, WEAK_MARK, SPACE):
            first.pop(0)
    subtitles: List[Subtitle] = []
    previous_end = group_start
    for piece in pieces:
        text = _join(piece)
        spoken = [u for u in piece if u.kind == SPOKEN]
        if not text or not spoken:
            continue
        start = group_start + spoken[0].start
        end = group_start + spoken[-1].end
        if end <= start:
            end = min(start + MIN_DISPLAY_S, group_end)
            if end <= start:
                start = max(end - MIN_DISPLAY_S, previous_end, group_start)
        subtitles.append((text, start, end))
        previous_end = end
    # The display repair above must not make a subtitle run into the next one of the group (4.9).
    for i in range(len(subtitles) - 1):
        text, start, end = subtitles[i]
        next_start = subtitles[i + 1][1]
        if end > next_start:
            subtitles[i] = (text, start, max(start, next_start))
    return subtitles


def group_rule(group_start: float, group_end: float, segments: Sequence[CrispSegment]) -> List[Subtitle]:
    """Cohere option C1: the group's text → one subtitle with the VAD group's start and end."""
    text = "".join(seg.text.strip() for seg in segments).strip()
    return [(text, group_start, group_end)] if text else []


def build(line_rule: str, group_start: float, group_end: float, segments: Sequence[CrispSegment],
          stats: Optional[LineStats] = None, cues: CueSettings = DEFAULT_CUES) -> List[Subtitle]:
    """The subtitles of one VAD group, by the pipeline's line rule; ``cues`` is used by "tokens" only."""
    if line_rule == "tokens":
        return tokens_rule(group_start, group_end, segments, stats, cues)
    if line_rule == "group":
        return group_rule(group_start, group_end, segments)
    raise ValueError(f"unknown line rule '{line_rule}' ({', '.join(LINE_RULES)})")
