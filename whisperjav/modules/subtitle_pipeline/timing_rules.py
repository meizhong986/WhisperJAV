"""
The timing model's rules: where a ChronosJAV subtitle starts and where a VAD group splits (item 12, C3).

Design note docs/plans/design/2026-10-28-chronosjav-timing-c3-c1.md, section 5.2 (rules T1-T8, as revised
after the owner's W255; T2 and T7 changed and T9 added after tuning, W296 and W300). The timing model is Whisper small run by CrispASR with DTW token times; its text is
thrown away. DTW gives where each Whisper segment's speech STARTS (its first text token's onset) but no end of
speech (probe P2c, section 14), so the rules use onsets only:

  T1  a Whisper segment counts if it has text, an onset, and no_speech_prob below 0.6 (Whisper's own
      no-speech threshold; a negative value means the binding has no figure, and counts);
  T2  the subtitle starts where the aligner puts its first character; only where there is no aligner time
      (text without word times, a collapsed scene) does it move later, to the first counted onset; never
      outside the VAD group; the end of the group's last subtitle stays where it is (the aligner already put
      it at the group's end). Owner W296 1: tuning against the Netflix reference found the onset start late
      (median +0.14 s on the same lines), settling W266 Q2;
  T3  nothing counted: nothing changes; the timing model never removes text;
  T4  a split falls only at a mark in ChronosJAV's text (a sentence-end mark, a closing mark such as 、, or a
      space between Japanese characters), at the mark whose aligner time lies nearest the next Whisper
      segment's onset, within the stretch from one onset to the next; no mark, no split (W203: no cut by a
      number of characters or a duration);
  T5  no split without word times or when the scene's alignment collapsed: the start only;
  T6  each part is its own subtitle: the next part starts at its first character's aligner time (at the next
      segment's onset where there is no aligner time), the part before ends where the next starts;
  T7  with regrouping on, the regrouping makes the splits and the start follows T2, so the timing model moves
      only the start of a frame without word times (owner W300 4);
  T8  a part's first words are moved to its start, so its displayed times follow (stable-ts takes a subtitle's
      times from its first and last word);
  T9  a split never leaves a part that a line filter the run applies after the timing step would remove (a
      moan alone, an ellipsis alone, a single non-verbal token such as はい。, punctuation alone); such a part
      stays with its neighbour. Owner W296 2 and W300 3; to be reopened if user feedback asks for it. A later
      step that drops a subtitle by its times (the scene-overlap resolver) is not covered.

Words are the orchestrator's dicts {"word", "start", "end"}, in seconds from the start of the frame (one VAD
group). Standard library only; no ASR import.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from whisperjav.crispasr.lines import SENTENCE_END_MARKS, is_closing_mark, is_japanese_char

# T1: Whisper's own no-speech threshold (whisper.cpp's default no_speech_thold).
NO_SPEECH_THRESHOLD = 0.6

Word = Dict[str, Any]


@dataclass(frozen=True)
class TimedSegment:
    onset: Optional[float]   # the DTW time of the segment's first text token, seconds from the frame's start
    no_speech_prob: float    # Whisper's figure; negative when the binding gives none
    has_text: bool


def counted(segments: Sequence[TimedSegment]) -> List[TimedSegment]:
    """T1: the segments that count, in order."""
    return [s for s in segments
            if s.has_text and s.onset is not None and s.no_speech_prob < NO_SPEECH_THRESHOLD]


def speech_start(segments: Sequence[TimedSegment], frame_s: float) -> Optional[float]:
    """T2: where the speech starts in the frame, or None when no segment counts or the onset lies past the end."""
    first = counted(segments)
    if not first:
        return None
    onset = max(0.0, float(first[0].onset))
    return onset if onset < frame_s else None


def _is_cut_after(words: Sequence[Word], i: int) -> bool:
    """T4: whether the text after word i holds a cut mark."""
    text = str(words[i]["word"])
    core = text.rstrip()
    if not core:
        return False
    if core[-1] in SENTENCE_END_MARKS or is_closing_mark(core[-1]):
        return True
    following = str(words[i + 1]["word"]).lstrip() if i + 1 < len(words) else ""
    return text != core and is_japanese_char(core[-1]) and bool(following) and is_japanese_char(following[0])


def split_points(words: Sequence[Word], segments: Sequence[TimedSegment]) -> List[Tuple[int, float]]:
    """T4: (index of the last word before a split, the next part's start), in order."""
    found: Dict[int, float] = {}
    marks = [i for i in range(len(words) - 1) if _is_cut_after(words, i)]
    segs = counted(segments)
    for before, after in zip(segs, segs[1:]):
        lo, hi = float(before.onset), float(after.onset)
        if hi <= lo:
            continue
        best: Optional[Tuple[float, int]] = None
        for i in marks:
            t = float(words[i]["end"])
            if lo <= t <= hi and (best is None or abs(hi - t) < best[0]):
                best = (abs(hi - t), i)
        if best is not None and best[1] not in found:
            found[best[1]] = hi
    return sorted(found.items())


def _text(part: Sequence[Word]) -> str:
    return "".join(str(w["word"]) for w in part).strip()


def pieces(words: Sequence[Word], segments: Sequence[TimedSegment], frame_s: float,
           allow_split: bool = True, aligner_start: bool = True,
           removable: Optional[Callable[[str], bool]] = None) -> List[List[Word]]:
    """T2-T9: the frame's words as one or more subtitles (lists of words), in frame time. New dicts are returned.

    ``aligner_start``: the words carry the aligner's times, so each part starts at its first word (T2, T6);
    without it the start moves to the Whisper onset. ``removable``: the later line filters' test (T9).
    """
    words = [dict(w) for w in words]
    if not words:
        return []
    groups: List[List[Word]] = []
    onsets: List[Optional[float]] = [speech_start(segments, frame_s)]
    begin = 0
    for index, next_start in (split_points(words, segments) if allow_split else []):
        groups.append(words[begin:index + 1])
        onsets.append(next_start)
        begin = index + 1
    groups.append(words[begin:])
    if removable is not None and len(groups) > 1:  # T9: no part the line filters would remove
        i = 0
        while i < len(groups) and len(groups) > 1:
            if removable(_text(groups[i])):
                if i + 1 < len(groups):
                    groups[i + 1] = groups[i] + groups[i + 1]
                    onsets[i + 1] = onsets[i]
                else:
                    groups[i - 1] = groups[i - 1] + groups[i]
                del groups[i], onsets[i]
                i = max(0, i - 1)
            else:
                i += 1
    starts: List[Optional[float]] = (
        [None] + [float(g[0]["start"]) for g in groups[1:]] if aligner_start else onsets)

    result: List[List[Word]] = []
    for group, start in zip(groups, starts):
        if result and (start is None or start <= result[-1][0]["start"] or start >= frame_s):
            result[-1].extend(group)  # T6: a split that would not start after the part before is not made
            continue
        if start is not None:
            for w in group:  # T8: words before the part's start move to it
                if w["start"] < start:
                    w["start"] = start
                    w["end"] = max(w["end"], start)
        result.append(group)
    for part, nxt in zip(result, result[1:]):  # T6: a part ends where the next starts
        last, next_start = part[-1], nxt[0]["start"]
        last["end"] = next_start
        last["start"] = min(last["start"], next_start)
    for part in result:
        for w in part:
            w["start"] = min(max(0.0, w["start"]), frame_s)
            w["end"] = min(max(w["start"], w["end"]), frame_s)
    return result
