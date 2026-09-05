"""Scene-overlap timestamp resolver for Qwen-family SRT output (v1.9.0).

The ChronosJAV / qwen pipeline uses semantic scene detection, which extracts
each scene with a ±0.35s buffer ("asr_processing" timestamps — see
whisperjav/vendor/semantic_audio_clustering.py). That buffer is intentional and
good for ASR (it captures soft Japanese onsets / trailing particles at scene
edges), but adjacent scenes therefore overlap by ~0.7s. Each scene is
transcribed independently and stitched using the buffered scene start as the
offset, so at every scene boundary the tail of scene N and the head of scene
N+1 land on overlapping timestamps. SRTStitcher does no overlap resolution and
the qwen pipeline skips the legacy sanitizer/TimingAdjuster, so the overlaps
reach the final SRT.

Two distinct artifacts result, handled by two rules:

  1. NESTED DUPLICATE (containment). When a scene boundary lands on an
     utterance onset, the earlier scene's 0.35s trailing pad captures only the
     onset -> a short fragment ("少し", "…先生。") that is FULLY NESTED inside the
     full sub the next scene produces ("少しだけだぞ…", "…先生私で…"). The fragment
     is a truncated duplicate. Rule: DROP the nested (shorter) entry, keep the
     longer one intact. A naive "clip earlier end to next start" would instead
     corrupt the good full sub, so containment must be detected first.

  2. PARTIAL OVERLAP. Otherwise the tail of one sub bleeds past the start of the
     next (the recurring ~0.7s). Rule: KEEP the earlier entry's end unchanged and
     PUSH the later entry's START forward to just after it. This matches how the
     semantic scene overlap actually arises — the earlier sub is the scene-A tail
     and has the accurate end (it captured the full line); it is the later sub
     (scene-B head) whose start is padded too early into the overlap zone. So the
     second line genuinely begins after the first finishes. (The previous logic
     clipped the earlier end instead, which made subtitles end ~0.7s too early
     while the speaker was still finishing the line.) Both entries are kept; only
     the second's start moves. In-line text is NOT edited (a duplicated boundary
     word stays in its line — removing words is riskier than it's worth).

No-op when scenes don't overlap (e.g. non-semantic scene detection), so it is
safe to run for every qwen backend. Operates in place on the stitched SRT,
mirroring the other Phase-8 filters (parse -> resolve -> renumber -> write).
"""

from pathlib import Path
from typing import Dict, Union

import pysrt

from whisperjav.utils.logger import logger


class SceneOverlapResolver:
    """Resolves scene-overlap timestamp collisions in a stitched SRT file."""

    # Gap (ms) left between an entry's end and the next entry's start.
    MIN_GAP_MS = 1

    def resolve_srt_file(self, srt_path: Union[str, Path]) -> Dict[str, int]:
        """Drop nested-duplicate entries + de-overlap partial overlaps, in place.

        Entries are sorted by start time (container before nested via a
        secondary end-descending sort), resolved sequentially against the last
        kept entry, renumbered, and written back. Sorting also repairs any
        non-start-ordered stitch output as a side effect.

        Returns:
            Stats dict: original_count, dropped_nested, shifted_starts,
            final_count.
        """
        path = Path(srt_path)
        stats = {
            "original_count": 0,
            "dropped_nested": 0,
            "shifted_starts": 0,
            "final_count": 0,
        }
        if not path.exists() or path.stat().st_size == 0:
            return stats

        try:
            subs = pysrt.open(str(path), encoding="utf-8")
        except Exception as e:  # pragma: no cover - defensive parse guard
            logger.warning(
                "[SceneOverlapResolver] failed to parse %s: %s", path, e
            )
            return stats

        stats["original_count"] = len(subs)
        if len(subs) < 2:
            stats["final_count"] = len(subs)
            return stats

        # Sort by start ascending; for equal starts, longer (later end) first so
        # a container is always seen before the entry nested inside it.
        ordered = sorted(subs, key=lambda s: (s.start.ordinal, -s.end.ordinal))

        kept: list = []
        for cur in ordered:
            if not kept:
                kept.append(cur)
                continue

            prev = kept[-1]
            ps, pe = prev.start.ordinal, prev.end.ordinal
            cs, ce = cur.start.ordinal, cur.end.ordinal

            # Rule 1a: cur fully nested inside prev -> drop cur (truncated dup).
            if cs >= ps and ce <= pe:
                stats["dropped_nested"] += 1
                continue

            # Rule 1b: same start, cur is the longer/fuller entry -> prev is the
            # nested fragment; keep cur instead of prev.
            if cs == ps and ce > pe:
                kept[-1] = cur
                stats["dropped_nested"] += 1
                continue

            # Rule 2: partial overlap -> KEEP prev's end (accurate scene-A tail),
            # push cur's START forward to just after prev ends. cur extends past
            # prev (containment handled by Rule 1), so cur stays valid. If the
            # push would collapse cur (its end is within one gap of prev's end),
            # it is a near-nested sliver -> drop it.
            if pe > cs:
                new_cs = pe + self.MIN_GAP_MS
                if new_cs < ce:
                    cur.start = pysrt.SubRipTime.from_ordinal(new_cs)
                    stats["shifted_starts"] += 1
                else:
                    stats["dropped_nested"] += 1
                    continue

            kept.append(cur)

        # Renumber surviving entries in start order (1, 2, 3, ...).
        for new_idx, sub in enumerate(kept, start=1):
            sub.index = new_idx

        pysrt.SubRipFile(items=kept).save(str(path), encoding="utf-8")

        stats["final_count"] = len(kept)
        if stats["dropped_nested"] or stats["shifted_starts"]:
            logger.info(
                "[SceneOverlapResolver] %s: %d -> %d entries "
                "(-%d nested duplicate, %d starts shifted)",
                path.name,
                stats["original_count"],
                stats["final_count"],
                stats["dropped_nested"],
                stats["shifted_starts"],
            )
        return stats
