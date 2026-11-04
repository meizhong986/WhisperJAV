"""Post-processing of the CrispASR pipelines' subtitles (1.10.1; owner W418 5, W420-W422).

Option B of docs/plans/design/2026-11-01-crispasr-postprocessing-options.md, approved by the owner (W421 i1, W422):
each pipeline's row (crispasr_pipeline.EngineRow) names its clean-up; this module runs it, built from the pieces the
other pipelines already use, with fidelity's names and terms (W421 i5) and no on/off switch (W422 3).

Two levels:
  - text level, on each VAD group's text before the aligner times it: ChronosJAV's text cleaner
    (modules/assembly_text_cleaner.py, as Qwen3-ASR runs it). Cohere-CrispASR only, and only while its aligner
    is on: the aligner times whatever text it is given, so the times follow the cleaned text.
  - SRT level, on the stitched subtitles: from fidelity's sanitizer (modules/subtitle_sanitizer.py) the sequential
    duplicate merge, the hallucination filter (the owner's lists) and the repetition cleaner, then the three filters
    the CrispASR pipelines ran before (non-verbal lines, non-linguistic lines, scene overlap).

Left out on purpose: the sanitizer's speed rules (more than 30 characters per second; 4 characters or fewer under 1 per
second), its 200-character cut and its timing adjuster -- the owner's rule against cutting by character count or
duration (W203) and the model's own timing as the start (W411); the CPS start retimer (W202 5c).

Nothing is removed without a trace (requirement R-RAW): the stitched subtitles are copied first to
raw_subs/<name>.pass<N>.<pipeline>_stitched.original.srt, and every removal and change -- of the sanitizer stages,
of the three filters (drops and moved starts) and of the text cleaner -- is listed in
raw_subs/<name>.pass<N>.<pipeline>_stitched.artifacts.srt, in the format of fidelity's artifacts file. The raw_subs
folder is the one beside the pass's subtitle file (the pass worker sets it, as for the ASR telemetry: in
--output-dir source mode the pipeline's own output folder is the temp folder). The pass and the pipeline are in the
names because both passes of an Ensemble run share the media file's name. Cohere-CrispASR's raw copy already holds
the text cleaner's text; the model's own text of each changed VAD group is in the artifacts file.

MOSS-CrispASR (owner W505 2 b-d; before it the three earlier filters only, without trace files, W421 i3): the
sequential duplicate merge, the three filters, then its interjection rules (INTERJECTIONS, below); not the
hallucination filter or the repetition cleaner, which remove real lines in MOSS's text ("痛い。" x8, "ちょっと。") and
delete 笑 inside words. Measured on his SSNI-939 MOSS file (docs/plans/design/2026-11-04-moss-token-limits-loops-
cleanup.md section 5): 1,981 lines -> 1,340, runs of 3+ identical lines 37 -> 9. The trace is kept as for the other
pipelines.
"""

import copy
import dataclasses
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from whisperjav.utils.logger import logger

# SRT-level stages, in the order they run.
DUPLICATES = "duplicates"
HALLUCINATION = "hallucination"
REPETITION = "repetition"
NONVERBAL = "nonverbal"
NONLINGUISTIC = "nonlinguistic"
SCENE_OVERLAP = "scene_overlap"
SRT_STAGES = (DUPLICATES, HALLUCINATION, REPETITION, NONVERBAL, NONLINGUISTIC, SCENE_OVERLAP)  # the GGUF pipelines'
SANITIZER_STAGES = (DUPLICATES, HALLUCINATION, REPETITION)
# The clean-up before 1.10.1 (the three filters; MOSS-CrispASR's until W505).
EARLIER_STAGES = (NONVERBAL, NONLINGUISTIC, SCENE_OVERLAP)
# MOSS-CrispASR's interjection rules (owner W501 M3, W505 2 c, 2 d), last, as measured.
INTERJECTIONS = "interjections"
ORDER = SRT_STAGES + (INTERJECTIONS,)
MOSS_STAGES = (DUPLICATES, NONVERBAL, NONLINGUISTIC, SCENE_OVERLAP, INTERJECTIONS)

# The interjection rules, as measured (token_caps\m3_test.py, 2026-11-04). The text without spaces and punctuation:
_KANA = re.compile(r"[぀-ヿ]")
_PUNCT = re.compile(r"[\s。、！？!?.,…‥・ー〜～「」『』（）()\-—~♪]+")
# (b) a line with no kana that is one interjection, Latin or Chinese: Ah, Ahh, Oh, Uh, Um, Umm, Uum, Mm, Hmm, Eh, Ha
# (Haha) and 啊 嗯 哦 呃 唔 哈 嘿 呀 哎 噢 喔 嗷; keeps "Honey.", "No!", "Oh, cute." (W505 2 c).
_LATIN_INTERJECTION = re.compile(r"^(?:a+h+|o+h+|u+h+|u*m+|h+m+|e+h+|ha(?:ha)*|ah+a*)$", re.I)
_CHINESE_INTERJECTION = re.compile(r"^[啊嗯哦呃唔哈嘿呀哎噢喔嗷]+$")
# (c) a line made only of ああ / あっ / うん / えっ, all such lines (W501 M3, W505 2 d; the Netflix reference, a
# drama, has 7 such lines of 68).
_JAPANESE_INTERJECTIONS = re.compile(r"^(?:あ+っ?|うん|えっ)+$")


def is_interjection_line(text: str) -> bool:
    """True for a line MOSS-CrispASR's interjection rules drop: rule (b) or rule (c)."""
    core = _PUNCT.sub("", text or "")
    if not core:
        return False
    if _JAPANESE_INTERJECTIONS.match(core):
        return True
    if _KANA.search(text):
        return False
    return bool(_LATIN_INTERJECTION.match(core) or _CHINESE_INTERJECTION.match(core))

# Text-level stage.
TEXT_CLEANER = "text_cleaner"
TEXT_STAGES = (TEXT_CLEANER,)

RAW_SUBS = "raw_subs"  # fidelity's folder (ProcessingConstants.RAW_SUBS_FOLDER)
NO_CUT = 10 ** 9       # the repetition cleaner's length limit, out of reach (W203: no cut by character count)


def file_stem(basename: str, pipeline: str, pass_tag: Optional[str] = None) -> str:
    """The stem of the raw copy and the artifacts file: <name>.pass<N>.<pipeline>_stitched (no pass outside an
    Ensemble pass worker)."""
    return f"{basename}.{pass_tag}.{pipeline}_stitched" if pass_tag else f"{basename}.{pipeline}_stitched"


@dataclasses.dataclass
class TextChange:
    """A VAD group's text as the model wrote it and as the text cleaner left it, in media-file time."""
    start: float
    end: float
    original: str
    cleaned: str


class TextCleaner:
    """ChronosJAV's text cleaner for Qwen3-ASR, on one VAD group's text (Cohere-CrispASR, before the aligner)."""

    def __init__(self, language: str = "ja"):
        from whisperjav.modules.assembly_text_cleaner import AssemblyCleanerConfig, AssemblyTextCleaner

        self._cleaner = AssemblyTextCleaner(config=AssemblyCleanerConfig(enabled=True), language=language)

    def clean(self, text: str) -> str:
        cleaned, _stats = self._cleaner.clean(text)
        return cleaned


def make_sanitizer(language: str = "ja", hallucination: bool = True):
    """Fidelity's sanitizer with its hallucination lists, and a repetition cleaner without the 200-character cut.

    hallucination=False (a clean-up without the hallucination filter: MOSS-CrispASR, owner W505 2 b): the
    sanitizer's records and constants only -- its hallucination lists are not loaded or fetched, so no
    "Hallucination filter: ... loaded" line suggests a filter that does not run."""
    from whisperjav.config.sanitization_config import SanitizationConfig
    from whisperjav.modules.repetition_cleaner import RepetitionCleaner
    from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer

    class _WithoutLists(SubtitleSanitizer):
        def _init_processors(self):
            self.hallucination_remover = None
            self.repetition_cleaner = RepetitionCleaner(self.constants["repetition"])
            self.timing_adjuster = None   # the CrispASR clean-up changes no times (W203, W411)

    kind = SubtitleSanitizer if hallucination else _WithoutLists
    sanitizer = kind(SanitizationConfig(primary_language=language, enable_exact_matching=True,
                                        enable_repetition_cleaning=True))
    sanitizer.repetition_cleaner = RepetitionCleaner(
        dataclasses.replace(sanitizer.constants["repetition"], MAX_SUBTITLE_TEXT_LENGTH=NO_CUT))
    return sanitizer


def _srt_time(seconds: float) -> str:
    import pysrt
    return str(pysrt.SubRipTime(milliseconds=int(round(max(0.0, seconds) * 1000))))


class SrtCleanup:
    """The SRT-level clean-up of one pass; the sanitizer (and its lists) is built once, on first use."""

    def __init__(self, stages: Sequence[str], language: str = "ja", label: str = ""):
        unknown = [s for s in stages if s not in ORDER]
        if unknown:
            raise ValueError(f"unknown clean-up stage(s): {', '.join(unknown)}")
        self.stages = tuple(s for s in ORDER if s in stages)  # always in the order above
        self.language = language
        self.label = label
        self._sanitizer = None

    @property
    def uses_sanitizer(self) -> bool:
        return any(s in SANITIZER_STAGES for s in self.stages)

    def _get_sanitizer(self):
        if self._sanitizer is None:
            self._sanitizer = make_sanitizer(self.language, hallucination=HALLUCINATION in self.stages)
        return self._sanitizer

    def run(self, stitched: Path, raw_dir: Path, stem: str, count: int,
            text_changes: Sequence[TextChange] = ()) -> Tuple[Dict[str, Any], int]:
        """Clean ``stitched`` in place. Returns the counts and the number of subtitles left.

        With a sanitizer stage (every CrispASR pipeline; MOSS-CrispASR since W505 2 b) the trace is kept: the raw
        copy first, then every removal and change of every stage -- the three filters' drops and moved starts too --
        in the artifacts file. An earlier run's trace files of the same name are removed first, so the two files
        always describe this run.
        """
        stats: Dict[str, Any] = {}
        trace = self.uses_sanitizer
        if trace:
            for old in (raw_dir / f"{stem}.original.srt", raw_dir / f"{stem}.artifacts.srt"):
                try:
                    old.unlink()
                except FileNotFoundError:
                    pass
        if count == 0 and not (trace and text_changes):
            return stats, 0
        began = time.time()
        sanitizer = None
        if trace:
            raw_dir.mkdir(parents=True, exist_ok=True)
            if count:
                original = raw_dir / f"{stem}.original.srt"
                logger.info(f"Post-processing: saving a copy of the original to {original}")
                shutil.copy2(stitched, original)
                stats["original_srt"] = str(original)
            from whisperjav.modules.subtitle_sanitizer import Phase1Stats
            sanitizer = self._get_sanitizer()
            sanitizer.artifact_entries.clear()
            sanitizer.phase1_stats = Phase1Stats(original_count=count)
            self._record_text_changes(sanitizer, text_changes)
        if count and trace:
            count = self._sanitize(sanitizer, stitched, stats)

        if count and NONVERBAL in self.stages:
            from whisperjav.modules.subtitle_pipeline.cleaners.nonverbal_line_filter import NonverbalLineFilter
            before = self._read(stitched) if trace else None
            r = NonverbalLineFilter().filter_srt_file(stitched)
            stats["nonverbal_dropped"] = r.get("dropped_nonverbal", 0)
            stats["empty_removed"] = stats.get("empty_removed", 0) + r.get("dropped_empty", 0)
            count = r.get("final_count", count)
            if trace:
                self._record_filter(sanitizer, before, self._read(stitched), "nonverbal_line")
        if count and NONLINGUISTIC in self.stages:
            from whisperjav.modules.subtitle_pipeline.cleaners.nonlinguistic_utterance_filter import (
                NonlinguisticUtteranceFilter)
            logger.info("Post-processing: looking for non-linguistic lines")
            before = self._read(stitched) if trace else None
            r = NonlinguisticUtteranceFilter().filter_srt_file(stitched)
            stats["nonlinguistic_dropped"] = r.get("dropped_nonlinguistic", 0)
            stats["empty_removed"] = stats.get("empty_removed", 0) + r.get("dropped_empty", 0)
            count = r.get("final_count", count)
            logger.info(f"Post-processing: dropped {stats['nonlinguistic_dropped']} non-linguistic lines")
            if trace:
                self._record_filter(sanitizer, before, self._read(stitched), "nonlinguistic_line")
        if count and SCENE_OVERLAP in self.stages:
            from whisperjav.modules.subtitle_pipeline.cleaners.scene_overlap_resolver import SceneOverlapResolver
            before = self._read(stitched) if trace else None
            r = SceneOverlapResolver().resolve_srt_file(stitched)
            stats["overlap_nested_dropped"] = r.get("dropped_nested", 0)
            stats["overlap_shifted"] = r.get("shifted_starts", 0)
            count = r.get("final_count", count)
            if trace:
                self._record_filter(sanitizer, before, self._read(stitched), "scene_overlap")
        if count and INTERJECTIONS in self.stages:
            before = self._read(stitched)
            kept = [s for s in before if not is_interjection_line(s.text)]
            stats["interjections_dropped"] = len(before) - len(kept)
            if len(kept) != len(before):
                import pysrt
                for n, sub in enumerate(kept, 1):
                    sub.index = n
                pysrt.SubRipFile(kept).save(str(stitched), encoding="utf-8")
            count = len(kept)
            logger.info(f"Post-processing: dropped {stats['interjections_dropped']} interjection-only lines")
            if trace:
                self._record_filter(sanitizer, before, self._read(stitched), "interjection_line")

        if sanitizer is not None and sanitizer.artifact_entries:
            artifacts = raw_dir / f"{stem}.artifacts.srt"
            logger.info(f"Post-processing: writing the artifacts file {artifacts.name}")
            sanitizer._save_artifacts_srt(artifacts, count)
            stats["artifacts_srt"] = str(artifacts)
        stats["seconds"] = round(time.time() - began, 2)
        return stats, count

    @staticmethod
    def _read(path: Path) -> List[Any]:
        import pysrt
        return list(pysrt.open(str(path), encoding="utf-8"))

    @staticmethod
    def _record_filter(sanitizer, before: List[Any], after: List[Any], reason: str) -> None:
        """One filter's drops ([REMOVED - <reason>]) and moved starts ([MODIFIED - <reason> Start Moved]) into the
        artifacts file: a line of ``before`` with no line of the same text and end in ``after`` was dropped; one
        whose start differs was moved (the scene-overlap resolver moves starts only)."""
        from whisperjav.modules.subtitle_sanitizer import ArtifactEntry

        remaining = list(after)
        for b in before:
            match = next((a for a in remaining if a.text == b.text and a.end == b.end), None)
            if match is None:
                sanitizer.artifact_entries.append(ArtifactEntry(
                    index=b.index, start_time=str(b.start), end_time=str(b.end), original_text=b.text,
                    modified_text=None, reason=reason, category=reason, confidence=1.0, pattern=None,
                    step="filter", additional_info={}))
                continue
            remaining.remove(match)
            if match.start != b.start:
                sanitizer.artifact_entries.append(ArtifactEntry(
                    index=b.index, start_time=str(b.start), end_time=str(b.end), original_text=b.text,
                    modified_text=f"{match.text}  (start {b.start} -> {match.start})",
                    reason=f"{reason}_start_moved", category=reason, confidence=1.0, pattern=None,
                    step="filter", additional_info={}))

    @staticmethod
    def _record_text_changes(sanitizer, text_changes: Sequence[TextChange]) -> None:
        from whisperjav.modules.subtitle_sanitizer import ArtifactEntry

        for change in text_changes:
            sanitizer.artifact_entries.append(ArtifactEntry(
                index=0, start_time=_srt_time(change.start), end_time=_srt_time(change.end),
                original_text=change.original, modified_text=change.cleaned or None,
                reason="text_cleaner", category="text_cleaner", confidence=1.0, pattern=None,
                step="text_cleaner_before_aligner", additional_info={}))

    def _sanitize(self, sanitizer, stitched: Path, stats: Dict[str, Any]) -> int:
        """The sanitizer stages on the stitched file, in place; every removal and change recorded."""
        import pysrt
        from whisperjav.modules.subtitle_sanitizer import _HAS_LINGUISTIC_CONTENT

        subs = list(pysrt.open(str(stitched), encoding="utf-8"))
        before = len(subs)
        if DUPLICATES in self.stages:
            subs = self._merge_duplicates(sanitizer, subs, stats)
        kept: List[Any] = []
        removed_hallucinations = removed_repetitions = changed = 0
        for sub in subs:
            text = sub.text
            mods: List[Dict] = []
            reason = ""
            if HALLUCINATION in self.stages:
                text, hall = sanitizer.hallucination_remover.remove_hallucinations(text, self.language)
                if hall:
                    mods.extend(hall)
                    reason = "hallucination"
            if REPETITION in self.stages and text.strip():
                text, rep = sanitizer.repetition_cleaner.clean_repetitions(text)
                if rep:
                    mods.extend(rep)
                    reason = reason or "repetition"
            if not mods:
                kept.append(sub)
                continue
            original = copy.deepcopy(sub)
            if not text.strip() or not _HAS_LINGUISTIC_CONTENT.search(text):
                why = reason if not text.strip() else "symbol_only_residue"
                sanitizer._record_removal(original, original.index - 1, why, "content_cleaning", mods)
                self._mark_hallucination(sanitizer, reason)
                if reason == "hallucination":
                    removed_hallucinations += 1
                else:
                    removed_repetitions += 1
                continue
            sub.text = text.strip()
            sanitizer._record_modification(original, sub, original.index - 1, mods, "content_cleaning")
            self._mark_hallucination(sanitizer, reason)
            changed += 1
            kept.append(sub)
        for n, sub in enumerate(kept, 1):
            sub.index = n
        pysrt.SubRipFile(kept).save(str(stitched), encoding="utf-8")
        stats["hallucinations_removed"] = removed_hallucinations
        stats["repetitions_removed"] = removed_repetitions
        stats["lines_changed"] = changed
        logger.info(f"Post-processing: cleaned {before} subtitles down to {len(kept)}")
        return len(kept)

    @staticmethod
    def _mark_hallucination(sanitizer, reason: str) -> None:
        """The artifacts summary counts entries whose category says "hallucination"; the filter's own categories
        (closing_phrase, context_caption, ...) do not, so the entry it just recorded is marked."""
        if reason == "hallucination" and sanitizer.artifact_entries:
            entry = sanitizer.artifact_entries[-1]
            if "hallucination" not in entry.category:
                entry.category = f"hallucination,{entry.category}"

    @staticmethod
    def _merge_duplicates(sanitizer, subs: List[Any], stats: Dict[str, Any]) -> List[Any]:
        """Fidelity's sequential duplicate merge (3 or more identical lines, each gap under 600 ms -> one line from
        the first start to the last end), each merge recorded."""
        import pysrt

        cross = sanitizer.constants["cross_subtitle"]
        merged: List[Any] = []
        merges = 0
        i = 0
        while i < len(subs):
            run = [subs[i]]
            j = i + 1
            while j < len(subs):
                gap = subs[j].start.ordinal - run[-1].end.ordinal
                if subs[j].text.strip() == subs[i].text.strip() and 0 <= gap < cross.MAX_GAP_MS:
                    run.append(subs[j])
                    j += 1
                else:
                    break
            if len(run) >= cross.DEDUP_THRESHOLD:
                first, last = run[0], run[-1]
                one = pysrt.SubRipItem(index=first.index, start=first.start, end=last.end, text=first.text)
                sanitizer._record_modification(
                    first, one, first.index - 1,
                    [{"type": "sequential_duplicates_merged", "category": "repetition_cleaning", "confidence": 1.0,
                      "pattern": f"{len(run)} identical lines"}], "content_cleaning")
                merged.append(one)
                merges += 1
                i = j
            else:
                merged.append(subs[i])
                i += 1
        stats["duplicates_merged"] = merges
        return merged
