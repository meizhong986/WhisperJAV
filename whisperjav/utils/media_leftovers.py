#!/usr/bin/env python3
"""Recognising WhisperJAV's own intermediate files among discovered media.

Lives in `whisperjav/utils/` rather than in `whisperjav/main.py` so it can be imported and
tested without pulling the ASR stack: importing `whisperjav.main` loads torch,
stable_whisper, transformers, faster_whisper and whisper, which CLAUDE.md forbids doing
merely to verify something.

This module's own body uses the standard library only. Importing it still executes
`whisperjav/utils/__init__.py`, which pulls numpy -- light, and nothing from the ASR stack.
"""
import os
from pathlib import Path

# Every folder WhisperJAV creates under its temp directory. A discovered file sitting under
# any of them is one of ours.
#   scenes, scene_srts, raw_subs   pipelines/base_pipeline.py
#   enhanced_scenes                modules/speech_enhancement/pipeline_helper.py
#   resampled_scenes               modules/speech_enhancement/pipeline_helper.py
#   crispasr_out                   pipelines/crispasr_pipeline.py
# main.py's cleanup routine imports THIS tuple rather than keeping its own copy: the two
# lists were maintained separately and both had drifted, each missing resampled_scenes and
# crispasr_out, so those folders were neither cleaned up nor labelled.
WHISPERJAV_WORK_DIRS = (
    "scenes",
    "enhanced_scenes",
    "resampled_scenes",
    "scene_srts",
    "raw_subs",
    "crispasr_out",
)

# Suffixes WhisperJAV gives to audio it writes itself:
#   <basename>_extracted.wav   BasePipeline._cleanup_temp_files
#   <basename>_enhanced.wav    speech enhancement pipeline helper
#   <basename>_resampled.wav   speech enhancement pipeline helper
WHISPERJAV_AUDIO_SUFFIXES = ("_extracted.wav", "_enhanced.wav", "_resampled.wav")

# Everything WhisperJAV writes into the ROOT of its temp directory, from
# BasePipeline._cleanup_temp_files plus the enhancement helper. Used to decide what a run
# may delete when tidying up: see is_whisperjav_temp_file.
WHISPERJAV_TEMP_SUFFIXES = WHISPERJAV_AUDIO_SUFFIXES + (
    "_raw.srt",
    "_stitched.srt",
    "_master.json",
)


def _same_place(a: Path, b: Path) -> bool:
    """True if two paths name the same folder, allowing for case and `..` on Windows."""
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except (OSError, ValueError):
        return False


def _contains(parent: Path, child: Path) -> bool:
    """True if `parent` is `child` or an ancestor of it."""
    try:
        p = Path(os.path.normcase(os.path.realpath(parent)))
        c = Path(os.path.normcase(os.path.realpath(child)))
    except (OSError, ValueError):
        return False
    return p == c or p in c.parents


def temp_dir_conflicts(temp_dir, media_paths=(), output_dir=None) -> list:
    """Reasons the chosen working folder must not be used, in plain words.

    The working folder is emptied at the end of a run. If it is also the folder holding the
    user's videos, or the folder their subtitles are written to, that cleanup reaches their
    files. Cleanup now only deletes what WhisperJAV itself wrote, but sharing the folder
    still invites trouble -- a video named like one of our intermediates would be removed --
    so the sturdier rule is simply to refuse the overlap.

    Also refuses a working folder that CONTAINS the videos: the hazard is the same, since
    the per-run subfolders are deleted outright.

    Returns a list of sentences to show the user. Empty means the folder is fine.
    """
    problems = []
    temp = Path(temp_dir)

    if output_dir and str(output_dir).strip().lower() != "source":
        if _same_place(temp, Path(output_dir)):
            problems.append(
                "It is the same folder your subtitles are saved to:\n    %s\n"
                "  WhisperJAV empties its working folder when a run finishes." % temp)

    seen = set()
    for media in media_paths:
        folder = Path(media).parent
        key = os.path.normcase(str(folder))
        if key in seen:
            continue
        seen.add(key)
        if _same_place(temp, folder):
            problems.append(
                "It is the same folder your videos are in:\n    %s\n"
                "  WhisperJAV empties its working folder when a run finishes." % folder)
        elif _contains(temp, folder):
            problems.append(
                "Your videos are inside it:\n    working folder: %s\n    your videos:    %s\n"
                "  WhisperJAV deletes folders inside its working folder when a run finishes."
                % (temp, folder))

    return problems


def is_whisperjav_temp_file(path: Path) -> bool:
    """True if a file at the root of a temp directory is one WhisperJAV wrote.

    Used by the end-of-run cleanup so that it removes its OWN intermediates and nothing
    else. The cleanup used to delete every file it found at that level, which destroyed a
    user's media whenever `--temp-dir` pointed at a folder of their own (see #TEMP-CLEANUP
    note in main.py). Leaving an unrecognised file behind is litter; deleting one is data
    loss, so anything not matched here is kept.
    """
    return path.name.lower().endswith(WHISPERJAV_TEMP_SUFFIXES)


def looks_like_whisperjav_leftover(path: Path) -> bool:
    """True if a discovered file is shaped like one WhisperJAV itself wrote earlier.

    Discovery walks folders recursively, so pointing it at a working folder can pick up
    intermediates from a previous run.

    This only LABELS a file. Nothing is ever removed from the input list on the strength
    of it -- a user's own file may legitimately carry any name or sit in any folder, and
    a folder genuinely called "scenes" is somebody's holiday footage as often as ours.
    """
    name = path.name.lower()
    if name.endswith(WHISPERJAV_AUDIO_SUFFIXES):
        return True
    lowered = {part.lower() for part in path.parent.parts}
    return any(work_dir in lowered for work_dir in WHISPERJAV_WORK_DIRS)
