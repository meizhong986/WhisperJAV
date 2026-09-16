#!/usr/bin/env python3
"""Recognising WhisperJAV's own intermediate files among discovered media.

Lives in `whisperjav/utils/` rather than in `whisperjav/main.py` so it can be imported and
tested without pulling the ASR stack: importing `whisperjav.main` loads torch,
stable_whisper, transformers, faster_whisper and whisper, which CLAUDE.md forbids doing
merely to verify something.

This module's own body uses the standard library only. Importing it still executes
`whisperjav/utils/__init__.py`, which pulls numpy -- light, and nothing from the ASR stack.
"""
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
