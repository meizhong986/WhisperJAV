#!/usr/bin/env python3
"""Recognising WhisperJAV's own intermediate files among discovered media.

Lives in `whisperjav/utils/` rather than in `whisperjav/main.py` so it can be imported and
tested without pulling the ASR stack: importing `whisperjav.main` loads torch,
stable_whisper, transformers, faster_whisper and whisper, which CLAUDE.md forbids doing
merely to verify something.

Standard library only.
"""
from pathlib import Path

# The folders WhisperJAV creates under its temp directory, from
# cleanup_temp_directory() in whisperjav/main.py. A discovered file sitting under any of
# them is one of ours.
WHISPERJAV_WORK_DIRS = ("scenes", "enhanced_scenes", "scene_srts", "raw_subs")

# Suffixes WhisperJAV gives to audio it writes itself:
#   <basename>_extracted.wav   BasePipeline._cleanup_temp_files
#   <basename>_enhanced.wav    speech enhancement pipeline helper
#   <basename>_resampled.wav   speech enhancement pipeline helper
WHISPERJAV_AUDIO_SUFFIXES = ("_extracted.wav", "_enhanced.wav", "_resampled.wav")


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
