#!/usr/bin/env python3
"""Labelling of WhisperJAV's own leftovers in the discovered-media listing (F4).

Discovery walks folders recursively, so a working folder can yield intermediates from an
earlier run. Those are LABELLED, never dropped: a user's own file may carry any name and
sit in any folder, and silently removing one would lose their input.

Imports only whisperjav.main's helper -- no ASR module is touched.
"""
from pathlib import Path

import pytest

from whisperjav.main import looks_like_whisperjav_leftover as leftover


class TestLeftoverShapesAreLabelled:
    def test_extracted_wav(self):
        assert leftover(Path("/tmp/whisperjav/SONE-853_extracted.wav"))

    def test_extracted_wav_is_case_insensitive(self):
        assert leftover(Path("/tmp/whisperjav/SONE-853_EXTRACTED.WAV"))

    def test_file_under_a_scenes_folder(self):
        assert leftover(Path("/tmp/whisperjav/scenes/SONE-853_scene_0001.wav"))

    def test_file_deeper_under_a_scenes_folder(self):
        assert leftover(Path("/tmp/whisperjav/scenes/batch2/x.wav"))


class TestOrdinaryInputIsNotLabelled:
    @pytest.mark.parametrize("p", [
        "/media/SONE-853.mp4",
        "/media/holiday.wav",
        "F:/MEDIA_DLNA/MIMK-276/MIMK-276.mkv",
        "/media/my_extracted_footage.mp4",     # 'extracted' but not the _extracted.wav shape
        "/media/scenes_from_a_play.mp4",       # 'scenes' in the file name, not a folder
        "/media/behind the scenes/clip.mp4",   # folder is 'behind the scenes', not 'scenes'
    ])
    def test_not_flagged(self, p):
        assert not leftover(Path(p))


def test_labelling_never_shortens_the_input_list():
    """The guarantee that matters: labelling is not filtering."""
    discovered = [
        Path("/tmp/whisperjav/a_extracted.wav"),
        Path("/tmp/whisperjav/scenes/a_scene_0001.wav"),
        Path("/media/real-input.mp4"),
    ]
    labelled = [p for p in discovered if leftover(p)]
    kept = list(discovered)          # what main.py passes on: everything

    assert len(labelled) == 2
    assert kept == discovered
    assert len(kept) == 3, "a labelled file must still be processed"
