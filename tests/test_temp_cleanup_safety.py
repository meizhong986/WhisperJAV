#!/usr/bin/env python3
"""End-of-run cleanup must delete only what WhisperJAV wrote.

`cleanup_temp_directory` used to delete EVERY file sitting at the root of the temp
directory. That is harmless for the default folder under the system temp, which WhisperJAV
owns — but `--temp-dir` can point anywhere, and pointing it at a folder of your own media
is a reasonable thing to do. On any run without `--keep-temp` that destroyed the lot.

Leaving an unrecognised file behind is litter; deleting one is data loss. These tests pin
that direction.

Imports `whisperjav.utils.media_leftovers` only — no ASR module, nothing from main.py.
"""
from pathlib import Path

import pytest

from whisperjav.utils.media_leftovers import (
    WHISPERJAV_TEMP_SUFFIXES,
    WHISPERJAV_WORK_DIRS,
    is_whisperjav_temp_file,
)


class TestFilesWhisperJavWrote:
    @pytest.mark.parametrize("name", [
        "SONE-853_extracted.wav",
        "SONE-853_enhanced.wav",
        "SONE-853_resampled.wav",
        "SONE-853_raw.srt",
        "SONE-853_stitched.srt",
        "SONE-853_master.json",
        "SONE-853_EXTRACTED.WAV",          # case-insensitive
    ])
    def test_is_deletable(self, name):
        assert is_whisperjav_temp_file(Path("/tmp/whisperjav") / name)

    @pytest.mark.parametrize("suffix", WHISPERJAV_TEMP_SUFFIXES)
    def test_every_listed_suffix_is_matched(self, suffix):
        assert is_whisperjav_temp_file(Path("/tmp/whisperjav/SONE-853" + suffix))


class TestUsersOwnFilesAreNeverDeleted:
    """The data-loss case: --temp-dir pointed at a folder of the user's own media."""

    @pytest.mark.parametrize("name", [
        "SONE-853.mp4",
        "SONE-853.mkv",
        "holiday.wav",
        "notes.txt",
        "my_subtitles.srt",
        "SONE-853.japanese.srt",           # a subtitle the user keeps
        "SONE-853.english.srt",            # a translation WhisperJAV produced as OUTPUT
        "extracted.wav",                   # no basename prefix, not our shape
        "master.json",
        "archive_master.json.bak",
    ])
    def test_is_not_deletable(self, name):
        assert not is_whisperjav_temp_file(Path("D:/MyVideos") / name), (
            "%s would have been deleted from the user's own folder" % name)


def test_a_finished_subtitle_is_not_mistaken_for_an_intermediate():
    """The file the user actually wants is the one that must survive."""
    assert not is_whisperjav_temp_file(Path("D:/MyVideos/SONE-853.whisperjav.srt"))


def test_cleanup_deletes_nothing_it_does_not_recognise():
    """Sort a realistic mixed folder and check which side each file lands on."""
    folder = [
        "SONE-853.mp4",                    # the user's video
        "SONE-853_extracted.wav",          # ours
        "SONE-853_master.json",            # ours
        "SONE-853.whisperjav.srt",         # the output the user came for
        "holiday-2019.mkv",                # unrelated
    ]
    deleted = [n for n in folder if is_whisperjav_temp_file(Path("D:/MyVideos") / n)]
    kept = [n for n in folder if not is_whisperjav_temp_file(Path("D:/MyVideos") / n)]

    assert deleted == ["SONE-853_extracted.wav", "SONE-853_master.json"]
    assert "SONE-853.mp4" in kept and "holiday-2019.mkv" in kept
    assert "SONE-853.whisperjav.srt" in kept


def test_main_no_longer_deletes_every_file_it_finds():
    """The blanket delete must not come back."""
    main_src = (Path(__file__).resolve().parents[1]
                / "whisperjav" / "main.py").read_text(encoding="utf-8")
    start = main_src.index("def cleanup_temp_directory")
    body = main_src[start:start + 3000]
    assert "is_whisperjav_temp_file(file)" in body, (
        "cleanup no longer checks whether it wrote the file")
    # the old shape: `if file.is_file(): file.unlink()` with no ownership test
    assert "if file.is_file():\n                    file.unlink()" not in body, (
        "the unconditional delete is back")


def test_work_dirs_are_shared_with_the_labeller():
    """Cleanup and labelling must not keep separate folder lists again."""
    assert "resampled_scenes" in WHISPERJAV_WORK_DIRS
    assert "crispasr_out" in WHISPERJAV_WORK_DIRS
