#!/usr/bin/env python3
"""Labelling of WhisperJAV's own leftovers in the discovered-media listing (F4).

Discovery walks folders recursively, so a working folder can yield intermediates from an
earlier run. Those are LABELLED, never dropped from the input list: a user's own file may
carry any name and sit in any folder, and silently removing one would lose their input.

Imports `whisperjav.utils.media_leftovers`, which is standard library only. It used to
import `whisperjav.main`, which pulls torch, stable_whisper, transformers, faster_whisper
and whisper -- CLAUDE.md forbids importing the ASR stack merely to verify something, and
the old docstring's claim that it did not was simply wrong.
"""
import re
from pathlib import Path

import pytest

from whisperjav.utils.media_leftovers import (
    WHISPERJAV_AUDIO_SUFFIXES,
    WHISPERJAV_WORK_DIRS,
    looks_like_whisperjav_leftover as leftover,
)


class TestLeftoverShapesAreLabelled:
    @pytest.mark.parametrize("p", [
        "/tmp/whisperjav/SONE-853_extracted.wav",
        "/tmp/whisperjav/SONE-853_enhanced.wav",
        "/tmp/whisperjav/SONE-853_resampled.wav",
        "/tmp/whisperjav/SONE-853_EXTRACTED.WAV",          # case-insensitive
    ])
    def test_audio_suffixes(self, p):
        assert leftover(Path(p))

    @pytest.mark.parametrize("folder", WHISPERJAV_WORK_DIRS)
    def test_each_listed_folder_is_recognised(self, folder):
        assert leftover(Path("/tmp/whisperjav") / folder / "SONE-853_scene_0001.wav")

    def test_the_list_matches_the_folders_the_code_actually_creates(self):
        """The list must not drift from the pipelines again.

        The previous version of this test parametrised over WHISPERJAV_WORK_DIRS and
        asserted the helper recognised them -- but the helper is implemented by iterating
        that same tuple, so it could not fail. Both the labelling list and main.py's
        cleanup list had meanwhile drifted: each was missing resampled_scenes and
        crispasr_out, so those folders were neither labelled nor cleaned up.

        This reads the folder names out of the source that creates them instead.
        """
        root = Path(__file__).resolve().parents[1] / "whisperjav"
        created = set()
        for src in root.rglob("*.py"):
            if "test" in src.parts:
                continue
            text = src.read_text(encoding="utf-8", errors="replace")
            # temp_dir / "name"  — how every working folder is built
            created.update(re.findall(r'temp_dir\s*/\s*[\'"]([a-z_]+)[\'"]', text))
        missing = created - set(WHISPERJAV_WORK_DIRS)
        assert not missing, (
            "these folders are created under the temp dir but are neither labelled nor "
            "cleaned up: %s" % sorted(missing))


    def test_nested_under_a_working_folder(self):
        assert leftover(Path("/tmp/whisperjav/scenes/batch2/x.wav"))


def test_main_cleanup_uses_the_shared_folder_list():
    """main.py must not keep a second copy of the folder names."""
    main_src = (Path(__file__).resolve().parents[1]
                / "whisperjav" / "main.py").read_text(encoding="utf-8")
    assert "for subdir in WHISPERJAV_WORK_DIRS:" in main_src, (
        "cleanup_temp_directory has its own folder list again")
    assert 'subdirs_to_clean = [' not in main_src, (
        "the old hand-kept subdirs_to_clean list is back")


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


class TestKnownFalsePositives:
    """Documented, accepted, and harmless because the label never removes anything.

    A user folder genuinely called "scenes" IS flagged. These tests exist so the
    behaviour is recorded rather than discovered, and so that a future attempt to make
    the helper stricter has to change a test that says why.
    """

    @pytest.mark.parametrize("p", [
        "D:/Media/Scenes/clip.mp4",
        "D:/Footage/raw_subs/interview.wav",
    ])
    def test_user_folder_sharing_our_name_is_flagged(self, p):
        assert leftover(Path(p)), "documented false positive; label only, never removal"


def test_the_listing_code_labels_without_dropping_anything():
    """The guarantee that matters: labelling is not filtering.

    The previous version of this built a list, copied it, and asserted the copy equalled
    the original -- it never touched main.py, so it proved nothing about the listing. This
    reads the listing block out of main.py and checks its shape instead: every discovered
    file reaches a logger call, and nothing is filtered, `continue`d or removed.
    """
    main_src = (Path(__file__).resolve().parents[1]
                / "whisperjav" / "main.py").read_text(encoding="utf-8")
    start = main_src.index('logger.info(f"Found {len(media_files)} media file(s) to process:")')
    block = main_src[start:main_src.index("# One summary line after the listing", start)]

    # The loop walks every discovered file...
    assert "for f in media_files:" in block
    # ...and both arms of the label decision log the path; neither skips it.
    assert block.count("logger.info(f\"  - {f['path']}") == 2, (
        "the listing no longer logs every file on both branches")
    for forbidden in ("continue", "media_files.remove", "media_files =", "del media_files"):
        assert forbidden not in block, (
            "the listing block now contains %r -- it must never drop a discovered file"
            % forbidden)


def test_a_labelled_file_is_still_a_normal_path():
    """Labelling attaches no state to the file and cannot be mistaken for exclusion."""
    p = Path("/tmp/whisperjav/a_extracted.wav")
    assert leftover(p)
    assert p.name == "a_extracted.wav" and p.exists() is False  # unchanged, just described


def test_suffix_table_is_not_silently_emptied():
    assert WHISPERJAV_AUDIO_SUFFIXES, "an empty suffix table would label nothing"
    assert all(s.endswith(".wav") for s in WHISPERJAV_AUDIO_SUFFIXES)
