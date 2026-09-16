#!/usr/bin/env python3
"""The working folder must not be the same as the video folder or the output folder.

The working folder is emptied when a run ends. Sharing it with the user's videos, or with
the folder their subtitles are written to, puts their files in the path of that cleanup.
Refusing the overlap outright is sturdier than trying to delete carefully.

Imports `whisperjav.utils.media_leftovers` only -- no ASR module, nothing from main.py.
"""
import os
from pathlib import Path

import pytest

from whisperjav.utils.media_leftovers import temp_dir_conflicts


@pytest.fixture
def media_folder(tmp_path):
    folder = tmp_path / "MyVideos"
    folder.mkdir()
    (folder / "SONE-853.mp4").write_text("x", encoding="utf-8")
    return folder


class TestRejected:
    def test_working_folder_is_the_video_folder(self, media_folder):
        problems = temp_dir_conflicts(media_folder, [media_folder / "SONE-853.mp4"])
        assert problems, "sharing the video folder must be refused"
        assert "same as the folder holding your videos" in problems[0]

    def test_working_folder_is_the_video_folder_named_differently(self, media_folder):
        """`..` and casing must not get round the rule."""
        awkward = media_folder.parent / "MYVIDEOS" if os.name == "nt" else \
            media_folder.parent / ".." / media_folder.parent.name / "MyVideos"
        assert temp_dir_conflicts(awkward, [media_folder / "SONE-853.mp4"])

    def test_working_folder_is_the_output_folder(self, tmp_path):
        out = tmp_path / "subs"
        out.mkdir()
        problems = temp_dir_conflicts(out, [], output_dir=str(out))
        assert problems
        assert "same as the output folder" in problems[0]

    def test_working_folder_contains_the_videos(self, tmp_path, media_folder):
        """`--temp-dir D:\\` with videos in D:\\MyVideos is the same hazard."""
        problems = temp_dir_conflicts(tmp_path, [media_folder / "SONE-853.mp4"])
        assert problems
        assert "contains your videos" in problems[0]

    def test_the_reported_data_loss_command(self, media_folder):
        """whisperjav --temp-dir D:\\MyVideos D:\\MyVideos is now refused outright."""
        assert temp_dir_conflicts(
            media_folder,
            [media_folder / "SONE-853.mp4"],
            output_dir=str(media_folder),
        )


class TestAccepted:
    def test_a_separate_folder(self, tmp_path, media_folder):
        work = tmp_path / "work"
        work.mkdir()
        assert temp_dir_conflicts(work, [media_folder / "SONE-853.mp4"]) == []

    def test_a_subfolder_of_the_video_folder(self, media_folder):
        """Allowed: cleanup only touches the working folder's own contents."""
        work = media_folder / "whisperjav-temp"
        work.mkdir()
        assert temp_dir_conflicts(work, [media_folder / "SONE-853.mp4"]) == []

    def test_output_set_to_source(self, tmp_path, media_folder):
        """'source' means beside each video, not a folder name -- must not false-positive."""
        work = tmp_path / "work"
        work.mkdir()
        assert temp_dir_conflicts(work, [media_folder / "SONE-853.mp4"],
                                  output_dir="source") == []

    def test_no_media_and_no_output(self, tmp_path):
        assert temp_dir_conflicts(tmp_path / "work") == []


def test_each_offending_folder_is_reported_once(tmp_path):
    """Many videos in one folder must not produce one complaint per file."""
    folder = tmp_path / "MyVideos"
    folder.mkdir()
    files = []
    for i in range(5):
        f = folder / ("clip%d.mp4" % i)
        f.write_text("x", encoding="utf-8")
        files.append(f)
    assert len(temp_dir_conflicts(folder, files)) == 1


def test_a_nonexistent_folder_does_not_raise(tmp_path):
    assert isinstance(temp_dir_conflicts(tmp_path / "nope", [tmp_path / "a.mp4"]), list)


def test_main_refuses_the_run():
    """The check is wired in, and it stops the run rather than warning."""
    main_src = (Path(__file__).resolve().parents[1]
                / "whisperjav" / "main.py").read_text(encoding="utf-8")
    assert "temp_dir_conflicts(" in main_src, "the check is not called"
    start = main_src.index("_temp_problems = temp_dir_conflicts(")
    block = main_src[start:start + 900]
    assert "sys.exit(1)" in block, "a conflict must stop the run, not just warn"
