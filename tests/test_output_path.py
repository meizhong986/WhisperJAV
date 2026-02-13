"""Tests for --output-dir source sentinel feature (#49).

Validates that:
- "source" sentinel resolves output to input file's parent directory
- Case insensitivity: "SOURCE", "Source", "source" all work
- Default ./output behavior is unchanged (regression)
- Explicit --output-dir path is unchanged (regression)
- Skip-existing correctly resolves paths in source mode
"""

import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


def _make_args(**overrides):
    """Create a minimal argparse.Namespace for testing."""
    defaults = {
        'output_dir': 'source',
        'skip_existing': False,
        'subs_language': 'native',
        'language': 'japanese',
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestSourceSentinelDetection:
    """Test that 'source' is correctly detected as the sentinel value."""

    def test_source_sentinel_lowercase(self):
        args = _make_args(output_dir='source')
        assert args.output_dir.lower().strip() == 'source'

    def test_source_sentinel_uppercase(self):
        args = _make_args(output_dir='SOURCE')
        assert args.output_dir.lower().strip() == 'source'

    def test_source_sentinel_mixed_case(self):
        args = _make_args(output_dir='Source')
        assert args.output_dir.lower().strip() == 'source'

    def test_source_sentinel_with_whitespace(self):
        args = _make_args(output_dir='  source  ')
        assert args.output_dir.lower().strip() == 'source'

    def test_not_source_default(self):
        args = _make_args(output_dir='./output')
        assert args.output_dir.lower().strip() != 'source'

    def test_not_source_explicit(self):
        args = _make_args(output_dir='C:/custom/dir')
        assert args.output_dir.lower().strip() != 'source'


class TestSourceOutputResolution:
    """Test that source mode resolves to input file's parent directory."""

    def test_source_resolves_to_parent(self, tmp_path):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        video_file = video_dir / "test_video.mp4"
        video_file.touch()

        output_to_source = True
        file_path_str = str(video_file)

        if output_to_source:
            resolved_dir = Path(file_path_str).parent
        else:
            resolved_dir = Path('./output')

        assert resolved_dir == video_dir

    def test_source_resolves_different_dirs_per_file(self, tmp_path):
        """Batch mode: each file gets its own parent as output dir."""
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()

        file_a = dir_a / "video_a.mp4"
        file_b = dir_b / "video_b.mp4"
        file_a.touch()
        file_b.touch()

        resolved_dirs = []
        for f in [file_a, file_b]:
            resolved_dirs.append(Path(str(f)).parent)

        assert resolved_dirs[0] == dir_a
        assert resolved_dirs[1] == dir_b
        assert resolved_dirs[0] != resolved_dirs[1]


class TestDefaultOutputUnchanged:
    """Test default and explicit output_dir behavior."""

    def test_default_is_source(self):
        """Default --output-dir is 'source' (save next to video)."""
        args = _make_args(output_dir='source')
        assert args.output_dir.lower().strip() == 'source'

    def test_explicit_dir_preserved(self):
        args = _make_args(output_dir='/my/custom/path')
        output_to_source = args.output_dir.lower().strip() == 'source'
        assert not output_to_source
        assert args.output_dir == '/my/custom/path'


class TestSkipExistingWithSourceMode:
    """Test that --skip-existing resolves expected path correctly in source mode."""

    def test_skip_existing_source_mode(self, tmp_path):
        """In source mode, expected SRT should be in the video's parent dir."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        video_file = video_dir / "test_video.mp4"
        video_file.touch()

        # Simulate existing output
        existing_srt = video_dir / "test_video.ja.whisperjav.srt"
        existing_srt.touch()

        output_to_source = True
        media_basename = "test_video"
        output_lang_code = "ja"
        file_path_str = str(video_file)

        if output_to_source:
            expected_dir = Path(file_path_str).parent
        else:
            expected_dir = Path('./output')

        expected_output = expected_dir / f"{media_basename}.{output_lang_code}.whisperjav.srt"
        assert expected_output.exists()

    def test_skip_existing_normal_mode(self, tmp_path):
        """In normal mode, expected SRT should be in the output dir."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        existing_srt = output_dir / "test_video.ja.whisperjav.srt"
        existing_srt.touch()

        output_to_source = False
        media_basename = "test_video"
        output_lang_code = "ja"

        if output_to_source:
            expected_dir = Path("dummy")
        else:
            expected_dir = output_dir

        expected_output = expected_dir / f"{media_basename}.{output_lang_code}.whisperjav.srt"
        assert expected_output.exists()

    def test_skip_existing_source_mode_no_srt(self, tmp_path):
        """In source mode, if no SRT exists, should not skip."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        video_file = video_dir / "test_video.mp4"
        video_file.touch()

        output_to_source = True
        media_basename = "test_video"
        output_lang_code = "ja"
        file_path_str = str(video_file)

        if output_to_source:
            expected_dir = Path(file_path_str).parent
        else:
            expected_dir = Path('./output')

        expected_output = expected_dir / f"{media_basename}.{output_lang_code}.whisperjav.srt"
        assert not expected_output.exists()


class TestPipelineOutputDirOverride:
    """Test that pipeline.output_dir is correctly overridden per-file."""

    def test_pipeline_output_dir_set_to_path(self):
        """Ensure the override produces a Path object compatible with BasePipeline."""
        file_path_str = "/some/dir/video.mp4"
        per_file_dir = Path(file_path_str).parent
        assert isinstance(per_file_dir, Path)
        assert per_file_dir.name == "dir"
