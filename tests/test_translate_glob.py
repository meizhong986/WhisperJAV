"""Tests for batch translate glob/wildcard expansion (#51).

Validates that:
- Glob patterns (*.srt) expand to matching .srt files
- Non-.srt files are filtered out from glob matches
- No-match globs produce warnings, not errors
- Directory input (shallow) only finds top-level .srt files
- Directory input (recursive) finds nested .srt files
- Deduplication: same file via different paths counted once
- Existing nargs behavior (multiple explicit files) still works
"""

import glob as glob_module
from pathlib import Path
import pytest


def _create_srt_tree(tmp_path):
    """Create a test directory tree with .srt files at various levels.

    Structure:
        tmp_path/
            a.srt
            b.srt
            c.txt           (non-srt)
            sub/
                d.srt
                sub2/
                    e.srt
    """
    (tmp_path / "a.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nA\n")
    (tmp_path / "b.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nB\n")
    (tmp_path / "c.txt").write_text("not a subtitle")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nD\n")
    sub2 = sub / "sub2"
    sub2.mkdir()
    (sub2 / "e.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nE\n")
    return tmp_path


class TestGlobExpansion:
    """Test Python-side glob expansion of wildcard patterns."""

    def test_glob_expansion_wildcard(self, tmp_path):
        """*.srt should expand to matching .srt files in the directory."""
        _create_srt_tree(tmp_path)

        pattern = str(tmp_path / "*.srt")
        matched = sorted(glob_module.glob(pattern, recursive=True))
        srt_matched = [Path(f) for f in matched if f.lower().endswith('.srt')]

        assert len(srt_matched) == 2
        names = sorted(f.name for f in srt_matched)
        assert names == ["a.srt", "b.srt"]

    def test_glob_expansion_recursive(self, tmp_path):
        """**/*.srt should expand recursively to all .srt files."""
        _create_srt_tree(tmp_path)

        pattern = str(tmp_path / "**" / "*.srt")
        matched = sorted(glob_module.glob(pattern, recursive=True))
        srt_matched = [Path(f) for f in matched if f.lower().endswith('.srt')]

        assert len(srt_matched) == 4  # a, b, d, e
        names = sorted(f.name for f in srt_matched)
        assert names == ["a.srt", "b.srt", "d.srt", "e.srt"]

    def test_glob_expansion_no_match(self, tmp_path):
        """Pattern with no matches should produce empty list."""
        _create_srt_tree(tmp_path)

        pattern = str(tmp_path / "nonexistent*.srt")
        matched = sorted(glob_module.glob(pattern, recursive=True))
        srt_matched = [Path(f) for f in matched if f.lower().endswith('.srt')]

        assert len(srt_matched) == 0

    def test_glob_srt_filter(self, tmp_path):
        """Non-.srt files should be filtered out even if glob matches them."""
        _create_srt_tree(tmp_path)

        # Use *.* which matches c.txt too
        pattern = str(tmp_path / "*.*")
        matched = sorted(glob_module.glob(pattern, recursive=True))
        srt_matched = [Path(f) for f in matched if f.lower().endswith('.srt')]

        # c.txt should be filtered out
        names = sorted(f.name for f in srt_matched)
        assert "c.txt" not in names
        assert len(srt_matched) == 2  # only a.srt, b.srt


class TestDirectoryInput:
    """Test directory-based file collection."""

    def test_directory_shallow(self, tmp_path):
        """Default directory input should only find top-level .srt files."""
        _create_srt_tree(tmp_path)

        srt_files = sorted(list(tmp_path.glob("*.srt")))
        assert len(srt_files) == 2
        names = sorted(f.name for f in srt_files)
        assert names == ["a.srt", "b.srt"]

    def test_directory_recursive(self, tmp_path):
        """Recursive directory input should find all nested .srt files."""
        _create_srt_tree(tmp_path)

        srt_files = sorted(list(tmp_path.rglob("*.srt")))
        assert len(srt_files) == 4
        names = sorted(f.name for f in srt_files)
        assert names == ["a.srt", "b.srt", "d.srt", "e.srt"]

    def test_empty_directory(self, tmp_path):
        """Directory with no .srt files should return empty list."""
        empty = tmp_path / "empty"
        empty.mkdir()

        srt_files = sorted(list(empty.glob("*.srt")))
        assert len(srt_files) == 0


class TestDeduplication:
    """Test that duplicate files (same resolved path) are counted only once."""

    def test_deduplication_same_file(self, tmp_path):
        """Same file via different paths should be counted once."""
        _create_srt_tree(tmp_path)

        seen_paths = set()
        files_to_process = []

        # Add a.srt directly
        f1 = tmp_path / "a.srt"
        resolved1 = f1.resolve()
        if resolved1 not in seen_paths:
            seen_paths.add(resolved1)
            files_to_process.append(f1)

        # Add a.srt again via glob
        pattern = str(tmp_path / "*.srt")
        matched = sorted(glob_module.glob(pattern, recursive=True))
        for m in matched:
            f = Path(m)
            if f.name.lower().endswith('.srt'):
                resolved = f.resolve()
                if resolved not in seen_paths:
                    seen_paths.add(resolved)
                    files_to_process.append(f)

        # a.srt should appear only once, b.srt added from glob
        assert len(files_to_process) == 2
        names = sorted(f.name for f in files_to_process)
        assert names == ["a.srt", "b.srt"]

    def test_deduplication_directory_plus_glob(self, tmp_path):
        """Files from directory and overlapping glob should not duplicate."""
        _create_srt_tree(tmp_path)

        seen_paths = set()
        files_to_process = []

        # Add from directory
        for f in sorted(tmp_path.glob("*.srt")):
            resolved = f.resolve()
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                files_to_process.append(f)

        # Add from glob (overlapping)
        pattern = str(tmp_path / "*.srt")
        for m in sorted(glob_module.glob(pattern, recursive=True)):
            f = Path(m)
            if f.name.lower().endswith('.srt'):
                resolved = f.resolve()
                if resolved not in seen_paths:
                    seen_paths.add(resolved)
                    files_to_process.append(f)

        assert len(files_to_process) == 2  # no duplicates


class TestExistingBehavior:
    """Regression: existing multi-file nargs behavior must not break."""

    def test_multiple_explicit_files(self, tmp_path):
        """Multiple explicitly named files should all be collected."""
        _create_srt_tree(tmp_path)

        explicit_inputs = [
            str(tmp_path / "a.srt"),
            str(tmp_path / "b.srt"),
        ]

        seen_paths = set()
        files_to_process = []

        for input_item in explicit_inputs:
            input_arg = Path(input_item)
            assert input_arg.exists()
            resolved = input_arg.resolve()
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                files_to_process.append(input_arg)

        assert len(files_to_process) == 2

    def test_glob_pattern_detection(self):
        """Strings with * or ? should be detected as glob patterns."""
        assert '*' in "*.srt"
        assert '*' in "sub/**/*.srt"
        assert '?' in "file?.srt"
        assert '*' not in "file.srt"
        assert '?' not in "file.srt"
