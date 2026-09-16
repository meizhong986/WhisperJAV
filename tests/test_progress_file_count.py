#!/usr/bin/env python3
"""The multi-file progress counter (issue #429).

`[839/1]` came from ProgressDisplayAdapter hard-coding total_files = 1 while the real
count was only ever set on the UnifiedProgressManager. These tests pin the count that
reaches UnifiedProgressManager.start_file_processing, which is what prints the line.

Imports only whisperjav.utils.progress_adapter -- no ASR module is touched.
"""
import pytest

from whisperjav.utils.progress_adapter import ProgressDisplayAdapter, create_progress_adapter


class RecordingManager:
    """Stands in for UnifiedProgressManager and records the counts it is handed."""

    def __init__(self):
        self.calls = []
        self.total_files = None

    def start_file_processing(self, filename, file_num, total_files):
        self.calls.append((filename, file_num, total_files))
        return "ctx-%d" % len(self.calls)


def test_count_from_constructor_reaches_the_printed_line():
    mgr = RecordingManager()
    adapter = ProgressDisplayAdapter(mgr, 2)

    adapter.set_current_file("a.mp4", 1)
    adapter.set_current_file("b.mp4", 2)

    assert mgr.calls == [("a.mp4", 1, 2), ("b.mp4", 2, 2)]


def test_falls_back_to_the_managers_count_when_not_passed():
    mgr = RecordingManager()
    mgr.total_files = 839
    adapter = ProgressDisplayAdapter(mgr)

    adapter.set_current_file("x.mp4", 839)

    assert mgr.calls == [("x.mp4", 839, 839)]


def test_single_file_run_still_reads_one_of_one():
    mgr = RecordingManager()
    adapter = ProgressDisplayAdapter(mgr, 1)

    adapter.set_current_file("only.mp4", 1)

    assert mgr.calls == [("only.mp4", 1, 1)]


def test_no_count_anywhere_defaults_to_one():
    mgr = RecordingManager()          # total_files stays None
    adapter = ProgressDisplayAdapter(mgr)

    adapter.set_current_file("only.mp4", 1)

    assert adapter.total_files == 1
    assert mgr.calls == [("only.mp4", 1, 1)]


def test_factory_passes_the_count_through():
    mgr = RecordingManager()
    adapter = create_progress_adapter(mgr, 3)

    adapter.set_current_file("c.mp4", 3)

    assert mgr.calls == [("c.mp4", 3, 3)]


def test_regression_the_839_of_1_shape_cannot_come_back():
    """The reported symptom: file 839 of a 900-file run printing as [839/1]."""
    mgr = RecordingManager()
    adapter = ProgressDisplayAdapter(mgr, 900)

    adapter.set_current_file("f839.mp4", 839)

    _, file_num, total = mgr.calls[0]
    assert (file_num, total) == (839, 900)
    assert total != 1
