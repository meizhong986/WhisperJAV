#!/usr/bin/env python3
"""Async processing: a task cancelled before it starts ends CANCELLED, the next one still runs, and the run summary
can be counted from the task objects (the old AttributeError crash point).

Rewritten in 1.10.2 (step 5): the earlier script passed max_workers to AsyncPipelineManager, which has not taken it
since v1.3 (one worker, fixed), and it would have run the real faster pipeline (a model load) with ./test_output in
the repository and the system temp folder. Here a stand-in pipeline holds the one worker until the test has
cancelled the first task; nothing is loaded and nothing is written.
"""

import threading
import time

from whisperjav.utils.async_processor import AsyncPipelineManager, ProcessingStatus
from whisperjav.utils.progress_aggregator import VerbosityLevel


class _StandInPipeline:
    """Built by the worker for each task, as a real pipeline is; the first build waits for the test."""

    release = threading.Event()

    def __init__(self, **kwargs):
        _StandInPipeline.release.wait(10)

    def process(self, media_info):
        return {"summary": {"final_subtitles_refined": 1}}

    def cleanup(self):
        pass


def _final(processor, task_id, deadline):
    """The task once its completion callback has set the status (it runs just after the result is set)."""
    task = processor.wait_for_task(task_id, timeout=10)
    while task.status == ProcessingStatus.RUNNING and time.time() < deadline:
        time.sleep(0.01)
        task = processor.get_task_status(task_id)
    return task


def test_async_cancellation():
    messages = []
    manager = AsyncPipelineManager(ui_update_callback=messages.append, verbosity=VerbosityLevel.SUMMARY)
    processor = manager.processor
    _StandInPipeline.release.clear()
    try:
        files = [{"path": f"test_video_{n}.mp4", "basename": f"test_video_{n}", "file_number": n, "total_files": 2}
                 for n in (1, 2)]
        ids = [processor.submit_task(info, _StandInPipeline, {}) for info in files]
        assert len(set(ids)) == 2

        assert processor.cancel_task(ids[0]) is True    # before its pipeline has been built
        _StandInPipeline.release.set()

        deadline = time.time() + 10
        tasks = [_final(processor, task_id, deadline) for task_id in ids]
        assert [t.status for t in tasks] == [ProcessingStatus.CANCELLED, ProcessingStatus.COMPLETED]

        # The run summary's count from the task objects (where the AttributeError was).
        assert sum(t.status == ProcessingStatus.COMPLETED for t in tasks) == 1
        assert sum(t.status == ProcessingStatus.FAILED for t in tasks) == 0
        assert sum(t.status == ProcessingStatus.CANCELLED for t in tasks) == 1
        # A task that has ended cannot be cancelled again.
        assert processor.cancel_task(ids[0]) is False
    finally:
        _StandInPipeline.release.set()
        manager.shutdown()
