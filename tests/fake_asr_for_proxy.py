"""A stand-in recogniser for RemoteFasterWhisperASR tests (loaded in the worker child).

Mirrors the FasterWhisperProASR surface the proxy relies on, with no ML imports.
Behaviour hooks come from the audio path so tests need no shared state:
    - a path containing "CRASH"  -> the worker exits natively (os._exit(3))
    - a path containing "RAISE"  -> transcribe raises (an ordinary per-scene error)
"""
from __future__ import annotations

import os
from pathlib import Path


class FakeASR:
    def __init__(self, model_config, params, task, tracer=None):
        self.model_name = model_config.get("model_name", "fake")
        # Pretend the int8 VRAM fallback fired, so the proxy can learn it.
        self.compute_type = "int8" if model_config.get("compute_type") == "float16" else model_config.get("compute_type", "auto")
        self.task = task
        self._filter = {"logprob_filtered": 0, "nonverbal_filtered": 0}
        self._vad = []

    def get_segmenter_name(self):
        return "fake-seg"

    def reset_statistics(self):
        self._filter = {"logprob_filtered": 0, "nonverbal_filtered": 0}

    def get_filter_statistics(self):
        return dict(self._filter)

    def get_last_vad_segments(self):
        return list(self._vad)

    def get_last_decode_stats(self):
        return [{"temperature": 0.0, "avg_logprob": -0.3, "pid": os.getpid()}]

    def transcribe_to_srt(self, audio_path, output_srt_path, **kwargs):
        s = str(audio_path)
        if "CRASH" in s:
            os._exit(3)
        if "RAISE" in s:
            raise ValueError("synthetic scene failure")
        self._filter["logprob_filtered"] += 1
        self._vad = [{"start_sec": 0.0, "end_sec": 1.0}]
        out = Path(output_srt_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"1\n00:00:00,000 --> 00:00:01,000\npid={os.getpid()} task={kwargs.get('task')}\n", encoding="utf-8")
        return out
