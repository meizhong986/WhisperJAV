"""Experiment hooks for the 1.9.4 character-accuracy step 2 (2026-10-05). A measuring device, not a product change:
nothing in whisperjav/ is edited; the changes exist only in a process started with this folder on PYTHONPATH.

It first loads the segmenter recorder (scripts/measure/segmenter_recorder/sitecustomize.py), so runs are recorded as
usual, then applies, from environment variables:
  WJ_EXP_LEADIN_MS   silence (zeros) prepended to every audio window handed to the ASR model
  WJ_EXP_TAIL_MS     silence appended to every window
  WJ_EXP_DIP_FROM    WhisperSeg hysteresis decoder, forced split: start the dip search at this fraction of the
                     longest segment (the product uses 0.6, i.e. only the last 40 %)
  WJ_EXP_DIP_ACCEPT_ALL=1  accept the lowest point found even if it is not below 0.85 x the window's mean (the
                     product then cuts exactly at the limit)
  WJ_EXP_REACHBACK_MS  each vad-grouped window starts this much earlier for the ASR model (it may reach into the
                     previous window; clipped at the scene start). The original start is recorded as the window's
                     "speech start", which the orchestrator uses as the displayed start (vad_only mode), so the
                     subtitle keeps its time. A window that already has a speech start keeps it.
Silence is added to in-memory windows only (the pipeline's default "pathless" mode). The default timestamp mode
takes subtitle times from the frames, not from the model, so added silence does not move times.
"""
import importlib.abc
import importlib.util
import inspect
import json
import os
import runpy
import sys
import textwrap
from pathlib import Path

_REC = Path(__file__).resolve().parents[5] / "scripts" / "measure" / "segmenter_recorder" / "sitecustomize.py"
if _REC.exists():
    runpy.run_path(str(_REC))

_OUT = os.environ.get("WJ_RECORD_OUT")
LEADIN = int(os.environ.get("WJ_EXP_LEADIN_MS", "0") or 0)
TAIL = int(os.environ.get("WJ_EXP_TAIL_MS", "0") or 0)
DIP_FROM = os.environ.get("WJ_EXP_DIP_FROM")
DIP_ALL = os.environ.get("WJ_EXP_DIP_ACCEPT_ALL") == "1"
REACHBACK = int(os.environ.get("WJ_EXP_REACHBACK_MS", "0") or 0)
SR = 16000


def _note(rec):
    if _OUT:
        with open(_OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _pad(audio):
    import numpy as np
    if not isinstance(audio, np.ndarray) or (LEADIN == 0 and TAIL == 0):
        return audio
    parts = []
    if LEADIN:
        parts.append(np.zeros(int(SR * LEADIN / 1000), dtype=audio.dtype))
    parts.append(audio)
    if TAIL:
        parts.append(np.zeros(int(SR * TAIL / 1000), dtype=audio.dtype))
    return np.concatenate(parts)


def _hook_anime(mod):
    cls = mod.AnimeWhisperGenerator
    orig = cls.generate

    def generate(self, audio_path, *a, **k):
        return orig(self, _pad(audio_path), *a, **k)

    cls.generate = generate          # generate_batch calls generate once per window
    _note({"kind": "experiment", "hook": "anime generate", "leadin_ms": LEADIN, "tail_ms": TAIL})


def _hook_qwen3(mod):
    cls = mod.Qwen3TextGenerator
    orig = cls.generate_batch

    def generate_batch(self, audio_paths, *a, **k):
        return orig(self, [_pad(x) for x in audio_paths], *a, **k)

    cls.generate_batch = generate_batch    # generate delegates to generate_batch
    _note({"kind": "experiment", "hook": "qwen3 generate_batch", "leadin_ms": LEADIN, "tail_ms": TAIL})


def _hook_split(mod):
    cls = mod.WhisperSegSpeechSegmenter
    if "dip_search_from" in inspect.signature(cls.__init__).parameters:
        # Since 2026-10-06 the product has these as settings (dip_search_from / dip_accept_any): set them.
        orig_init = cls.__init__

        def __init__(self, *a, **k):
            if DIP_FROM:
                k["dip_search_from"] = float(DIP_FROM)
            if DIP_ALL:
                k["dip_accept_any"] = True
            orig_init(self, *a, **k)

        cls.__init__ = __init__
        _note({"kind": "experiment", "hook": "whisperseg dip split (settings)", "dip_from": DIP_FROM,
               "accept_all": DIP_ALL})
        return
    src = inspect.getsource(mod)
    a = src.index("    def _probs_to_segments(")
    b = src.index("    def _pad_and_convert(")
    method = textwrap.dedent(src[a:b])
    old_win = "window_start = int(max_speech_frames * 0.6)"
    old_acc = "if min_prob < mean_prob * 0.85:"
    assert method.count(old_win) == 1 and method.count(old_acc) == 1, "WhisperSeg split code changed"
    if DIP_FROM:
        method = method.replace(old_win, f"window_start = int(max_speech_frames * {float(DIP_FROM)})")
    if DIP_ALL:
        method = method.replace(old_acc, "if True:")
    ns = {}
    exec(compile(method, mod.__file__ + " [experiment]", "exec"), mod.__dict__, ns)
    setattr(cls, "_probs_to_segments", ns["_probs_to_segments"])
    _note({"kind": "experiment", "hook": "whisperseg dip split", "dip_from": DIP_FROM, "accept_all": DIP_ALL})


def _hook_reachback(mod):
    cls = mod.VadGroupedFramer
    orig = cls.frame
    r = REACHBACK / 1000.0

    def frame(self, *a, **k):
        res = orig(self, *a, **k)
        frames = list(res.frames)
        starts = list(res.metadata.get("speech_starts") or [None] * len(frames))
        moved = 0
        for i, f in enumerate(frames):
            new_start = max(0.0, f.start - r)
            if new_start < f.start:
                if i < len(starts) and starts[i] is None:
                    starts[i] = f.start
                f.start = new_start
                moved += 1
        res.metadata["speech_starts"] = starts
        _note({"kind": "experiment", "hook": "reachback", "ms": REACHBACK, "frames": len(frames), "moved": moved})
        return res

    cls.frame = frame


_TARGETS = {}
if REACHBACK:
    _TARGETS["whisperjav.modules.subtitle_pipeline.framers.vad_grouped"] = _hook_reachback
if LEADIN or TAIL:
    _TARGETS["whisperjav.modules.subtitle_pipeline.generators.anime_whisper"] = _hook_anime
    _TARGETS["whisperjav.modules.subtitle_pipeline.generators.qwen3"] = _hook_qwen3
if DIP_FROM or DIP_ALL:
    _TARGETS["whisperjav.modules.speech_segmentation.backends.whisperseg"] = _hook_split


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module
        hook = _TARGETS[fullname]

        def exec_module(module):
            orig_exec(module)
            hook(module)

        spec.loader.exec_module = exec_module
        return spec


if _TARGETS:
    sys.meta_path.insert(0, _Finder())
