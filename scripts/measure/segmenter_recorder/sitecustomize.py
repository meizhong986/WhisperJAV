"""Speech segmenter recorder. A measuring instrument, not a change to WhisperJAV.

Python loads this file automatically when its folder is on PYTHONPATH (run_chronosjav_reference_runs.py --record
does that). When the environment variable WJ_RECORD_OUT names a file, it records, as JSON lines, the values the
ChronosJAV framing step produced, without changing any of them:
- "scenes": the scene list of each media file (path, start, end, duration);
- "segmenter": the speech segmenter's class and its settings, once per run;
- "frame_call": per scene, the speech segmenter's speech segments (with their metadata), its groups, and the frames
  the framer kept;
- "probs": WhisperSeg's per-frame speech probabilities, before they are turned into speech segments.

It works by wrapping three WhisperJAV functions when their modules are imported: each wrapper calls the original
function unchanged, then writes what it returned. A run with the recorder should produce byte-identical subtitles to
a run without it (check with --compare-with). Does nothing when WJ_RECORD_OUT is unset.
"""
import importlib.abc
import importlib.util
import json
import os
import sys

_OUT = os.environ.get("WJ_RECORD_OUT")


def _write(rec):
    rec["pid"] = os.getpid()
    with open(_OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _scalar_attrs(obj):
    return {k: v for k, v in vars(obj).items() if isinstance(v, (int, float, str, bool)) or v is None}


def _record_vad_grouped(mod):
    """VadGroupedFramer.frame/reframe: record the speech segmenter's result and the frames kept."""
    cls = mod.VadGroupedFramer

    def _wrap(name):
        orig = getattr(cls, name)

        def wrapped(self, audio, sample_rate, *a, **k):
            self._wj_record_last = None
            if name == "frame":
                # frame() calls _ensure_segmenter() first itself; it is idempotent, so calling it here changes nothing.
                self._ensure_segmenter()
                seg = self._segmenter
                if seg is not None and not getattr(seg, "_wj_record_wrapped", False):
                    orig_segment = seg.segment

                    def segment(*sa, _o=orig_segment, _self=self, **sk):
                        res = _o(*sa, **sk)
                        _self._wj_record_last = res
                        return res

                    seg.segment = segment
                    seg._wj_record_wrapped = True
                    _write({"kind": "segmenter", "cls": type(seg).__name__, "attrs": _scalar_attrs(seg),
                            "framer": _scalar_attrs(self)})
            fr = orig(self, audio, sample_rate, *a, **k)
            res = getattr(self, "_wj_record_last", None)
            segs, groups = [], None
            if res is not None:
                index = {}
                for i, s in enumerate(res.segments):
                    index[id(s)] = i
                    md = {kk: vv for kk, vv in (s.metadata or {}).items()
                          if isinstance(vv, (int, float, str, bool)) or vv is None}
                    segs.append({"s": s.start_sec, "e": s.end_sec, "c": s.confidence, "md": md})
                groups = [[index.get(id(s), {"s": s.start_sec, "e": s.end_sec}) for s in g] for g in res.groups]
            _write({"kind": "frame_call", "call": name,
                    "dur": len(audio) / sample_rate if sample_rate else None,
                    "segments": segs, "groups": groups,
                    "frames": [[f.start, f.end] for f in fr.frames],
                    "speech_starts": fr.metadata.get("speech_starts")})
            return fr

        setattr(cls, name, wrapped)

    _wrap("frame")
    if hasattr(cls, "reframe"):
        _wrap("reframe")


def _record_scene_list(mod):
    """Scene detection result: record the scene list each time it is converted for the pipeline."""
    for name in dir(mod):
        cls = getattr(mod, name)
        if isinstance(cls, type) and "to_legacy_tuples" in vars(cls):
            orig = cls.to_legacy_tuples

            def to_legacy_tuples(self, _orig=orig):
                t = _orig(self)
                _write({"kind": "scenes", "scenes": [[str(p), s, e, d] for p, s, e, d in t]})
                return t

            cls.to_legacy_tuples = to_legacy_tuples


def _record_whisperseg_probs(mod):
    """WhisperSeg: record the per-frame speech probabilities passed to either decoder."""
    cls = mod.WhisperSegSpeechSegmenter
    for name in ("_probs_to_segments", "_probs_to_segments_offline"):
        orig = getattr(cls, name)

        def wrapped(self, speech_probs, audio_duration_sec, *a, _o=orig, _n=name, **k):
            _write({"kind": "probs", "decoder": _n, "frame_ms": self._frame_duration_ms,
                    "probs": [round(float(x), 3) for x in speech_probs]})
            return _o(self, speech_probs, audio_duration_sec, *a, **k)

        setattr(cls, name, wrapped)


_TARGETS = {
    "whisperjav.modules.speech_segmentation.backends.whisperseg": _record_whisperseg_probs,
    "whisperjav.modules.subtitle_pipeline.framers.vad_grouped": _record_vad_grouped,
    "whisperjav.modules.scene_detection_backends.base": _record_scene_list,
}


class _Finder(importlib.abc.MetaPathFinder):
    """Adds the recording wrapper right after one of the target modules is imported."""

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
        add_recording = _TARGETS[fullname]

        def exec_module(module):
            orig_exec(module)
            add_recording(module)
            _write({"kind": "recording_added", "module": fullname})

        spec.loader.exec_module = exec_module
        return spec


if _OUT:
    sys.meta_path.insert(0, _Finder())
