"""Call tracer: the threshold and padding values at the calls that USE them. A measuring instrument, not a change
to WhisperJAV; every wrapper calls the original unchanged and only writes what it saw.

Python loads this file when its folder is on PYTHONPATH. When WJ_TRACE_OUT names a file, it writes JSON lines, each
distinct record once per process:
- TEN:        the arguments of ten_vad.TenVad(...) (the native model that compares each frame with the threshold),
              and, in TenSpeechSegmenter._apply_padding, the pads it read and the largest shift it actually applied.
- FireRedVAD: the values inside the upstream VadPostprocessor built by FireRedVad.from_pretrained (speech threshold,
              smoothing, frame counts), and, in FireRedVadSpeechSegmenter._apply_padding, pads read and applied.
- WhisperSeg: in the decoder (_probs_to_segments / _probs_to_segments_offline), the onset threshold, the offset
              threshold it computes (neg_threshold, or max(threshold - 0.15, 0.01) when unset), the grow floor and
              the display-start threshold; in _pad_and_convert, the pads it read, the whole frames they become, and
              the largest shift it actually applied.
"""
import importlib.abc
import importlib.util
import json
import os
import sys

_OUT = os.environ.get("WJ_TRACE_OUT")
_SEEN = set()


def _write(rec):
    key = json.dumps(rec, sort_keys=True, default=str)
    if key in _SEEN:
        return
    _SEEN.add(key)
    rec = dict(rec, pid=os.getpid())
    with open(_OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _scalars(obj):
    return {k: v for k, v in vars(obj).items() if isinstance(v, (int, float, str, bool)) or v is None}


def _ms(x):
    return round(x * 1000) if x is not None else None


def _trace_ten_vad(mod):
    orig = mod.TenVad.__init__

    def __init__(self, hop_size=256, threshold=0.5, *a, **k):
        orig(self, hop_size, threshold, *a, **k)
        _write({"call": "ten_vad.TenVad.__init__", "hop_size": hop_size, "threshold": threshold})

    mod.TenVad.__init__ = __init__


def _trace_padding(cls, label):
    """_apply_padding(segments, duration) of TEN and FireRedVAD: dicts with start/end in, start/end/raw_* out."""
    orig = cls._apply_padding

    def _apply_padding(self, segments, audio_duration, *a, **k):
        out = orig(self, segments, audio_duration, *a, **k)
        shifts_s = [p["raw_start"] - p["start"] for p in out]
        shifts_e = [p["end"] - p["raw_end"] for p in out]
        _write({"call": f"{label}._apply_padding", "start_pad_ms_read": self.start_pad_ms,
                "end_pad_ms_read": self.end_pad_ms,
                "largest_start_shift_applied_ms": _ms(max(shifts_s)) if shifts_s else None,
                "largest_end_shift_applied_ms": _ms(max(shifts_e)) if shifts_e else None})
        return out

    cls._apply_padding = _apply_padding


def _trace_ten(mod):
    _trace_padding(mod.TenSpeechSegmenter, "TenSpeechSegmenter")


def _trace_firered_backend(mod):
    _trace_padding(mod.FireRedVadSpeechSegmenter, "FireRedVadSpeechSegmenter")


def _trace_fireredvad(mod):
    cls = mod.FireRedVad
    orig = cls.__dict__["from_pretrained"].__func__

    def from_pretrained(klass, model_dir, *a, **k):
        inst = orig(klass, model_dir, *a, **k)
        for name, val in vars(inst).items():
            if "postprocessor" in type(val).__name__.lower():
                _write({"call": f"fireredvad.FireRedVad.from_pretrained -> {type(val).__name__}",
                        "values": _scalars(val)})
        return inst

    cls.from_pretrained = classmethod(from_pretrained)


def _trace_whisperseg(mod):
    cls = mod.WhisperSegSpeechSegmenter
    for name in ("_probs_to_segments", "_probs_to_segments_offline"):
        orig = getattr(cls, name)

        def wrapped(self, speech_probs, audio_duration_sec, *a, _o=orig, _n=name, **k):
            neg = self.neg_threshold if self.neg_threshold is not None else max(float(self.threshold) - 0.15, 0.01)
            _write({"call": f"WhisperSegSpeechSegmenter.{_n}", "threshold": self.threshold,
                    "neg_threshold_set": self.neg_threshold, "offset_threshold_used": round(neg, 4),
                    "grow_floor": getattr(self, "grow_floor", None),
                    "speech_start_threshold": self.speech_start_threshold, "decoder": self.segmentation_decoder})
            return _o(self, speech_probs, audio_duration_sec, *a, **k)

        setattr(cls, name, wrapped)

    orig_pad = cls._pad_and_convert

    def _pad_and_convert(self, speeches, n_frames, audio_duration_sec, *a, **k):
        before = [(s["start"], s["end"]) for s in speeches]
        out = orig_pad(self, speeches, n_frames, audio_duration_sec, *a, **k)
        fm = float(self._frame_duration_ms)
        sh_s = [(b[0] - s["start"]) * fm for b, s in zip(before, speeches)]
        sh_e = [(s["end"] - b[1]) * fm for b, s in zip(before, speeches)]
        _write({"call": "WhisperSegSpeechSegmenter._pad_and_convert", "frame_ms": fm,
                "start_pad_ms_read": self.start_pad_ms, "end_pad_ms_read": self.end_pad_ms,
                "start_pad_as_frames_ms": int(self.start_pad_ms / fm) * fm,
                "end_pad_as_frames_ms": int(self.end_pad_ms / fm) * fm,
                "largest_start_shift_applied_ms": max(sh_s) if sh_s else None,
                "largest_end_shift_applied_ms": max(sh_e) if sh_e else None})
        return out

    cls._pad_and_convert = _pad_and_convert


def _trace_anime_whisper(mod):
    """anime-whisper lead-in silence: the value read and the silence actually put before the window (ms)."""
    cls = mod.AnimeWhisperGenerator
    orig = cls.with_leading_silence

    def with_leading_silence(self, audio, *a, **k):
        out = orig(self, audio, *a, **k)
        _write({"call": "AnimeWhisperGenerator.with_leading_silence",
                "leading_silence_ms_read": self._config.get("leading_silence_ms", 0),
                "silence_added_ms": round((len(out) - len(audio)) / 16)})
        return out

    cls.with_leading_silence = with_leading_silence


_TARGETS = {
    "whisperjav.modules.subtitle_pipeline.generators.anime_whisper": _trace_anime_whisper,
    "ten_vad": _trace_ten_vad,
    "fireredvad": _trace_fireredvad,
    "whisperjav.modules.speech_segmentation.backends.ten": _trace_ten,
    "whisperjav.modules.speech_segmentation.backends.firered_vad": _trace_firered_backend,
    "whisperjav.modules.speech_segmentation.backends.whisperseg": _trace_whisperseg,
}


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
        add = _TARGETS[fullname]

        def exec_module(module):
            orig_exec(module)
            add(module)
            _write({"call": "tracer_added", "module": fullname})

        spec.loader.exec_module = exec_module
        return spec


if _OUT:
    sys.meta_path.insert(0, _Finder())
