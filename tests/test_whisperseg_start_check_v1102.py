"""WhisperSeg without transformers stops at the start-of-run check (1.10.2, owner W622 item 9).

Before: the check looked for onnxruntime only, so on a minimal install (no transformers) the run went on and
every scene failed, exit 0, no subtitles. Now WhisperSeg needs onnxruntime, transformers and huggingface_hub, and
the check stops the run before any audio is read, as it does for every other speech segmenter whose package is
missing. Nothing is installed or removed: importlib.util.find_spec is patched for one package at a time.
"""
import importlib.util

import pytest

from whisperjav.modules.speech_segmentation.factory import SpeechSegmenterFactory
from whisperjav.utils import preflight_check

_real_find_spec = importlib.util.find_spec


def _hide(monkeypatch, missing):
    def fake(name, *args, **kwargs):
        if name == missing:
            return None
        return _real_find_spec(name, *args, **kwargs)
    monkeypatch.setattr(importlib.util, "find_spec", fake)


@pytest.mark.parametrize("missing", ["onnxruntime", "transformers", "huggingface_hub"])
def test_whisperseg_unavailable_without_each_package(monkeypatch, missing):
    _hide(monkeypatch, missing)
    available, hint = SpeechSegmenterFactory.is_backend_available("whisperseg")
    assert available is False
    assert "whisperseg" in hint


def test_start_check_stops_whisperseg_without_transformers(monkeypatch, capsys):
    _hide(monkeypatch, "transformers")
    with pytest.raises(SystemExit) as stop:
        preflight_check.ensure_segmenter_backend_available("whisperseg")
    assert stop.value.code == 1
    out = capsys.readouterr().out
    assert "The whisperseg speech detector cannot run on this machine" in out
    assert "Nothing has been transcribed." in out


@pytest.mark.parametrize("backend", ["ten", "firered-vad", "silero", "none"])
def test_other_segmenters_unaffected(monkeypatch, backend):
    # These do not use transformers: hiding it changes nothing for them.
    before = SpeechSegmenterFactory.is_backend_available(backend)
    _hide(monkeypatch, "transformers")
    assert SpeechSegmenterFactory.is_backend_available(backend) == before
