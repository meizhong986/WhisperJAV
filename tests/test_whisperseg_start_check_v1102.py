"""WhisperSeg without transformers stops at the start-of-run check (1.10.2, owner W622 item 9).

Before: the check looked for onnxruntime only, so on a minimal install (no transformers) the run went on and
every scene failed, exit 0, no subtitles. Now WhisperSeg needs onnxruntime, transformers and huggingface_hub, and
the check stops the run before any audio is read, as it does for every other speech segmenter whose package is
missing. Nothing is installed or removed: importlib.util.find_spec is patched to hide packages.

The install hint (owner W624 D3, W625 2) names only the missing packages, with WhisperJAV's own version limits: the
hint of 1.10.1, "pip install whisperjav[whisperseg]", could fetch WhisperJAV itself again from PyPI, with a PyTorch
that runs on the CPU only on Windows, and would put onnxruntime over a user's onnxruntime-gpu.
"""
import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

from whisperjav.modules.speech_segmentation.factory import SpeechSegmenterFactory
from whisperjav.utils import preflight_check

_real_find_spec = importlib.util.find_spec
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
TRANSFORMERS = '"transformers>=5.20,<6"'
HUB = '"huggingface-hub>=1.0,<2"'


def _hide(monkeypatch, *missing):
    def fake(name, *args, **kwargs):
        if name in missing:
            return None
        return _real_find_spec(name, *args, **kwargs)
    monkeypatch.setattr(importlib.util, "find_spec", fake)


@pytest.mark.parametrize("missing, hint", [
    (("onnxruntime",), "pip install onnxruntime"),
    (("transformers",), f"pip install {TRANSFORMERS}"),
    (("huggingface_hub",), f"pip install {HUB}"),
    (("transformers", "huggingface_hub"), f"pip install {TRANSFORMERS} {HUB}"),   # a minimal install
], ids=["onnxruntime", "transformers", "huggingface_hub", "minimal install"])
def test_whisperseg_unavailable_and_the_hint_names_only_what_is_missing(monkeypatch, missing, hint):
    _hide(monkeypatch, *missing)
    assert SpeechSegmenterFactory.is_backend_available("whisperseg") == (False, hint)


def test_the_hint_never_installs_whisperjav_or_pytorch(monkeypatch):
    _hide(monkeypatch, "onnxruntime", "transformers", "huggingface_hub")
    _ok, hint = SpeechSegmenterFactory.is_backend_available("whisperseg")
    assert "whisperjav" not in hint.lower() and "torch" not in hint.lower()


def test_the_hints_limits_are_pyprojects():
    extra = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]["huggingface"]
    assert TRANSFORMERS.strip('"') in extra
    assert HUB.strip('"') in extra


def _box(out):
    return [line for line in re.sub(r"\x1b\[[0-9;]*m", "", out).splitlines() if line.startswith(("║", "╔", "╚"))]


def test_start_check_stops_whisperseg_on_a_minimal_install_with_the_hint_on_its_own_line(monkeypatch, capsys):
    _hide(monkeypatch, "transformers", "huggingface_hub")
    with pytest.raises(SystemExit) as stop:
        preflight_check.ensure_segmenter_backend_available("whisperseg")
    assert stop.value.code == 1
    box = _box(capsys.readouterr().out)
    text = [line.strip("║").rstrip() for line in box]
    assert "  The whisperseg speech detector cannot run on this machine" in text
    assert "  Nothing has been transcribed. Without a working speech detector" in text
    i = text.index("    - Install them, once:")
    assert text[i + 1] == f"        pip install {TRANSFORMERS} {HUB}"
    assert {len(line) for line in box} == {preflight_check.BOX_WIDTH}   # the frame holds


def test_a_short_hint_stays_on_the_install_line(monkeypatch, capsys):
    _hide(monkeypatch, "onnxruntime")
    with pytest.raises(SystemExit):
        preflight_check.ensure_segmenter_backend_available("whisperseg")
    text = [line.strip("║").rstrip() for line in _box(capsys.readouterr().out)]
    assert "    - Install it, once:  pip install onnxruntime" in text


@pytest.mark.parametrize("package, line", [
    ("onnxruntime", "Install with: pip install onnxruntime"),
    ("transformers", f"Install with: pip install {TRANSFORMERS}"),
])
def test_whisperseg_own_import_errors_give_the_same_safe_line(monkeypatch, package, line):
    # Reached when a package is there but cannot be imported (the start check finds it, the import fails).
    import sys
    from whisperjav.modules.speech_segmentation.backends.whisperseg import WhisperSegSpeechSegmenter, _install_line
    monkeypatch.setitem(sys.modules, package, None)   # "import <package>" now raises ImportError
    with pytest.raises(ImportError) as err:
        WhisperSegSpeechSegmenter()._ensure_model()
    assert str(err.value).endswith(line)
    assert _install_line("huggingface_hub") == f"pip install {HUB}"


@pytest.mark.parametrize("backend", ["ten", "firered-vad", "silero", "none"])
def test_other_segmenters_unaffected(monkeypatch, backend):
    # These do not use transformers: hiding it changes nothing for them.
    before = SpeechSegmenterFactory.is_backend_available(backend)
    _hide(monkeypatch, "transformers")
    assert SpeechSegmenterFactory.is_backend_available(backend) == before
