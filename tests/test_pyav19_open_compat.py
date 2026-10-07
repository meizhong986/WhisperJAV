"""PS-001: PyAV 19 removed av.open's metadata_errors / metadata_encoding; faster-whisper still passes metadata_errors.
whisperjav.utils.pyav_compat wraps av.open so the call is repeated without them. A stand-in 'av' module plays PyAV 19
here; the real PyAV 19.0.1 check (Python 3.12, faster-whisper 1.2.1 decode_audio) is in the commit message.
"""
import sys
import types

import pytest

from whisperjav.utils import pyav_compat


def _fake_av(accepts_metadata_args: bool):
    calls = []

    def open(file, mode="r", **kwargs):
        calls.append(dict(kwargs))
        for k in kwargs:
            if not accepts_metadata_args or k not in ("metadata_errors", "metadata_encoding"):
                raise TypeError(f"open() got an unexpected keyword argument '{k}'")
        return "container"

    return types.SimpleNamespace(open=open, __version__="19.0.1" if not accepts_metadata_args else "17.1.0"), calls


@pytest.fixture
def fake_av(monkeypatch):
    def install(accepts):
        mod, calls = _fake_av(accepts)
        monkeypatch.setitem(sys.modules, "av", mod)
        return mod, calls
    return install


def test_pyav19_call_is_repeated_without_the_removed_arguments(fake_av):
    av, calls = fake_av(accepts=False)
    assert pyav_compat.ensure_av_open_compat()
    assert av.open("x.mkv", mode="r", metadata_errors="ignore") == "container"
    assert calls == [{"metadata_errors": "ignore"}, {}]


def test_older_pyav_is_called_once_unchanged(fake_av):
    av, calls = fake_av(accepts=True)
    pyav_compat.ensure_av_open_compat()
    assert av.open("x.mkv", mode="r", metadata_errors="ignore") == "container"
    assert calls == [{"metadata_errors": "ignore"}]


def test_other_wrong_arguments_still_raise_and_wrapping_happens_once(fake_av):
    av, calls = fake_av(accepts=False)
    pyav_compat.ensure_av_open_compat()
    first = av.open
    pyav_compat.ensure_av_open_compat()
    assert av.open is first
    with pytest.raises(TypeError, match="no_such_argument"):
        av.open("x.mkv", no_such_argument=1)


def test_every_faster_whisper_site_installs_it():
    from pathlib import Path
    root = Path(pyav_compat.__file__).resolve().parents[1] / "modules"
    for rel in ("faster_whisper_pro_asr.py", "kotoba_faster_whisper_asr.py", "analytics.py",
                "speech_segmentation/backends/whisper_vad.py"):
        assert "ensure_av_open_compat()" in (root / rel).read_text(encoding="utf-8"), rel
