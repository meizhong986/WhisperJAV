"""openai-whisper's word-timing step on the Apple GPU (1.10.3, owner W750 A4; #515).

No Apple GPU here: an Apple-GPU tensor is stood in for by an object whose device type is "mps" and whose .cpu() gives
the real table; it fails, as on a Mac, if anything converts it to float64 before moving it. The replacement must give
the same alignment as openai-whisper's own CPU path, and leave every other tensor to openai-whisper's dtw().

Run with: python -m pytest tests/test_mps_word_timing.py -q
"""
import types

import numpy as np
import pytest
import torch

timing = pytest.importorskip("whisper.timing")

from whisperjav.modules import mps_word_timing  # noqa: E402


class AppleGpuTable:
    """Stands in for a tensor on the Apple GPU: float64 there is refused, as MPS refuses it."""

    is_cuda = False

    def __init__(self, cpu_tensor):
        self._t = cpu_tensor
        self.device = types.SimpleNamespace(type="mps")

    def double(self):
        raise TypeError("Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64.")

    def cpu(self):
        return self._t


@pytest.fixture
def table():
    rng = np.random.default_rng(1)
    return torch.tensor(rng.random((12, 40)), dtype=torch.float32)


@pytest.fixture(autouse=True)
def restore_dtw():
    before = timing.dtw
    yield
    timing.dtw = before


def test_openai_whispers_own_dtw_fails_on_the_apple_gpu(table):
    with pytest.raises(TypeError, match="float64"):
        getattr(timing.dtw, "_whisperjav_original", timing.dtw)(AppleGpuTable(table))


def test_the_replacement_aligns_the_apple_gpu_table_on_the_cpu(table):
    assert mps_word_timing.install()
    got = timing.dtw(AppleGpuTable(table))
    expected = timing.dtw_cpu(table.double().numpy())
    assert [np.array_equal(a, b) for a, b in zip(got, expected)] == [True, True]


def test_other_tensors_take_openai_whispers_own_dtw(table):
    original = getattr(timing.dtw, "_whisperjav_original", timing.dtw)
    mps_word_timing.install()
    got, expected = timing.dtw(table), original(table)
    assert [np.array_equal(a, b) for a, b in zip(got, expected)] == [True, True]


def test_install_twice_wraps_once():
    mps_word_timing.install()
    first = timing.dtw
    mps_word_timing.install()
    assert timing.dtw is first
