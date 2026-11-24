"""ZipEnhancer starts again for a run's second file (1.10.3; requirements of 1.10.0, section 5, F2).

The pipelines make a new ZipEnhancer for each file. PyTorch accepts torch.set_num_interop_threads only once per
process, so before 1.10.3 the second ZipEnhancer failed to start ("cannot set number of interop threads after parallel
work has started or set_num_interop_threads called") and the file kept its original audio.

The first two tests use the real PyTorch in a fresh process (the setting is once per process); the third makes two
ZipEnhancers in this process with ModelScope's pipeline replaced by a stand-in, so no model and no network are needed.
"""
import subprocess
import sys
import textwrap
import types

import pytest

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
                          cwd=str(REPO_ROOT), timeout=300)


def test_pytorch_refuses_a_second_interop_setting():
    """The premise: PyTorch's own second call raises."""
    done = _run("""
        import torch
        torch.set_num_interop_threads(2)
        torch.set_num_interop_threads(2)
    """)
    assert done.returncode != 0
    assert "cannot set number of interop threads" in done.stderr


def test_the_helper_takes_a_second_call():
    done = _run("""
        import torch
        from whisperjav.modules.speech_enhancement.backends.zipenhancer import _set_interop_threads
        _set_interop_threads(torch, 2)
        _set_interop_threads(torch, 3)
        print("count", torch.get_num_interop_threads())
    """)
    assert done.returncode == 0, done.stderr
    assert "count 2" in done.stdout


@pytest.fixture
def fake_modelscope(monkeypatch):
    """ModelScope's pipeline() replaced by a stand-in; counts how often a pipeline was made."""
    made = []

    def pipeline(task, model=None, device=None):
        made.append((task, model, device))
        return lambda path: {"output_pcm": b""}

    root = types.ModuleType("modelscope")
    pipelines = types.ModuleType("modelscope.pipelines")
    pipelines.pipeline = pipeline
    utils = types.ModuleType("modelscope.utils")
    constant = types.ModuleType("modelscope.utils.constant")
    constant.Tasks = types.SimpleNamespace(acoustic_noise_suppression="acoustic-noise-suppression")
    for name, module in (("modelscope", root), ("modelscope.pipelines", pipelines),
                         ("modelscope.utils", utils), ("modelscope.utils.constant", constant)):
        monkeypatch.setitem(sys.modules, name, module)
    return made


def test_a_second_zipenhancer_in_one_process_starts(fake_modelscope, monkeypatch):
    import torch
    from whisperjav.modules.speech_enhancement.backends import zipenhancer

    calls = []

    def interop(n):   # as PyTorch: the first call is taken, every later one raises
        calls.append(n)
        if len(calls) > 1:
            raise RuntimeError("Error: cannot set number of interop threads after parallel work has started "
                               "or set_num_interop_threads called")

    monkeypatch.setattr(torch, "set_num_interop_threads", interop)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)

    for _file in (1, 2):
        enhancer = zipenhancer.ZipEnhancerBackend(model="torch", device="cpu")
        assert enhancer._ensure_initialized() is True
        enhancer.cleanup()
    assert len(fake_modelscope) == 2
    assert len(calls) == 2
