"""--model on balanced, fast and faster keeps the resolver's device (1.10.2, still-open item 7 of the 1.10.1 notes).

The resolver gives these modes "cuda" only when CTranslate2 itself sees a CUDA device (resolver_v3.py). The --model
override took PyTorch's device alone (get_best_device), so on a PyTorch whose GPU CTranslate2 cannot use (Own
PyTorch, ROCm) a run with --model went to "cuda". The probe stands in such a computer: PyTorch says "cuda",
CTranslate2 says "cpu". It runs main() with --dump-params in a child process (-X utf8: no UTF-8 relaunch) and reads
the model block of the dump. No model is loaded.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

PROBE = textwrap.dedent("""
    import json, sys
    out, extra = sys.argv[1], json.loads(sys.argv[2])
    sys.argv = ["whisperjav", "--dump-params", out] + extra

    import whisperjav.main as M
    import whisperjav.config.resolver_v3 as R

    M.get_best_device = lambda *a, **k: "cuda"          # PyTorch sees a GPU ...
    R.get_best_device = lambda *a, **k: "cuda"
    R.ctranslate2_device = lambda *a, **k: "cpu"        # ... that CTranslate2 cannot use
    try:
        M.main()
    except SystemExit as e:
        print("EXIT=" + str(e.code))
""")


def _model_block(node):
    if isinstance(node, dict):
        if "device" in node and "model_name" in node:
            return node
        for value in node.values():
            hit = _model_block(value)
            if hit:
                return hit
    return None


def _dump(tmp_path, extra):
    out = tmp_path / "dump.json"
    done = subprocess.run([sys.executable, "-X", "utf8", "-c", PROBE, str(out), json.dumps(extra)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                          cwd=str(REPO))
    assert "EXIT=0" in done.stdout, (done.stdout + done.stderr)[-2000:]
    return _model_block(json.loads(out.read_text(encoding="utf-8")))


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["balanced", "fast", "faster"])
def test_model_override_keeps_the_ctranslate2_device(tmp_path, mode):
    without = _dump(tmp_path, ["--mode", mode, "--sensitivity", "balanced"])
    with_model = _dump(tmp_path, ["--mode", mode, "--sensitivity", "balanced", "--model", "large-v2"])
    assert without["device"] == "cpu"
    assert with_model["device"] == "cpu" and with_model["model_name"] == "large-v2"


@pytest.mark.slow
def test_an_explicit_device_still_wins(tmp_path):
    block = _dump(tmp_path, ["--mode", "balanced", "--sensitivity", "balanced", "--model", "large-v2",
                             "--device", "cuda"])
    assert block["device"] == "cuda"
