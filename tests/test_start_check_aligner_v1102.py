"""The CrispASR aligner a GGUF pass asks for is downloaded at the start-of-run check (1.10.2, owner W622 item 5).

Before, the pass downloaded it after its first file's ASR, and a failed download let the pass go on without it
(subtitles at the VAD groups' edges or the model's own times, with a note). Now a pass with
--passN-word-timestamps aligner has the aligner downloaded with its ASR model, before any audio is read, and a
failure stops the run with the start check's box, exit status 1. A pass whose pipeline runs the aligner by default
(Cohere-CrispASR, Qwen3-ASR-CrispASR) has it downloaded there too, without the switch (owner W624 B).

No download and no CrispASR call: install's steps are replaced, as in tests/test_crispasr_notes_and_check.py. The
main() probe runs in a child process with -X utf8 (no UTF-8 relaunch), and its start check is a stand-in.
"""
import ast
import importlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from whisperjav.crispasr import install, manifest
from whisperjav.utils import preflight_check as pf

REPO = Path(__file__).resolve().parents[1]
CHOICES = (None, "model", "aligner")


def _pipeline_classes():
    """The pass worker's own table, read from its source so the worker (and its ASR imports) is not loaded."""
    tree = ast.parse((REPO / "whisperjav" / "ensemble" / "pass_worker.py").read_text(encoding="utf-8"))
    imports = {a.asname or a.name: node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and node.module and node.module.startswith("whisperjav.pipelines.") for a in node.names}
    table = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and getattr(node.targets[0], "id", "") == "PIPELINE_CLASSES")
    return {k.value: getattr(importlib.import_module(imports[v.id]), v.id)
            for k, v in zip(table.keys, table.values) if k.value.endswith("-crispasr")}


def test_aligner_for_says_what_each_pipeline_runs():
    classes = _pipeline_classes()
    assert tuple(classes) == manifest.CRISPASR_PIPELINES
    for name, cls in classes.items():
        for choice in CHOICES:
            assert manifest.aligner_for(name, choice) == cls.engine_row(choice).aligner, (name, choice)


def test_aligner_for_names_the_default_aligner_or_none():
    assert manifest.aligner_for("parakeet-crispasr", "aligner") == manifest.DEFAULT_ALIGNER == "wav2vec2-ja"
    assert manifest.aligner_for("parakeet-crispasr") is None
    assert manifest.aligner_for("cohere-crispasr") == "wav2vec2-ja"
    assert manifest.aligner_for("cohere-crispasr", "model") is None


class TestEnsureCrispASRReadyWithAligner:
    PASSES = [(1, "parakeet-crispasr", None), (2, "cohere-crispasr", "cohere-asr-ja")]
    REASON = "Downloading the model wav2vec2 large XLSR-53 ja (q4_k) failed: stand-in connection refused"

    @pytest.fixture
    def steps(self, monkeypatch):
        state = SimpleNamespace(asked=[], fail_on=None)

        def forbidden(*args, **kwargs):
            raise AssertionError("the start-up check's test must not download or load anything")

        for name in ("_hf_download", "available_backends"):
            monkeypatch.setattr(install, name, forbidden)

        def prepare_runtime(device, model):
            state.asked.append(f"runtime {model.name}")
            return install.RuntimeChoice(True, "cuda", None, False, "0.8.53+cuda", "cuda")

        def ensure_model(model, offline=None):
            state.asked.append(model.name)
            if state.fail_on == model.name:
                raise install.CrispASRSetupError(self.REASON)
            return Path("model.gguf")

        def stage_model(path, temp_dir, sha256):
            state.asked.append("stage")
            return Path(path)

        monkeypatch.setattr(install, "prepare_runtime", prepare_runtime)
        monkeypatch.setattr(install, "ensure_model", ensure_model)
        monkeypatch.setattr(install, "stage_model", stage_model)
        monkeypatch.setattr(install, "hub_offline", lambda: False)
        return state

    def test_the_aligner_is_downloaded_after_its_pass_model(self, steps, capsys, tmp_path):
        assert pf.ensure_crispasr_ready(self.PASSES, temp_dir=str(tmp_path), aligners={1: "wav2vec2-ja"}) is True
        assert steps.asked == ["runtime parakeet-tdt-0.6b-ja", "parakeet-tdt-0.6b-ja", "stage", "wav2vec2-ja",
                               "runtime cohere-asr-ja", "cohere-asr-ja", "stage"]
        assert capsys.readouterr().out == ""

    def test_without_aligners_nothing_more_is_downloaded(self, steps, tmp_path):
        pf.ensure_crispasr_ready(self.PASSES, temp_dir=str(tmp_path))
        assert "wav2vec2-ja" not in steps.asked

    def test_a_failed_aligner_download_stops_the_run_before_any_audio(self, steps, capsys, tmp_path):
        steps.fail_on = "wav2vec2-ja"
        with pytest.raises(SystemExit) as stop:
            pf.ensure_crispasr_ready(self.PASSES, temp_dir=str(tmp_path), aligners={1: "wav2vec2-ja"})
        assert stop.value.code == 1
        # pass 2 is not reached: the run stops at the first file that cannot be had
        assert steps.asked[-1] == "wav2vec2-ja" and "cohere-asr-ja" not in steps.asked
        out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
        text = " ".join(" ".join(line.strip("║ ") for line in out.splitlines()).split())
        assert "CrispASR is not ready for this run" in text
        # The box names the aligner of the pass that has one (owner W624 D1); the no-aligner sentence is T4 of
        # tests/test_crispasr_notes_and_check.py.
        assert ("This run uses parakeet-crispasr (pass 1) with the aligner wav2vec2-ja, cohere-crispasr (pass 2). "
                "WhisperJAV checks CrispASR and downloads its models the first time they are needed. "
                "That did not work:") in text
        assert self.REASON in text
        assert "Nothing has been transcribed." in text


# main() with two CrispASR passes; the start check stands in and records what it was asked, then stops the run.
STARTUP_PROBE = textwrap.dedent("""
    import json, sys
    from pathlib import Path

    root = Path(sys.argv[1])
    extra = json.loads(sys.argv[2])
    media = root / "media" / "a.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"x")
    sys.argv = ["whisperjav", str(media), "--ensemble", "--pass1-pipeline", "parakeet-crispasr",
                "--pass2-pipeline", "cohere-crispasr",
                "--output-dir", str(root / "out"), "--temp-dir", str(root / "tmp"), "--accept-cpu-mode",
                "--no-signature", "--no-progress"] + extra

    import whisperjav.main as M

    asked = {}

    def crispasr_ready(passes, **kwargs):
        asked["passes"] = [list(p) for p in passes]
        asked["aligners"] = {str(k): v for k, v in (kwargs.get("aligners") or {}).items()}
        raise SystemExit(1)

    M.ensure_segmenter_backend_available = lambda backend, **kwargs: True
    M.ensure_segmenter_model_available = lambda backend, **kwargs: True
    M.ensure_crispasr_ready = crispasr_ready
    try:
        M.main()
        code = 0
    except SystemExit as e:
        code = e.code
    print("RESULT=" + json.dumps({"exit": code, "asked": asked}))
""")


def _probe(tmp_path, extra):
    done = subprocess.run([sys.executable, "-X", "utf8", "-c", STARTUP_PROBE, str(tmp_path), json.dumps(extra)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                          cwd=str(REPO))
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT=")]
    assert lines, (done.stdout + done.stderr)[-2000:]
    return json.loads(lines[-1][len("RESULT="):])


@pytest.mark.slow
@pytest.mark.parametrize("extra, aligners", [
    ([], {"2": "wav2vec2-ja"}),                                     # Cohere-CrispASR's default aligner (W624 B)
    (["--pass1-word-timestamps", "aligner"], {"1": "wav2vec2-ja", "2": "wav2vec2-ja"}),
    (["--pass2-word-timestamps", "aligner"], {"2": "wav2vec2-ja"}),
    (["--pass1-word-timestamps", "model", "--pass2-word-timestamps", "model"], {}),
], ids=["no switch", "pass 1 aligner", "pass 2 aligner", "both model"])
def test_main_hands_the_asked_for_aligner_to_the_start_check(tmp_path, extra, aligners):
    result = _probe(tmp_path, extra)
    assert result["exit"] == 1
    assert result["asked"]["passes"] == [[1, "parakeet-crispasr", None], [2, "cohere-crispasr", None]]
    assert result["asked"]["aligners"] == aligners
