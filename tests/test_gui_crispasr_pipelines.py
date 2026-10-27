"""The GUI and the two CrispASR pipelines (1.10.0; owner W118, W129 7, W133 1, W193 CLA2, W194 3).

What is pinned here:
  - the removed 1.9.x "bring your own CrispASR" mode has left the GUI completely: no "crispasr" mode, no Executable,
    Backend or Extra Args fields, no "External (CrispASR)" choice, no file dialog or saved preferences for it, and no
    CrispASR key in the shipped asr_config.json (W118: "Delete the old ... panel text fields ... from the GUI completely");
  - Parakeet-CrispASR and Cohere-CrispASR are offered in the Ensemble tab only, in the Pass 1 and Pass 2 menus, under
    the group "GGUF" (W195 I2; "CrispASR" in W194 3), with their clean names (W118, W129 7);
  - the model menus in app.js are the manifest's offered models, the default first;
  - each pipeline's default speech segmenter in app.js is the command line's (W202 2-3); the GGUF pipelines carry
    the CJK folder-name note (W203 O14, T24); a status panel under each pass row shows the model and the device
    (W133 2, T16), asked from api.get_crispasr_status, which never downloads;
  - the command line the GUI builds for such a pass carries the model, the speech segmenter and the sensitivity, and
    no scene detector, speech enhancer or 1.9.x flag; and WhisperJAV's own command line accepts it
    (--dump-params, which runs no ASR).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from whisperjav.crispasr.manifest import CRISPASR_PIPELINES, PIPELINE_DEFAULT_MODEL, PIPELINE_MODELS
from whisperjav.webview_gui.api import WhisperJAVAPI

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "whisperjav" / "webview_gui" / "assets"
HTML = (ASSETS / "index.html").read_text(encoding="utf-8")
JS = (ASSETS / "app.js").read_text(encoding="utf-8")
LABELS = {"parakeet-crispasr": "Parakeet-CrispASR", "cohere-crispasr": "Cohere-CrispASR"}


@pytest.fixture
def api():
    return WhisperJAVAPI()


def _select(html, select_id):
    return re.search(r'<select[^>]*id="%s"[^>]*>(.*?)</select>' % re.escape(select_id), html, re.S).group(1)


def _twopass(tmp_path, pass1, pass2=None):
    inp = tmp_path / "a.mp4"
    inp.write_bytes(b"x")
    return {"inputs": [str(inp)], "output_dir": str(tmp_path / "out"), "source_language": "japanese",
            "subs_language": "native", "pass1": pass1, "pass2": pass2 or {"enabled": False}}


def _value(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


# ---------------------------------------------------------------------------------------------------------------
# The 1.9.x mode is gone
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("gone", ["crispasrModeRow", "crispasr-settings-panel", "crispasr-exe-path",
                                  "crispasr-extra-args", "crispasr-browse-btn", "crispasr-mode-backend",
                                  "crispasr-mode-exe-path", "crispasr-mode-extra-args", "crispasr-mode-browse-btn",
                                  "External (CrispASR)", 'value="crispasr"', "set exe path"])
def test_old_fields_are_gone_from_the_page(gone):
    assert gone not in HTML


@pytest.mark.parametrize("gone", ["select_crispasr_exe", "get_crispasr_preferences", "save_crispasr_preferences",
                                  "crispasrExePath", "crispasrExtraArgs", "crispasrBackend", "crispasrExe",
                                  "crispasrArgs", "updateCrispasrPanel", "=== 'crispasr'"])
def test_old_code_is_gone_from_app_js(gone):
    assert gone not in JS


@pytest.mark.parametrize("gone", ["_build_crispasr_args", "select_crispasr_exe", "get_crispasr_preferences",
                                  "save_crispasr_preferences"])
def test_old_functions_are_gone_from_the_api(gone):
    assert not hasattr(WhisperJAVAPI, gone)


def test_the_shipped_settings_hold_no_crispasr_key():
    config = json.loads((ROOT / "whisperjav" / "config" / "asr_config.json").read_text(encoding="utf-8"))
    assert "crispasr" not in json.dumps(config).lower()


def test_the_transcription_mode_tab_offers_no_crispasr():
    assert "crispasr" not in _select(HTML, "mode").lower()


# ---------------------------------------------------------------------------------------------------------------
# The two pipelines are offered in the Ensemble tab
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("select_id", ["pass1-pipeline", "pass2-pipeline"])
def test_both_passes_offer_both_pipelines_under_gguf(select_id):
    group = re.search(r'<optgroup label="GGUF">(.*?)</optgroup>', _select(HTML, select_id), re.S)
    assert group, f"{select_id} has no GGUF group"
    options = re.findall(r'<option value="([^"]+)"[^>]*>([^<]+)</option>', group.group(1))
    assert options == [(p, LABELS[p]) for p in CRISPASR_PIPELINES]


@pytest.mark.parametrize("pipeline, list_name", [("parakeet-crispasr", "parakeetCrispasrModels"),
                                                 ("cohere-crispasr", "cohereCrispasrModels")])
def test_model_menus_are_the_manifests_offered_models_default_first(pipeline, list_name):
    block = re.search(re.escape(list_name) + r":\s*\[(.*?)\]", JS, re.S).group(1)
    values = re.findall(r"value:\s*'([^']+)'", block)
    assert set(values) == set(PIPELINE_MODELS[pipeline])
    assert values[0] == PIPELINE_DEFAULT_MODEL[pipeline]


# ---------------------------------------------------------------------------------------------------------------
# The command line the GUI builds
# ---------------------------------------------------------------------------------------------------------------

def test_a_crispasr_pass_sends_model_segmenter_and_sensitivity_only(api, tmp_path):
    pass1 = {"pipeline": "parakeet-crispasr", "isCrispasr": True, "sensitivity": "aggressive",
             "speechSegmenter": "ten", "model": "parakeet-tdt-0.6b-ja", "sceneDetector": "auditok",
             "speechEnhancer": "zipenhancer", "enhanceForVad": True}
    pass2 = {"enabled": True, "pipeline": "cohere-crispasr", "isCrispasr": True, "sensitivity": "conservative",
             "speechSegmenter": "whisperseg", "model": "cohere-asr-ja", "sceneDetector": "silero",
             "speechEnhancer": "ffmpeg-dsp"}
    args = api._build_twopass_args(_twopass(tmp_path, pass1, pass2))
    assert _value(args, "--pass1-pipeline") == "parakeet-crispasr"
    assert _value(args, "--pass1-model") == "parakeet-tdt-0.6b-ja"
    assert _value(args, "--pass1-speech-segmenter") == "ten"
    assert _value(args, "--pass1-sensitivity") == "aggressive"
    assert _value(args, "--pass2-pipeline") == "cohere-crispasr"
    assert _value(args, "--pass2-model") == "cohere-asr-ja"
    assert _value(args, "--pass2-speech-segmenter") == "whisperseg"
    assert _value(args, "--pass2-sensitivity") == "conservative"
    assert _value(args, "--language") == "japanese"
    # design 22.9.1 (owner W212 D3, W214 2, W232 8): scene detector, enhancer, enhance-for-vad
    assert _value(args, "--pass1-scene-detector") == "auditok"
    assert _value(args, "--pass1-speech-enhancer") == "zipenhancer"
    assert "--pass1-enhance-for-vad" in args
    assert _value(args, "--pass2-scene-detector") == "silero"
    assert _value(args, "--pass2-speech-enhancer") == "ffmpeg-dsp:loudnorm"
    for flag in ("--pass1-qwen-params", "--pass1-params", "--pass1-hf-params"):
        assert flag not in args
    assert not [a for a in args if a.startswith("--crispasr")]


def test_none_and_missing_values_are_not_sent(api, tmp_path):
    pass1 = {"pipeline": "cohere-crispasr", "isCrispasr": True, "speechSegmenter": "none"}
    args = api._build_twopass_args(_twopass(tmp_path, pass1))
    assert _value(args, "--pass1-pipeline") == "cohere-crispasr"
    for flag in ("--pass1-speech-segmenter", "--pass1-model", "--pass1-sensitivity"):
        assert flag not in args


def test_whisperjav_accepts_the_command_line_the_gui_builds(api, tmp_path):
    """--dump-params runs main.py's usage checks (validate_crispasr_passes) and exits 0 without ASR."""
    pass1 = {"pipeline": "parakeet-crispasr", "isCrispasr": True, "sensitivity": "balanced",
             "speechSegmenter": "whisperseg", "model": "parakeet-tdt-0.6b-ja"}
    pass2 = {"enabled": True, "pipeline": "cohere-crispasr", "isCrispasr": True, "sensitivity": "balanced",
             "speechSegmenter": "whisperseg", "model": "cohere-asr-ja-v0.1"}
    args = api._build_twopass_args(_twopass(tmp_path, pass1, pass2))
    dump = tmp_path / "dump.json"
    p = subprocess.run([sys.executable, "-m", "whisperjav.main", *args, "--dump-params", str(dump)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                       cwd=str(ROOT))
    assert p.returncode == 0, p.stderr[-2000:]
    ensemble = json.loads(dump.read_text(encoding="utf-8"))["ensemble_config"]
    assert ensemble["pass1"]["crispasr_effective"]["model"] == "parakeet-tdt-0.6b-ja"
    assert ensemble["pass2"]["crispasr_effective"]["model"] == "cohere-asr-ja-v0.1"


# ---------------------------------------------------------------------------------------------------------------
# Route B (1.10.0): the default speech segmenters, the CJK note, the status panel
# ---------------------------------------------------------------------------------------------------------------

def test_the_default_segmenters_in_app_js_are_the_command_lines():
    from whisperjav.config.segmenter_presets import CRISPASR_DEFAULT_SEGMENTERS

    body = re.search(r"crispasrDefaultSegmenter\(pipeline\) \{(.*?)\n    \},", JS, re.S).group(1)
    assert "pipeline === 'cohere-crispasr' ? 'firered-vad' : 'whisperseg'" in body
    assert CRISPASR_DEFAULT_SEGMENTERS == {"parakeet-crispasr": "whisperseg", "cohere-crispasr": "firered-vad"}


@pytest.mark.parametrize("select_id", ["pass1-pipeline", "pass2-pipeline"])
def test_the_gguf_pipelines_carry_the_cjk_note(select_id):
    note = ("Experimental in 1.10.0 (early release). "   # owner W212 D4 1
            "The GGUF (CrispASR) pipelines do not support CJK (Chinese, Japanese, Korean) characters in "
            "folder names.")
    group = re.search(r'<optgroup label="GGUF">(.*?)</optgroup>', _select(HTML, select_id), re.S).group(1)
    assert group.count(f'title="{note}"') == 2
    # and on the Pipeline dropdown itself while a GGUF pipeline is selected
    assert f"CRISPASR_CJK_NOTE: '{note}'" in JS
    assert "pipelineSelect.title = passState.isCrispasr ? this.CRISPASR_CJK_NOTE : '';" in JS


def test_the_sensitivity_tooltip_names_the_preset():
    # owner W212 D4 2: sensitivity picks a preset, not only the detection threshold
    assert ("sensitivitySelect.title = 'Picks the speech segmenter\\'s preset: the detection threshold and "
            "related settings';") in JS


@pytest.mark.parametrize("pass_key", ["pass1", "pass2"])
def test_each_pass_has_a_hidden_status_panel(pass_key):
    assert f'<div class="crispasr-status-panel" id="{pass_key}-crispasr-status" style="display: none;"></div>' in HTML
    assert f"this.updateCrispasrStatus('{pass_key}');" in JS  # refreshed when the model changes


class TestTheStatusPanel:
    @pytest.fixture
    def stubbed(self, api, monkeypatch):
        from whisperjav.crispasr import install

        state = {"downloaded": False, "runtime": {"device": "cuda", "reason": None}}
        monkeypatch.setattr(install, "model_status",
                            lambda model: install.ModelStatus(model, state["downloaded"]))
        monkeypatch.setattr(install, "ensure_model", lambda *a, **k: pytest.fail("the panel must not download"))
        monkeypatch.setattr(type(api), "_crispasr_runtime", lambda self: state["runtime"])
        return state

    def test_a_model_not_downloaded_and_the_nvidia_gpu(self, api, stubbed):
        result = api.get_crispasr_status("cohere-crispasr", "")
        assert result == {"success": True, "can_run": True, "lines": [
            "Model: Cohere Transcribe ja v0.1 (q8_0) — not downloaded yet (2,424 MB); downloaded when the run starts",
            "GPU: asks for the NVIDIA GPU (CUDA)"]}

    def test_a_downloaded_model_on_the_processor(self, api, stubbed):
        stubbed["downloaded"] = True
        stubbed["runtime"] = {"device": "cpu", "reason": "no NVIDIA GPU was found"}
        result = api.get_crispasr_status("parakeet-crispasr", "parakeet-tdt-0.6b-ja")
        assert result["lines"] == ["Model: Parakeet TDT 0.6B ja (q8_0) — downloaded",
                                   "CPU (not accelerated), several times slower: no NVIDIA GPU was found"]

    def test_apple_metal(self, api, stubbed):
        stubbed["runtime"] = {"device": "metal", "reason": None}
        assert api.get_crispasr_status("parakeet-crispasr")["lines"][1] == \
            "GPU: asks for the Apple GPU (Metal), not yet tested on a Mac"

    def test_crispasr_that_cannot_run(self, api, stubbed):
        stubbed["runtime"] = {"problem": "CrispASR is not installed in this WhisperJAV. Run the WhisperJAV installer again."}
        result = api.get_crispasr_status("parakeet-crispasr")
        assert result["can_run"] is False
        assert result["lines"][1] == ("CrispASR cannot run here: CrispASR is not installed in this WhisperJAV. "
                                      "Run the WhisperJAV installer again.")

    def test_an_unknown_model_is_an_answer_not_a_crash(self, api, stubbed):
        assert api.get_crispasr_status("parakeet-crispasr", "nope")["success"] is False

    def test_the_device_is_asked_in_a_child_process_once(self, api, monkeypatch):
        calls = []

        def run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout='RESULT={"device": "cpu", "reason": "x"}\n', stderr="")

        monkeypatch.setattr("whisperjav.webview_gui.api.subprocess.run", run)
        assert api._crispasr_runtime() == {"device": "cpu", "reason": "x"}
        assert api._crispasr_runtime() == {"device": "cpu", "reason": "x"}
        assert len(calls) == 1 and calls[0][0] == sys.executable

