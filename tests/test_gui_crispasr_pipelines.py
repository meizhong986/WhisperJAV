"""The GUI and the two CrispASR pipelines (1.10.0; owner W118, W129 7, W133 1, W193 CLA2, W194 3).

What is pinned here:
  - the removed 1.9.x "bring your own CrispASR" mode has left the GUI completely: no "crispasr" mode, no Executable,
    Backend or Extra Args fields, no "External (CrispASR)" choice, no file dialog or saved preferences for it, and no
    CrispASR key in the shipped asr_config.json (W118: "Delete the old ... panel text fields ... from the GUI completely");
  - Parakeet-CrispASR and Cohere-CrispASR are offered in the Ensemble tab only, in the Pass 1 and Pass 2 menus, under
    the group "GGUF via crispasr", named "parakeet" and "cohere" (W343; "GGUF" W195 I2 until then); since 1.10.1 the
    five new pipelines (W397, W401) below a divider line, in their own group "Experimental, open to feedback", MOSS's
    name in capitals (owner W520 CLA2 S1-S2, W521); the greyed-out "Cohere-Transcribe (not available yet)" entry is
    gone (W520 S3);
  - the Ensemble tab opens on cohere / balanced / FireRedVAD in pass 1 and parakeet / aggressive / TEN VAD in pass 2,
    and picking either pipeline sets its own sensitivity and segmenter, as the command line does (W343, W344);
  - the model menus in app.js are the manifest's offered models, the default first, each named by the manifest's
    menu_label: its quantization and the GPU memory it needs (W400);
  - each pipeline's default speech segmenter in app.js is the command line's (W202 2-3); the GGUF pipelines carry
    the CJK folder-name note (W203 O14, T24); a status panel under each pass row shows the model and the device
    (W133 2, T16), asked from api.get_crispasr_status, which never downloads;
  - the command line the GUI builds for such a pass carries the model, the speech segmenter (None included, owner
    W457, W458: each scene goes to the model whole) and the sensitivity, and
    no scene detector, speech enhancer or 1.9.x flag; and WhisperJAV's own command line accepts it
    (--dump-params, which runs no ASR).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from whisperjav.crispasr.manifest import (CRISPASR_PIPELINES, MODELS, PIPELINE_DEFAULT_MODEL, PIPELINE_MODELS,
                                          menu_label)
from whisperjav.webview_gui.api import WhisperJAVAPI

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "whisperjav" / "webview_gui" / "assets"
HTML = (ASSETS / "index.html").read_text(encoding="utf-8")
JS = (ASSETS / "app.js").read_text(encoding="utf-8")
LABELS = {"parakeet-crispasr": "parakeet", "cohere-crispasr": "cohere",   # W343
          "reazonspeech-crispasr": "reazonspeech", "kotoba-crispasr": "kotoba",  # W401
          "whisper-turbo-crispasr": "whisper-turbo", "whisper-large-crispasr": "whisper-large",
          "moss-crispasr": "MOSS"}                                             # W520 S2
GROUP = "GGUF via crispasr"                                                # W343
NEW_GROUP = "Experimental, open to feedback"                               # W520 CLA2 S1, W521
FIRST_TWO = ("parakeet-crispasr", "cohere-crispasr")                        # 1.10.0's; the other five new in 1.10.1


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
    menu = _select(HTML, select_id)
    groups = re.search(r'<optgroup label="%s">(.*?)</optgroup>\s*(?:<!--.*?-->\s*)?<hr>\s*'
                       r'<optgroup label="%s">(.*?)</optgroup>' % (GROUP, NEW_GROUP), menu, re.S)
    assert groups, f"{select_id} has no {GROUP} group, divider line and {NEW_GROUP} group in a row"
    options = [re.findall(r'<option value="([^"]+)"[^>]*>([^<]+)</option>', g) for g in groups.groups()]
    assert options[0] == [(p, LABELS[p]) for p in CRISPASR_PIPELINES if p in FIRST_TWO]
    assert options[1] == [(p, LABELS[p]) for p in CRISPASR_PIPELINES if p not in FIRST_TWO]
    assert "Cohere-Transcribe (not available yet)" not in menu and 'value="cohere"' not in menu   # W520 S3


@pytest.mark.parametrize("pass_key, pipeline, sensitivity, segmenter",
                         [("pass1", "cohere-crispasr", "balanced", "firered-vad"),
                          ("pass2", "parakeet-crispasr", "aggressive", "ten")])
def test_the_ensemble_tab_opens_on_cohere_and_parakeet(pass_key, pipeline, sensitivity, segmenter):
    # W343: the selected options in index.html are what the tab opens on (EnsembleManager.init reads them)
    def selected(select_id):
        return re.findall(r'<option value="([^"]+)"[^>]*\bselected\b', _select(HTML, select_id))
    assert selected(f"{pass_key}-pipeline") == [pipeline]
    assert selected(f"{pass_key}-sensitivity") == [sensitivity]
    assert selected(f"{pass_key}-segmenter") == [segmenter]
    # and the start-up state in app.js says the same
    block = re.search(r"\n        %s: \{(.*?)\n        \}," % pass_key, JS, re.S).group(1)
    assert f"pipeline: '{pipeline}'" in block and f"sensitivity: '{sensitivity}'" in block
    assert f"speechSegmenter: '{segmenter}'" in block and "isCrispasr: true" in block


def _crispasr_menus():
    block = re.search(r"\n    crispasrModels: \{(.*?)\n    \},", JS, re.S).group(1)
    return {p: re.findall(r"\{ value: '([^']+)', label: '([^']+)' \}", body)
            for p, body in re.findall(r"'([a-z-]+)': \[(.*?)\]", block, re.S)}


def test_every_crispasr_pipeline_has_a_model_menu():
    assert list(_crispasr_menus()) == list(CRISPASR_PIPELINES)


@pytest.mark.parametrize("pipeline", CRISPASR_PIPELINES)
def test_model_menus_are_the_manifests_offered_models_default_first(pipeline):
    menu = _crispasr_menus()[pipeline]
    values = [value for value, _ in menu]
    assert set(values) == set(PIPELINE_MODELS[pipeline])
    assert values[0] == PIPELINE_DEFAULT_MODEL[pipeline]
    # W400: each name says the quantization and the GPU memory
    assert [label for _, label in menu] == [menu_label(MODELS[v]) for v in values]
    for _, label in menu:
        assert re.fullmatch(r".+ \((f16|q8_0|q5_0)\), ~\d+\.\d GB VRAM", label), label


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


def test_missing_values_are_not_sent_and_the_segmenter_none_is(api, tmp_path):
    pass1 = {"pipeline": "cohere-crispasr", "isCrispasr": True, "speechSegmenter": "none"}
    args = api._build_twopass_args(_twopass(tmp_path, pass1))
    assert _value(args, "--pass1-pipeline") == "cohere-crispasr"
    # owner W457, W458: None as the speech segmenter is sent; each scene goes to the model whole
    assert _value(args, "--pass1-speech-segmenter") == "none"
    for flag in ("--pass1-model", "--pass1-sensitivity"):
        assert flag not in args
    args = api._build_twopass_args(_twopass(tmp_path, {"pipeline": "cohere-crispasr", "isCrispasr": True}))
    assert "--pass1-speech-segmenter" not in args


def test_the_segmenter_none_is_not_greyed_for_a_crispasr_pass():
    assert "if (noneSegmenter) noneSegmenter.disabled = false;" in JS
    assert "noneSegmenter.disabled = !!passState.isCrispasr" not in JS


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
# 1.10.1 (Part B; owner W411-W414): the Timing tab and --passN-word-timestamps
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("choice", ["model", "aligner"])
def test_a_timing_tab_choice_is_sent_as_the_flag(api, tmp_path, choice):
    pass1 = {"pipeline": "kotoba-crispasr", "isCrispasr": True, "wordTimestamps": choice}
    pass2 = {"enabled": True, "pipeline": "moss-crispasr", "isCrispasr": True, "wordTimestamps": choice}
    args = api._build_twopass_args(_twopass(tmp_path, pass1, pass2))
    assert _value(args, "--pass1-word-timestamps") == choice
    assert _value(args, "--pass2-word-timestamps") == choice


@pytest.mark.parametrize("value", [None, "", "bogus"])
def test_no_choice_sends_no_flag(api, tmp_path, value):
    pass1 = {"pipeline": "cohere-crispasr", "isCrispasr": True, "wordTimestamps": value}
    assert "--pass1-word-timestamps" not in api._build_twopass_args(_twopass(tmp_path, pass1))


def test_whisperjav_accepts_the_word_timestamps_the_gui_sends(api, tmp_path):
    pass1 = {"pipeline": "kotoba-crispasr", "isCrispasr": True, "wordTimestamps": "aligner"}
    pass2 = {"enabled": True, "pipeline": "cohere-crispasr", "isCrispasr": True, "wordTimestamps": "model"}
    args = api._build_twopass_args(_twopass(tmp_path, pass1, pass2))
    dump = tmp_path / "dump.json"
    p = subprocess.run([sys.executable, "-m", "whisperjav.main", *args, "--dump-params", str(dump)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                       cwd=str(ROOT))
    assert p.returncode == 0, p.stderr[-2000:]
    ensemble = json.loads(dump.read_text(encoding="utf-8"))["ensemble_config"]
    assert ensemble["pass1"]["crispasr_effective"]["word_timestamps"] == "aligner"
    assert ensemble["pass2"]["crispasr_effective"]["word_timestamps"] == "model"


def test_the_timing_tab_is_hidden_until_a_crispasr_pass_opens_customize():
    # owner W413 2: a "Timing" tab in a CrispASR pass's Customize window
    assert '<button class="modal-tab" data-tab="timing" style="display: none;">Timing</button>' in HTML
    assert '<div id="tab-timing" class="modal-tab-panel">' in HTML
    opener = re.search(r"\n    async openCustomize\(passKey\) \{(.*?)\n    \},", JS, re.S).group(1)
    assert opener.index("this.showTimingTab(false);") < opener.index("this.openTransformersCustomize(passKey)")
    crispasr = re.search(r"\n    async openCrispasrCustomize\(passKey\) \{(.*?)\n    \},", JS, re.S).group(1)
    assert "this.generateCrispasrTimingTab(passKey);" in crispasr and "this.showTimingTab(true);" in crispasr
    close = re.search(r"\n    closeModal\(\) \{(.*?)\n    \},", JS, re.S).group(1)
    assert "this.showTimingTab(false);" in close
    # every opener goes through openCustomize
    for name in ("openTransformersCustomize", "openQwenCustomize", "openCrispasrCustomize"):
        assert JS.count(f"this.{name}(") == 1


def test_the_timing_tab_texts_are_the_approved_ones():
    # owner W413 2: "Word timestamps": "From the model" / "From the aligner (wav2vec2-ja)", default marked
    assert "label.textContent = 'Word timestamps';" in JS
    assert ("model: { text: 'From the model', defaultText: 'From the model (default)' }," in JS)
    assert ("aligner: { text: 'From the aligner (wav2vec2-ja)', "
            "defaultText: 'From the aligner (wav2vec2-ja) (default)' }" in JS)


def test_a_loaded_preset_sets_the_crispasr_mark_and_the_timing_tab_by_its_pipeline():
    # Found by the call-chain review (2026-11-01): a preset of another pipeline left the tab and the choice stale
    loader = re.search(r"\n    async loadPresetIntoModal\(name\) \{(.*?)\n    \},", JS, re.S).group(1)
    assert "passState.isCrispasr = this.isCrispasrPipeline(passState.pipeline);" in loader
    assert "this.generateCrispasrTimingTab(passKey);" in loader and "this.showTimingTab(false);" in loader


def test_a_preset_keeps_the_timing_value():
    # owner W418 3: "saved presets should save the value for timing"
    collect = re.search(r"\n    _collectPresetData\(\) \{(.*?)\n    \},", JS, re.S).group(1)
    assert "wordTimestamps: this._timingForPreset(passState)," in collect
    helper = re.search(r"\n    _timingForPreset\(passState\) \{(.*?)\n    \},", JS, re.S).group(1)
    assert "if (!passState.isCrispasr) return null;" in helper
    assert "document.getElementById('crispasrWordTimestamps')" in helper     # as the window shows it
    loader = re.search(r"\n    async loadPresetIntoModal\(name\) \{(.*?)\n    \},", JS, re.S).group(1)
    assert "const savedTiming = preset.wordTimestamps;" in loader
    assert ("passState.wordTimestamps = (passState.isCrispasr && (savedTiming === 'model' || savedTiming === "
            "'aligner'))") in loader
    assert "this.updateCrispasrStatus(passKey);   // the aligner line follows the loaded value" in loader


def test_the_timing_value_survives_the_preset_files_round_trip(api, monkeypatch, tmp_path):
    from whisperjav.settings import presets

    monkeypatch.setattr(presets, "get_presets_dir", lambda: tmp_path)
    data = {"pipeline": "kotoba-crispasr", "customized": True, "params": {}, "wordTimestamps": "aligner"}
    assert api.save_preset("k", data)["success"]
    stored = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert stored["word_timestamps"] == "aligner"                              # snake_case on disk, as the others
    assert api.load_preset("k")["preset"]["wordTimestamps"] == "aligner"


def test_the_choice_is_kept_apart_from_the_params_and_reset_with_the_pipeline():
    assert JS.count("wordTimestamps: null\n") + JS.count("wordTimestamps: null\r\n") == 2   # both passes start on null
    assert JS.count("passState.wordTimestamps = null;  // the new pipeline's own default (1.10.1)") == 2
    for n in (1, 2):
        assert (f"wordTimestamps: this.state.pass{n}.isCrispasr ? (this.state.pass{n}.wordTimestamps || null) : null,"
                in JS)
    assert "await pywebview.api.get_crispasr_status(pipeline, model, wordTimestamps);" in JS


# ---------------------------------------------------------------------------------------------------------------
# Route B (1.10.0): the default speech segmenters, the CJK note, the status panel
# ---------------------------------------------------------------------------------------------------------------

def test_the_default_segmenters_in_app_js_are_the_command_lines():
    from whisperjav.config.segmenter_presets import CRISPASR_DEFAULT_SEGMENTERS, CRISPASR_DEFAULT_SENSITIVITY

    block = re.search(r"\n    crispasrDefaults: \{(.*?)\n    \},", JS, re.S).group(1)
    from whisperjav.crispasr.manifest import word_timestamps_for

    from whisperjav.crispasr.pass_settings import scene_detector_for

    in_js = {p: (seg, sens, wt, scene or "semantic") for p, seg, sens, wt, scene in
             re.findall(r"'([a-z-]+)': \{ segmenter: '([^']+)', sensitivity: '([^']+)', "
                        r"wordTimestamps: '([^']+)'(?:, scene: '([^']+)')? \}", block)}
    # 1.10.1 (owner W411-W414): each pipeline's default word timestamps, as the command line's; and its scene
    # detector (semantic for every one; W482 for MOSS-CrispASR)
    assert in_js == {p: (CRISPASR_DEFAULT_SEGMENTERS[p], CRISPASR_DEFAULT_SENSITIVITY[p], word_timestamps_for(p),
                         scene_detector_for(p)) for p in CRISPASR_PIPELINES}
    assert "return (defaults && defaults.scene) || 'semantic';" in JS
    assert "return this.crispasrDefaults[pipeline].segmenter;" in JS
    assert "return this.crispasrDefaults[pipeline].sensitivity;" in JS
    assert "return defaults ? defaults.wordTimestamps : null;   // null: not a CrispASR pipeline" in JS
    # W344, and the 1.10.1 pipelines as Parakeet (ReazonSpeech) or Cohere (the others) (W401); MOSS FireRedVAD (W482)
    assert CRISPASR_DEFAULT_SEGMENTERS == {"parakeet-crispasr": "ten", "cohere-crispasr": "firered-vad",
                                           "reazonspeech-crispasr": "ten", "kotoba-crispasr": "firered-vad",
                                           "whisper-turbo-crispasr": "firered-vad",
                                           "whisper-large-crispasr": "firered-vad", "moss-crispasr": "firered-vad"}
    assert CRISPASR_DEFAULT_SENSITIVITY == {"parakeet-crispasr": "aggressive", "cohere-crispasr": "balanced",
                                            "reazonspeech-crispasr": "aggressive", "kotoba-crispasr": "balanced",
                                            "whisper-turbo-crispasr": "balanced",
                                            "whisper-large-crispasr": "balanced", "moss-crispasr": "balanced"}


@pytest.mark.parametrize("select_id", ["pass1-pipeline", "pass2-pipeline"])
def test_the_gguf_pipelines_carry_the_cjk_note(select_id):
    tail = ("The GGUF (CrispASR) pipelines need folder names in plain Western letters (ASCII). When your user "
            "folder has Chinese, Japanese or Korean characters, WhisperJAV links or copies the model to a "
            "plain-named folder (the original stays where it is).")
    note = "Experimental (early release). " + tail                    # owner W212 D4 1; versions W501 4
    new_note = "New in 1.10.1, experimental (early release). " + tail  # the five pipelines new in 1.10.1
    menu = _select(HTML, select_id)
    for pipeline in CRISPASR_PIPELINES:
        group = GROUP if pipeline in FIRST_TWO else NEW_GROUP
        options = re.search(r'<optgroup label="%s">(.*?)</optgroup>' % group, menu, re.S).group(1)
        option = re.search(r'<option value="%s"[^>]*>' % pipeline, options).group(0)
        assert f'title="{note if pipeline in FIRST_TWO else new_note}"' in option, option
    assert "Experimental in 1.10.0" not in HTML
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
            # Cohere-CrispASR aligns its text with wav2vec2-ja (item 12)
            "Aligner: wav2vec2 large XLSR-53 ja (q4_k) — not downloaded yet (232 MB); downloaded the first time "
            "it is used",
            "GPU: asks for the NVIDIA GPU (CUDA)"]}

    def test_a_chronosjav_pass_shows_its_aligner_and_timing_model(self, api, stubbed):
        stubbed["downloaded"] = True
        result = api.get_chronosjav_crispasr_status("crispasr", "qwen3-fa-gguf", "whisper-small",
                                                    "aligner_vad_fallback")
        assert result == {"success": True, "can_run": True, "lines": [
            "Aligner and timing model:",
            "Aligner: Qwen3-ForcedAligner 0.6B (q4_k) — downloaded",
            "Timing model: Whisper small — downloaded",
            "GPU: asks for the NVIDIA GPU (CUDA)"]}

    def test_a_chronosjav_pass_without_crispasr_parts_shows_nothing(self, api, stubbed):
        assert api.get_chronosjav_crispasr_status("qwen3", "Qwen/Qwen3-ForcedAligner-0.6B", "none",
                                                  "aligner_vad_fallback")["lines"] == []
        # vad_only loads no aligner; the timing model still runs
        lines = api.get_chronosjav_crispasr_status("crispasr", "wav2vec2-ja", "whisper-small", "vad_only")["lines"]
        assert [line.split(":")[0] for line in lines] == ["Aligner and timing model", "Timing model", "GPU"]

    def test_a_downloaded_model_on_the_processor(self, api, stubbed):
        stubbed["downloaded"] = True
        stubbed["runtime"] = {"device": "cpu", "reason": "no NVIDIA GPU was found"}
        result = api.get_crispasr_status("parakeet-crispasr", "parakeet-tdt-0.6b-ja")
        assert result["lines"] == ["Model: Parakeet TDT 0.6B ja (q8_0) — downloaded",
                                   "CPU (not accelerated), several times slower: no NVIDIA GPU was found"]

    def test_apple_metal(self, api, stubbed):
        stubbed["runtime"] = {"device": "metal", "reason": None}
        assert api.get_crispasr_status("parakeet-crispasr")["lines"][1] == \
            "GPU: asks for the Apple GPU (Metal), not yet tested on a Mac"   # Parakeet-CrispASR: no aligner

    def test_crispasr_that_cannot_run(self, api, stubbed):
        stubbed["runtime"] = {"problem": "CrispASR is not installed in this WhisperJAV. Run the WhisperJAV installer again."}
        result = api.get_crispasr_status("parakeet-crispasr")
        assert result["can_run"] is False
        assert result["lines"][1] == ("CrispASR cannot run here: CrispASR is not installed in this WhisperJAV. "
                                      "Run the WhisperJAV installer again.")

    # 1.10.1 (owner W413 2): the aligner line follows the pass's Timing tab choice.
    @pytest.mark.parametrize("pipeline, choice, aligner_line", [
        ("kotoba-crispasr", None, False), ("kotoba-crispasr", "", False), ("kotoba-crispasr", "aligner", True),
        ("parakeet-crispasr", "aligner", True), ("cohere-crispasr", None, True), ("cohere-crispasr", "model", False),
        ("moss-crispasr", "model", False), ("moss-crispasr", "aligner", True)])
    def test_the_aligner_line_follows_the_word_timestamps_choice(self, api, stubbed, pipeline, choice, aligner_line):
        lines = api.get_crispasr_status(pipeline, "", choice)["lines"]
        assert any(line.startswith("Aligner: wav2vec2") for line in lines) is aligner_line

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

