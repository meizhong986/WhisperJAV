"""The GUI entry of the rscn scene detector (R-SCN; 1.10.1; owner W448-W455; design
docs/plans/design/2026-11-02-rscn-design.md section 15).

What is pinned here:
  - both passes' Scene Detector dropdowns hold an R-SCN option, disabled in the page, which app.js enables only for a
    pass whose pipeline accepts rscn (EnsembleManager.rscnPipelines), the same list the command line checks
    (crispasr/pass_settings.py RSCN_PIPELINES: moss-crispasr in this build);
  - the Customize Scene tab offers R-SCN under the same rule, and draws its controls from the rscn tool YAML with the
    defaults of the detector's own settings (rscn/settings.py), so the tab, the preset and the run cannot drift;
  - a MOSS pass with rscn and customized values is sent as --passN-scene-detector rscn with the rscn_ keys in
    --passN-params, and WhisperJAV's own command line accepts it (--dump-params, which runs no ASR);
  - the tab is seeded with the preset: Apply without a change changes nothing.

Run alone: python -m pytest tests/test_gui_rscn.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from whisperjav.crispasr.pass_settings import RSCN_PIPELINES, rscn_problem
from whisperjav.modules.scene_detection_backends.rscn.settings import PREFIX, RSCNSettings
from whisperjav.modules.scene_detection_backends.rscn.speech import known_backend
from whisperjav.webview_gui.api import WhisperJAVAPI

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "whisperjav" / "webview_gui" / "assets"
HTML = (ASSETS / "index.html").read_text(encoding="utf-8")
JS = (ASSETS / "app.js").read_text(encoding="utf-8")
YAML_FILE = ROOT / "whisperjav" / "config" / "v4" / "ecosystems" / "tools" / "rscn-scene-detection.yaml"
TOOLTIP = ("R-SCN cuts scenes where the sound changes and at black picture confirmed by the sound, 70-420 s long, "
           "never dropping audio. For the MOSS pipeline in this version.")
GROUPS = {"length", "change", "cut_placement", "digital_silence", "speech_measure", "picture"}


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
# The row: the option and the rule that enables it
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("select_id", ["pass1-scene", "pass2-scene"])
def test_both_scene_dropdowns_hold_a_disabled_rscn_option(select_id):
    options = re.findall(r'<option value="([^"]+)"([^>]*)>([^<]+)</option>', _select(HTML, select_id))
    assert [v for v, _, _ in options] == ["auditok", "silero", "semantic", "rscn", "none"]
    attrs, text = next((a, t) for v, a, t in options if v == "rscn")
    assert "disabled" in attrs and f'title="{TOOLTIP}"' in attrs and text == "R-SCN"


def test_the_gui_offers_rscn_for_the_pipelines_the_command_line_accepts():
    m = re.search(r"rscnPipelines: \[([^\]]*)\]", JS)
    assert m, "EnsembleManager.rscnPipelines is missing"
    assert tuple(re.findall(r"'([a-z-]+)'", m.group(1))) == RSCN_PIPELINES == ("moss-crispasr",)
    assert "updateRscnOption(passKey) {" in JS
    # at start for both passes, after a pipeline change (both branches) and after a preset is loaded
    assert JS.count("this.updateRscnOption(") >= 5
    # the Customize Scene tab follows the same rule and asks the schema for the pass's pipeline
    assert "if (customizePass && this.offersRscn(customizePass.pipeline)) {" in JS
    assert "{ value: 'rscn', label: 'R-SCN (changes in the sound, picture breaks)' }" in JS
    assert "pywebview.api.get_scene_detector_schema(backend, customizePass ? customizePass.pipeline : '')" in JS


def test_moss_s_default_scene_detector_is_still_silero():
    # D10 (design section 14): the user chooses R-SCN; nothing is changed by default
    from whisperjav.crispasr.pass_settings import scene_detector_for
    assert scene_detector_for("moss-crispasr") == "silero"
    assert "'moss-crispasr': { segmenter: 'silero-v6.2', sensitivity: 'balanced', wordTimestamps: 'model', scene: 'silero' }" in JS


# ---------------------------------------------------------------------------------------------------------------
# The Customize Scene tab: controls from the YAML, defaults from the detector's own settings
# ---------------------------------------------------------------------------------------------------------------

def test_the_rscn_yaml_holds_controls_only():
    doc = yaml.safe_load(YAML_FILE.read_text(encoding="utf-8"))
    assert doc["tool_type"] == "scene_detection" and "spec" not in doc
    names = RSCNSettings.names()
    for key, hints in doc["gui"].items():
        assert key.startswith(PREFIX) and key[len(PREFIX):] in names, key
        assert hints["widget"] in ("number", "dropdown") and hints["label"] and hints["description"], key
        assert hints["group"] in GROUPS, key
    assert len(doc["gui"]) == 19


def test_the_registry_loads_the_rscn_yaml_with_the_other_tools():
    from whisperjav.config.v4.schemas import ToolConfig
    doc = yaml.safe_load(YAML_FILE.read_text(encoding="utf-8"))
    tool = ToolConfig(**doc)
    assert tool.tool_type == "scene_detection" and tool.spec == {} and len(tool.gui) == 19


def test_the_schema_defaults_are_the_detectors_own(api):
    schema = api.get_scene_detector_schema("rscn", "moss-crispasr")
    assert schema["success"], schema
    assert schema["display_name"] == "R-SCN (changes in the sound, picture breaks)"
    preset = RSCNSettings.for_pipeline("moss-crispasr").as_dict()
    assert schema["defaults"] == {k: preset[k[len(PREFIX):]] for k in schema["parameters"]}
    assert set(schema["groups"]) == GROUPS
    for key, control in schema["parameters"].items():
        default = schema["defaults"][key]
        assert control["default"] == default
        if control["type"] == "dropdown":
            assert control["original_type"] == "str" and default in control["options"], key
        else:
            assert control["type"] == "number" and control["original_type"] == "float", key
            assert control["min"] <= default <= control["max"], (key, default)
    # the same with no pipeline named (the Scene tab of a pass whose state is unknown): the default preset
    assert api.get_scene_detector_schema("rscn")["defaults"] == schema["defaults"]


def test_the_choices_of_the_dropdowns(api):
    parameters = api.get_scene_detector_schema("rscn", "moss-crispasr")["parameters"]
    measures = parameters["rscn_speech_measure"]["options"]
    assert "none" in measures and "firered-vad" in measures and all(known_backend(o) for o in measures)
    assert parameters["rscn_picture"]["options"] == ["auto", "on", "off"]


def test_the_tab_is_seeded_with_the_preset_and_apply_without_a_change_changes_nothing(api):
    r = api.get_crispasr_customize_defaults("moss-crispasr", "balanced", "", "rscn")
    assert r["success"] and r["scene_detection_method"] == "rscn", r
    assert r["scene"] == api.get_scene_detector_schema("rscn", "moss-crispasr")["defaults"]
    assert RSCNSettings.for_pipeline("moss-crispasr", r["scene"]) == RSCNSettings.for_pipeline("moss-crispasr")
    # what the GUI sends after Apply (segmenter and scene values together) passes the start-of-run check
    assert rscn_problem(1, "moss-crispasr", "rscn", json.dumps({**r["segmenter"], **r["scene"]})) is None


def test_the_other_scene_tabs_are_unchanged(api):
    r = api.get_crispasr_customize_defaults("moss-crispasr", "balanced", "", "silero")
    assert r["success"] and r["scene"]["max_duration_s"] == 420
    assert not any(str(k).startswith(PREFIX) for k in r["scene"])
    assert api.get_scene_detector_schema("silero")["success"]
    assert api.get_scene_detector_schema("none")["parameters"] == {}


# ---------------------------------------------------------------------------------------------------------------
# The command line the GUI builds
# ---------------------------------------------------------------------------------------------------------------

def _moss_rscn_pass(params):
    return {"pipeline": "moss-crispasr", "isCrispasr": True, "sensitivity": "balanced",
            "speechSegmenter": "silero-v6.2", "model": "moss-transcribe-diarize-0.9b-f16", "sceneDetector": "rscn",
            "customized": True, "params": params}


def test_a_moss_pass_with_rscn_sends_the_detector_and_its_values(api, tmp_path):
    params = {"rscn_min_scene_s": 90, "rscn_picture": "off", "rscn_speech_measure": "none", "silero_threshold": 0.5}
    args = api._build_twopass_args(_twopass(tmp_path, _moss_rscn_pass(params)))
    assert _value(args, "--pass1-pipeline") == "moss-crispasr"
    assert _value(args, "--pass1-scene-detector") == "rscn"
    assert json.loads(_value(args, "--pass1-params")) == params
    assert rscn_problem(1, "moss-crispasr", "rscn", _value(args, "--pass1-params")) is None


def test_the_same_choice_for_another_pipeline_is_refused_which_is_why_the_option_is_greyed():
    problem = rscn_problem(1, "kotoba-crispasr", "rscn", None)
    assert problem and "available only when" in problem and "moss-crispasr" in problem


def test_whisperjav_accepts_the_command_line_the_gui_builds(api, tmp_path):
    """--dump-params runs main.py's usage checks (validate_crispasr_passes, rscn_problem) and exits 0 without ASR."""
    params = {"rscn_min_scene_s": 90, "rscn_preferred_min_s": 120, "rscn_picture": "off", "rscn_speech_measure": "none"}
    args = api._build_twopass_args(_twopass(tmp_path, _moss_rscn_pass(params)))
    dump = tmp_path / "dump.json"
    p = subprocess.run([sys.executable, "-m", "whisperjav.main", *args, "--dump-params", str(dump)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                       cwd=str(ROOT))
    assert p.returncode == 0, p.stderr[-2000:]
    effective = json.loads(dump.read_text(encoding="utf-8"))["ensemble_config"]["pass1"]["crispasr_effective"]
    assert effective["scene_detector"] == "rscn"
    assert effective["rscn_settings"]["min_scene_s"] == 90
    assert effective["rscn_settings"]["picture"] == "off"
    assert effective["rscn_settings"]["speech_measure"] == "none"
    assert effective["rscn_settings"]["max_scene_s"] == 420
