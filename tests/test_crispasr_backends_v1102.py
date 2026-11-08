"""
1.10.2: CrispASR's speech segmenters and speech enhancers in WhisperJAV (owner W594, W596, W598, W599).

- Speech segmenter backends: sortformer (NVIDIA Nemotron-3-Diarization, on the GPU of the install's CrispASR build)
  and pyannote (segmentation-3.0, on the CPU); the same values at every sensitivity (W596 4).
- Speech enhancer backends: mel-band-roformer and mini-bs-roformer (W594, W596); a name WhisperJAV does not offer
  stops the run (W603; tests/test_unknown_enhancer_v1102.py).
- Enhance for VAD only is the default wherever a speech enhancer runs in a pipeline with a separate speech segmenter
  (W596 3); the GUI's tick box sits in the Speech Enhancer menu, ticked.
- The GUI menus (W598 2, W599): the Speech Enhancer menu is None, then the sections Vocal Separation and DSP; the
  Speech Segmenter menu is FireRedVAD, TEN VAD, WhisperSeg, the Silero builds, then "Experimental, open to feedback"
  with Sortformer and pyannote, then None.

No model is loaded or downloaded here.
"""
import argparse
import json
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "whisperjav" / "webview_gui" / "assets"
TOOLS = REPO / "whisperjav" / "config" / "v4" / "ecosystems" / "tools"


# ---------------------------------------------------------------------------
# The model files, pinned
# ---------------------------------------------------------------------------

class TestTheModelFiles:
    def test_the_speech_segmenters(self):
        from whisperjav.crispasr.manifest import SEGMENTER_MODELS
        assert sorted(SEGMENTER_MODELS) == ["pyannote", "sortformer"]
        sortformer, pyannote = SEGMENTER_MODELS["sortformer"], SEGMENTER_MODELS["pyannote"]
        assert (sortformer.repo_id, sortformer.filename) == ("nvidia/Nemotron-3-Diarization",
                                                             "Nemotron-3-Diarization.q8_0.gguf")
        assert (pyannote.repo_id, pyannote.filename) == ("cstr/pyannote-v3-segmentation-GGUF",
                                                         "pyannote-seg-3.0.gguf")

    def test_the_speech_enhancers(self):
        from whisperjav.crispasr.manifest import ENHANCER_MODELS
        assert sorted(ENHANCER_MODELS) == ["mel-band-roformer", "mini-bs-roformer"]
        assert ENHANCER_MODELS["mel-band-roformer"].repo_id == "cstr/mel-band-roformer-vocals-GGUF"
        assert ENHANCER_MODELS["mini-bs-roformer"].repo_id == "cstr/mini-bs-roformer-GGUF"

    @pytest.mark.parametrize("table", ["SEGMENTER_MODELS", "ENHANCER_MODELS"])
    def test_every_file_is_pinned(self, table):
        from whisperjav.crispasr import manifest
        for name, model in getattr(manifest, table).items():
            assert model.name == name
            assert len(model.revision) == 40, name      # a commit, not a branch
            assert len(model.sha256) == 64, name
            assert model.size > 1_000_000, name


# ---------------------------------------------------------------------------
# Speech segmenter: sortformer and pyannote
# ---------------------------------------------------------------------------

class TestTheSpeechSegmenters:
    @pytest.mark.parametrize("name", ["sortformer", "pyannote"])
    def test_offered_by_the_factory_and_the_command_line(self, name):
        from whisperjav.config.segmenter_presets import SEGMENTER_TOOL_NAMES
        from whisperjav.modules.speech_segmentation.factory import SpeechSegmenterFactory
        assert name in SpeechSegmenterFactory.list_backends()
        assert SEGMENTER_TOOL_NAMES[name] == f"{name}-speech-segmentation"
        listed = {b["name"]: b for b in SpeechSegmenterFactory.get_available_backends()}
        assert name in listed

    @pytest.mark.parametrize("name", ["sortformer", "pyannote"])
    def test_the_same_values_at_every_sensitivity(self, name):
        """Owner W596 4: no grey-out; all three sensitivities give the same values."""
        from whisperjav.config.segmenter_presets import resolve_segmenter_sensitivity
        values = [resolve_segmenter_sensitivity(name, s) for s in ("conservative", "balanced", "aggressive")]
        assert values[0] == values[1] == values[2]
        assert values[0]["max_group_duration_s"] == 5 and values[0]["chunk_threshold_s"] == 1.0

    def test_their_names_match_the_gui(self):
        """One name per segmenter (W599): the menu, the Customize window and the factory's listing."""
        from whisperjav.modules.speech_segmentation.factory import SpeechSegmenterFactory
        listed = {b["name"]: b["display_name"] for b in SpeechSegmenterFactory.get_available_backends()}
        for name, label in (("sortformer", "Sortformer (NVIDIA Nemotron)"), ("pyannote", "pyannote 3.0")):
            assert listed[name] == label
            assert f'displayName: "{label}"' in (TOOLS / f"{name}-speech-segmentation.yaml").read_text(encoding="utf-8")

    def test_fidelity_keeps_them(self):
        """Owner W596 2, "All pipelines": a single fidelity run keeps them instead of falling back."""
        from whisperjav.config.segmenter_presets import SINGLE_PASS_EXTERNAL_OK
        assert {"sortformer", "pyannote"} <= SINGLE_PASS_EXTERNAL_OK["fidelity"]


class TestTheStartUpModelCheck:
    """A model that cannot be fetched stops the run before any audio is read (every scene would fail)."""

    @pytest.fixture
    def failing_fetch(self, monkeypatch):
        from whisperjav.utils import preflight_check

        def _fail(backend, download):
            raise OSError("no route to host")
        monkeypatch.setattr(preflight_check, "_ensure_crispasr_segmenter_model", _fail)
        return preflight_check

    @pytest.mark.parametrize("name,display", [("sortformer", "Sortformer"), ("pyannote", "pyannote")])
    def test_it_stops_with_status_1(self, failing_fetch, name, display, capsys):
        with pytest.raises(SystemExit) as excinfo:
            failing_fetch.ensure_segmenter_model_available(name)
        assert excinfo.value.code == 1
        out = capsys.readouterr()
        assert f"{display}'s speech-detection model is not on this machine" in out.out + out.err

    def test_check_gets_false_back(self, failing_fetch):
        assert failing_fetch.ensure_segmenter_model_available("sortformer", download=False,
                                                              exit_on_fail=False) is False

    def test_a_fetched_model_passes(self, monkeypatch):
        from whisperjav.utils import preflight_check
        monkeypatch.setattr(preflight_check, "_ensure_crispasr_segmenter_model", lambda backend, download: None)
        assert preflight_check.ensure_segmenter_model_available("pyannote") is True


# ---------------------------------------------------------------------------
# Speech enhancer: mel-band-roformer, mini-bs-roformer
# ---------------------------------------------------------------------------

class TestTheSpeechEnhancers:
    def test_the_backends(self):
        from whisperjav.main import SPEECH_ENHANCER_CHOICES
        from whisperjav.modules.speech_enhancement.factory import SpeechEnhancerFactory
        expected = ["none", "ffmpeg-dsp", "zipenhancer", "htdemucs", "mel-band-roformer", "mini-bs-roformer"]
        assert sorted(SpeechEnhancerFactory.list_backends()) == sorted(expected)
        assert sorted(SPEECH_ENHANCER_CHOICES) == sorted(expected)

    def test_the_old_backend_file_is_gone(self):
        assert not (REPO / "whisperjav" / "modules" / "speech_enhancement" / "backends" / "bs_roformer.py").exists()

    @pytest.mark.parametrize("value", ["bs-roformer", "bs-roformer:vocals", " BS-RoFormer", "clearvoice"])
    def test_an_old_name_is_an_unknown_name(self, value):
        """W603: no name of a speech enhancer WhisperJAV no longer includes is special any more."""
        from whisperjav.modules.speech_enhancement.factory import unknown_backend_message
        name = value.split(":", 1)[0].strip()
        assert unknown_backend_message(value, "--pass1-speech-enhancer") == (
            f"'{name}' is not a speech enhancer in WhisperJAV. Choose one of none, ffmpeg-dsp, zipenhancer, htdemucs, "
            "mel-band-roformer or mini-bs-roformer for --pass1-speech-enhancer and run again.")

    @pytest.mark.parametrize("args,flag", [
        (["--ensemble", "--pass1-pipeline", "fidelity", "--pass1-speech-enhancer", "bs-roformer"],
         "--pass1-speech-enhancer"),
        (["--mode", "qwen", "--qwen-enhancer", "bs-roformer:vocals"], "--qwen-enhancer"),
    ])
    def test_the_command_line_stops_with_exit_2(self, args, flag):
        result = subprocess.run([sys.executable, "-m", "whisperjav.main", "dummy.mp4", *args],
                                cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=300)
        assert result.returncode == 2, (result.stdout + result.stderr)[-800:]
        assert f"argument {flag}: invalid choice" in result.stderr


# ---------------------------------------------------------------------------
# Enhance for VAD only: on by default (owner W596 3)
# ---------------------------------------------------------------------------

def _ensemble(p1, e1, p2=None, e2=None, **flags):
    ns = argparse.Namespace(ensemble=True, enhance_for_vad=None, pass1_pipeline=p1, pass1_speech_enhancer=e1,
                            pass2_pipeline=p2, pass2_speech_enhancer=e2, pass1_enhance_for_vad=None,
                            pass2_enhance_for_vad=None)
    for key, value in flags.items():
        setattr(ns, key, value)
    return ns


class TestEnhanceForVadDefault:
    @pytest.mark.parametrize("pipeline,enhancer,expected", [
        ("fidelity", "zipenhancer", True),
        ("qwen", "mel-band-roformer", True),
        ("parakeet-crispasr", "mini-bs-roformer", True),
        ("moss-crispasr", "htdemucs", True),
        ("fidelity", "none", False),
        ("fidelity", None, False),
        ("transformers", "zipenhancer", False),
        ("faster", "zipenhancer", False),
        ("balanced", "zipenhancer", False),
    ])
    def test_an_ensemble_pass(self, pipeline, enhancer, expected):
        from whisperjav.main import apply_enhance_for_vad_default
        args = _ensemble(pipeline, enhancer)
        apply_enhance_for_vad_default(args)
        assert args.pass1_enhance_for_vad is expected
        assert args.enhance_for_vad is False   # folded into the passes

    def test_the_user_can_switch_it_off_per_pass(self):
        from whisperjav.main import apply_enhance_for_vad_default
        args = _ensemble("fidelity", "zipenhancer", "qwen", "zipenhancer", pass2_enhance_for_vad=False)
        apply_enhance_for_vad_default(args)
        assert (args.pass1_enhance_for_vad, args.pass2_enhance_for_vad) == (True, False)

    def test_the_global_switch_reaches_both_passes(self):
        from whisperjav.main import apply_enhance_for_vad_default
        args = _ensemble("fidelity", "zipenhancer", "qwen", "zipenhancer", enhance_for_vad=False)
        apply_enhance_for_vad_default(args)
        assert (args.pass1_enhance_for_vad, args.pass2_enhance_for_vad) == (False, False)

    @pytest.mark.parametrize("enhancer,expected", [("htdemucs", True), ("none", False), (None, False)])
    def test_a_qwen_run(self, enhancer, expected):
        from whisperjav.main import apply_enhance_for_vad_default
        args = argparse.Namespace(ensemble=False, enhance_for_vad=None, mode="qwen", qwen_enhancer=enhancer,
                                  pipeline=None, pipeline_config=None)
        apply_enhance_for_vad_default(args)
        assert args.enhance_for_vad is expected

    def test_decoupled_keeps_its_own_default(self):
        from whisperjav.main import apply_enhance_for_vad_default
        args = argparse.Namespace(ensemble=False, enhance_for_vad=None, mode="balanced", qwen_enhancer=None,
                                  pipeline="decoupled", pipeline_config=None)
        apply_enhance_for_vad_default(args)
        assert args.enhance_for_vad is None

    def test_from_the_command_line_to_the_passes(self, tmp_path):
        out = tmp_path / "params.json"
        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "dummy.mp4", "--ensemble",
             "--pass1-pipeline", "fidelity", "--pass1-speech-enhancer", "zipenhancer",
             "--pass2-pipeline", "parakeet-crispasr", "--pass2-speech-enhancer", "mini-bs-roformer",
             "--no-pass2-enhance-for-vad", "--dump-params", str(out)],
            cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        assert result.returncode == 0, (result.stdout + result.stderr)[-800:]
        passes = json.loads(out.read_text(encoding="utf-8"))["ensemble_config"]
        assert passes["pass1"]["enhance_for_vad"] is True
        assert passes["pass2"]["enhance_for_vad"] is False


class TestTheGuiSendsIt:
    @pytest.mark.parametrize("pipeline", ["fidelity", "qwen", "parakeet-crispasr", "anime-whisper"])
    def test_ticked_and_unticked(self, pipeline):
        from whisperjav.webview_gui.api import _enhance_for_vad_args
        assert _enhance_for_vad_args(1, {"pipeline": pipeline, "enhanceForVad": True}) == ["--pass1-enhance-for-vad"]
        assert _enhance_for_vad_args(2, {"pipeline": pipeline, "enhanceForVad": False}) == \
            ["--no-pass2-enhance-for-vad"]

    def test_ticked_when_never_touched(self):
        from whisperjav.webview_gui.api import _enhance_for_vad_args
        assert _enhance_for_vad_args(1, {"pipeline": "fidelity"}) == ["--pass1-enhance-for-vad"]

    @pytest.mark.parametrize("pipeline", ["balanced", "transformers", "xxl", "fast", "faster"])
    def test_nothing_for_one_track(self, pipeline):
        from whisperjav.webview_gui.api import _enhance_for_vad_args
        assert _enhance_for_vad_args(1, {"pipeline": pipeline, "enhanceForVad": True}) == []

    def test_the_box_starts_ticked(self):
        html = (ASSETS / "index.html").read_text(encoding="utf-8")
        for n in (1, 2):
            assert f'<input type="checkbox" id="pass{n}-enhance-for-vad" checked>' in html


class TestStepSevenFixes:
    """The wiring review (step 7, owner W601): three GUI gaps closed."""

    def test_no_tick_box_for_fast_and_faster(self):
        """They run no speech enhancer and no separate speech segmenter; the GUI sends no flag for them."""
        from whisperjav.webview_gui.api import ONE_TRACK_PIPELINES
        js = (ASSETS / "app.js").read_text(encoding="utf-8")
        body = js[js.index("updateEnhanceForVadCheckbox(passId) {"):][:2500]
        assert "const isFastOrFaster = ['fast', 'faster'].includes(this.state[passId]?.pipeline);" in body
        assert "!isFastOrFaster" in body
        assert {"fast", "faster"} <= ONE_TRACK_PIPELINES

    def test_remember_settings_keeps_the_tick_box(self):
        from whisperjav.settings.gui_settings import DEFAULT_GUI_SETTINGS
        from whisperjav.webview_gui.api import WhisperJAVAPI
        js = (ASSETS / "app.js").read_text(encoding="utf-8")
        for n in (1, 2):
            assert f"'pass{n}-enhance-for-vad': {{ key: 'pass{n}EnhanceForVad', prop: 'checked' }}," in js
            assert WhisperJAVAPI._GUI_SETTINGS_MAP[f"pass{n}_enhance_for_vad"] == f"pass{n}EnhanceForVad"
            assert DEFAULT_GUI_SETTINGS[f"pass{n}_enhance_for_vad"] is True   # = the page's ticked box

    def test_available_backends_keep_their_hover_text(self):
        js = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert js.count("if (option.dataset.pageTitle === undefined) option.dataset.pageTitle = option.title || '';") == 2
        assert js.count("option.title = option.dataset.pageTitle;") == 2
        assert "option.title = '';" not in js


# ---------------------------------------------------------------------------
# The GUI menus (owner W598 2, W599)
# ---------------------------------------------------------------------------

class _SelectReader(HTMLParser):
    """The children of one <select>, in order: ("option", value, text), ("hr",), ("group", label, [options])."""

    def __init__(self, select_id):
        super().__init__()
        self.select_id = select_id
        self.inside = False
        self.items = []
        self.group = None
        self.option = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "select" and attrs.get("id") == self.select_id:
            self.inside = True
        elif not self.inside:
            return
        elif tag == "optgroup":
            self.group = ("group", attrs.get("label"), [])
            self.items.append(self.group)
        elif tag == "hr":
            self.items.append(("hr",))
        elif tag == "option":
            self.option = [attrs.get("value"), ""]

    def handle_data(self, data):
        if self.inside and self.option is not None:
            self.option[1] += data

    def handle_endtag(self, tag):
        if not self.inside:
            return
        if tag == "option" and self.option is not None:
            entry = ("option", self.option[0], self.option[1].strip())
            (self.group[2] if self.group else self.items).append(entry)
            self.option = None
        elif tag == "optgroup":
            self.group = None
        elif tag == "select":
            self.inside = False


def _menu(select_id):
    reader = _SelectReader(select_id)
    reader.feed((ASSETS / "index.html").read_text(encoding="utf-8"))
    return reader.items


class TestTheMenus:
    @pytest.mark.parametrize("n", [1, 2])
    def test_the_speech_enhancer_menu(self, n):
        assert _menu(f"pass{n}-enhancer") == [
            ("option", "none", "None"),
            ("hr",),
            ("group", "Vocal Separation", [
                ("option", "mel-band-roformer", "Mel-Band-RoFormer"),
                ("option", "mini-bs-roformer", "Mini-BS-RoFormer"),
                ("option", "htdemucs", "htdemucs (Demucs v4)"),
            ]),
            ("hr",),
            ("group", "Noise Suppression", [("option", "zipenhancer", "ZipEnhancer")]),   # owner W600 1
            ("hr",),
            ("group", "DSP", [("option", "ffmpeg-dsp", "FFmpeg Audio Filters...")]),
        ]

    @pytest.mark.parametrize("n,ten_label", [(1, "TEN VAD"), (2, "TEN VAD (default)")])
    def test_the_speech_segmenter_menu(self, n, ten_label):
        assert _menu(f"pass{n}-segmenter") == [
            ("option", "firered-vad", "FireRedVAD"),
            ("option", "ten", ten_label),
            ("option", "whisperseg", "WhisperSeg (JA-ASMR)"),
            ("option", "silero-v3.1", "Silero v3.1"),
            ("option", "silero", "Silero v4.0"),
            ("option", "silero-v6.2", "Silero v6.2"),
            ("hr",),
            ("group", "Experimental, open to feedback", [
                ("option", "sortformer", "Sortformer (NVIDIA Nemotron)"),
                ("option", "pyannote", "pyannote 3.0"),
            ]),
            ("option", "none", "None"),
        ]

    def test_the_closed_enhancer_menu_shows_the_entry_not_the_section(self):
        js = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert "const shown = this._short(optText);" in js
        assert "VAD_LABEL: 'For VAD only'," in js

    def test_the_new_texts_have_chinese_or_are_kept(self):
        """The full check is test_gui_i18n_zh.py; this names the menu texts W598/W599 brought in."""
        zh = (ASSETS / "i18n_zh.js").read_text(encoding="utf-8")
        assert '"Vocal Separation": "人声分离",' in zh
        assert '"Noise Suppression": "噪声抑制",' in zh
        assert '"FFmpeg Audio Filters...": "FFmpeg 音频滤镜...",' in zh
        for kept in ("DSP", "Mel-Band-RoFormer", "Mini-BS-RoFormer", "Sortformer (NVIDIA Nemotron)", "pyannote 3.0"):
            assert f'  "{kept}",' in zh, kept


# ---------------------------------------------------------------------------
# Owner W602: the step-7 findings he decided (2 b, 3, 4 a, 5)
# ---------------------------------------------------------------------------

class TestOldPresetsLoadTicked:
    """W602 2: a preset saved before 1.10.2 holds the old default (false); it is read as not set."""

    def test_the_preset_version_moved_on(self):
        from whisperjav.settings.presets import PRESET_SCHEMA_VERSION
        assert PRESET_SCHEMA_VERSION == "1.1.0"

    def test_the_gui_reads_an_old_false_as_not_set(self):
        js = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert "const savedBefore1102 = !preset.schemaVersion || preset.schemaVersion === '1.0.0';" in js
        assert "passState.enhanceForVad = (savedBefore1102 ? undefined : preset.enhanceForVad) ?? true;" in js

    def test_the_version_reaches_the_gui_under_that_name(self):
        from whisperjav.webview_gui.api import WhisperJAVAPI
        assert WhisperJAVAPI._PRESET_MAP["schema_version"] == "schemaVersion"


class TestEnhancerModelAtStartUp:
    """W602 3: Mel-Band-RoFormer's and Mini-BS-RoFormer's model files are fetched before any audio is read."""

    @staticmethod
    def _ready(monkeypatch):
        from whisperjav.modules.speech_enhancement.factory import SpeechEnhancerFactory
        monkeypatch.setattr(SpeechEnhancerFactory, "is_backend_available", staticmethod(lambda name: (True, "")))

    @pytest.mark.parametrize("value,display", [("mel-band-roformer", "Mel-Band-RoFormer"),
                                               ("mini-bs-roformer:vocals", "Mini-BS-RoFormer")])
    def test_a_failed_download_stops_the_run(self, value, display, monkeypatch, capsys):
        from whisperjav.crispasr import install
        from whisperjav.utils import preflight_check
        self._ready(monkeypatch)

        def _fail(model, offline=None):
            raise RuntimeError("no route to host")
        monkeypatch.setattr(install, "ensure_model", _fail)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        with pytest.raises(SystemExit) as excinfo:
            preflight_check.ensure_speech_enhancer_available(value)
        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert f"{display}'s model is not on this machine" in out
        assert "Or pick another speech enhancer for this pass." in out
        assert "HF_HUB_OFFLINE" not in out

    def test_offline_says_so(self, monkeypatch, capsys):
        from whisperjav.crispasr import install
        from whisperjav.utils import preflight_check
        self._ready(monkeypatch)

        def _fail(model, offline=None):
            raise OSError("offline")
        monkeypatch.setattr(install, "ensure_model", _fail)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with pytest.raises(SystemExit):
            preflight_check.ensure_speech_enhancer_available("mel-band-roformer")
        assert "Downloads are switched off for this run (HF_HUB_OFFLINE is set," in capsys.readouterr().out

    def test_a_fetched_model_passes_and_the_others_fetch_nothing(self, monkeypatch):
        from whisperjav.crispasr import install
        from whisperjav.utils import preflight_check
        self._ready(monkeypatch)
        fetched = []

        def _fetch(model, offline=None):
            fetched.append(model.name)
            return Path("model.gguf")
        monkeypatch.setattr(install, "ensure_model", _fetch)
        assert preflight_check.ensure_speech_enhancer_available("mini-bs-roformer") is True
        assert preflight_check.ensure_speech_enhancer_available("zipenhancer") is True
        assert preflight_check.ensure_speech_enhancer_available("htdemucs") is True
        assert fetched == ["mini-bs-roformer"]


class TestNoThresholdForCrispasrSegmenters:
    """W602 4: Sortformer and pyannote take no VAD threshold; not offered, and a given one gets a line."""

    @pytest.mark.parametrize("seg,name", [("sortformer", "Sortformer"), ("pyannote", "pyannote")])
    def test_the_line(self, seg, name):
        from whisperjav.config.segmenter_presets import vad_threshold_unused_line
        assert vad_threshold_unused_line(seg, "--pass2-vad-threshold") == (
            f"--pass2-vad-threshold is not used by {name}; CrispASR uses its own threshold (0.5).")

    @pytest.mark.parametrize("seg", ["ten", "firered-vad", "whisperseg", "silero-v6.2", None])
    def test_no_line_for_the_others(self, seg):
        from whisperjav.config.segmenter_presets import segmenter_accepts, vad_threshold_unused_line
        assert vad_threshold_unused_line(seg, "--vad-threshold") is None
        if seg:
            assert segmenter_accepts(seg, "threshold")

    def test_every_threshold_flag_is_checked(self):
        main = (REPO / "whisperjav" / "main.py").read_text(encoding="utf-8")
        assert 'vad_threshold_unused_line(_qwen_segmenter, "--qwen-vad-threshold")' in main
        assert 'resolved_config["params"]["speech_segmenter"].get("backend"), "--vad-threshold")' in main
        assert 'vad_threshold_unused_line(_seg, f"--pass{_n}-vad-threshold")' in main

    def test_the_command_line_says_it(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "dummy.mp4", "--mode", "fidelity", "--speech-segmenter",
             "sortformer", "--vad-threshold", "0.4", "--dump-params", str(tmp_path / "p.json")],
            cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        assert result.returncode == 0, (result.stdout + result.stderr)[-800:]
        assert ("--vad-threshold is not used by Sortformer; CrispASR uses its own threshold (0.5)."
                in result.stdout + result.stderr)

    @pytest.mark.parametrize("seg,offered", [("sortformer", False), ("pyannote", False), ("ten", True),
                                             ("firered-vad", True), ("whisperseg", True)])
    def test_the_customize_window(self, seg, offered):
        from whisperjav.webview_gui.api import WhisperJAVAPI
        result = WhisperJAVAPI().get_qwen_schema("balanced", "qwen3", seg)
        assert result["success"]
        assert ("vad_threshold" in result["schema"]["audio"]) is offered


class TestEnhancerTabWithoutSettings:
    """W602 5: the Customize window's Enhancer tab for the speech enhancers without settings."""

    @pytest.mark.parametrize("backend,display", [("mel-band-roformer", "Mel-Band-RoFormer"),
                                                 ("mini-bs-roformer", "Mini-BS-RoFormer"),
                                                 ("htdemucs", "htdemucs (Demucs v4)")])
    def test_the_tab(self, backend, display):
        from whisperjav.webview_gui.api import WhisperJAVAPI
        result = WhisperJAVAPI().get_enhancer_schema(backend)
        assert result["success"] and result["display_name"] == display and result["parameters"] == {}
        assert result["info_message"] == "This speech enhancer has no settings."

    def test_the_line_has_chinese(self):
        zh = (ASSETS / "i18n_zh.js").read_text(encoding="utf-8")
        assert '"This speech enhancer has no settings.": "此语音增强器没有可调设置。",' in zh
