"""MOSS-CrispASR built after the owner's reference script (1.10.1; owner W425, W426, W428).

The stretches (the shortest-group setting of the shared grouping), the reference's instruction, the "segments" line
rule, the engine's set_ask and token limit, the segmenter schema files, the GUI back end and the command line.
No ASR import, no model.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

from whisperjav.crispasr import lines, moss_prompt  # noqa: E402
from whisperjav.crispasr.engine import BindingEngine, CrispSegment  # noqa: E402
from whisperjav.crispasr.moss_prompt import PROMPT_EN, PROMPT_ZH, MossSettings  # noqa: E402
from whisperjav.modules.speech_segmentation.base import SpeechSegment  # noqa: E402
from whisperjav.modules.speech_segmentation.backends.ten import group_segments, join_short_groups  # noqa: E402


def S(a, b):
    return SpeechSegment(start_sec=a, end_sec=b)


def spans(groups):
    return [(g[0].start_sec, g[-1].end_sec) for g in groups]


class TestShortestGroup:
    """Owner W426 3: a group shorter than the shortest joins the next while both fit under the longest."""

    SEGS = [S(0, 10), S(20, 30), S(40, 50), S(60, 100), S(110, 115), S(130, 200), S(300, 310)]

    def test_off_by_default(self):
        assert spans(group_segments(self.SEGS, 120, 5)) == spans(group_segments(self.SEGS, 120, 5, None)) \
            == spans(group_segments(self.SEGS, 120, 5, 0)) == [(s.start_sec, s.end_sec) for s in self.SEGS]

    def test_short_groups_join_the_next_while_both_fit(self):
        # 0-10 + 20-30 + 40-50 + 60-100 = 0-100 (100 s <= 120); 110-115 joins 130-200 (90 s); 300-310 is last
        assert spans(group_segments(self.SEGS, 120, 5, 70)) == [(0, 100), (110, 200), (300, 310)]

    def test_a_join_that_would_pass_the_longest_is_not_made(self):
        segs = [S(0, 30), S(50, 160)]
        assert spans(group_segments(segs, 120, 5, 70)) == [(0, 30), (50, 160)]

    def test_no_speech_is_dropped_or_moved(self):
        groups = group_segments(self.SEGS, 120, 5, 70)
        assert [s for g in groups for s in g] == self.SEGS

    def test_join_short_groups_alone(self):
        assert spans(join_short_groups([[S(0, 5)], [S(8, 20)], [S(25, 26)]], 10, 30)) == [(0, 20), (25, 26)]

    @pytest.mark.parametrize("backend", ["silero-v6.2", "ten", "firered-vad", "whisperseg"])
    def test_the_factory_hands_it_to_each_segmenter_that_uses_the_shared_grouping(self, backend):
        from whisperjav.modules.speech_segmentation.factory import _PARAM_SCHEMAS
        from whisperjav.config.segmenter_presets import SEGMENTER_PARAMS
        assert "min_group_duration_s" in _PARAM_SCHEMAS[backend]
        assert "min_group_duration_s" in SEGMENTER_PARAMS
        source = {"silero-v6.2": "silero_v6.py", "ten": "ten.py", "firered-vad": "firered_vad.py",
                  "whisperseg": "whisperseg.py"}[backend]
        text = (REPO / "whisperjav" / "modules" / "speech_segmentation" / "backends" / source).read_text(
            encoding="utf-8")
        assert 'self.min_group_duration_s = float(kwargs.get("min_group_duration_s") or 0.0)' in text
        assert '"min_group_duration_s": self.min_group_duration_s,' in text


class TestSchemaFiles:
    """The Segmenter tab is drawn from these files: the setting is shown, and the sliders reach MOSS's values."""

    TOOLS = REPO / "whisperjav" / "config" / "v4" / "ecosystems" / "tools"

    @pytest.mark.parametrize("name", ["silero-scene-detection.yaml", "auditok-scene-detection.yaml"])
    def test_the_scene_sliders_reach_moss_s_420_s(self, name):
        # review finding 1: a range slider clamps to its max; Apply then sent 120 s instead of 420 s
        doc = yaml.safe_load((self.TOOLS / name).read_text(encoding="utf-8"))
        assert doc["gui"]["max_duration_s"]["max"] >= 420

    def test_no_moss_customize_value_is_outside_its_control(self):
        from whisperjav.webview_gui.api import WhisperJAVAPI
        api = object.__new__(WhisperJAVAPI)
        for sensitivity in ("conservative", "balanced", "aggressive"):
            r = api.get_crispasr_customize_defaults("moss-crispasr", sensitivity)
            values = {**r["segmenter"], **r["scene"]}
            schemas = {**api.get_segmenter_schema(r["segmenter_backend"])["parameters"],
                       **api.get_scene_detector_schema(r["scene_detection_method"])["parameters"]}
            for key, schema in schemas.items():
                v = values.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    assert schema.get("min", v) <= v <= schema.get("max", v), (sensitivity, key, v)

    @pytest.mark.parametrize("name", ["silero-v6-speech-segmentation.yaml", "ten-speech-segmentation.yaml",
                                      "firered-vad-speech-segmentation.yaml", "whisperseg-speech-segmentation.yaml"])
    def test_each_has_the_shortest_group_off_and_room_for_moss(self, name):
        doc = yaml.safe_load((self.TOOLS / name).read_text(encoding="utf-8"))
        assert doc["spec"]["min_group_duration_s"] == 0
        gui = doc["gui"]
        assert gui["min_group_duration_s"]["label"] == "Shortest Group (s)"
        assert gui["min_group_duration_s"]["group"] == "grouping"
        assert gui["max_group_duration_s"]["max"] >= 240     # MOSS's longest group, W482 (a slider clamps to its max)
        assert gui["chunk_threshold_s"]["max"] >= 5.0        # MOSS's 5 s of silence
        assert gui["min_group_duration_s"]["min"] == 0       # MOSS has no shortest group (W482): 0 is in range


class TestReferenceInstruction:
    """The reference's build_prompt (C:\\BIN\\MyScripts\\Python\\moss_transcribe_diarize_hybrid_v3_grok.py)."""

    def test_jav_optimized_with_a_language_hint(self):
        assert MossSettings().prompt() == (
            "Transcribe the audio. For each segment, start with the timestamp and speaker ID "
            "([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp."
            " Language hint: Japanese.")

    def test_hotwords(self):
        s = MossSettings.from_params({"hotwords": " 監督, 村西 "})
        assert s.prompt() == PROMPT_EN + " Language hint: Japanese. Hotwords: 監督, 村西"

    def test_chinese(self):
        s = MossSettings.from_params({"prompt_language": "zh", "hotwords": "監督"})
        assert s.prompt() == PROMPT_ZH + " 语言提示：Japanese 热词提示：監督"

    def test_no_hint_no_hotwords(self):
        assert MossSettings.from_params({"language_hint": ""}).prompt() == PROMPT_EN

    def test_the_reference_scripts_own_build_prompt_gives_the_same_text(self):
        script = Path(r"C:\BIN\MyScripts\Python\moss_transcribe_diarize_hybrid_v3_grok.py")
        if not script.is_file():
            pytest.skip("the owner's reference script is not on this computer")
        import ast
        tree = ast.parse(script.read_text(encoding="utf-8"))
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_prompt")
        consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name) and t.id == "DEFAULT_PROMPT_EN"}
        namespace = {"DEFAULT_PROMPT_EN": consts["DEFAULT_PROMPT_EN"], "DEFAULT_PROMPT_ZH": PROMPT_ZH,
                     "ProcessingConfig": object}   # the annotation of build_prompt's argument
        exec(compile(ast.Module([func], []), str(script), "exec"), namespace)

        class Cfg:
            prompt = None
            prompt_lang = "en"
            language_hint = "Japanese"
            hotwords = "監督, 村西"
        for lang in ("en", "zh"):
            Cfg.prompt_lang = lang
            ours = MossSettings.from_params({"prompt_language": lang, "hotwords": "監督, 村西"}).prompt()
            assert namespace["build_prompt"](Cfg) == ours

    @pytest.mark.parametrize("params, key", [({"prompt_language": "fr"}, "prompt_language"),
                                             ({"max_new_tokens": "many"}, "max_new_tokens"),
                                             ({"max_new_tokens": -5}, "max_new_tokens")])
    def test_a_value_it_cannot_use_is_refused(self, params, key):
        with pytest.raises(ValueError, match=key):
            MossSettings.from_params(params)

    def test_other_keys_are_ignored(self):
        assert MossSettings.from_params({"threshold": 0.5, "max_speech_duration_s": 5}) == MossSettings()

    # 1.10.1 (owner W492 M1, W501, W502): the decoding preset and beam size
    def test_the_decoding_preset(self):
        assert MossSettings().decoding() == {"do_sample": True, "temperature": 0.7, "top_k": 50, "seed": 42,
                                             "beam_size": 1}
        assert moss_prompt.DEFAULTS["do_sample"] is True and moss_prompt.DEFAULTS["seed"] == 42

    def test_decoding_values_from_the_tab(self):
        s = MossSettings.from_params({"do_sample": False, "temperature": "0.5", "top_k": 0, "seed": 0,
                                      "beam_size": 4})
        assert s.decoding() == {"do_sample": False, "temperature": 0.5, "top_k": 0, "seed": 0, "beam_size": 4}
        assert MossSettings.from_params({"do_sample": "false"}).do_sample is False
        # temperature counts only with sampling on (review piece 5, finding 6)
        assert MossSettings.from_params({"do_sample": False, "temperature": 0}).do_sample is False
        # the GUI's shared limits hold temperature at 1.0 at most on Apply (app.js), so the tab does too
        assert moss_prompt.GUI_SCHEMA["temperature"]["max"] == moss_prompt.MAX_TEMPERATURE == 1.0
        assert set(moss_prompt.DECODING) <= set(s.as_dict())

    @pytest.mark.parametrize("params,key", [({"temperature": 0}, "temperature"), ({"temperature": 1.2}, "temperature"),
                                            ({"temperature": True}, "temperature"),
                                            ({"top_k": -1}, "top_k"), ({"seed": -1}, "seed"),
                                            ({"seed": 2 ** 31}, "seed"), ({"beam_size": 0}, "beam_size"),
                                            ({"beam_size": 9}, "beam_size"), ({"beam_size": 2.5}, "beam_size"),
                                            ({"do_sample": "maybe"}, "do_sample")])
    def test_a_decoding_value_it_cannot_use_is_refused(self, params, key):
        with pytest.raises(ValueError, match=key):
            MossSettings.from_params(params)

    def test_the_decoding_controls(self):
        g = moss_prompt.GUI_SCHEMA
        assert g["do_sample"]["type"] == "toggle" and g["do_sample"]["label"] == "Sampling"
        assert (g["beam_size"]["min"], g["beam_size"]["max"]) == (1, moss_prompt.MAX_BEAM)
        for key in moss_prompt.DECODING:
            default = moss_prompt.DEFAULTS[key]
            if g[key]["type"] == "number":
                assert g[key]["min"] <= default <= g[key]["max"], key

    def test_the_gui_tab_has_one_control_per_setting(self):
        assert set(moss_prompt.GUI_SCHEMA) == set(moss_prompt.DEFAULTS) == set(moss_prompt.PARAMS)


class TestSegmentsRule:
    def test_each_segment_is_one_subtitle_at_its_own_times(self):
        segs = [CrispSegment("[S01]はい。", 0.5, 2.0), CrispSegment("[S02]行きます[3.4]", 2.5, 4.0)]
        assert lines.build("segments", 100.0, 110.0, segs) == [("はい。", 100.5, 102.0), ("行きます", 102.5, 104.0)]

    def test_empty_and_label_only_segments_give_nothing(self):
        segs = [CrispSegment("", 0.0, 1.0), CrispSegment("[S01]", 1.0, 2.0), CrispSegment("  ", 2.0, 3.0)]
        assert lines.build("segments", 0.0, 10.0, segs) == []

    def test_times_are_clamped_to_the_group_and_put_in_order(self):
        stats = lines.LineStats()
        segs = [CrispSegment("一", -1.0, 3.0), CrispSegment("二", 2.0, 50.0), CrispSegment("三", 1.0, 1.0)]
        subs = lines.build("segments", 10.0, 20.0, segs, stats)
        assert subs == [("一", 10.0, 11.0), ("三", 11.0, 11.1), ("二", 12.0, 20.0)]
        assert stats.clamped_times == 2

    def test_overlapping_segments_are_all_shown(self):
        # review finding 2: two speakers at once; cutting each to the next start would leave zero-length lines
        segs = [CrispSegment("A", 10.0, 14.0), CrispSegment("B", 10.0, 12.0), CrispSegment("C", 9.0, 9.5)]
        subs = lines.build("segments", 100.0, 120.0, segs)
        assert [t for t, _s, _e in subs] == ["C", "A", "B"]
        assert all(e - s >= 0.1 - 1e-9 for _t, s, e in subs)
        assert subs[0] == ("C", 109.0, 109.5)
        assert {subs[1][1], subs[2][1]} == {110.0}        # A and B start together and overlap, as written

    def test_a_raw_text_answer_is_split_as_the_reference_splits_it(self):
        # review finding 3: CrispASR returns the raw text as one segment over the whole call when its parser finds
        # nothing (no space allowed between "]" and "[", no newline inside a segment)
        raw = "[0.00] [S01]はい。\n行きます[3.20][3.50][S02]待って[6.0]"
        stats = lines.LineStats()
        subs = lines.build("segments", 100.0, 218.0, [CrispSegment(raw, 0.0, 118.0)], stats)
        assert subs == [("はい。\n行きます", 100.0, 103.2), ("待って", 103.5, 106.0)]
        assert stats.fallback_segments == 1

    def test_an_unclosed_last_segment_is_kept_to_the_end_of_the_call(self):
        raw = "[0.00][S01]はい[1.0][2.00][S01]まだ続く"
        subs = lines.build("segments", 0.0, 30.0, [CrispSegment(raw, 0.0, 30.0)])
        assert subs == [("はい", 0.0, 1.0), ("まだ続く", 2.0, 30.0)]

    def test_no_cut_by_characters_or_duration(self):
        long_text = "あ" * 300
        assert lines.build("segments", 0.0, 120.0, [CrispSegment(long_text, 0.0, 90.0)]) == [(long_text, 0.0, 90.0)]


class TestMossSegmenterValues:
    def test_the_silero_threshold_only_for_a_silero_segmenter(self):
        # review finding 10: 0.7 is a Silero speech probability; another segmenter keeps its own threshold
        from whisperjav.crispasr.pass_settings import segmenter_settings
        from whisperjav.config.segmenter_presets import resolve_segmenter_sensitivity
        s = segmenter_settings("moss-crispasr", "firered-vad", "balanced")
        assert s.config["threshold"] == resolve_segmenter_sensitivity("firered-vad", "balanced")["threshold"]
        assert (s.max_group_duration_s, s.chunk_threshold_s, s.config["min_group_duration_s"]) == (240.0, 5.0, 0.0)
        assert segmenter_settings("moss-crispasr", "silero-v6.2", "aggressive").config["threshold"] == 0.7


class TestEngine:
    class Session:
        def __init__(self):
            self.calls = []

        def set_max_new_tokens(self, n):
            self.calls.append(("tokens", n))

        def set_ask(self, text):
            self.calls.append(("ask", text))

        def transcribe(self, pcm, sample_rate, language):
            self.calls.append(("transcribe", len(pcm)))
            return []

    def engine(self, **kwargs):
        e = BindingEngine(Path("m.gguf"), "moss-diarize", True, **kwargs)
        e._session = self.Session()
        return e

    def test_the_instruction_and_the_limit_are_given_before_every_call(self):
        import numpy as np
        e = self.engine(tokens_per_second=moss_prompt.TOKENS_PER_SECOND, ask="ASK")
        e.transcribe(np.zeros(16000 * 120, dtype=np.float32))
        e.transcribe(np.zeros(16000 * 60, dtype=np.float32))
        assert e._session.calls == [("tokens", 2622), ("ask", "ASK"), ("transcribe", 16000 * 120),
                                    ("tokens", 1311), ("ask", "ASK"), ("transcribe", 16000 * 60)]

    def test_a_fixed_limit_wins_and_no_instruction_means_the_models_own(self):
        import numpy as np
        e = self.engine(tokens_per_second=moss_prompt.TOKENS_PER_SECOND, max_new_tokens=65536)
        e.transcribe(np.zeros(16000, dtype=np.float32))
        assert e._session.calls == [("tokens", 65536), ("transcribe", 16000)]

    class DecodingSession(Session):
        backend = "moss-diarize"

        def set_beam_size(self, n):
            self.calls.append(("beam", n))

        def set_temperature(self, t, seed=0):
            self.calls.append(("temperature", t, seed))

        def set_top_k(self, k):
            self.calls.append(("top_k", k))

        def set_do_sample(self, on):
            self.calls.append(("do_sample", on))

    def opened(self, monkeypatch, decoding):
        import types
        made = self.DecodingSession()
        fake = types.SimpleNamespace(Session=lambda *a, **k: made)
        monkeypatch.setitem(sys.modules, "crispasr", fake)
        monkeypatch.setattr("whisperjav.crispasr.engine.native_log.quiet_debug_lines", lambda lib: None)
        e = BindingEngine(Path("m.gguf"), "moss-diarize", True, decoding=decoding)
        e.open()
        return made.calls

    def test_the_preset_is_set_on_the_session_once_it_is_open(self, monkeypatch):
        assert self.opened(monkeypatch, MossSettings().decoding()) == [
            ("beam", 1), ("temperature", 0.7, 42), ("top_k", 50), ("do_sample", True)]

    def test_greedy_and_beam_search(self, monkeypatch):
        calls = self.opened(monkeypatch, MossSettings.from_params({"do_sample": False, "beam_size": 3}).decoding())
        assert calls == [("beam", 3), ("do_sample", False)]

    def test_no_decoding_leaves_the_session_as_it_is(self, monkeypatch):
        assert self.opened(monkeypatch, None) == []

    def test_a_failed_decoding_setup_closes_the_session(self, monkeypatch):
        import types
        from whisperjav.crispasr.errors import CrispASRSetupError

        class Broken(self.DecodingSession):
            closed = False

            def set_beam_size(self, n):
                raise RuntimeError("set_beam_size failed (rc=-1)")

            def close(self):
                Broken.closed = True

        made = Broken()
        monkeypatch.setitem(sys.modules, "crispasr", types.SimpleNamespace(Session=lambda *a, **k: made))
        monkeypatch.setattr("whisperjav.crispasr.engine.native_log.quiet_debug_lines", lambda lib: None)
        e = BindingEngine(Path("m.gguf"), "moss-diarize", True, decoding=MossSettings().decoding())
        with pytest.raises(CrispASRSetupError):
            e.open()
        assert Broken.closed and not e.is_open

    def test_the_pipeline_hands_moss_s_decoding_to_the_engine(self):
        src = (REPO / "whisperjav" / "pipelines" / "crispasr_pipeline.py").read_text(encoding="utf-8")
        assert "decoding=(settings.decoding() if settings is not None" in src


class TestGuiBackEnd:
    def test_moss_customize_values(self):
        from whisperjav.webview_gui.api import WhisperJAVAPI
        r = WhisperJAVAPI.get_crispasr_customize_defaults(object.__new__(WhisperJAVAPI), "moss-crispasr", "balanced")
        assert r["success"], r
        assert r["segmenter_backend"] == "firered-vad"   # W482
        seg = r["segmenter"]
        from whisperjav.config.segmenter_presets import resolve_segmenter_sensitivity
        assert (seg["threshold"], seg["chunk_threshold_s"], seg["max_group_duration_s"],
                seg["min_group_duration_s"]) == (resolve_segmenter_sensitivity("firered-vad", "balanced")["threshold"],
                                                 5.0, 240.0, 0.0)
        assert r["scene_detection_method"] == "semantic"
        assert r["scene"] == {"min_duration": 90, "max_duration": 240}
        assert r["model"]["title"] == "MOSS"
        assert r["model"]["defaults"] == moss_prompt.DEFAULTS
        assert r["model"]["schema"] == moss_prompt.GUI_SCHEMA

    def test_the_other_pipelines_have_no_model_tab_and_keep_their_scenes(self):
        from whisperjav.webview_gui.api import WhisperJAVAPI
        r = WhisperJAVAPI.get_crispasr_customize_defaults(object.__new__(WhisperJAVAPI), "cohere-crispasr", "balanced")
        assert r["model"] is None
        assert r["scene_detection_method"] == "semantic" and r["scene"] == {"min_duration": 12, "max_duration": 48}

    def test_app_js_draws_the_moss_tab_and_resets_it(self):
        js = (REPO / "whisperjav" / "webview_gui" / "assets" / "app.js").read_text(encoding="utf-8")
        assert "if (modelTab) this.generateCrispasrModelTab(modelTab, currentValues);" in js
        assert "const container = document.getElementById('tab-context');" in js
        assert "const tabs = ['model', 'quality', 'segmenter', 'enhancer', 'scene', 'context'];" in js
        # applyCustomization collects the context panel, so the MOSS values reach --passN-params
        assert "const tabs = ['model', 'quality', 'segmenter', 'enhancer', 'scene', 'context'];\n        tabs.forEach" \
            in js

    def test_chinese_texts(self):
        zh = (REPO / "whisperjav" / "webview_gui" / "assets" / "i18n_zh.js").read_text(encoding="utf-8")
        for schema in moss_prompt.GUI_SCHEMA.values():
            assert json.dumps(schema["label"], ensure_ascii=False) + ":" in zh
            assert json.dumps(schema["description"], ensure_ascii=False) + ":" in zh
        assert '"Shortest Group (s)":' in zh


class TestCommandLine:
    PY = sys.executable

    def run(self, *args):
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        return subprocess.run([self.PY, "-m", "whisperjav.main", *args], cwd=str(REPO), capture_output=True,
                              text=True, encoding="utf-8", env=env, timeout=600)

    def test_dump_params_shows_the_moss_settings_and_the_instruction(self, tmp_path):
        out = tmp_path / "dump.json"
        r = self.run("--dump-params", str(out), "--ensemble", "--pass1-pipeline", "moss-crispasr",
                     "--pass1-params", json.dumps({"hotwords": "監督"}, ensure_ascii=False), "x.mp4")
        assert r.returncode == 0, r.stderr[-2000:]
        eff = json.loads(out.read_text(encoding="utf-8"))["ensemble_config"]["pass1"]["crispasr_effective"]
        assert (eff["speech_segmenter"], eff["scene_detector"], eff["word_timestamps"]) == \
            ("firered-vad", "semantic", "model")   # W482
        assert eff["moss_settings"]["instruction"] == PROMPT_EN + " Language hint: Japanese. Hotwords: 監督"
        # the decoding preset (W492 M1, W502)
        assert {k: eff["moss_settings"][k] for k in moss_prompt.DECODING} == MossSettings().decoding()

    def test_dump_params_shows_the_pass_s_own_segmenter_values(self, tmp_path):
        # review finding 8: the run uses --passN-params' Segmenter values; the dump shows them too
        out = tmp_path / "dump.json"
        r = self.run("--dump-params", str(out), "--ensemble", "--pass1-pipeline", "moss-crispasr",
                     "--pass1-params", json.dumps({"min_group_duration_s": 30, "max_group_duration_s": 90}), "x.mp4")
        assert r.returncode == 0, r.stderr[-2000:]
        eff = json.loads(out.read_text(encoding="utf-8"))["ensemble_config"]["pass1"]["crispasr_effective"]
        s = eff["speech_segmenter_settings"]
        assert (s["min_group_duration_s"], s["max_group_duration_s"], s["chunk_threshold_s"]) == (30, 90, 5.0)

    def test_a_moss_setting_it_cannot_use_stops_the_run(self, tmp_path):
        r = self.run("--dump-params", str(tmp_path / "d.json"), "--ensemble", "--pass1-pipeline", "moss-crispasr",
                     "--pass1-params", json.dumps({"prompt_language": "fr"}), "x.mp4")
        assert r.returncode == 2
        assert "--pass1-params: prompt_language 'fr' is not one of: en, zh" in r.stderr
