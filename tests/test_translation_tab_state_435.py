"""#435: translation choices of the Ensemble row and the AI SRT Translate tab were kept only in browser storage, which
the GUI's private_mode clears at every launch. They are now saved in the translate settings file under
"gui_tab_state". The settings path is redirected to a temporary folder; the GUI is not started."""
import json
from pathlib import Path

import pytest

import whisperjav.translate.settings as ts


@pytest.fixture
def api(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(ts, "get_settings_path", lambda: path)
    from whisperjav.webview_gui.api import WhisperJAVAPI
    return WhisperJAVAPI.__new__(WhisperJAVAPI), path


def test_round_trip_keeps_only_the_listed_fields(api):
    a, path = api
    r = a.save_translation_tab_state("srt", {"provider": "custom", "customModel": "oc/deepseek-flash/none",
                                             "customEndpoint": "http://127.0.0.1:8090/v1", "targetLang": "chinese",
                                             "maxBatchSize": 20, "apiKey": "secret", "movieTitle": "x"})
    assert r["success"]
    a.save_translation_tab_state("ensemble", {"provider": "gemini", "model": "gemini-3.6-flash", "modelOverride": ""})
    state = a.get_translation_tab_state()["state"]
    assert state["srt"] == {"provider": "custom", "customModel": "oc/deepseek-flash/none",
                            "customEndpoint": "http://127.0.0.1:8090/v1", "targetLang": "chinese", "maxBatchSize": "20"}
    assert state["ensemble"] == {"provider": "gemini", "model": "gemini-3.6-flash"}
    assert "secret" not in path.read_text(encoding="utf-8")


def test_other_settings_in_the_file_are_kept(api):
    a, path = api
    a.save_translation_settings({"apiKey": "k", "tone": "contextual"})
    a.save_translation_tab_state("ensemble", {"provider": "deepseek"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["tone"] == "contextual" and data["gui_tab_state"]["ensemble"] == {"provider": "deepseek"}


def test_unknown_tab_is_refused(api):
    a, _ = api
    assert not a.save_translation_tab_state("tab9", {"provider": "x"})["success"]


def test_gui_restores_from_the_backend_and_saves_on_change():
    js = (Path(__file__).resolve().parents[1] / "whisperjav" / "webview_gui" / "assets" / "app.js").read_text(
        encoding="utf-8")
    assert "pywebview.api.get_translation_tab_state()" in js
    assert "pywebview.api.save_translation_tab_state(tab, this.collect(tab))" in js
    assert "TranslationTabState.apply(tabState);" in js and "TranslationTabState.bind();" in js
