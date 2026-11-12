"""1.10.2 guide points (c) and (d): two GUI labels said what is not so.

(c) "Smart Merge (Best Quality)": no measurement shows it gives the best subtitles; now "Smart Merge".
(d) anime-whisper's Text Cleaner showed "Passthrough (no cleaning)", but anime-whisper always runs its own text cleaner
    (qwen_pipeline.py picks the cleaner by model); the menu now names that, and the value sent is unchanged.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "whisperjav" / "webview_gui" / "assets"


def _schema(backend):
    from whisperjav.webview_gui.api import WhisperJAVAPI
    api = WhisperJAVAPI.__new__(WhisperJAVAPI)
    return WhisperJAVAPI.get_qwen_schema(api, "balanced", backend)["schema"]


def test_anime_whisper_text_cleaner_names_what_runs():
    ac = _schema("anime-whisper")["alignment"]["assembly_cleaner"]
    assert ac["options"] == [{"value": "passthrough", "label": "anime-whisper's own text cleaner"}]
    assert ac["default"] == "passthrough"   # what the GUI sends: unchanged


def test_qwen3_text_cleaner_menu_unchanged():
    ac = _schema("qwen3")["alignment"]["assembly_cleaner"]
    assert [o["value"] for o in ac["options"]] == ["qwen3", "passthrough"]
    assert ac["default"] == "qwen3"


def test_anime_whisper_pipeline_still_picks_its_own_cleaner():
    src = (REPO / "whisperjav" / "pipelines" / "qwen_pipeline.py").read_text(encoding="utf-8")
    assert 'if self.generator_backend == "anime-whisper":\n            cleaner = TextCleanerFactory.create("anime-whisper")' in src


def test_smart_merge_label_without_quality_claim():
    from whisperjav.ensemble.merge import get_available_strategies
    smart = next(s for s in get_available_strategies() if s["name"] == "smart_merge")
    assert smart["label"] == "Smart Merge"
    for f in ("index.html", "i18n_zh.js"):
        text = (ASSETS / f).read_text(encoding="utf-8")
        assert "Best Quality" not in text and "质量最佳" not in text
    zh = (ASSETS / "i18n_zh.js").read_text(encoding="utf-8")
    assert '"anime-whisper\'s own text cleaner": "anime-whisper 自带的文本清理器"' in zh
