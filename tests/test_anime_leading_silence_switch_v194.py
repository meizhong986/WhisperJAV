"""v1.9.4 (owner, 2026-10-06): an off switch for the anime-whisper lead-in silence (200 ms by default).

The value travels: CLI --qwen-leading-silence (--mode qwen) / ensemble qwen-params "leading_silence_ms" / GUI
Customize "Silence Before Each Window (ms)" -> QwenPipeline(anime_leading_silence_ms=...) -> AnimeWhisperGenerator
(leading_silence_ms=...). 0 turns it off. Qwen3-ASR does not use it.
"""
import inspect
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_single_default_and_pipeline_setting():
    from whisperjav.config.anime_whisper_vad import ANIME_WHISPER_LEADING_SILENCE_MS
    from whisperjav.pipelines.qwen_pipeline import QwenPipeline
    assert ANIME_WHISPER_LEADING_SILENCE_MS == 200
    assert inspect.signature(QwenPipeline.__init__).parameters["anime_leading_silence_ms"].default == 200
    src = (REPO / "whisperjav/pipelines/qwen_pipeline.py").read_text(encoding="utf-8")
    assert "leading_silence_ms=self.anime_leading_silence_ms," in src


def test_ensemble_and_gui_key_reaches_the_pipeline():
    src = (REPO / "whisperjav/ensemble/pass_worker.py").read_text(encoding="utf-8")
    assert '"leading_silence_ms": "qwen_leading_silence_ms"' in src
    assert 'qwen_pipeline_params["anime_leading_silence_ms"]' in src


def test_cli_flag_in_help():
    out = subprocess.run([sys.executable, "-m", "whisperjav.main", "--help"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", cwd=str(REPO), timeout=300)
    assert out.returncode == 0
    assert "--qwen-leading-silence MS" in out.stdout
    src = (REPO / "whisperjav/main.py").read_text(encoding="utf-8")
    assert '"anime_leading_silence_ms": args.qwen_leading_silence' in src


def test_gui_dialog_offers_it_for_anime_whisper_only():
    from whisperjav.webview_gui.api import WhisperJAVAPI
    api = WhisperJAVAPI.__new__(WhisperJAVAPI)
    gen = WhisperJAVAPI.get_qwen_schema(api, "aggressive", "anime-whisper")["schema"]["generation"]
    assert gen["leading_silence_ms"]["default"] == 200 and gen["leading_silence_ms"]["min"] == 0
    assert "leading_silence_ms" not in WhisperJAVAPI.get_qwen_schema(api, "balanced", "qwen3")["schema"]["generation"]
    js = (REPO / "whisperjav/webview_gui/assets/app.js").read_text(encoding="utf-8")
    assert "'leading_silence_ms', lsDef.label" in js                       # drawn when present
    assert "defaults.leading_silence_ms = ls.default" in js                 # Reset restores it
