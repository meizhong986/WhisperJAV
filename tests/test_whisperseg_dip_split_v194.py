"""v1.9.4 (owner, 2026-10-06): WhisperSeg hysteresis forced split, two settings.

dip_search_from (default 0.6) and dip_accept_any (default False) keep the old rule unless set: search the last
40 % of max_speech_duration_s for the lowest point, take it only if it is below 0.85 x the window's mean, else cut
exactly at the limit. anime-whisper conservative/balanced set 0.3 and True. Measured: docs/measurements/v1.9.4,
section 2.10. No model is loaded: the decoder is driven with made-up probabilities.
"""
import numpy as np
import pytest

from whisperjav.modules.speech_segmentation.backends.whisperseg import WhisperSegSpeechSegmenter

FRAME_MS = 20


def _segmenter(**kw):
    seg = WhisperSegSpeechSegmenter(threshold=0.25, max_speech_duration_s=4.0, min_speech_duration_ms=100,
                                    min_silence_duration_ms=100, start_pad_ms=0, end_pad_ms=0, **kw)
    seg._frame_duration_ms = FRAME_MS
    return seg


def _speech_with_gentle_dip():
    """6 s of steady speech (0.9) with a gentle dip to 0.80 at 1.6 s: below the mean, but not below 0.85 x mean."""
    p = np.full(int(6000 / FRAME_MS), 0.9, dtype=np.float32)
    i = int(1600 / FRAME_MS)
    p[i - 2:i + 3] = 0.80
    return p


def _first_cut(seg, probs):
    segments = seg._probs_to_segments(probs, len(probs) * FRAME_MS / 1000)
    return round(segments[0].end_sec, 2)


def test_default_rule_is_unchanged_and_cuts_at_the_limit():
    seg = _segmenter()
    assert seg.dip_search_from == 0.6 and seg.dip_accept_any is False
    assert _first_cut(seg, _speech_with_gentle_dip()) == pytest.approx(4.0, abs=0.05)


def test_v194_anime_rule_cuts_at_the_dip():
    seg = _segmenter(dip_search_from=0.3, dip_accept_any=True)
    assert _first_cut(seg, _speech_with_gentle_dip()) == pytest.approx(1.6, abs=0.06)


def test_search_start_alone_does_not_accept_a_shallow_dip():
    seg = _segmenter(dip_search_from=0.3)
    assert _first_cut(seg, _speech_with_gentle_dip()) == pytest.approx(4.0, abs=0.05)


def test_parameters_are_reported():
    params = _segmenter(dip_search_from=0.3, dip_accept_any=True)._get_parameters()
    assert params["dip_search_from"] == 0.3 and params["dip_accept_any"] is True


def test_settings_reach_the_segmenter_through_the_anime_table():
    from whisperjav.config.chronosjav_vad import resolve_chronosjav_segmenter_config
    for sens in ("conservative", "balanced"):
        cfg = resolve_chronosjav_segmenter_config("anime-whisper", "whisperseg", sens)
        assert cfg["dip_search_from"] == 0.3 and cfg["dip_accept_any"] is True
    agg = resolve_chronosjav_segmenter_config("anime-whisper", "whisperseg", "aggressive")
    assert "dip_search_from" not in agg            # offline decoder: not used
    for sens in ("conservative", "balanced", "aggressive"):
        q = resolve_chronosjav_segmenter_config("qwen3", "whisperseg", sens)
        assert "dip_search_from" not in q and "dip_accept_any" not in q   # Qwen3-ASR keeps the defaults


def test_anime_whisper_leading_silence():
    from pathlib import Path
    from whisperjav.modules.subtitle_pipeline.generators.anime_whisper import AnimeWhisperGenerator
    assert AnimeWhisperGenerator()._config["leading_silence_ms"] == 0          # off unless asked for
    src = (Path(__file__).resolve().parents[1] / "whisperjav/pipelines/qwen_pipeline.py").read_text(encoding="utf-8")
    assert "leading_silence_ms=200," in src                                       # ChronosJAV asks for 200 ms
    gen = AnimeWhisperGenerator(leading_silence_ms=200)
    out = gen.with_leading_silence(np.ones(1600, dtype=np.float32))
    assert len(out) == 1600 + 3200 and not out[:3200].any() and out[3200:].all() and out.dtype == np.float32
    off = AnimeWhisperGenerator(leading_silence_ms=0)
    a = np.ones(10, dtype=np.float32)
    assert off.with_leading_silence(a) is a
