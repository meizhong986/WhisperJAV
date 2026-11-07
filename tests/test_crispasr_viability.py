"""The CrispASR viability suite's helpers on made-up data (no CrispASR call, no model, no GPU)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("crispasr_viability", ROOT / "tools" / "crispasr_viability.py")
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


def test_data_file_entries_are_complete():
    cands = json.loads((ROOT / "tools" / "crispasr_viability.json").read_text(encoding="utf-8"))["candidates"]
    assert "baseline" in cands
    for name, c in cands.items():
        assert c["kind"] in ("wj", "crisp-separator", "crisp-s2s", "crisp-vad", "wj-enhancer-swap",
                             "wj-segmenter-swap"), name
        if c.get("same_file_as"):
            assert cands[c["same_file_as"]].get("repo"), name
        elif c["kind"].startswith("crisp") and not c.get("path"):
            assert c.get("repo") and c.get("file") and len(c.get("revision", "")) == 40, name
        if c.get("twin"):
            assert c["twin"] in cands, name
        if c["kind"] == "crisp-vad":
            assert c.get("twin_segmenter"), name


def test_pad_spans_pads_clamps_and_meets_in_the_gap():
    # 1.0-2.0 and 2.1-3.0 padded by 0.1 / 0.2 would overlap (2.2 > 2.0): they meet at the middle of 2.0-2.1
    out = cv.pad_spans([(2.1, 3.0), (1.0, 2.0)], 0.1, 0.2, 3.1)
    assert out[0][0] == 0.9 and abs(out[0][1] - 2.05) < 1e-9
    assert abs(out[1][0] - 2.05) < 1e-9 and abs(out[1][1] - 3.1) < 1e-9   # clamped to the audio
    assert cv.pad_spans([(0.0, 0.5)], 0.3, 0.0, 10.0) == [(0.0, 0.5)]       # clamped at 0


def test_output_rate_from_lengths():
    assert cv.output_rate(16000, 16000, 48000) == 48000
    assert cv.output_rate(16000, 16000, 47990) == 48000
    assert cv.output_rate(0, 16000, 5) == 16000


def test_vad_settings_twin_and_own():
    twin = {"threshold": 0.3, "min_speech_duration_ms": 150, "min_silence_duration_ms": 150, "start_pad_ms": 50,
            "end_pad_ms": 100, "max_speech_duration_s": 6.0, "max_group_duration_s": 6.0, "chunk_threshold_s": 1.0}
    s = cv.crisp_vad_settings({"settings": "twin"}, twin)
    assert (s["threshold"], s["min_speech_ms"], s["start_pad_ms"], s["end_pad_ms"]) == (0.3, 150, 50, 100)
    assert (s["max_speech_s"], s["max_group_duration_s"], s["chunk_threshold_s"]) == (6.0, 6.0, 1.0)
    silero = {"threshold": 0.12, "speech_pad_ms": 150, "max_speech_duration_s": 6.0, "max_group_duration_s": 6.0,
              "chunk_threshold_s": 2.0}
    s = cv.crisp_vad_settings({"settings": "twin"}, silero)
    assert s["start_pad_ms"] == s["end_pad_ms"] == 150 and s["chunk_threshold_s"] == 2.0
    s = cv.crisp_vad_settings({"settings": "twin", "model_threshold": True}, twin)
    assert s["threshold"] == 0.0
    s = cv.crisp_vad_settings({"settings": "own"}, twin)
    assert (s["threshold"], s["min_speech_ms"], s["min_silence_ms"], s["start_pad_ms"]) == (0.0, 250, 100, 30)
    assert s["max_group_duration_s"] == 6.0


def test_verdict_rule():
    base = {"pooled": {"cer": 0.400, "missing_rate": 0.10}}
    assert cv.verdict({"pooled": {"cer": 0.390, "missing_rate": 0.10}}, base, 0.005) == "better"
    assert cv.verdict({"pooled": {"cer": 0.404, "missing_rate": 0.104}}, base, 0.005) == "keeps quality"
    assert cv.verdict({"pooled": {"cer": 0.404, "missing_rate": 0.110}}, base, 0.005) == "worse"
    assert cv.verdict({"pooled": {"cer": 0.410, "missing_rate": 0.10}}, base, 0.005) == "worse"
    assert cv.verdict({"pooled": {}}, base, 0.005) == "not scored"


def test_same_srt(tmp_path):
    a, b = tmp_path / "a.srt", tmp_path / "b.srt"
    a.write_text("1\n00:00:01,000 --> 00:00:02,000\nあ\n\n", encoding="utf-8")
    b.write_text("1\n00:00:01,000 --> 00:00:02,000\nあ\n\n", encoding="utf-8")
    assert cv.same_srt(a, b)[0]
    b.write_text("1\n00:00:01,000 --> 00:00:02,002\nあ\n\n", encoding="utf-8")
    assert not cv.same_srt(a, b)[0]


def test_chain_settings():
    assert cv.chain_settings("crisp-x", {"kind": "crisp-separator"}, None) == ("crisp-x", "ten", False)
    assert cv.chain_settings("crisp-x", {"kind": "crisp-separator"}, "vad") == ("crisp-x", "ten", True)
    assert cv.chain_settings("crisp-v", {"kind": "crisp-vad", "twin_segmenter": "firered-vad"}, None) == \
        ("none", "firered-vad", False)
    assert cv.chain_settings("wj-a", {"kind": "wj", "enhancer": "htdemucs"}, None) == ("htdemucs", "ten", False)
