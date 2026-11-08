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
        assert c["kind"] in ("wj", "crisp-separator", "crisp-s2s", "crisp-vad", "crisp-diar", "wj-enhancer-swap",
                             "wj-segmenter-swap"), name
        if c.get("same_file_as"):
            assert cands[c["same_file_as"]].get("repo"), name
        elif c["kind"].startswith("crisp") and not c.get("path"):
            assert c.get("repo") and c.get("file") and len(c.get("revision", "")) == 40, name
        if c.get("twin"):
            assert c["twin"] in cands, name
        if c["kind"] == "crisp-vad":
            assert c.get("twin_segmenter"), name
        if c["kind"] == "crisp-diar":
            assert c.get("method") in ("sortformer", "pyannote", "ten-frames") and c.get("twin_segmenter") == "ten", name


def test_union_spans_joins_overlapping_speaker_turns():
    # two speakers' turns overlap (1.0-2.0 with 1.5-3.0); a touching span joins; a separate one stays
    assert cv.union_spans([(4.0, 5.0), (1.5, 3.0), (1.0, 2.0), (3.0, 3.5), (6.0, 6.0)]) == [(1.0, 3.5), (4.0, 5.0)]
    assert cv.union_spans([]) == []


def test_frame_runs_and_spans_to_flags_round_trip():
    flags = [0, 1, 1, 0, 0, 1, 0, 1]
    spans = cv.frame_runs(flags, 0.016)
    assert [(round(a, 3), round(b, 3)) for a, b in spans] == [(0.016, 0.048), (0.08, 0.096), (0.112, 0.128)]
    assert cv.spans_to_flags(spans, len(flags), 0.016) == flags


def test_spans_to_flags_marks_frames_whose_middle_is_inside():
    # 16 ms frames: middles at 8, 24, 40, 56 ms; the span 20-45 ms holds the middles of frames 1 and 2
    assert cv.spans_to_flags([(0.020, 0.045)], 4, 0.016) == [0, 1, 1, 0]
    assert cv.spans_to_flags([(0.0, 1.0)], 3, 0.016) == [1, 1, 1]       # clamped to the grid
    assert cv.spans_to_flags([], 2, 0.016) == [0, 0]


def test_diar_segmenter_uses_ten_post_processing():
    """The method's spans go through TEN's own merge, pads, cut and grouping, with TEN's settings."""
    import numpy as np
    from whisperjav.modules.speech_segmentation.backends.ten import TenSpeechSegmenter
    ten = TenSpeechSegmenter(min_speech_duration_ms=100, min_silence_duration_ms=100, max_speech_duration_s=6.0,
                             chunk_threshold_s=1.0, max_group_duration_s=6.0, start_pad_ms=0, end_pad_ms=100)
    seg = cv.CrispDiarSegmenter("x", {"method": "sortformer"}, "unused", ten)
    # 1.00-2.00 and 2.05-3.00: the 50 ms gap is under 100 ms, so they merge; 3.50-3.55 is under 100 ms, dropped;
    # 5.0-14.0 is longer than 6 s and is cut
    seg.speech_spans = lambda pcm, duration: ([(1.0, 2.0), (2.05, 3.0), (3.5, 3.55), (5.0, 14.0)], 4)
    r = seg.segment(np.zeros(16 * 16000, dtype=np.float32), sample_rate=16000)
    assert r.error is None
    spans = [(round(s.start_sec, 2), round(s.end_sec, 2)) for s in r.segments]
    assert abs(spans[0][0] - 1.0) < 0.02 and abs(spans[0][1] - 3.1) < 0.02         # merged, end pad 100 ms
    assert all(not (3.4 < a < 3.6) for a, _ in spans)                              # the short one is gone
    assert all(b - a <= 6.0 + 1e-6 for a, b in spans)                              # every piece at most 6 s
    assert len(r.groups) >= 3 and seg.record["turns"] == 4 and seg.record["scenes"] == 1


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


def test_stage_record_adds_up_across_files():
    rec = {}
    a = cv.CrispEnhancer("crisp-x", {"kind": "crisp-separator"}, "x.gguf", record=rec)
    a.record["scenes"] += 3
    a.record["failed"] += 1
    cv.CrispEnhancer("crisp-x", {"kind": "crisp-separator"}, "x.gguf", record=rec)   # the next file's enhancer
    assert rec["scenes"] == 3 and rec["failed"] == 1
    seg = {}
    s = cv.CrispSegmenter("crisp-v", {"settings": "own"}, "v.gguf", record=seg, max_group_duration_s=6.0)
    s.record["scenes"] += 2
    cv.CrispSegmenter("crisp-v", {"settings": "own"}, "v.gguf", record=seg, max_group_duration_s=6.0)
    assert seg["scenes"] == 2 and seg["settings"]["max_group_duration_s"] == 6.0


def test_chain_settings():
    assert cv.chain_settings("crisp-x", {"kind": "crisp-separator"}, None) == ("crisp-x", "ten", False)
    assert cv.chain_settings("crisp-x", {"kind": "crisp-separator"}, "vad") == ("crisp-x", "ten", True)
    assert cv.chain_settings("crisp-v", {"kind": "crisp-vad", "twin_segmenter": "firered-vad"}, None) == \
        ("none", "firered-vad", False)
    assert cv.chain_settings("wj-a", {"kind": "wj", "enhancer": "htdemucs"}, None) == ("htdemucs", "ten", False)
