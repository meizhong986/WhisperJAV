"""tools/crispasr_model_test.py: the checks on CrispASR's returned object and the helpers, on made-up results, without
CrispASR (owner W568, W569)."""
import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "crispasr_model_test.py"
spec = importlib.util.spec_from_file_location("crispasr_model_test", TOOL)
cmt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmt)


def seg(text, start, end, tokens=(), words=(), speaker=""):
    return {"text": text, "start": start, "end": end, "no_speech_prob": -1.0, "speaker": speaker,
            "words": [list(w) for w in words], "tokens": [list(t) for t in tokens]}


def tok(text, start, end, conf=0.9, raw=None):
    return [text, start, end, conf, (raw if raw is not None else text.encode("utf-8")).hex()]


def test_rebuild_token_text_joins_a_character_split_over_two_tokens():
    b = "あ".encode("utf-8")                       # three bytes
    tokens = [["�", 0.0, 0.1, 0.9, b[:2].hex()], ["�", 0.1, 0.2, 0.9, b[2:].hex()], tok("い", 0.2, 0.3)]
    assert cmt.rebuild_token_text(tokens) == "あい"


def test_rebuild_token_text_without_bytes_keeps_the_text():
    assert cmt.rebuild_token_text([["こん", 0, 0.1, 0.5], ["にちは", 0.1, 0.2, 0.5]]) == "こんにちは"


def test_call_checks_a_clean_result():
    rec = {"seconds": 3.0, "segments": [seg("こんにちは", 0.1, 1.0, [tok("こん", 0.1, 0.4), tok("にちは", 0.4, 1.0)])]}
    c = cmt.call_checks(rec)
    assert c["segments"] == 1 and c["tokens"] == 2 and c["tokens_start"] == 2 and c["tokens_end"] == 2
    assert c["seg_out_of_bounds"] == 0 and c["tokens_out_of_bounds"] == 0 and c["tokens_backwards"] == 0
    assert c["token_text_equal"] is True and c["text_chars"] == 5


def test_call_checks_finds_each_fault():
    rec = {"seconds": 2.0, "segments": [
        seg("あ", 1.0, 2.5, [tok("あ", 1.0, 2.5, conf=-1.0)]),            # ends after the audio
        # unordered, end before start, a token going back in time within the segment, text differs
        seg("い", 0.5, 0.4, [tok("う", 0.6, None), tok("え", 0.3, 0.35)]),
    ]}
    c = cmt.call_checks(rec)
    assert c["seg_out_of_bounds"] == 1 and c["seg_end_before_start"] == 1 and c["seg_unordered"] == 1
    assert c["tokens_out_of_bounds"] == 1 and c["tokens_backwards"] == 1 and c["token_conf_none"] == 1
    assert c["tokens_end"] == 2 and c["token_text_equal"] is False
    assert c["seg_overlap_max_s"] == pytest.approx(2.0)


def test_call_checks_an_error_counts_nothing():
    c = cmt.call_checks({"seconds": 1.0, "error": "RuntimeError: x"})
    assert c["error"] and c["segments"] == 0


def test_time_kind():
    assert cmt.time_kind(0, 0, 0) == "none (not returned)"
    assert cmt.time_kind(10, 10, 10) == "start and end"
    assert cmt.time_kind(10, 10, 0) == "start only"
    assert cmt.time_kind(10, 0, 0) == "none (no times)"
    assert cmt.time_kind(10, 5, 5).startswith("some")


@pytest.mark.parametrize("totals, spc, rule, onsets, aligner", [
    ({"tokens": 10, "tokens_start": 10, "tokens_end": 10}, 1, "tokens", False, None),
    ({"tokens": 10, "tokens_start": 10, "tokens_end": 0}, 1, "tokens", True, None),
    ({"tokens": 0, "words": 4, "words_start": 4, "words_end": 4}, 1, "tokens", False, None),
    ({"tokens": 3, "tokens_start": 0, "tokens_end": 0}, 5, "segments", False, None),
    ({"tokens": 0, "words": 1, "words_start": 0, "words_end": 0}, 1, "group", False, "wav2vec2-ja"),
])
def test_propose_row_follows_the_line_rules(totals, spc, rule, onsets, aligner):
    r = cmt.propose_row(totals, spc)
    assert (r["line_rule"], r["onsets"], r["aligner"]) == (rule, onsets, aligner)


def test_total_and_compare_records():
    a = {"seconds": 3.0, "segments": [seg("あい", 0, 1, [tok("あ", 0.0, 0.5), tok("い", 0.5, 1.0)])]}
    b = {"seconds": 3.0, "segments": [seg("あい", 0, 1, [tok("あ", 0.02, 0.5), tok("い", 0.5, 1.0)])]}
    t = cmt.total([cmt.call_checks(a), cmt.call_checks(b)])
    assert t["tokens"] == 4 and t["calls_failed"] == 0 and t["token_text_equal"] is True
    c = cmt.compare_records(a, b)
    assert c["text_equal"] and c["similarity"] == 1.0 and c["token_start_max_diff_s"] == pytest.approx(0.02)
    assert c["time_max_diff_s"] == pytest.approx(0.02)
    d = cmt.compare_records(a, {"segments": [seg("あう", 0, 1)]})
    assert not d["text_equal"] and d["token_start_max_diff_s"] is None and d["time_max_diff_s"] is None


def test_compare_records_sees_every_time_not_only_token_starts():
    a = {"segments": [seg("あ。", 0.0, 4.16, [tok("あ", 1.0, 1.2), tok("。", 3.84, 4.16)])]}
    b = {"segments": [seg("あ。", 0.0, 4.32, [tok("あ", 1.0, 1.2), tok("。", 3.84, 4.32)])]}
    c = cmt.compare_records(a, b)
    assert c["token_start_max_diff_s"] == 0.0 and c["time_max_diff_s"] == pytest.approx(0.16)
    e = cmt.compare_records(a, {"segments": [seg("あ。", 0.0, 4.16, [tok("あ", 1.0, None), tok("。", 3.84, 4.16)])]})
    assert e["time_max_diff_s"] is None   # a time present in one and missing in the other


def test_pooled_weighs_by_reference_characters():
    p = cmt.pooled([{"ref_chars": 100, "edits": 30, "wrong_rate": 0.1, "missing_rate": 0.1, "extra_rate": 0.1,
                     "missed_lines": 2, "hallucination_lines": 1, "test_lines": 10, "ref_lines": 12, "start_measured": 8},
                    {"ref_chars": 300, "edits": 30, "wrong_rate": 0.0, "missing_rate": 0.1, "extra_rate": 0.0,
                     "missed_lines": 1, "hallucination_lines": 0, "test_lines": 20, "ref_lines": 22, "start_measured": 18},
                    {"error": "x"}])
    assert p["clips"] == 2 and p["cer"] == pytest.approx(60 / 400) and p["wrong_rate"] == pytest.approx(10 / 400)
    assert p["missed_lines"] == 3 and p["test_lines"] == 30


def test_speech_to_text_list_of_the_data_file():
    models = cmt.load_models()
    assert not cmt.is_speech_to_text("kokoro", models)
    assert not cmt.is_speech_to_text("chatterbox-turbo", models)
    assert cmt.is_speech_to_text("haas-ja", models) and cmt.is_speech_to_text("parakeet", models)
    for name, e in models["models"].items():
        assert {"repo", "file", "backend", "row"} <= set(e), name
        assert e["row"]["line_rule"] in ("tokens", "group", "segments")


def test_manifest_files_reads_whisperjav_s_files_without_importing():
    files = cmt.manifest_files()
    assert files["parakeet-tdt-0.6b-ja-q8_0.gguf"] == ("parakeet-tdt-0.6b-ja", "cstr/parakeet-tdt-0.6b-ja-GGUF")


def test_baseline_entry_matches_whisperjav_s_manifest():
    e = cmt.load_models()["models"]["parakeet-tdt-0.6b-ja"]
    text = cmt.MANIFEST.read_text(encoding="utf-8")
    assert e["file"] in text and e["revision"] in text and e["repo"] in text


def test_srt_round_trip(tmp_path):
    p = tmp_path / "a.srt"
    cmt.write_srt(p, [(0.0, 1.25, "あ"), (61.5, 3725.001, "い\nう")])
    assert cmt.read_srt(p) == [(0.0, 1.25, "あ"), (61.5, 3725.001, "い\nう")]


def test_find_clips_pairs_media_with_their_reference(tmp_path):
    for n in ("A.Scene.1.mkv", "A.Scene.1.ja.srt", "B.mkv", "B.srt", "B.wav", "C.mkv", "notes.txt"):
        (tmp_path / n).write_text("x", encoding="utf-8")
    got = [(m.name, r.name) for m, r in cmt.find_clips(tmp_path)]
    assert got == [("A.Scene.1.mkv", "A.Scene.1.ja.srt"), ("B.mkv", "B.srt")]


def test_quiet_pieces_cut_in_the_quietest_place():
    import numpy as np
    audio = np.ones(cmt.SR * 25, dtype=np.float32)
    audio[int(8.5 * cmt.SR):int(8.6 * cmt.SR)] = 0.0       # a silence in the last quarter of the first 10 s
    pieces = cmt.quiet_pieces(audio, 10.0)
    assert pieces[0][0] == 0.0 and pieces[0][1] == pytest.approx(8.55, abs=0.06)
    assert all(e - s <= 10.0 + 1e-9 for s, e in pieces) and pieces[-1][1] == pytest.approx(25.0)


def test_probe_calls_cover_the_fixed_test_audio():
    calls = cmt.probe_calls("clip.mkv", [(1.0, 2.0, "x"), (5.0, 6.0, "y"), (9.0, 10.0, "z")], 200.0, noise_at=150.0)
    ids = [c["id"] for c in calls]
    assert ids[:3] == ["zeros_0.2s", "zeros_0.5s", "zeros_5.0s"]
    assert {"speech0_3s", "speech0_6s", "speech2_6s", "long_30s", "long_60s", "long_120s", "noise_8s"} <= set(ids)
    s0 = next(c for c in calls if c["id"] == "speech0_6s")
    assert (s0["start"], s0["end"]) == (0.8, 6.8)


def test_pipeline_groups_reads_the_debug_record(tmp_path):
    p = tmp_path / "x.parakeet-crispasr.groups.jsonl"
    p.write_text(json.dumps({"scene": 2, "start": 1.5, "end": 3.0, "aligned": False,
                             "segments": [{"text": "あ", "start": 0, "end": 1, "words": [], "tokens": []}]}) + "\n",
                 encoding="utf-8")
    g = cmt.pipeline_groups(p)
    assert g == [{"scene": 2, "start": 1.5, "end": 3.0, "aligned": False,
                  "segments": [{"text": "あ", "start": 0, "end": 1, "words": [], "tokens": []}]}]


def test_whisperjav_subtitles_use_whisperjav_s_own_line_rule(tmp_path):
    """The read-only path: a made-up Parakeet-like result becomes the subtitles of lines.build("tokens") and the
    clean-up with no stages."""
    rec = {"seconds": 4.0, "segments": [seg("こんにちは。元気？", 0.0, 3.0, [
        tok("こんにちは", 0.1, 0.9), tok("。", 0.9, 1.0), tok("元気", 1.5, 2.4), tok("？", 2.4, 2.5)])]}
    g = {"scene": 0, "start": 10.0, "end": 14.0}
    row = {"line_rule": "tokens", "onsets": False, "cut_at_pauses": True, "cleanup": "none"}
    srt, raw_n, final_n, _ = cmt.whisperjav_subtitles([(g, rec)], row, tmp_path, "clip")
    subs = cmt.read_srt(srt)
    assert raw_n == 2 and [s[2] for s in subs] == ["こんにちは。", "元気？"]
    assert subs[0][0] == pytest.approx(10.1) and subs[1][1] == pytest.approx(12.4)
