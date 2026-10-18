"""
CrispASR's two file formats (whisperjav/crispasr/ojf.py; master plan 4.2, 3.4).

- The cut-point file WhisperJAV writes for ``--vad-import`` (kind ``chunks``). The program uses it
  exactly as given, so a bad list is refused, never repaired: a slice the program dropped would
  shift every later ``chunk_id``.
- The ``-ojf`` JSON the program writes. Everything in it is kept (blank tokens, zero-length tokens,
  speaker tags); a file that does not parse or has no transcription list counts as no output, and
  so does one that names another back end, model or language. Each segment is tied to its slice: by
  ``chunk_id`` when that fits, else by its start time (a note), else it is dropped (counted).

The sample documents are written by hand in the shapes CrispASR 0.8.43 writes -- the key sets of
segments, words and tokens are those of the 2026-10-17 study outputs -- and hold no film text.

Run with: python -m pytest tests/test_crispasr_ojf.py -q
"""
import json
import os
import sys

import pytest

from whisperjav.crispasr.ojf import (
    CHUNK_CS,
    SAMPLE_RATE,
    SLICE_TOLERANCE_S,
    CrispSegment,
    CutPoint,
    header_mismatch,
    parse_ojf,
    place_in_slices,
    read_ojf,
    write_vad_import,
)

PARAKEET_MODEL = "C:\\m\\parakeet-tdt-0.6b-ja-q8_0.gguf"

# Parakeet TDT: words and tokens, a blank " " token before the first word, chunk_id on every segment.
PARAKEET_TDT = {
    "crispasr": {"backend": "parakeet", "model": PARAKEET_MODEL, "language": "ja"},
    "transcription": [
        {"timestamps": {"from": "00:00:25,160", "to": "00:00:26,880"},
         "offsets": {"from": 25160, "to": 26880},
         "text": "テストです。", "chunk_id": 0,
         "words": [
             {"text": "テスト", "t0": 2516, "t1": 2548, "offsets": {"from": 25160, "to": 25480}},
             {"text": "です。", "t0": 2548, "t1": 2688, "offsets": {"from": 25480, "to": 26880}}],
         "tokens": [
             {"text": " ", "p": 0.99, "t0": 2500, "t1": 2516, "offsets": {"from": 25000, "to": 25160}},
             {"text": "テスト", "p": 0.99, "t0": 2516, "t1": 2548, "offsets": {"from": 25160, "to": 25480}},
             {"text": "です", "p": 0.87, "t0": 2548, "t1": 2620, "offsets": {"from": 25480, "to": 26200}},
             {"text": "。", "p": 0.6, "t0": 2620, "t1": 2688, "offsets": {"from": 26200, "to": 26880}}]},
        {"timestamps": {"from": "00:00:31,040", "to": "00:00:32,000"},
         "offsets": {"from": 31040, "to": 32000},
         "text": "はい", "chunk_id": 1,
         "words": [{"text": "はい", "t0": 3104, "t1": 3200, "offsets": {"from": 31040, "to": 32000}}],
         "tokens": [{"text": "はい", "p": 0.95, "t0": 3104, "t1": 3200,
                     "offsets": {"from": 31040, "to": 32000}}]},
    ],
}

# Cohere: one "word" holding the whole text; tokens may be zero-length (t0 == t1).
COHERE = {
    "crispasr": {"backend": "cohere", "model": "C:\\m\\cohere-asr-ja-q8_0.gguf", "language": "ja"},
    "transcription": [
        {"timestamps": {"from": "00:00:01,000", "to": "00:00:03,500"},
         "offsets": {"from": 1000, "to": 3500},
         "text": "テストです", "chunk_id": 0,
         "words": [{"text": "テストです", "t0": 100, "t1": 350, "offsets": {"from": 1000, "to": 3500}}],
         "tokens": [
             {"text": "テスト", "p": 0.91, "t0": 100, "t1": 100, "offsets": {"from": 1000, "to": 1000}},
             {"text": "です", "p": 0.88, "t0": 100, "t1": 350, "offsets": {"from": 1000, "to": 3500}}]},
    ],
}

# VibeVoice and MOSS: a speaker tag, no words and no tokens.
VIBEVOICE = {
    "crispasr": {"backend": "vibevoice", "model": "C:\\m\\vibevoice-asr-q4_k.gguf", "language": "ja"},
    "transcription": [
        {"timestamps": {"from": "00:00:05,000", "to": "00:00:07,250"},
         "offsets": {"from": 5000, "to": 7250},
         "text": "テストです", "speaker": "(Speaker 1) ", "chunk_id": 0},
    ],
}

# The CTC Parakeet: tokens of one 80 ms frame each, a blank one among them, and no words.
PARAKEET_CTC = {
    "crispasr": {"backend": "fastconformer-ctc", "model": "C:\\m\\parakeet-ctc-1.1b-ja-q8_0.gguf",
                 "language": "ja"},
    "transcription": [
        {"timestamps": {"from": "00:00:10,000", "to": "00:00:10,320"},
         "offsets": {"from": 10000, "to": 10320},
         "text": "テスト", "chunk_id": 0,
         "tokens": [
             {"text": "テ", "p": 0.97, "t0": 1000, "t1": 1008, "offsets": {"from": 10000, "to": 10080}},
             {"text": "ス", "p": 0.96, "t0": 1008, "t1": 1016, "offsets": {"from": 10080, "to": 10160}},
             {"text": " ", "p": 0.5, "t0": 1016, "t1": 1024, "offsets": {"from": 10160, "to": 10240}},
             {"text": "ト", "p": 0.95, "t0": 1024, "t1": 1032, "offsets": {"from": 10240, "to": 10320}}]},
    ],
}


def _segment(**fields):
    """One Parakeet-shaped segment of a run with a single slice, with fields replaced or added."""
    seg = {"timestamps": {"from": "00:00:02,000", "to": "00:00:03,000"},
           "offsets": {"from": 2000, "to": 3000}, "text": "です",
           "words": [{"text": "です", "t0": 200, "t1": 300, "offsets": {"from": 2000, "to": 3000}}],
           "tokens": [{"text": "です", "p": 0.9, "t0": 200, "t1": 300, "offsets": {"from": 2000, "to": 3000}}]}
    seg.update(fields)
    return seg


def _doc(*segments, backend="parakeet", model=PARAKEET_MODEL):
    return {"crispasr": {"backend": backend, "model": model, "language": "ja"},
            "transcription": list(segments)}


def _texts(items):
    return [i.text for i in items]


# ---------------------------------------------------------------------------------------------------
# The cut-point file
# ---------------------------------------------------------------------------------------------------

class TestTheCutPointFile:
    def test_the_file_is_exactly_the_chunks_import_the_program_reads(self, tmp_path):
        path = tmp_path / "cuts.json"
        write_vad_import(path, [CutPoint(0.0, 12.5), CutPoint(12.5, 30.0), CutPoint(41.23456, 58.0)])
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "crispasr_vad": {
                "version": 1, "kind": "chunks", "sample_rate": 16000, "chunk_cs": 3000, "num_slices": 3,
                "slices": [
                    {"start": 0, "end": 200000, "t0_cs": 0, "t1_cs": 1250},
                    {"start": 200000, "end": 480000, "t0_cs": 1250, "t1_cs": 3000},
                    {"start": 659753, "end": 928000, "t0_cs": 4123, "t1_cs": 5800},
                ],
            }
        }

    def test_the_constants_are_the_programs_defaults(self):
        # --chunk-seconds 30 is the program's default; writing it avoids its mismatch warning.
        assert (SAMPLE_RATE, CHUNK_CS) == (16000, 3000)

    def test_every_number_in_a_slice_is_an_integer(self, tmp_path):
        path = tmp_path / "cuts.json"
        write_vad_import(path, [CutPoint(0.33333, 1.66667), CutPoint(2.5, 7.123456)])
        doc = json.loads(path.read_text(encoding="utf-8"))["crispasr_vad"]
        assert type(doc["num_slices"]) is int and type(doc["sample_rate"]) is int
        for s in doc["slices"]:
            assert sorted(s) == ["end", "start", "t0_cs", "t1_cs"]
            assert all(type(v) is int for v in s.values()), s

    @pytest.mark.parametrize("cut, expected", [
        # 100.6 cs becomes 101 and 0.64 samples becomes 1: rounded to the nearest, not truncated.
        (CutPoint(1.006, 2.004), {"start": 16096, "end": 32064, "t0_cs": 101, "t1_cs": 200}),
        (CutPoint(0.00004, 0.5), {"start": 1, "end": 8000, "t0_cs": 0, "t1_cs": 50}),
    ])
    def test_times_are_rounded_to_the_nearest_sample_and_centisecond(self, tmp_path, cut, expected):
        path = tmp_path / "cuts.json"
        write_vad_import(path, [cut])
        assert json.loads(path.read_text(encoding="utf-8"))["crispasr_vad"]["slices"] == [expected]

    def test_another_sample_rate_is_written_and_used(self, tmp_path):
        path = tmp_path / "cuts.json"
        write_vad_import(path, [CutPoint(1.0, 2.5)], sample_rate=8000)
        doc = json.loads(path.read_text(encoding="utf-8"))["crispasr_vad"]
        assert doc["sample_rate"] == 8000
        assert doc["slices"] == [{"start": 8000, "end": 20000, "t0_cs": 100, "t1_cs": 250}]

    def test_touching_cuts_are_allowed(self, tmp_path):
        path = tmp_path / "cuts.json"
        write_vad_import(path, [CutPoint(0.0, 5.0), CutPoint(5.0, 9.0), CutPoint(9.0, 9.5)])
        slices = json.loads(path.read_text(encoding="utf-8"))["crispasr_vad"]["slices"]
        assert [(s["start"], s["end"]) for s in slices] == [(0, 80000), (80000, 144000), (144000, 152000)]

    def test_a_cut_ending_at_the_end_of_the_audio_is_allowed(self, tmp_path):
        write_vad_import(tmp_path / "a.json", [CutPoint(0.0, 30.0)], audio_seconds=30.0)
        # A float hair past the end (1e-6 s) is still the end of the audio.
        write_vad_import(tmp_path / "b.json", [CutPoint(0.0, 30.0000005)], audio_seconds=30.0)
        assert (tmp_path / "a.json").is_file() and (tmp_path / "b.json").is_file()

    def test_without_the_audio_length_no_end_is_checked(self, tmp_path):
        write_vad_import(tmp_path / "cuts.json", [CutPoint(0.0, 99999.0)])
        assert (tmp_path / "cuts.json").is_file()

    @pytest.mark.parametrize("cuts, audio_seconds, words", [
        ([], None, "no cut points"),
        ([CutPoint(1.0, 1.0)], None, "empty or negative"),
        ([CutPoint(2.0, 1.0)], None, "empty or negative"),
        ([CutPoint(-0.5, 1.0)], None, "empty or negative"),
        ([CutPoint(0.0, 1.0), CutPoint(1.0, 1.0)], None, "cut point 1 is empty"),
        ([CutPoint(5.0, 6.0), CutPoint(1.0, 2.0)], None, "starts before cut point 0 ends"),
        ([CutPoint(0.0, 5.0), CutPoint(4.9, 8.0)], None, "starts before cut point 0 ends"),
        ([CutPoint(0.0, 5.0), CutPoint(5.0, 30.01)], 30.0, "cut point 1 ends after the audio"),
    ], ids=["no cuts", "empty", "reversed", "negative", "empty later cut", "unsorted", "overlapping",
            "beyond the audio"])
    def test_a_list_the_program_would_not_use_as_given_is_refused_and_nothing_written(
            self, tmp_path, cuts, audio_seconds, words):
        path = tmp_path / "cuts.json"
        with pytest.raises(ValueError, match=words):
            write_vad_import(path, cuts, audio_seconds=audio_seconds)
        assert not path.exists()


# ---------------------------------------------------------------------------------------------------
# Reading the -ojf JSON
# ---------------------------------------------------------------------------------------------------

class TestReadingTheFile:
    def test_a_complete_file_is_returned_as_written(self, tmp_path):
        path = tmp_path / "out.json"
        path.write_text(json.dumps(PARAKEET_TDT, ensure_ascii=False), encoding="utf-8")
        assert read_ojf(path) == PARAKEET_TDT

    def test_a_complete_file_with_no_segments_is_still_a_file(self, tmp_path):
        path = tmp_path / "out.json"
        path.write_text(json.dumps(_doc()), encoding="utf-8")
        assert read_ojf(path) == _doc()

    def test_a_missing_file_is_no_output(self, tmp_path):
        assert read_ojf(tmp_path / "out.json") is None

    def test_a_file_cut_off_while_being_written_is_no_output(self, tmp_path):
        path = tmp_path / "out.json"
        text = json.dumps(PARAKEET_TDT, ensure_ascii=False)
        path.write_text(text[: len(text) // 2], encoding="utf-8")
        assert read_ojf(path) is None

    def test_a_file_that_is_not_utf8_is_no_output(self, tmp_path):
        path = tmp_path / "out.json"
        path.write_bytes(b'{"crispasr": {}, "transcription": [{"text": "\xff\xfe"}]}')
        assert read_ojf(path) is None

    @pytest.mark.parametrize("content", [
        [PARAKEET_TDT],
        "transcription",
        {"crispasr": {"backend": "parakeet"}},
        {"crispasr": {"backend": "parakeet"}, "transcription": {"0": {"text": "テスト"}}},
        {"crispasr": {"backend": "parakeet"}, "transcription": None},
    ], ids=["a list", "a string", "no transcription", "transcription not a list", "transcription null"])
    def test_a_document_without_a_transcription_list_is_no_output(self, tmp_path, content):
        path = tmp_path / "out.json"
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        assert read_ojf(path) is None


class TestParsing:
    def test_the_header_is_read(self):
        t = parse_ojf(PARAKEET_TDT)
        assert (t.backend, t.model, t.language) == ("parakeet", PARAKEET_MODEL, "ja")
        assert t.malformed == 0

    def test_segment_times_come_from_offsets_in_milliseconds(self):
        first, second = parse_ojf(PARAKEET_TDT).segments
        assert (first.start, first.end) == (pytest.approx(25.16), pytest.approx(26.88))
        assert (second.start, second.end) == (pytest.approx(31.04), pytest.approx(32.0))
        assert (first.text, second.text) == ("テストです。", "はい")
        assert (first.chunk_id, second.chunk_id) == (0, 1)

    def test_words_and_tokens_are_kept_in_order_with_times_and_p(self):
        seg = parse_ojf(PARAKEET_TDT).segments[0]
        assert _texts(seg.words) == ["テスト", "です。"]
        assert [(w.start, w.end) for w in seg.words] == [
            (pytest.approx(25.16), pytest.approx(25.48)), (pytest.approx(25.48), pytest.approx(26.88))]
        assert _texts(seg.tokens) == [" ", "テスト", "です", "。"]
        assert [t.p for t in seg.tokens] == [pytest.approx(0.99), pytest.approx(0.99),
                                             pytest.approx(0.87), pytest.approx(0.6)]
        assert (seg.tokens[0].start, seg.tokens[0].end) == (pytest.approx(25.0), pytest.approx(25.16))

    def test_a_blank_token_is_kept_for_the_pipeline_to_judge(self):
        tokens = parse_ojf(PARAKEET_TDT).segments[0].tokens
        assert tokens[0].text == " "
        ctc_tokens = parse_ojf(PARAKEET_CTC).segments[0].tokens
        assert " " in _texts(ctc_tokens)

    def test_cohere_keeps_its_one_whole_text_word_and_its_zero_length_token(self):
        seg = parse_ojf(COHERE).segments[0]
        assert _texts(seg.words) == ["テストです"]
        assert (seg.words[0].start, seg.words[0].end) == (pytest.approx(1.0), pytest.approx(3.5))
        zero = seg.tokens[0]
        assert (zero.text, zero.start, zero.end) == ("テスト", pytest.approx(1.0), pytest.approx(1.0))
        assert len(seg.tokens) == 2

    def test_a_speaker_tag_is_kept_exactly_and_a_segment_needs_no_words_or_tokens(self):
        t = parse_ojf(VIBEVOICE)
        assert t.malformed == 0
        seg = t.segments[0]
        assert seg.speaker == "(Speaker 1) "
        assert seg.text == "テストです"
        assert (seg.words, seg.tokens) == ([], [])
        assert (seg.start, seg.end) == (pytest.approx(5.0), pytest.approx(7.25))

    def test_ctc_tokens_of_one_frame_each_and_no_words(self):
        seg = parse_ojf(PARAKEET_CTC).segments[0]
        assert seg.words == []
        assert _texts(seg.tokens) == ["テ", "ス", " ", "ト"]
        assert all(t.end - t.start == pytest.approx(0.08) for t in seg.tokens)

    def test_no_speaker_field_means_none(self):
        assert parse_ojf(PARAKEET_TDT).segments[0].speaker is None

    def test_a_speaker_field_that_is_not_text_is_none(self):
        assert parse_ojf(_doc(_segment(speaker=1))).segments[0].speaker is None

    def test_a_run_with_one_slice_has_no_chunk_id_and_reads_as_slice_0(self):
        seg = parse_ojf(_doc(_segment())).segments[0]
        assert seg.chunk_id == 0

    @pytest.mark.parametrize("value", ["x", None, [1], {"i": 1}], ids=["text", "null", "list", "object"])
    def test_a_chunk_id_that_is_not_a_number_reads_as_minus_1(self, value):
        assert parse_ojf(_doc(_segment(chunk_id=value))).segments[0].chunk_id == -1

    def test_a_fractional_chunk_id_reads_as_minus_1_not_as_a_slice(self):
        # A chunk_id of 1.5 names no slice; truncating it to 1 would tie the segment to slice 1
        # without the re-tie note place_in_slices gives every chunk_id that does not fit.
        assert parse_ojf(_doc(_segment(chunk_id=1.5))).segments[0].chunk_id == -1

    def test_times_fall_back_to_t0_and_t1_in_centiseconds_without_offsets(self):
        seg = _segment(t0=250, t1=375,
                       words=[{"text": "です", "t0": 260, "t1": 300}],
                       tokens=[{"text": "です", "p": 0.5, "t0": 260, "t1": 300, "offsets": {"from": 2600}}])
        del seg["offsets"]
        t = parse_ojf(_doc(seg))
        assert t.malformed == 0
        s = t.segments[0]
        assert (s.start, s.end) == (pytest.approx(2.5), pytest.approx(3.75))
        assert (s.words[0].start, s.words[0].end) == (pytest.approx(2.6), pytest.approx(3.0))
        # offsets without both ends do not count; t0/t1 are used.
        assert (s.tokens[0].start, s.tokens[0].end) == (pytest.approx(2.6), pytest.approx(3.0))

    def test_offsets_win_over_t0_and_t1(self):
        seg = _segment(t0=999, t1=1000)
        s = parse_ojf(_doc(seg)).segments[0]
        assert (s.start, s.end) == (pytest.approx(2.0), pytest.approx(3.0))

    def test_a_token_without_p_or_with_p_not_a_number_has_none(self):
        seg = _segment(tokens=[{"text": "テ", "t0": 200, "t1": 250, "offsets": {"from": 2000, "to": 2500}},
                               {"text": "ス", "p": "0.5", "t0": 250, "t1": 300,
                                "offsets": {"from": 2500, "to": 3000}},
                               {"text": "ト", "p": 1, "t0": 250, "t1": 300,
                                "offsets": {"from": 2500, "to": 3000}}])
        tokens = parse_ojf(_doc(seg)).segments[0].tokens
        assert [t.p for t in tokens] == [None, None, 1.0]

    def test_malformed_segments_words_and_tokens_are_counted_and_skipped(self):
        good = _segment(
            chunk_id=0,
            words=[{"text": "テ"},                                                  # no times
                   "テスト",                                                         # not an object
                   {"text": "スト", "offsets": {"from": 2000, "to": 3000}}],
            tokens=[{"text": "テ", "p": 0.9, "offsets": {"from": None, "to": 2100}},  # from is null
                    5,                                                                # not an object
                    {"text": "ス", "p": 0.9, "t0": 210, "t1": 220}])
        doc = _doc("not a segment",
                   None,
                   {"text": "テスト"},                                                # no times
                   {"offsets": {"from": "abc", "to": 5000}, "text": "テスト"},        # not a number
                   good)
        t = parse_ojf(doc)
        assert t.malformed == 4 + 2 + 2
        assert len(t.segments) == 1
        assert _texts(t.segments[0].words) == ["スト"]
        assert _texts(t.segments[0].tokens) == ["ス"]

    def test_a_null_text_reads_as_empty_never_as_the_word_none(self):
        seg = _segment(text=None,
                       words=[{"text": None, "t0": 200, "t1": 300, "offsets": {"from": 2000, "to": 3000}}],
                       tokens=[{"text": None, "p": 0.9, "t0": 200, "t1": 300,
                                "offsets": {"from": 2000, "to": 3000}}])
        s = parse_ojf(_doc(seg)).segments[0]
        assert (s.text, s.words[0].text, s.tokens[0].text) == ("", "", "")

    def test_a_document_without_a_header_has_empty_header_fields(self):
        t = parse_ojf({"transcription": [_segment()]})
        assert (t.backend, t.model, t.language) == ("", "", "")
        assert len(t.segments) == 1

    def test_no_segments(self):
        t = parse_ojf(_doc())
        assert (t.segments, t.malformed) == ([], 0)


class TestHeaderCheck:
    def test_the_requested_back_end_and_model_match(self):
        assert header_mismatch(parse_ojf(PARAKEET_TDT), "parakeet", PARAKEET_MODEL) is None

    def test_another_back_end_is_no_output(self):
        why = header_mismatch(parse_ojf(PARAKEET_TDT), "fastconformer-ctc", PARAKEET_MODEL)
        assert why is not None
        assert "'parakeet'" in why and "'fastconformer-ctc'" in why

    def test_another_model_is_no_output(self):
        other = "C:\\m\\parakeet-ctc-1.1b-ja-q8_0.gguf"
        why = header_mismatch(parse_ojf(PARAKEET_TDT), "parakeet", other)
        assert why is not None
        assert PARAKEET_MODEL in why and other in why

    def test_an_empty_model_in_the_json_is_no_output(self):
        assert header_mismatch(parse_ojf(_doc(model="")), "parakeet", PARAKEET_MODEL) is not None

    def test_the_short_name_of_a_staged_model_matches(self):
        assert header_mismatch(parse_ojf(_doc(model="model.gguf")), "parakeet", "model.gguf") is None

    def test_the_same_path_written_with_a_redundant_part_matches(self):
        written = os.path.join("models", "parakeet.gguf")
        requested = os.path.join("models", ".", "parakeet.gguf")
        assert header_mismatch(parse_ojf(_doc(model=written)), "parakeet", requested) is None

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows paths ignore case and accept either slash")
    def test_on_windows_case_and_slashes_do_not_make_a_mismatch(self):
        requested = "c:/M/PARAKEET-tdt-0.6b-ja-q8_0.GGUF"
        assert header_mismatch(parse_ojf(PARAKEET_TDT), "parakeet", requested) is None

    def test_the_relative_name_of_a_model_staged_beside_the_run_folder_matches(self):
        # A model staged once per pass is named relative to the film's run folder (plan 1.7).
        staged = os.path.join("..", "parakeet-tdt-0.6b-ja-q8_0.gguf")
        assert header_mismatch(parse_ojf(_doc(model=staged)), "parakeet", staged, language="ja") is None

    @pytest.mark.parametrize("requested", ["ja", "JA", " ja "])
    def test_the_requested_language_matches_in_any_letter_case(self, requested):
        assert header_mismatch(parse_ojf(PARAKEET_TDT), "parakeet", PARAKEET_MODEL, language=requested) is None

    def test_another_language_is_no_output(self):
        doc = dict(PARAKEET_TDT, crispasr={"backend": "parakeet", "model": PARAKEET_MODEL, "language": "en"})
        why = header_mismatch(parse_ojf(doc), "parakeet", PARAKEET_MODEL, language="ja")
        assert why == "the JSON names language 'en', not 'ja'"

    def test_a_json_without_a_language_is_no_output_when_one_was_asked_for(self):
        doc = {"crispasr": {"backend": "parakeet", "model": PARAKEET_MODEL}, "transcription": []}
        assert header_mismatch(parse_ojf(doc), "parakeet", PARAKEET_MODEL, language="ja") is not None

    def test_without_a_requested_language_the_language_is_not_checked(self):
        doc = dict(PARAKEET_TDT, crispasr={"backend": "parakeet", "model": PARAKEET_MODEL, "language": "en"})
        assert header_mismatch(parse_ojf(doc), "parakeet", PARAKEET_MODEL) is None
        assert header_mismatch(parse_ojf(doc), "parakeet", PARAKEET_MODEL, language=None) is None

    def test_the_back_end_is_named_first_when_several_things_differ(self):
        doc = {"crispasr": {"backend": "cohere", "model": "other.gguf", "language": "en"}, "transcription": []}
        why = header_mismatch(parse_ojf(doc), "parakeet", PARAKEET_MODEL, language="ja")
        assert why == "the JSON names back end 'cohere', not 'parakeet'"


# ---------------------------------------------------------------------------------------------------
# Tying segments to their slices
# ---------------------------------------------------------------------------------------------------

CUTS = [CutPoint(0.0, 10.0), CutPoint(20.0, 30.0)]


def _seg(start, chunk_id):
    return CrispSegment(start=start, end=start + 1.0, text="テスト", chunk_id=chunk_id)


class TestPlacingSegmentsInSlices:
    def test_a_chunk_id_that_fits_places_the_segment(self):
        a, b = _seg(2.0, 0), _seg(25.0, 1)
        result = place_in_slices([a, b], CUTS)
        assert result.placed == [(0, a), (1, b)]
        assert (result.retied, result.dropped) == (0, 0)

    @pytest.mark.parametrize("chunk_id", [2, 7, -1], ids=["one past the end", "far past", "unreadable"])
    def test_a_chunk_id_out_of_range_is_retied_by_time(self, chunk_id):
        s = _seg(25.0, chunk_id)
        result = place_in_slices([s], CUTS)
        assert result.placed == [(1, s)]
        assert (result.retied, result.dropped) == (1, 0)

    def test_a_start_outside_the_named_slice_is_retied_to_the_slice_holding_it(self):
        s = _seg(25.0, 0)
        result = place_in_slices([s], CUTS)
        assert result.placed == [(1, s)]
        assert (result.retied, result.dropped) == (1, 0)

    @pytest.mark.parametrize("start, chunk_id", [(15.0, 0), (15.0, 1), (35.0, 1), (-1.0, 0)],
                             ids=["in the gap, named 0", "in the gap, named 1", "after the end", "before 0"])
    def test_a_start_in_no_slice_is_dropped_and_counted(self, start, chunk_id):
        result = place_in_slices([_seg(start, chunk_id)], CUTS)
        assert result.placed == []
        assert (result.retied, result.dropped) == (0, 1)

    def test_the_tolerance_holds_at_both_edges_of_a_slice(self):
        assert SLICE_TOLERANCE_S == 0.05
        early = _seg(20.0 - SLICE_TOLERANCE_S, 1)
        late = _seg(30.0 + SLICE_TOLERANCE_S, 1)
        first = _seg(0.0 - SLICE_TOLERANCE_S, 0)
        result = place_in_slices([early, late, first], CUTS)
        assert result.placed == [(1, early), (1, late), (0, first)]
        assert (result.retied, result.dropped) == (0, 0)

    def test_just_past_the_tolerance_is_outside(self):
        too_early = _seg(20.0 - SLICE_TOLERANCE_S - 0.001, 1)
        too_late = _seg(30.0 + SLICE_TOLERANCE_S + 0.001, 1)
        result = place_in_slices([too_early, too_late], CUTS)
        assert result.placed == []
        assert result.dropped == 2

    def test_a_start_within_tolerance_of_another_slice_is_retied_there(self):
        s = _seg(10.0 + SLICE_TOLERANCE_S, 1)  # names slice 1, but lies at slice 0's end
        result = place_in_slices([s], CUTS)
        assert result.placed == [(0, s)]
        assert result.retied == 1

    def test_another_tolerance_can_be_given(self):
        s = _seg(19.6, 1)
        assert place_in_slices([s], CUTS).dropped == 1
        assert place_in_slices([s], CUTS, tolerance=0.5).placed == [(1, s)]

    def test_the_programs_order_is_kept_and_counts_add_up(self):
        segs = [_seg(25.0, 1), _seg(2.0, 0), _seg(15.0, 0), _seg(3.0, 4), _seg(28.0, 1)]
        result = place_in_slices(segs, CUTS)
        assert result.placed == [(1, segs[0]), (0, segs[1]), (0, segs[3]), (1, segs[4])]
        assert (result.retied, result.dropped) == (1, 1)

    def test_with_no_slices_everything_is_dropped(self):
        result = place_in_slices([_seg(1.0, 0), _seg(2.0, 0)], [])
        assert (result.placed, result.dropped) == ([], 2)

    def test_a_parsed_file_is_placed_in_the_slices_it_was_cut_with(self, tmp_path):
        cuts = [CutPoint(24.0, 28.0), CutPoint(30.5, 33.0)]
        write_vad_import(tmp_path / "cuts.json", cuts)
        result = place_in_slices(parse_ojf(PARAKEET_TDT).segments, cuts)
        assert [(i, s.text) for i, s in result.placed] == [(0, "テストです。"), (1, "はい")]
        assert (result.retied, result.dropped) == (0, 0)
