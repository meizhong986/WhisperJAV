"""tools/scene_inspector.py: the calls a CrispASR pass makes and the PotPlayer markers (owner W477, W478, W482, W483).

No audio is read and no model is loaded: the functions under test work on scene and call records. Files are written
under tmp_path only.
"""
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("scene_inspector", REPO / "tools" / "scene_inspector.py")
si = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(si)


def _call(call_id, scene, start, end):
    return {"call_id": call_id, "scene_index": scene, "start_s": start, "end_s": end, "duration_s": round(end - start, 3),
            "start_hms": si.hms(start), "end_hms": si.hms(end)}


def _scene(i, start, end):
    return {"scene_index": i, "start_s": start, "end_s": end, "duration_s": round(end - start, 3)}


class TestTheCopiedPipelineValuesStayEqual:
    """The tool copies two class values of the CrispASR pipeline (importing it would load CrispASR)."""

    def test_semantic_minimum_and_shortest_group_match_the_pipeline(self):
        text = (REPO / "whisperjav" / "pipelines" / "crispasr_pipeline.py").read_text(encoding="utf-8")
        assert float(re.search(r"^\s*SCENE_MIN_S\s*=\s*([\d.]+)", text, re.M).group(1)) == si.CRISPASR_SEMANTIC_MIN_S
        assert float(re.search(r"^\s*MIN_GROUP_S\s*=\s*([\d.]+)", text, re.M).group(1)) == si.CRISPASR_MIN_GROUP_S


class TestThePassSettings:
    def test_moss_semantic_scenes_take_the_pipeline_bounds_then_the_customize_values(self):
        kw = si.pipeline_scene_kwargs("moss-crispasr", "semantic", {})
        assert kw == {"method": "semantic", "min_duration": 12, "max_duration": 420}
        kw = si.pipeline_scene_kwargs("moss-crispasr", "semantic", {"min_duration": 90, "max_duration": 240})
        assert (kw["min_duration"], kw["max_duration"]) == (90, 240)

    def test_silero_and_auditok_get_only_the_longest_scene(self):
        assert si.pipeline_scene_kwargs("moss-crispasr", "silero", {}) == {"method": "silero", "max_duration": 420}
        assert si.pipeline_scene_kwargs("cohere-crispasr", "auditok", {}) == {"method": "auditok", "max_duration": 48}

    def test_moss_default_segmenter_and_the_w482_values(self):
        default = si.pipeline_segmenter("moss-crispasr", None, None, {})
        assert (default.backend, default.max_group_duration_s, default.chunk_threshold_s) == ("silero-v6.2", 120.0, 5.0)
        assert default.config["min_group_duration_s"] == 70.0
        w482 = si.pipeline_segmenter("moss-crispasr", "firered-vad", None,
                                     {"max_group_duration_s": 240, "chunk_threshold_s": 5, "min_group_duration_s": 0})
        assert (w482.backend, w482.max_group_duration_s, w482.chunk_threshold_s) == ("firered-vad", 240, 5)
        assert w482.config["min_group_duration_s"] == 0

    def test_no_segmenter_takes_no_values(self):
        none = si.pipeline_segmenter("moss-crispasr", "none", None, {"max_group_duration_s": 240})
        assert none.backend == "none" and none.max_group_duration_s is None


class TestWhatEndedEachCall:
    def test_silence_length_scene_cut_and_end_of_file(self):
        calls = [_call("1.1", 1, 0.0, 120.0),      # next starts 1 s later, same scene: the longest length
                 _call("1.2", 1, 121.0, 150.0),    # next starts 9 s later: a silence over the 5 s gap
                 _call("1.3", 1, 159.0, 180.0),    # next is in scene 2, 2 s later: the scene cut ended it
                 _call("2.1", 2, 182.0, 200.0)]    # last
        si.classify_call_ends(calls, 5.0)
        assert [c["ended_by"] for c in calls] == ["length", "silence", "scene cut", "end of file"]

    def test_without_a_segmenter_each_whole_scene_ends_at_its_cut(self):
        calls = [_call("1.1", 1, 0.0, 95.0), _call("2.1", 2, 95.0, 190.0), _call("3.1", 3, 189.7, 300.0)]
        si.classify_call_ends(calls, None)
        assert [c["ended_by"] for c in calls] == ["scene cut", "scene cut", "end of file"]


class TestCallStatistics:
    def test_buckets_long_calls_endings_and_scenes_without_calls(self):
        calls = [_call("1.1", 1, 0.0, 5.0), _call("1.2", 1, 20.0, 100.0), _call("3.1", 3, 200.0, 450.0)]
        si.classify_call_ends(calls, 5.0)
        st = si.call_stats(calls, 500.0, scene_count=4, not_sent=2, failed_scenes=[4])
        assert st["call_count"] == 3 and st["calls_total_s"] == 335.0
        assert st["histogram"] == {"0.2-10s": 1, "10-30s": 0, "30-70s": 0, "70-120s": 1, "120-240s": 0, ">240s": 1}
        assert st["calls_70s_or_longer"] == 2
        assert st["ended_by"] == {"silence": 2, "length": 0, "scene cut": 0, "end of file": 1}
        assert st["scenes_without_calls"] == [2] and st["scenes_failed"] == [4]
        assert st["pieces_not_sent_under_0_2s"] == 2


class TestPotPlayerBookmarks:
    def test_format_numbering_and_merged_marks(self):
        scenes = [_scene(1, 0.0, 95.5), _scene(2, 95.5, 200.0)]
        calls = [_call("1.1", 1, 0.0, 90.0), _call("2.1", 2, 101.25, 199.0)]
        si.classify_call_ends(calls, 5.0)
        text = si.pbf_text(si.marker_marks(scenes, calls, "both"))
        lines = text.split("\r\n")
        assert lines[0] == "[Bookmark]"
        assert lines[1] == "0=0*Scene 1 (95.5 s) | Call 1.1 (90.0 s, ends: silence)*"
        assert lines[2] == "1=95500*Scene 2 (104.5 s)*"
        assert lines[3] == "2=101250*Call 2.1 (97.8 s, ends: end of file)*"
        assert text.endswith("\r\n") and lines[4] == ""

    def test_levels_and_no_star_in_a_name(self):
        scenes, calls = [_scene(1, 0.0, 10.0)], [_call("1.1", 1, 1.0, 9.0)]
        si.classify_call_ends(calls, 5.0)
        assert [n for _, n in si.marker_marks(scenes, calls, "scenes")] == ["Scene 1 (10.0 s)"]
        assert [n for _, n in si.marker_marks(scenes, calls, "calls")] == ["Call 1.1 (8.0 s, ends: end of file)"]
        assert "*a b*" not in si.pbf_text([(1.0, "a*b\nc")]) and "1000*axb c*" in si.pbf_text([(1.0, "a*b\nc")])


class TestOnScreenLabel:
    def test_time_format(self):
        assert si.ass_time(3725.456) == "1:02:05.46"
        assert si.ass_time(0) == "0:00:00.00"

    def test_scene_and_call_lines_and_the_stretches_not_sent(self):
        scenes = [_scene(1, 0.0, 100.0)]
        calls = [_call("1.1", 1, 10.0, 80.0)]
        si.classify_call_ends(calls, 5.0)
        text = si.ass_text(100.0, scenes, calls, "moss-crispasr")
        assert text.startswith("[Script Info]")
        events = [l for l in text.split("\r\n") if l.startswith("Dialogue:")]
        assert len(events) == 3
        assert events[0].startswith("Dialogue: 0,0:00:00.00,0:00:10.00,Marker,,0,0,0,,Scene 1 of 1: ")
        assert events[0].endswith("\\NNot sent to moss-crispasr (no VAD group here)")
        assert "\\NCall 1.1 to moss-crispasr: 70.0 s, ends at the end of the file" in events[1]
        assert events[2].startswith("Dialogue: 0,0:01:20.00,0:01:40.00,")

    def test_without_calls_only_the_scene_is_named(self):
        text = si.ass_text(50.0, [_scene(1, 0.0, 50.0)], None, None)
        events = [l for l in text.split("\r\n") if l.startswith("Dialogue:")]
        assert len(events) == 1 and "\\N" not in events[0]


class TestMarkerFilesAreNeverOverwritten:
    def test_written_beside_the_media_as_utf8_with_bom(self, tmp_path):
        media = tmp_path / "film.mkv"
        media.write_bytes(b"")
        pbf, ass = si.write_markers(media, None, "[Bookmark]\r\n", "[Script Info]\r\n")
        assert (pbf.name, ass.name) == ("film.pbf", "film.ass")
        assert pbf.read_bytes().startswith(b"\xef\xbb\xbf[Bookmark]")
        assert ass.read_bytes().startswith(b"\xef\xbb\xbf[Script Info]")

    @pytest.mark.parametrize("existing", ["film.pbf", "film.ass"])
    def test_an_existing_file_stops_both(self, tmp_path, existing):
        media = tmp_path / "film.mkv"
        media.write_bytes(b"")
        (tmp_path / existing).write_bytes(b"his own")
        with pytest.raises(FileExistsError, match="never overwritten"):
            si.write_markers(media, None, "[Bookmark]\r\n", "[Script Info]\r\n")
        assert (tmp_path / existing).read_bytes() == b"his own"
        other = "film.ass" if existing == "film.pbf" else "film.pbf"
        assert not (tmp_path / other).exists()

    def test_another_folder(self, tmp_path):
        media = tmp_path / "film.mkv"
        media.write_bytes(b"")
        (tmp_path / "film.pbf").write_bytes(b"his own")
        pbf, ass = si.write_markers(media, tmp_path / "marks", "[Bookmark]\r\n", "[Script Info]\r\n")
        assert pbf.parent == ass.parent == tmp_path / "marks"


class TestCommandLineChecks:
    def test_segmenter_options_need_a_pipeline(self, tmp_path, capsys):
        assert si.main([str(tmp_path / "film.mkv"), "--speech-segmenter", "ten"]) == 2
        assert "need --pipeline" in capsys.readouterr().err

    def test_markers_and_label_take_one_backend(self, tmp_path, capsys):
        assert si.main([str(tmp_path / "film.mkv"), "--backend", "auditok,semantic", "--markers"]) == 2
        assert si.main([str(tmp_path / "film.mkv"), "--backend", "auditok,semantic", "--label", "x"]) == 2

    def test_moss_runs_its_own_scene_detector_by_default(self, tmp_path, capsys):
        # The media does not exist: the run stops at the file check, after the settings were resolved.
        assert si.main([str(tmp_path / "film.mkv"), "--pipeline", "moss-crispasr", "--quiet"]) == 1
        assert "not found" in capsys.readouterr().err
