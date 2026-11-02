"""R-SCN scene detector (owner W448-W452; docs/plans/design/2026-11-02-rscn-design.md).

Synthetic audio only; no model is loaded (the speech measure is `none` or a fake behind the contract). FFmpeg-made tiny
files for the media record and the black picture (skipped when FFmpeg is missing). Run alone:
python -m pytest tests/test_rscn_scene_detector.py
"""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from whisperjav.modules.media_record import MediaRecord, probe
from whisperjav.modules.scene_detection_backends import SceneDetectionError, SceneDetectorFactory
from whisperjav.modules.scene_detection_backends.rscn import analysis, partition, picture
from whisperjav.modules.scene_detection_backends.rscn.backend import RSCNSceneDetector
from whisperjav.modules.scene_detection_backends.rscn.settings import PRESETS, RSCNSettings, overrides_from_params
from whisperjav.modules.scene_detection_backends.rscn.speech import (SegmenterSpeechMeasure, SpeechMeasurement,
                                                                     SpeechStretch, create_speech_measure, merge)

SR = 16000
RNG = np.random.default_rng(7)
QUIET = RSCNSettings(speech_measure="none", picture="off")
HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def noise(sec, level=0.05, sr=SR):
    return (level * RNG.standard_normal(int(round(sec * sr)))).astype(np.float32)


def tone(sec, f=440.0, level=0.05, sr=SR):
    t = np.arange(int(round(sec * sr))) / sr
    return (level * np.sqrt(2) * np.sin(2 * np.pi * f * t)).astype(np.float32)


def babble(sec, level=0.05, sr=SR):
    x = noise(sec, level, sr)
    t = np.arange(len(x)) / sr
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4 * t)
    env[(t % 3.0) < 0.4] = 0.02
    return (x * env).astype(np.float32)


def detect(tmp_path, audio, sr=SR, settings=QUIET, media=None, measure=None, name="a"):
    wav = tmp_path / f"{name}.wav"
    sf.write(wav, audio, sr)
    det = RSCNSceneDetector(settings=settings, speech_measure=measure)
    return det.detect_scenes(wav, tmp_path / f"{name}_scenes", name, media=media)


def assert_exact(result, total_samples, sr):
    """Every sample in exactly one scene; WAV files hold exactly the scene's samples."""
    at = 0
    for sc in result.scenes:
        a, b = round(sc.start_sec * sr), round(sc.end_sec * sr)
        assert a == at and b > a
        info = sf.info(str(sc.scene_path))
        assert info.samplerate == sr and info.frames == b - a
        at = b
    assert at == total_samples


class FakeMeasure:
    """A speech measure behind the contract, with fixed stretches."""
    name = "fake"

    def __init__(self, stretches):
        self.stretches = stretches
        self.closed = False

    def measure(self, audio_16k):
        return SpeechMeasurement(merge(self.stretches), "fake")

    def close(self):
        self.closed = True


# --- Settings ------------------------------------------------------------------------------------------------------

def test_defaults_are_the_owners_lengths():
    s = RSCNSettings()
    assert (s.min_scene_s, s.preferred_min_s, s.preferred_max_s, s.max_scene_s) == (70, 120, 360, 420)
    assert s.speech_measure == "firered-vad" and s.picture == "auto"
    assert PRESETS["moss-crispasr"] == s


@pytest.mark.parametrize("bad", [
    dict(min_scene_s=130),                       # min above the preferred minimum
    dict(preferred_max_s=430),                   # preferred maximum above the maximum
    dict(max_scene_s=130),                       # below twice the minimum
    dict(picture="sometimes"),
    dict(feature_hop_ms=50),                     # hop longer than the frame
    dict(novelty_scale_weights=(1.0,)),          # one weight per scale
    dict(novelty_scales_s=(200.0,), novelty_scale_weights=(1.0,)),  # windows longer than a scene
    dict(speech_measure_overlap_s=200),          # overlap longer than a piece
    dict(grid_step_s=40),                        # above half the minimum
    dict(change_threshold=0),
])
def test_invalid_settings_are_refused(bad):
    with pytest.raises(ValueError):
        RSCNSettings(**bad)


def test_overrides_from_params_take_only_rscn_keys():
    params = {"rscn_min_scene_s": 60, "rscn_novelty_scales_s": [10, 30], "rscn_novelty_scale_weights": [1, 2],
              "rscn_picture": "OFF", "max_duration_s": 29, "hotwords": "x"}
    s = RSCNSettings.for_pipeline("moss-crispasr", params)
    assert s.min_scene_s == 60 and s.novelty_scales_s == (10.0, 30.0) and s.picture_mode == "off"
    assert overrides_from_params({"max_duration_s": 29}) == {}


@pytest.mark.parametrize("params", [{"rscn_nonsense": 1}, {"rscn_min_scene_s": "long"},
                                    {"rscn_novelty_smooth_frames": 2.5}, {"rscn_min_scene_s": True}])
def test_unknown_or_bad_rscn_key_names_the_key(params):
    with pytest.raises(ValueError, match=list(params)[0]):
        RSCNSettings.for_pipeline("moss-crispasr", params)


def test_settings_record_round_trips_as_json():
    d = RSCNSettings().as_dict()
    assert json.loads(json.dumps(d))["novelty_scales_s"] == [20.0, 60.0]


# --- Division -----------------------------------------------------------------------------------------------------

def test_cut_goes_to_the_change_and_scenes_divide_exactly(tmp_path):
    audio = np.concatenate([babble(200), tone(250) + noise(250, 0.01)])
    r = detect(tmp_path, audio)
    assert_exact(r, len(audio), SR)
    assert len(r.scenes) == 2
    cut = r.scenes[0].end_sec
    assert abs(cut - 200) <= QUIET.max_cut_shift_s
    assert r.scenes[0].metadata["end_cut"] == "change"
    assert r.scenes[0].metadata["end_strength"] >= QUIET.change_threshold


def test_uniform_sound_is_cut_only_to_fit_the_length(tmp_path):
    audio = noise(1000)
    r = detect(tmp_path, audio)
    assert_exact(r, len(audio), SR)
    lengths = [sc.duration_sec for sc in r.scenes]
    assert all(QUIET.min_scene_s <= x <= QUIET.max_scene_s for x in lengths)
    assert len(r.scenes) == 3
    assert {sc.metadata["end_cut"] for sc in r.scenes[:-1]} == {"to_fit_length"}


def test_uniform_sound_inside_the_preferred_range_is_one_scene(tmp_path):
    audio = noise(330)
    r = detect(tmp_path, audio)
    assert len(r.scenes) == 1 and r.scenes[0].metadata["end_cut"] == "file_end"


def test_uniform_sound_past_the_preferred_maximum_is_cut_into_the_range(tmp_path):
    # 400 s is within the hard maximum but past the preferred 360 s (W451 C2): one cut to fit the length.
    r = detect(tmp_path, noise(400))
    assert len(r.scenes) == 2 and r.scenes[0].metadata["end_cut"] == "to_fit_length"
    assert all(120 - 1e-6 <= sc.duration_sec <= 360 + 1e-6 for sc in r.scenes)


def test_length_does_not_decide_where_inside_the_preferred_range(tmp_path):
    # The reference's 120 s target cut 210 s of unchanged sound in two (study, section 2); the range does not.
    audio = np.concatenate([noise(150), tone(210, level=0.05)])
    r = detect(tmp_path, audio)
    assert len(r.scenes) == 2
    assert abs(r.scenes[0].end_sec - 150) <= QUIET.max_cut_shift_s


def test_a_short_file_is_one_scene_marked(tmp_path):
    audio = noise(40)
    r = detect(tmp_path, audio)
    assert len(r.scenes) == 1 and r.scenes[0].metadata["duration_exception"] == "short_file"


def test_every_scene_within_the_bounds_on_mixed_material(tmp_path):
    parts = [babble(95), tone(40), noise(130, 0.02), babble(300, 0.1), tone(75, 880), noise(260), babble(33)]
    audio = np.concatenate(parts)
    r = detect(tmp_path, audio)
    assert_exact(r, len(audio), SR)
    for sc in r.scenes:
        assert sc.duration_sec <= QUIET.max_scene_s + 1e-6
        assert sc.duration_sec >= QUIET.min_scene_s - 1e-6 or sc.metadata["duration_exception"]


def test_digital_silence_is_its_own_scene_and_handed_on(tmp_path):
    audio = np.concatenate([babble(150), np.zeros(int(2.5 * SR), np.float32), babble(30), tone(200)])
    r = detect(tmp_path, audio)
    assert_exact(r, len(audio), SR)
    kinds = [sc.metadata["rscn_kind"] for sc in r.scenes]
    assert "digital_silence" in kinds
    silence = r.scenes[kinds.index("digital_silence")]
    assert silence.duration_sec == pytest.approx(2.5, abs=1e-3)
    assert silence.metadata["duration_exception"] == "digital_silence"
    # the 30 s stretch between the silence and the change keeps its own scene only if the act allows: no loss either way
    assert sum(sc.duration_sec for sc in r.scenes) == pytest.approx(len(audio) / SR, abs=1e-6)


def test_long_digital_silence_is_divided_within_the_maximum(tmp_path):
    audio = np.concatenate([noise(100), np.zeros(int(900 * SR), np.float32), noise(100)])
    r = detect(tmp_path, audio)
    assert_exact(r, len(audio), SR)
    silent = [sc for sc in r.scenes if sc.metadata["rscn_kind"] == "digital_silence"]
    assert len(silent) == 3 and all(sc.duration_sec <= 420 for sc in silent)


def test_48_khz_audio_keeps_its_rate_and_is_divided_exactly(tmp_path):
    sr = 48000
    audio = np.concatenate([babble(150, sr=sr), tone(150, sr=sr) + noise(150, 0.01, sr=sr)])
    r = detect(tmp_path, audio, sr=sr)
    assert_exact(r, len(audio), sr)
    assert len(r.scenes) == 2 and abs(r.scenes[0].end_sec - 150) <= QUIET.max_cut_shift_s


def test_run_record_holds_settings_scenes_and_validation(tmp_path):
    audio = np.concatenate([babble(200), tone(200)])
    r = detect(tmp_path, audio)
    record = json.loads((tmp_path / "a_scenes" / "a_rscn.json").read_text(encoding="utf-8"))
    assert record["settings"]["min_scene_s"] == 70
    assert record["validation"]["scene_samples"] == len(audio)
    assert len(record["scenes"]) == len(r.scenes)
    assert r.parameters["record"].endswith("a_rscn.json")


# --- Speech measure (contract) --------------------------------------------------------------------------------------

def test_cut_is_kept_out_of_speech(tmp_path):
    # The change is at 200 s, inside a speech stretch 196-203.5 s; the cut must leave it.
    audio = np.concatenate([noise(200), tone(200)])
    fake = FakeMeasure([SpeechStretch(100, 196, 0.9), SpeechStretch(196, 203.5, 0.9), SpeechStretch(205, 300, 0.9)])
    s = RSCNSettings(speech_measure="none", picture="off")
    r = detect(tmp_path, audio, settings=s, measure=fake)
    cut = r.scenes[0].end_sec
    assert not (196 - s.speech_guard_s <= cut <= 203.5 + s.speech_guard_s)
    assert abs(cut - 200) <= s.max_cut_shift_s
    assert r.scenes[0].metadata["end_placement"] == "speech_gap"
    assert r.scenes[0].metadata["speech_share"] is not None


def test_speech_marks(tmp_path):
    audio = np.concatenate([noise(200), tone(200)])
    fake = FakeMeasure([SpeechStretch(10, 150, 0.8)])
    r = detect(tmp_path, audio, measure=fake)
    first, second = r.scenes
    assert first.metadata["speech_share"] == pytest.approx(140 / first.duration_sec, abs=0.01)
    assert second.metadata["little_or_no_dialogue"] is True


def test_segmenter_adapter_runs_in_pieces_and_offsets_times():
    calls = []

    class Seg:
        def segment(self, audio, sample_rate=16000):
            calls.append(len(audio) / sample_rate)
            return SimpleNamespace(error=None, segments=[SimpleNamespace(start_sec=1.0, end_sec=2.0, confidence=0.7)])

        def cleanup(self):
            calls.append("closed")

    factory = SimpleNamespace(create=lambda name, **kw: Seg())
    m = SegmenterSpeechMeasure("fake-vad", piece_s=10, factory=factory)
    out = m.measure(np.zeros(25 * SR, np.float32))
    m.close()
    assert calls[:3] == [10.0, 10.0, 5.0] and calls[-1] == "closed"
    assert [(x.start_s, x.end_s) for x in out.stretches] == [(1.0, 2.0), (11.0, 12.0), (21.0, 22.0)]
    assert out.stretches[0].probability == pytest.approx(0.7)


def test_segmenter_adapter_records_a_failed_piece():
    class Seg:
        n = 0

        def segment(self, audio, sample_rate=16000):
            Seg.n += 1
            if Seg.n == 2:
                raise RuntimeError("boom")
            return SimpleNamespace(error=None, segments=[])

        def cleanup(self):
            pass

    m = SegmenterSpeechMeasure("fake-vad", piece_s=10, factory=SimpleNamespace(create=lambda name, **kw: Seg()))
    out = m.measure(np.zeros(30 * SR, np.float32))
    assert len(out.failed_pieces) == 1 and out.failed_pieces[0][:2] == (10.0, 20.0)


def test_a_speech_measure_that_cannot_be_built_fails_the_file(tmp_path):
    def broken(name, **kw):
        raise ImportError("no such backend")

    measure = SegmenterSpeechMeasure("fake-vad", factory=SimpleNamespace(create=broken))
    with pytest.raises(SceneDetectionError, match="speech measure"):
        detect(tmp_path, noise(100), measure=measure)


def test_create_speech_measure_none_and_named():
    assert create_speech_measure("none") is None
    m = create_speech_measure("firered-vad", 120, {"threshold": 0.3})
    assert isinstance(m, SegmenterSpeechMeasure) and m.backend == "firered-vad" and m.piece_s == 120


# --- Picture -------------------------------------------------------------------------------------------------------

def _black(start, end):
    return picture.BlackStretch(start, end, start, end)


def test_confirmed_black_is_a_major_break(tmp_path):
    # Two different sounds meet at 250 s with black over 249-251 s and no speech there.
    audio = np.concatenate([babble(250), tone(250)])
    s = RSCNSettings(speech_measure="none", picture="off")
    m = analysis.measures(audio, s)
    plans, record = partition.plan(audio, SR, s, m, [], [_black(249.0, 251.0)])
    assert record["black"][0]["confirmed"] is True
    cuts = [p for p in plans if p.end_cut.reason == "major_break"]
    assert len(cuts) == 1 and 249.0 <= cuts[0].end_sample / SR <= 251.0


def test_black_with_speech_over_it_is_not_a_break(tmp_path):
    audio = np.concatenate([babble(250), tone(250)])
    s = RSCNSettings(speech_measure="none", picture="off")
    m = analysis.measures(audio, s)
    plans, record = partition.plan(audio, SR, s, m, [SpeechStretch(245, 255)], [_black(249.0, 251.0)])
    assert record["black"][0]["confirmed"] is False
    assert all(p.end_cut.reason != "major_break" for p in plans)


def test_black_over_unchanged_sound_is_not_a_break():
    audio = noise(500)
    s = RSCNSettings(speech_measure="none", picture="off")
    m = analysis.measures(audio, s)
    plans, record = partition.plan(audio, SR, s, m, [], [_black(249.0, 251.0)])
    assert record["black"][0]["confirmed"] is False


def test_picture_modes():
    audio_only = MediaRecord(path="x.m4a", probed=True, has_picture=False)
    film = MediaRecord(path="x.mkv", probed=True, has_picture=True, picture_stream=0)
    assert picture.decide("off", film) == (False, "picture analysis off by setting")
    assert picture.decide("auto", audio_only)[0] is False
    assert picture.decide("auto", None)[0] is False
    assert picture.decide("auto", film)[0] is True
    with pytest.raises(picture.PictureError):
        picture.decide("on", audio_only)
    with pytest.raises(picture.PictureError):
        picture.decide("on", None)


def test_picture_on_without_a_picture_fails_the_file(tmp_path):
    s = RSCNSettings(speech_measure="none", picture="on")
    with pytest.raises(SceneDetectionError, match="no picture"):
        detect(tmp_path, noise(100), settings=s, media=MediaRecord(path="x.m4a", probed=True))


def test_media_record_time_base():
    m = MediaRecord(path="x", probed=True, audio_start_s=2.936, container_start_s=0.0)
    assert m.audio_offset_s == pytest.approx(2.936)
    assert m.to_wav_time(6.0) == pytest.approx(3.064)
    assert MediaRecord(path="x").audio_offset_s == 0.0


def test_media_record_save_and_load(tmp_path):
    m = MediaRecord(path="x.mkv", probed=True, has_picture=True, picture_stream=0, audio_start_s=1.0,
                    container_start_s=0.5)
    m2 = MediaRecord.load(m.save(tmp_path / "m.json"))
    assert m2 == m


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-500:]


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not found")
def test_media_record_and_black_picture_on_a_real_file(tmp_path):
    # 20 s of grey picture, black 6-9 s; the sound starts 3 s after the picture.
    v, a, mkv = tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "m.mkv"
    sf.write(a, noise(17), SR)
    _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=160x120:r=25:d=20",
          "-vf", "drawbox=c=black:t=fill:enable='between(t,6,9)'", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(v)])
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(v), "-itsoffset", "3", "-i", str(a), "-map", "0:v", "-map", "1:a",
          "-c:v", "copy", "-c:a", "pcm_s16le", str(mkv)])
    record = probe(mkv)
    assert record.probed and record.has_picture and record.picture_stream == 0
    assert record.audio_offset_s == pytest.approx(3.0, abs=0.05)
    found, _ = picture.black_stretches(record, RSCNSettings(black_min_s=1.0), wav_duration_s=17.0)
    assert len(found) == 1
    assert found[0].container_start_s == pytest.approx(6.0, abs=0.05)
    assert found[0].start_s == pytest.approx(3.0, abs=0.1) and found[0].end_s == pytest.approx(6.0, abs=0.15)


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not found")
def test_an_audio_file_has_no_picture_and_a_cover_image_is_not_one(tmp_path):
    a, cover, m4a = tmp_path / "a.wav", tmp_path / "c.png", tmp_path / "c.m4a"
    sf.write(a, noise(3), SR)
    assert probe(a).has_picture is False
    _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=32x32:d=1", "-frames:v", "1", str(cover)])
    # An .m4a with the cover as an attached picture: ffprobe lists it as a video stream.
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(a), "-i", str(cover), "-map", "0:a", "-map", "1:v", "-c:a", "aac",
          "-c:v", "png", "-disposition:v:0", "attached_pic", str(m4a)])
    record = probe(m4a)
    assert any(s["type"] == "video" for s in record.streams)
    assert record.has_picture is False


# --- Registration and wiring ---------------------------------------------------------------------------------------

def test_factory_knows_rscn():
    assert "rscn" in SceneDetectorFactory.list_backends()
    assert SceneDetectorFactory.is_backend_available("rscn") == (True, "")
    det = SceneDetectorFactory.create_from_legacy_kwargs(method="rscn", pipeline="moss-crispasr",
                                                         rscn_min_scene_s=60, max_duration=420)
    assert det.name == "rscn" and det.settings.min_scene_s == 60


def test_invalid_rscn_setting_is_not_hidden_by_the_fallback_to_auditok():
    with pytest.raises(ValueError):
        SceneDetectorFactory.safe_create_from_legacy_kwargs(method="rscn", rscn_max_scene_s=10)


def test_crispasr_pipeline_passes_pipeline_and_rscn_keys(tmp_path):
    from whisperjav.pipelines.moss_crispasr_pipeline import MossCrispASRPipeline

    pipe = MossCrispASRPipeline(output_dir=str(tmp_path / "o"), temp_dir=str(tmp_path / "t"), scene_detector="rscn",
                                scene_params={"rscn_min_scene_s": 80})
    kwargs = pipe.scene_detector_kwargs()
    assert kwargs == {"method": "rscn", "pipeline": "moss-crispasr", "rscn_min_scene_s": 80}


def test_crispasr_pipeline_hands_the_media_record_to_rscn(tmp_path, monkeypatch):
    from whisperjav.pipelines import crispasr_pipeline
    from whisperjav.pipelines.moss_crispasr_pipeline import MossCrispASRPipeline

    pipe = MossCrispASRPipeline(output_dir=str(tmp_path / "o"), temp_dir=str(tmp_path / "t"), scene_detector="rscn")
    seen = {}

    class Det:
        def detect_scenes(self, wav, out, base, **kw):
            seen.update(kw)
            from whisperjav.modules.scene_detection_backends import SceneDetectionResult, SceneInfo
            return SceneDetectionResult([SceneInfo(0.0, 1.0, tmp_path / "s.wav")], "rscn",
                                        parameters={"record": "r.json"})

        def cleanup(self):
            pass

    from whisperjav.modules import media_record
    from whisperjav.modules.scene_detection_backends import factory
    made = {}

    def create(name, **k):
        made.update(name=name, **k)
        return Det()

    # rscn is built with create(), never safe_create(): no silent fall-back to auditok (review finding 5).
    monkeypatch.setattr(factory.SceneDetectorFactory, "create", staticmethod(create))
    monkeypatch.setattr(factory.SceneDetectorFactory, "safe_create_from_legacy_kwargs",
                        staticmethod(lambda **k: pytest.fail("rscn must not go through the fall-back")))
    monkeypatch.setattr(media_record, "probe", lambda p: MediaRecord(path=str(p), probed=True))
    md = {"stages": {}, "input_file": str(tmp_path / "film.mkv")}
    pipe._detect_scenes(tmp_path / "x.wav", "film", md)
    assert isinstance(seen.get("media"), MediaRecord)
    assert Path(md["media_record"]).name == "film_media.json" and md["rscn_record"] == "r.json"
    assert made == {"name": "rscn", "pipeline": "moss-crispasr"}
    assert md["rscn"]["scenes"][0]["end_s"] == 1.0


def test_other_scene_detectors_get_no_media_record(tmp_path):
    from whisperjav.pipelines.moss_crispasr_pipeline import MossCrispASRPipeline

    pipe = MossCrispASRPipeline(output_dir=str(tmp_path / "o"), temp_dir=str(tmp_path / "t"))
    assert pipe.scene_detector == "silero"  # MOSS's default is unchanged (W426 4)
    assert pipe._media_record("film", {"input_file": "film.mkv"}) is None


def test_pass_check_refuses_rscn_outside_the_gguf_pipelines_and_bad_settings():
    from whisperjav.crispasr.pass_settings import rscn_problem

    assert rscn_problem(1, "moss-crispasr", "rscn", None) is None
    assert rscn_problem(1, "moss-crispasr", "silero", '{"rscn_bogus": 1}') is None
    assert "only when --pass1-pipeline" in rscn_problem(1, "balanced", "rscn", None)
    assert "only when --pass1-pipeline is moss-crispasr" in rscn_problem(1, "cohere-crispasr", "rscn", None)
    assert "rscn_bogus" in rscn_problem(2, "moss-crispasr", "rscn", '{"rscn_bogus": 1}')
    assert "rscn_speech_measure" in rscn_problem(1, "moss-crispasr", "rscn", '{"rscn_speech_measure": "nope"}')


# --- Review findings (adversary review of the design, 2026-11-02) ----------------------------------------------------

def test_confirmed_black_too_near_the_start_is_only_a_candidate():
    # A logo then black at 12-14 s: a fixed cut there would make a 13 s scene; it becomes a candidate instead.
    audio = np.concatenate([tone(13), babble(300)])
    s = RSCNSettings(speech_measure="none", picture="off")
    m = analysis.measures(audio, s)
    plans, record = partition.plan(audio, SR, s, m, [], [_black(12.0, 14.0)])
    assert all(p.end_sample - p.start_sample >= s.min_scene_s * SR for p in plans)
    assert "candidate" in record["black"][0]["why"] or record["black"][0]["confirmed"] is False


def test_black_next_to_digital_silence_is_part_of_that_silence():
    audio = np.concatenate([babble(150), np.zeros(3 * SR, np.float32), tone(150)])
    s = RSCNSettings(speech_measure="none", picture="off")
    m = analysis.measures(audio, s)
    plans, record = partition.plan(audio, SR, s, m, [], [_black(152.5, 154.5)])
    assert record["black"][0]["why"].startswith("next to digital silence")
    assert all(p.end_cut.reason != "major_break" for p in plans)


def test_a_click_between_two_silences_joins_the_silence(tmp_path):
    click = noise(0.2)
    audio = np.concatenate([babble(150), np.zeros(3 * SR, np.float32), click, np.zeros(3 * SR, np.float32),
                            tone(150)])
    r = detect(tmp_path, audio)
    assert_exact(r, len(audio), SR)
    silent = [sc for sc in r.scenes if sc.metadata["rscn_kind"] == "digital_silence"]
    assert len(silent) == 1 and silent[0].duration_sec == pytest.approx(6.2, abs=1e-3)
    assert silent[0].metadata["short_audio_inside_s"] == pytest.approx(0.2, abs=1e-3)
    assert all(sc.duration_sec >= 1.0 for sc in r.scenes)


def test_speech_pieces_overlap_and_are_merged():
    seen = []

    class Seg:
        def segment(self, audio, sample_rate=16000):
            seen.append(len(audio) / sample_rate)
            # speech over the whole piece: across every edge
            return SimpleNamespace(error=None, segments=[SimpleNamespace(start_sec=0.0, end_sec=len(audio) / sample_rate,
                                                                         confidence=0.9)])

        def cleanup(self):
            pass

    m = SegmenterSpeechMeasure("fake-vad", piece_s=10, overlap_s=2, factory=SimpleNamespace(create=lambda n, **k: Seg()))
    out = m.measure(np.zeros(25 * SR, np.float32))
    assert seen == [12.0, 12.0, 5.0]
    assert [(x.start_s, x.end_s) for x in out.stretches] == [(0.0, 25.0)]


def test_a_preferred_range_scene_is_not_cut_short_for_a_weak_change(tmp_path):
    # Ordinary variation (babble at two levels 3 dB apart) is not a strong change: 300 s stays one scene.
    audio = np.concatenate([babble(150, 0.05), babble(150, 0.07)])
    r = detect(tmp_path, audio)
    assert len(r.scenes) == 1


def test_picture_auto_carries_on_when_the_picture_cannot_be_read(tmp_path, monkeypatch):
    def broken(media, s, duration):
        raise picture.PictureError("FFmpeg's black-picture detection failed: damaged stream")

    monkeypatch.setattr(picture, "black_stretches", broken)
    film = MediaRecord(path="x.mkv", probed=True, has_picture=True, picture_stream=0)
    r = detect(tmp_path, noise(100), settings=RSCNSettings(speech_measure="none", picture="auto"), media=film)
    assert len(r.scenes) == 1
    with pytest.raises(SceneDetectionError, match="damaged"):
        detect(tmp_path, noise(100), settings=RSCNSettings(speech_measure="none", picture="on"), media=film,
               name="b")


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not found")
def test_time_base_of_an_mpeg_ts_file(tmp_path):
    # MPEG-TS starts its clock above 0 (usually 1.4 s): the black times and the sound start share it.
    v, a, ts = tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "m.ts"
    sf.write(a, noise(18), SR)
    _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=160x120:r=25:d=20",
          "-vf", "drawbox=c=black:t=fill:enable='between(t,6,9)'", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(v)])
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(v), "-itsoffset", "2", "-i", str(a), "-map", "0:v", "-map", "1:a",
          "-c:v", "copy", "-c:a", "mp2", str(ts)])
    record = probe(ts)
    assert record.container_start_s and record.container_start_s > 0
    found, _ = picture.black_stretches(record, RSCNSettings(black_min_s=1.0), wav_duration_s=18.0)
    assert len(found) == 1
    # black 6-9 s of picture time; the sound starts 2 s after the picture, so 4-7 s of WAV time
    assert found[0].start_s == pytest.approx(4.0, abs=0.15) and found[0].end_s == pytest.approx(7.0, abs=0.2)
