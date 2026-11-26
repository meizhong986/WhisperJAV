"""The shortest scene (Min Duration) groups; no scene detector drops a shorter piece (1.10.3, owner W808 D1, W813, W814).

W808 D1: "The objective of  Min Duration is that smaller pieces would be grouped until it reaches above more than the
Minimum Duration ... If any of the scene detectors drop scenes less than that, it means that is a bug". Before 1.10.3
the auditok and silero scene detectors dropped a piece shorter than Min Duration, with its speech (a 1.10.2 known
fault for ChronosJAV and decoupled with 12 s); now a piece shorter than Min Duration is joined to the neighbour with the
shorter silence between them (utils.join_short_scenes), first within Max Duration, else past it by at most Min Duration
(W813 Q2 (a)); a scene's length runs from its start to its end (Q3 (a)); ChronosJAV, decoupled and the GGUF pipelines
give their minimum to every scene detector (Q4 (a)).

Run with: python -m pytest tests/test_short_scenes_kept.py -q
"""
import wave
from pathlib import Path

import numpy as np
import pytest

from whisperjav.modules import scene_detection_backends
from whisperjav.modules.scene_detection_backends import SceneDetectorFactory
from whisperjav.modules.scene_detection_backends.utils import brute_force_split, join_short_scenes

REPO = Path(__file__).resolve().parents[1]
SR = 16000


# ---------------------------------------------------------------------------
# The rule: join_short_scenes
# ---------------------------------------------------------------------------

def _flat(groups):
    return sorted(i for g in groups for i in g)


def test_a_short_piece_is_joined_never_dropped():
    groups = join_short_scenes([(0, 5), (6, 30)], 12, 48)
    assert groups == [[0, 1]]


def test_it_joins_the_neighbour_with_the_shorter_silence():
    # the 3 s piece is 2 s after the first scene and 15 s before the third
    groups = join_short_scenes([(0, 20), (22, 25), (40, 60)], 12, 48)
    assert groups == [[0, 1], [2]]
    # on a tie, the one before
    assert join_short_scenes([(0, 20), (25, 28), (33, 60)], 12, 48) == [[0, 1], [2]]


def test_short_pieces_are_grouped_until_the_minimum_is_reached():
    spans = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (40, 70)]
    groups = join_short_scenes(spans, 12, 48)
    assert _flat(groups) == list(range(len(spans)))
    first = groups[0]
    assert spans[first[-1]][1] - spans[first[0]][0] >= 12


def test_past_max_duration_by_at_most_min_duration():
    # W813 Q2 (a): joined past the longest scene, by at most the shortest scene's length
    assert join_short_scenes([(0, 45), (46, 50)], 12, 48) == [[0, 1]]          # 50 s <= 48 + 12
    # a piece that cannot be joined within that stays a scene of its own (not dropped)
    assert join_short_scenes([(0, 45), (70, 74)], 12, 48) == [[0], [1]]          # 74 s > 60


def test_within_max_first():
    # the short piece has a closer neighbour that would pass the maximum and a farther one that would not
    groups = join_short_scenes([(0, 46), (47, 50), (55, 80)], 12, 48)
    assert groups == [[0], [1, 2]]


def test_one_piece_and_no_minimum():
    assert join_short_scenes([(0, 3)], 12, 48) == [[0]]
    assert join_short_scenes([(0, 3), (4, 5)], 0, 48) == [[0], [1]]
    assert join_short_scenes([], 12, 48) == []


def test_the_last_fixed_length_piece_is_kept():
    pieces = brute_force_split(0.0, 50.0, 20.0, 12.0)
    assert [(p.start_sec, p.end_sec) for p in pieces] == [(0.0, 20.0), (20.0, 40.0), (40.0, 50.0)]


# ---------------------------------------------------------------------------
# The detectors, on made-up audio
# ---------------------------------------------------------------------------

def _write_wav(path: Path) -> Path:
    """30 s: 1 s silence, 15 s of loud noise (a long scene), 4 s silence, 5 s of noise (a short scene), 5 s silence."""
    rng = np.random.default_rng(0)
    audio = np.zeros(30 * SR, dtype=np.float32)
    audio[1 * SR:16 * SR] = rng.uniform(-0.4, 0.4, 15 * SR).astype(np.float32)
    audio[20 * SR:25 * SR] = rng.uniform(-0.4, 0.4, 5 * SR).astype(np.float32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    return path


def _scenes(tmp_path, method, **kwargs):
    wav = _write_wav(tmp_path / "short.wav")
    out = tmp_path / (method + "_" + "_".join(f"{k}{v}" for k, v in sorted(kwargs.items())))
    out.mkdir()
    detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(method=method, **kwargs)
    return detector.detect_scenes(wav, out, "short")


def _covers(result, t):
    return any(sc.start_sec <= t <= sc.end_sec for sc in result.scenes)


@pytest.mark.parametrize("method", ["auditok", "silero"])
def test_a_12s_minimum_keeps_the_short_scene_by_joining_it(tmp_path, method):
    result = _scenes(tmp_path, method, max_duration=48, min_duration=12)
    assert _covers(result, 8.0) and _covers(result, 22.5)          # the 1.10.2 fault lost the second one
    assert len(result.scenes) == 1
    scene = result.scenes[0]
    assert scene.duration_sec >= 12 and scene.metadata.get("joined_pieces") == 2
    assert Path(scene.scene_path).is_file()


@pytest.mark.parametrize("method", ["auditok", "silero"])
def test_with_a_small_minimum_both_scenes_stay_apart(tmp_path, method):
    result = _scenes(tmp_path, method, max_duration=48, min_duration=0.2)
    assert _covers(result, 8.0) and _covers(result, 22.5)
    assert len(result.scenes) == 2


# ---------------------------------------------------------------------------
# The pipelines give their minimum to every scene detector (W813 Q4 (a))
# ---------------------------------------------------------------------------

def test_the_special_case_is_gone():
    assert not hasattr(scene_detection_backends, "MERGING_SCENE_DETECTORS")


def test_chronosjav_and_decoupled_give_the_minimum_to_every_detector():
    qwen = (REPO / "whisperjav/pipelines/qwen_pipeline.py").read_text(encoding="utf-8")
    assert qwen.count('scene_detector_kwargs["min_duration"] = min_dur') == 1
    assert "MERGING_SCENE_DETECTORS" not in qwen and "the scene detector's own" not in qwen
    decoupled = (REPO / "whisperjav/pipelines/decoupled_pipeline.py").read_text(encoding="utf-8")
    assert decoupled.count('scene_detector_kwargs["min_duration"] = self.scene_min_duration') == 1
    assert "MERGING_SCENE_DETECTORS" not in decoupled and "the scene detector's own" not in decoupled


def test_the_gguf_pipelines_give_it_to_auditok_and_silero():
    src = (REPO / "whisperjav/pipelines/crispasr_pipeline.py").read_text(encoding="utf-8")
    assert 'if self.scene_detector in ("semantic", "auditok", "silero"):' in src
    api = (REPO / "whisperjav/webview_gui/api.py").read_text(encoding="utf-8")
    assert '"min_duration": shortest})' in api


def test_the_texts():
    """W814: the approved texts."""
    main = (REPO / "whisperjav/main.py").read_text(encoding="utf-8")
    assert ('"Shortest scene in seconds; a shorter piece is joined to its neighbour, never "\n'
            '                                "dropped (default: 12)"') in main
    api = (REPO / "whisperjav/webview_gui/api.py").read_text(encoding="utf-8")
    zh = (REPO / "whisperjav/webview_gui/assets/i18n_zh.js").read_text(encoding="utf-8")
    text = "Shortest scene: a shorter piece is joined to its neighbour (default: 12 s)"
    assert f'"description": "{text}"' in api
    assert f'"{text}": "最短场景长度：更短的片段会并入相邻片段（默认：12 秒）"' in zh
    assert "Minimum scene length (default: 12s)" not in api + zh
    guide = (REPO / "docs/en/pipelines/crispasr.md").read_text(encoding="utf-8")
    assert ("`semantic` (default), `auditok`, `silero`: scenes of normally 12 to 48 s, a shorter piece joined to its "
            "neighbour (MOSS-CrispASR: 90 to 240 s)") in guide
    shipped = (REPO / "whisperjav/config/components/features/scene_detection.py").read_text(encoding="utf-8")
    assert "DISCARD filter" not in shipped
