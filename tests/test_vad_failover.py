import sys
import types

if 'stable_whisper' not in sys.modules:
    stub = types.ModuleType('stable_whisper')

    class _DummyResult:  # minimal placeholder for type hints
        pass

    def _unavailable(*args, **kwargs):  # prevents accidental runtime use
        raise RuntimeError("stable_whisper is not available in test environment")

    stub.WhisperResult = _DummyResult
    stub.load_model = _unavailable
    stub.load_faster_whisper = _unavailable
    sys.modules['stable_whisper'] = stub

from whisperjav.modules.vad_failover import should_force_full_transcribe


def test_should_force_full_when_no_segments_and_long_clip():
    assert should_force_full_transcribe([], audio_duration=600.0)


def test_should_not_force_for_short_clip_without_segments():
    assert not should_force_full_transcribe([], audio_duration=60.0)


def test_should_force_when_coverage_ratio_too_low():
    vad_segments = [[{'start_sec': 10.0, 'end_sec': 12.0}]]
    assert should_force_full_transcribe(vad_segments, audio_duration=1200.0)
