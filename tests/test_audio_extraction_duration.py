#!/usr/bin/env python3
"""
Tests for how AudioExtractor works out the duration of the audio it extracted.

Until v1.9.3 it ran FFmpeg a second time with "-f null -", which decodes the
whole extracted file and discards the samples, purely to read one line off the
front of the log. These tests pin the replacement: the header is read, nothing is
decoded, and the answer is the same one the old method gave.

Run with: pytest tests/test_audio_extraction_duration.py -v
"""

import shutil
import subprocess

import pytest

from whisperjav.modules.audio_extraction import AudioExtractor

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="FFmpeg is not installed")

TONE_SECONDS = 5


@pytest.fixture(scope="module")
def extractor():
    return AudioExtractor()


@pytest.fixture(scope="module")
def source_media(tmp_path_factory, extractor):
    """A real, short media file with a known duration."""
    path = tmp_path_factory.mktemp("media") / "tone.mp4"
    subprocess.run([
        extractor.ffmpeg_path, "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={TONE_SECONDS}",
        "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=5:duration={TONE_SECONDS}",
        "-c:a", "aac", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-shortest", str(path)
    ], check=True, capture_output=True)
    return path


class TestDuration:
    def test_extract_reports_the_real_duration(self, extractor, source_media, tmp_path):
        out = tmp_path / "extracted.wav"
        path, duration = extractor.extract(source_media, out)
        assert path == out and out.is_file()
        assert duration == pytest.approx(TONE_SECONDS, abs=0.2)

    def test_it_matches_what_a_full_decode_would_say(self, extractor, source_media, tmp_path):
        """The old method's answer, computed the old way, for comparison."""
        out = tmp_path / "extracted.wav"
        extractor.extract(source_media, out)

        old = subprocess.run(
            [extractor.ffmpeg_path, "-i", str(out), "-hide_banner", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        from_full_decode = AudioExtractor._parse_ffmpeg_duration(old.stderr)

        assert extractor._get_audio_duration(out) == pytest.approx(from_full_decode, abs=0.05)

    def test_the_fallback_reads_the_extraction_log(self, extractor, source_media, tmp_path):
        """
        With no ffprobe on the machine, the duration comes out of the log FFmpeg
        already produced while extracting -- still without decoding anything.
        """
        out = tmp_path / "extracted.wav"
        extractor.extract(source_media, out)
        probe_output = subprocess.run(
            [extractor.ffmpeg_path, "-i", str(source_media)],
            capture_output=True, text=True, encoding="utf-8", errors="replace").stderr

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(extractor, "_find_ffprobe", lambda: None)
            duration = extractor._get_audio_duration(out, probe_output)
        assert duration == pytest.approx(TONE_SECONDS, abs=0.2)

    def test_it_returns_zero_when_nothing_can_tell_it(self, extractor, tmp_path):
        """Unreadable file, no log: the same answer as before -- 0.0, not a crash."""
        missing = tmp_path / "not-audio.wav"
        missing.write_bytes(b"this is not a wav file")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(extractor, "_find_ffprobe", lambda: None)
            assert extractor._get_audio_duration(missing) == 0.0


class TestDurationParsing:
    def test_reads_the_duration_line(self):
        assert AudioExtractor._parse_ffmpeg_duration(
            "  Duration: 01:23:45.67, start: 0.000000, bitrate: 256 kb/s"
        ) == pytest.approx(5025.67)

    def test_handles_an_unknown_duration(self):
        assert AudioExtractor._parse_ffmpeg_duration(
            "  Duration: N/A, start: 0.000000, bitrate: N/A") == 0.0

    def test_handles_output_with_no_duration_at_all(self):
        assert AudioExtractor._parse_ffmpeg_duration("ffmpeg version 7.1\n") == 0.0
