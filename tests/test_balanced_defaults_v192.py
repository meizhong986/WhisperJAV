"""
v1.9.2: Balanced pipeline speech-segmenter default and single-pass preset resolution.

Owner decisions: CFF3 (2026-09-05) made FireRedVAD the Balanced default with a
fallback chain; the owner reversed the default on 2026-09-05 (N3) — Balanced runs
faster-whisper's built-in VAD again, as in v1.9.0/v1.9.1. Kept from CFF3 (owner,
2026-09-06, "default only"): the single-pass path resolves a WhisperJAV segmenter's
per-sensitivity YAML preset, and firered-vad / ten are exempt from the routing-guard
downgrade on --mode balanced, so an explicit choice honours --sensitivity. Test-D
grouping stays on top of the preset (D3).

Two layers are covered without importing any ML stack:
1. `whisperjav.config.segmenter_presets` — the shared constants and resolver the
   entry points (main.py, pass_worker, GUI api) use.
2. The real CLI through `--dump-params`, which exercises main.py's resolution
   order backend → YAML preset → Test-D → CLI override, and the routing guard.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from whisperjav.config.segmenter_presets import (
    BALANCED_DEFAULT_SEGMENTER,
    BALANCED_SINGLE_PASS_EXTERNAL,
    SEGMENTER_PARAMS,
    resolve_segmenter_sensitivity,
)
from whisperjav.utils.output_coverage import PASSTHROUGH_SEGMENTERS


# ─── helper layer ────────────────────────────────────────────────────────────

class TestBalancedDefault:
    def test_default_is_the_built_in_vad(self):
        # Owner N3 (2026-09-05): reversal of CFF3.
        assert BALANCED_DEFAULT_SEGMENTER == "faster-whisper"

    def test_built_in_vad_has_no_external_preset(self):
        # Consequence for the run-outcome contract: under the built-in VAD the
        # recogniser reports its segmenter as "none" (NullSpeechSegmenter), which
        # SpeechPositiveEmptyStreak treats as a passthrough, so a zero-cue Balanced
        # file is `empty`, never `suspect`, unless a WhisperJAV segmenter is chosen
        # (tests/test_output_coverage.py covers the counter itself).
        assert resolve_segmenter_sensitivity(BALANCED_DEFAULT_SEGMENTER, "balanced") == {}
        assert "none" in PASSTHROUGH_SEGMENTERS

    def test_guard_exemption_is_firered_and_ten_only(self):
        assert BALANCED_SINGLE_PASS_EXTERNAL == frozenset({"firered-vad", "ten"})
        assert BALANCED_DEFAULT_SEGMENTER not in BALANCED_SINGLE_PASS_EXTERNAL

    def test_exempt_backends_are_real_segmenters_with_presets(self):
        for name in BALANCED_SINGLE_PASS_EXTERNAL:
            assert name.lower() not in PASSTHROUGH_SEGMENTERS
            assert resolve_segmenter_sensitivity(name, "balanced")  # non-empty preset


class TestFireRedPresetResolution:
    @pytest.mark.parametrize("sensitivity,threshold,max_speech,end_pad", [
        ("conservative", 0.5, 7, 250),
        ("balanced", 0.4, 6, 150),
        ("aggressive", 0.3, 5, 100),
    ])
    def test_firered_yaml_rows(self, sensitivity, threshold, max_speech, end_pad):
        preset = resolve_segmenter_sensitivity("firered-vad", sensitivity)
        assert preset["threshold"] == threshold
        assert preset["max_speech_duration_s"] == max_speech
        assert preset["end_pad_ms"] == end_pad
        assert set(preset) <= SEGMENTER_PARAMS

    def test_native_and_none_resolve_to_nothing(self):
        assert resolve_segmenter_sensitivity("faster-whisper", "aggressive") == {}
        assert resolve_segmenter_sensitivity("none", "aggressive") == {}


# ─── CLI layer (--dump-params, no model load) ───────────────────────────────

def _dump(tmp_path: Path, *args: str) -> tuple[dict, str]:
    out = tmp_path / f"dump_{abs(hash(args))}.json"
    proc = subprocess.run(
        [sys.executable, "-m", "whisperjav.main", "--dump-params", str(out), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    assert out.exists(), f"dump not written; stderr:\n{proc.stderr[-2000:]}"
    return json.loads(out.read_text(encoding="utf-8")), proc.stdout + proc.stderr


def _segmenter(dump: dict) -> dict:
    return dump["resolved_config"]["params"].get("speech_segmenter") or {}


@pytest.mark.slow
class TestBalancedCliDefaults:
    @pytest.mark.parametrize("sensitivity", ["conservative", "balanced", "aggressive"])
    def test_default_is_built_in_vad_with_native_preset(self, tmp_path, sensitivity):
        dump, _ = _dump(tmp_path, "--mode", "balanced", "--sensitivity", sensitivity)
        assert _segmenter(dump)["backend"] == "faster-whisper"
        # The native faster_whisper_vad preset is kept (scale-correct for the
        # bundled Silero; the firewall mirror only clears it for external backends).
        vad = dump["resolved_config"]["params"]["vad"]
        assert vad["threshold"] > 0
        assert "_dump_note" not in dump["resolved_config"]["params"]

    @pytest.mark.parametrize("sensitivity,threshold", [
        ("conservative", 0.5), ("balanced", 0.4), ("aggressive", 0.3),
    ])
    def test_explicit_firered_gets_preset_then_test_d(self, tmp_path, sensitivity, threshold):
        dump, log = _dump(tmp_path, "--mode", "balanced", "--speech-segmenter", "firered-vad",
                          "--sensitivity", sensitivity)
        ss = _segmenter(dump)
        assert ss["backend"] == "firered-vad"                 # not downgraded
        assert ss["threshold"] == threshold                  # YAML preset
        assert ss["max_group_duration_s"] == 9.0             # Test-D (D3) overrides YAML 7/6/5
        assert ss["chunk_threshold_s"] == 0.1                # Test-D (D3) overrides YAML 1.0
        assert "vad" not in dump["resolved_config"]["params"]
        assert "_dump_note" in dump["resolved_config"]["params"]
        assert "Falling back to silero-v3.1" not in log

    def test_ten_is_exempt_from_the_routing_guard_on_balanced(self, tmp_path):
        dump, log = _dump(tmp_path, "--mode", "balanced", "--speech-segmenter", "ten",
                          "--sensitivity", "aggressive")
        ss = _segmenter(dump)
        assert ss["backend"] == "ten"
        assert "threshold" in ss                             # preset resolved
        assert ss["max_group_duration_s"] == 9.0
        assert "Falling back to silero-v3.1" not in log

    def test_whisperseg_still_downgrades_on_single_pass_balanced(self, tmp_path):
        dump, log = _dump(tmp_path, "--mode", "balanced", "--speech-segmenter", "whisperseg")
        assert _segmenter(dump)["backend"] == "silero-v3.1"
        assert "Falling back to silero-v3.1" in log

    def test_cli_overrides_win_over_preset_and_test_d(self, tmp_path):
        dump, _ = _dump(tmp_path, "--mode", "balanced", "--speech-segmenter", "firered-vad",
                        "--vad-threshold", "0.6", "--max-group-duration", "6")
        ss = _segmenter(dump)
        assert ss["backend"] == "firered-vad"
        assert ss["threshold"] == 0.6
        assert ss["max_group_duration_s"] == 6.0

    def test_fidelity_default_unchanged(self, tmp_path):
        dump, _ = _dump(tmp_path, "--mode", "fidelity")
        assert _segmenter(dump)["backend"] == "silero-v3.1"
