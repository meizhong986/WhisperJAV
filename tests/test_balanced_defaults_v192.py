"""
v1.9.2: Balanced pipeline speech-segmenter default and single-pass preset resolution.

Owner decisions: CFF3 (2026-09-05) made FireRedVAD the Balanced default with a
fallback chain; the owner reversed the default on 2026-09-05 (N3) — Balanced runs
faster-whisper's built-in VAD again, as in v1.9.0/v1.9.1. His balanced requirements
(S2/S9, 2026-09-09) then went further: Balanced accepts **no** external speech
segmenter at all, so `--speech-segmenter` with `--mode balanced` is now a usage error
and the firered-vad / ten routing-guard exemption that CFF3 added has been removed as
unreachable. Both backends remain available through --ensemble; the per-sensitivity
YAML resolution and the Test-D grouping overlay are unchanged there.

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
        # recogniser reports its segmenter as "none" -- v1.9.2 builds no external
        # segmenter there at all -- so a zero-cue Balanced file is `empty`, never
        # `suspect`. Nothing corroborates it, because there is no external speech
        # signal to corroborate against.
        assert resolve_segmenter_sensitivity(BALANCED_DEFAULT_SEGMENTER, "balanced") == {}
        assert "none" in PASSTHROUGH_SEGMENTERS

    def test_the_balanced_exemption_constant_is_gone(self):
        """v1.9.2 S2/S9: nothing exempts an external segmenter on Balanced any more."""
        import whisperjav.config.segmenter_presets as sp
        assert not hasattr(sp, "BALANCED_SINGLE_PASS_EXTERNAL")

    def test_firered_and_ten_are_still_real_segmenters_with_presets(self):
        """They keep their presets — they are simply reached through --ensemble now."""
        for name in ("firered-vad", "ten"):
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

    @pytest.mark.parametrize("segmenter", ["firered-vad", "ten", "whisperseg", "silero-v3.1", "none"])
    def test_balanced_rejects_every_external_segmenter(self, tmp_path, segmenter):
        """S2/S9: the whole option is gone on Balanced, not just the awkward ones.

        Before v1.9.2 these either resolved (firered-vad, ten), silently downgraded to
        silero-v3.1 (whisperseg), or quietly changed the pipeline. Now the run stops.
        """
        out = tmp_path / "never_written.json"
        proc = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "--dump-params", str(out),
             "--mode", "balanced", "--speech-segmenter", segmenter],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        assert proc.returncode == 2
        assert "--vad-version" in (proc.stdout + proc.stderr)
        assert not out.exists()

    def test_the_native_vad_is_not_a_segmenter_choice_any_more(self):
        """'faster-whisper' was only ever meaningful on Balanced, which now rejects it."""
        proc = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "--speech-segmenter", "faster-whisper", "x.wav"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        assert proc.returncode == 2
        assert "invalid choice" in (proc.stdout + proc.stderr)

    def test_ensemble_still_resolves_firered_preset_then_test_d(self, tmp_path):
        """The CFF3 behaviour lives on where it was always fully wired: --ensemble."""
        dump, log = _dump(tmp_path, "--ensemble", "--pass1-pipeline", "fidelity",
                          "--pass1-speech-segmenter", "firered-vad",
                          "--pass1-sensitivity", "aggressive")
        p1 = dump["ensemble_config"]["pass1"]
        assert p1["speech_segmenter"] == "firered-vad"
        assert p1["sensitivity"] == "aggressive"

    def test_fidelity_still_downgrades_an_unwired_segmenter(self, tmp_path):
        dump, log = _dump(tmp_path, "--mode", "fidelity", "--speech-segmenter", "whisperseg")
        assert _segmenter(dump)["backend"] == "silero-v3.1"
        assert "Falling back to silero-v3.1" in log

    def test_cli_overrides_still_win_on_a_mode_that_takes_a_segmenter(self, tmp_path):
        dump, _ = _dump(tmp_path, "--mode", "fidelity", "--speech-segmenter", "silero-v3.1",
                        "--vad-threshold", "0.6")
        assert _segmenter(dump)["threshold"] == 0.6

    def test_fidelity_default_unchanged(self, tmp_path):
        dump, _ = _dump(tmp_path, "--mode", "fidelity")
        assert _segmenter(dump)["backend"] == "silero-v3.1"
