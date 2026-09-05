"""
v1.9.2 CFF3: Balanced pipeline defaults to a WhisperJAV external speech segmenter.

Owner decisions (2026-09-05): FireRedVAD first, then TEN, then silero-v3.1 when a
package is missing (D7 — never faster-whisper's internal VAD); Test-D grouping
stays on top of the segmenter's YAML preset (D3); the corroboration counter is
active on balanced as a consequence (D6).

Two layers are covered without importing any ML stack:
1. `whisperjav.config.segmenter_presets` — the shared helper the three entry
   points (main.py, pass_worker, GUI api) call.
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
    BALANCED_DEFAULT_SEGMENTER_CHAIN,
    BALANCED_SINGLE_PASS_EXTERNAL,
    SEGMENTER_PARAMS,
    pick_balanced_default_segmenter,
    resolve_segmenter_sensitivity,
)
from whisperjav.utils.output_coverage import PASSTHROUGH_SEGMENTERS


# ─── helper layer ────────────────────────────────────────────────────────────

class TestDefaultChain:
    def test_chain_is_firered_then_ten_then_silero(self):
        assert BALANCED_DEFAULT_SEGMENTER_CHAIN == ("firered-vad", "ten", "silero-v3.1")

    def test_native_vad_is_never_in_the_chain(self):
        assert "faster-whisper" not in BALANCED_DEFAULT_SEGMENTER_CHAIN
        assert "none" not in BALANCED_DEFAULT_SEGMENTER_CHAIN

    def test_firered_when_available(self):
        assert pick_balanced_default_segmenter(is_available=lambda n: True) == "firered-vad"

    def test_falls_back_to_ten_when_firered_missing(self):
        avail = {"firered-vad": False, "ten": True}
        assert pick_balanced_default_segmenter(is_available=lambda n: avail.get(n, True)) == "ten"

    def test_last_member_is_returned_unconditionally(self):
        assert pick_balanced_default_segmenter(is_available=lambda n: False) == "silero-v3.1"

    def test_missing_firered_warns_with_pip_command(self, monkeypatch):
        # The project logger has its own handler and does not propagate, so
        # record the call instead of relying on caplog.
        import whisperjav.config.segmenter_presets as sp
        calls = []
        monkeypatch.setattr(sp.logger, "warning", lambda msg, *a: calls.append(msg % a if a else msg))
        picked = pick_balanced_default_segmenter(is_available=lambda n: n != "firered-vad")
        assert picked == "ten"
        assert any("pip install fireredvad" in c for c in calls), calls

    def test_default_uses_the_factory_availability_check(self):
        # No injected predicate: the real check runs (find_spec, no import).
        picked = pick_balanced_default_segmenter()
        assert picked in BALANCED_DEFAULT_SEGMENTER_CHAIN

    def test_every_chain_member_is_a_real_segmenter(self):
        # D6: the balanced default must be a real detector so the #394
        # corroboration counter is trustworthy (native VAD reports "none").
        for name in BALANCED_DEFAULT_SEGMENTER_CHAIN:
            assert name.lower() not in PASSTHROUGH_SEGMENTERS

    def test_guard_exemption_covers_exactly_the_chain(self):
        assert BALANCED_SINGLE_PASS_EXTERNAL == frozenset(BALANCED_DEFAULT_SEGMENTER_CHAIN)


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
    @pytest.mark.parametrize("sensitivity,threshold", [
        ("conservative", 0.5), ("balanced", 0.4), ("aggressive", 0.3),
    ])
    def test_default_is_firered_with_preset_then_test_d(self, tmp_path, sensitivity, threshold):
        dump, _ = _dump(tmp_path, "--mode", "balanced", "--sensitivity", sensitivity)
        ss = _segmenter(dump)
        assert ss["backend"] == "firered-vad"
        assert ss["threshold"] == threshold                 # YAML preset (step 1)
        assert ss["max_group_duration_s"] == 9.0             # Test-D (D3) overrides YAML 7/6/5
        assert ss["chunk_threshold_s"] == 0.1                # Test-D (D3) overrides YAML 1.0
        # The firewall mirror clears the resolver's silero vad block for a
        # non-silero backend, and says so.
        assert "vad" not in dump["resolved_config"]["params"]
        assert "_dump_note" in dump["resolved_config"]["params"]

    def test_native_vad_still_selectable(self, tmp_path):
        dump, _ = _dump(tmp_path, "--mode", "balanced", "--speech-segmenter", "faster-whisper",
                        "--sensitivity", "balanced")
        assert _segmenter(dump)["backend"] == "faster-whisper"
        # Native preset (scale-correct 0.40 for the bundled Silero) is kept.
        assert dump["resolved_config"]["params"]["vad"]["threshold"] == 0.40

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
        dump, _ = _dump(tmp_path, "--mode", "balanced", "--vad-threshold", "0.6",
                        "--max-group-duration", "6")
        ss = _segmenter(dump)
        assert ss["backend"] == "firered-vad"
        assert ss["threshold"] == 0.6
        assert ss["max_group_duration_s"] == 6.0

    def test_fidelity_default_unchanged(self, tmp_path):
        dump, _ = _dump(tmp_path, "--mode", "fidelity")
        assert _segmenter(dump)["backend"] == "silero-v3.1"
