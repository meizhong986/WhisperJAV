# Unified Segmenter Param Routing (v1.9.0 P0)

Status: IMPLEMENTED (this branch). Default flip to whisperseg in simple modes is
NOT included — gated on GT acceptance runs (F4/F6, `tools/fw_diagnostic_suite.py`).

## Problem

`LEGACY_PIPELINES` hardcodes `vad: "silero-v3.1"`, so the resolver always emits
Silero preset values (including the backend-agnostic grouping keys
`chunk_threshold_s` / `max_group_duration_s`) into `params["vad"]`. When the
runtime speech segmenter is non-Silero, the CONSTRUCTOR FIREWALL in
`whisper_pro_asr.py` / `faster_whisper_pro_asr.py` blanks `params["vad"]`
entirely — losing the grouping params. WhisperSeg then falls back to its 29s
`max_group_duration_s` default, triggering the Whisper repetition pathology on
JAV content (F4/F6 catastrophic empty output). `main.py` therefore force-
downgraded non-Silero segmenters to silero-v3.1 in simple balanced/fidelity
modes (issue #323).

## Design

One canonical location: **`params["speech_segmenter"]`** carries `backend`,
backend-specific detection params, and the shared grouping params. Silero-only
detection params remain in `params["vad"]` (populated only for Silero backends).
All segmenter consumers read grouping from the merged segmenter config.

Components:

1. **`whisperjav/config/segmenter_resolution.py`** (new) — single source of
   truth, shared by the legacy resolver and the ensemble path:
   - `SEGMENTER_TOOL_NAMES` (backend name → YAML tool), formerly
     `pass_worker._SEGMENTER_TOOL_NAMES`.
   - `SEGMENTER_PARAM_KEYS`, formerly `pass_worker.SEGMENTER_PARAMS`.
   - `GROUPING_KEYS = {"chunk_threshold_s", "max_group_duration_s"}`.
   - `resolve_segmenter_config(backend, sensitivity, overrides)` — resolves the
     backend's YAML tool preset (spec < preset < overrides), formerly
     `pass_worker.resolve_qwen_sensitivity`.

2. **`SegmenterGroupingOptions`** (`config/schemas/vad.py`) — Pydantic model for
   the two grouping keys, documenting the split from `SileroVADOptions`.

3. **Resolver** — `resolve_legacy_pipeline(..., speech_segmenter=...)` performs
   segmenter routing at resolution time:
   - Silero backend (or unset): grouping keys are MOVED from `params["vad"]` to
     `params["speech_segmenter"]`; Silero detection keys stay in `params["vad"]`.
   - Non-Silero backend: `params["vad"]` is emptied at the source (no firewall
     needed) and `params["speech_segmenter"]` is filled from the backend's own
     YAML sensitivity preset — same values the ensemble/qwen paths use.
   - `backend` is recorded in `params["speech_segmenter"]["backend"]`.

4. **`main.py`** — segmenter selection happens BEFORE resolution and is passed
   into `resolve_legacy_pipeline`. The forced downgrade of explicit non-Silero
   choices is REMOVED: `--mode balanced --speech-segmenter ten|whisperseg|...`
   now routes correctly. The scoped default (silero-v3.1 for simple modes)
   is retained until acceptance runs pass; flipping it is a one-line change.

5. **ASR constructors** — the firewall blanking block is removed. The
   defense-in-depth merge guard remains: Silero backends merge
   `{**vad_params, **speech_segmenter_config}`; non-Silero backends use
   `speech_segmenter_config` alone, so stale Silero params passed via direct
   module instantiation still cannot contaminate non-Silero backends.

6. **`ensemble/pass_worker.py`** — imports the shared constants/function from
   `config/segmenter_resolution.py` (`resolve_qwen_sensitivity` kept as alias).

## Compatibility

- CLI knobs (`--vad-threshold`, `--speech-pad-ms`) already dual-route into both
  `params["vad"]` and `params["speech_segmenter"]` post-resolution — unchanged.
- Silero backends receive grouping keys via the constructor merge, so behavior
  for balanced/fidelity + silero is value-identical to v1.8.14.
- Direct module instantiation without resolver: constructor fallback default
  (whisperseg) and merge guard behave as before.

## Follow-ups (not in this change)

- Flip simple-mode default silero-v3.1 → whisperseg after F4/F6 GT acceptance.
- Re-tune aggressive preset for large-v3 (separate P0).
- Migrate ensemble param routing to the canonical location wholesale.
