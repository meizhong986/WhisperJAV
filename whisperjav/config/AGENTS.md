# AGENTS.md — Config area dossier

> Loaded when working under `whisperjav/config/`. This is the **highest-risk area** in the
> repo (a v1.7.3 multi-source conflict caused a 20% subtitle regression). Inducted 2026-06-20.
> Authority doc: `docs/en/architecture/CONFIG_SOURCES_HIERARCHY.md` (updated 2026-03-16).
> Cross-area contract: `docs/architecture/MODULE_CONTRACTS.md` C4.

## ⚠️ The single most important correction

`CLAUDE.md` and `memory/MEMORY.md` describe a **4-source hierarchy** ("Pydantic >
asr_config.json > v4 YAML > module defaults"). **That model was ELIMINATED in v1.8.9.**
Do not reason from it. Current reality below. (Correcting those two docs is an open cleanup item.)

## Current model (v1.8.9+)

- **Legacy modes** (`balanced/faster/fast/fidelity/kotoba`):
  `resolve_legacy_pipeline(mode, sensitivity)` → `resolve_config_v3()` →
  **Pydantic component presets = SINGLE source of truth**:
  - `config/components/asr/faster_whisper.py` — decoder/transcriber/engine (incl. `chunk_length`)
  - `config/components/asr/stable_ts.py`, `.../openai_whisper.py`
  - `config/components/vad/silero.py` — threshold, speech_pad_ms, max_group_duration_s, …
  - `config/components/features/scene_detection.py`
  - `component.get_preset(sensitivity).model_dump()`, then **CLI overrides only if non-None**.
- **Modern modes** (`transformers/qwen/crispasr`): **direct CLI args → pipeline**. No legacy
  resolution, no Pydantic. Qwen sensitivity uses the **v4** path
  (`ConfigManager.get_tool_config()` → `config/v4/manager.py:109-160`, order:
  ecosystem → model spec → preset → overrides). v4 YAML governs qwen/tools, NOT legacy modes.
- **`asr_config.json`**: only `version` + `ui_preferences` (console verbosity), via
  `ConfigManager.get_ui_preferences()`. NOT a parameter source. (Its `ui_preferences.crispasr`
  exe-path leak is the intentionally-uncommitted change on `dev_v1.9.0`.)

## Fixed (NOT sensitivity-controlled)

`MAX_SAFE_CPS=30.0`, `MIN_SUBTITLE_DURATION=0.3` (`config/sanitization_constants.py`);
`start_pad_samples=11200` / `end_pad_samples=20800` (silero only,
`modules/speech_segmentation/factory.py`); hardcoded segmenter fallback `silero-v4.0`
(`modules/faster_whisper_pro_asr.py`).

## How to inspect (NEVER heavy-import to check a value)

```bash
python -m whisperjav.main --dump-params /dev/null --mode balanced --sensitivity aggressive
```
Trace order when a value looks wrong: (1) Pydantic preset for the mode/sensitivity →
(2) was a CLI flag explicitly passed? → (3) module `.get()` fallback (should match the
balanced preset). This is exactly what the `config-tracer` agent automates.

## Tests
`tests/test_config_v4.py` (YAML load, schema validation, merge, preset application),
`tests/test_qwen_sensitivity.py`, `tests/test_ensemble_params.py`.

## Historical
v1.0–1.6 JSON was truth → v1.7.0 Pydantic added (JSON still read) → v1.7.3 conflict =
20% regression → **v1.8.9 JSON stripped, Pydantic sole truth**. Audit trail:
`docs/audit/config_resolution_per_pipeline.md`.
