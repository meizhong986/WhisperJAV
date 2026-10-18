# MODULE_CONTRACTS.md — interfaces between WhisperJAV areas

> The "API between developers." When an agent's task would **change** anything here, it
> **stops and escalates to the orchestrator** (see `AGENTS.md` §1). Each contract cites the
> primary source and, where one exists, the **test that pins it** (the test is the real
> source of truth — prose drifts, tests fail).
>
> Verified by code-induction 2026-06-20. Re-verify a contract by reading the cited lines
> before relying on it; do **not** trust this doc over the code.

---

## C1 — Pipeline result dict (the master contract)

Every `BasePipeline.process(media_info: Dict) -> Dict` (`whisperjav/pipelines/base_pipeline.py:45`)
MUST return a dict that the ensemble worker can consume
(`whisperjav/ensemble/pass_worker.py:724-738, 786-792`):

| Key path | Type | Required | Consumer behavior |
|----------|------|----------|-------------------|
| `output_files.final_srt` | path-like | **YES** | `pass_worker.py:724-726` raises `ValueError` if missing; `:733-738` raises `FileNotFoundError` if the path doesn't exist on disk |
| `summary.final_subtitles_refined` | int | recommended | read at `:786` (defaults to 0) |
| `summary.total_processing_time_seconds` | float | recommended | read at `:787` (defaults to 0.0) |
| `summary.degradations` | list of str | optional | read at `:790-791`; any entry makes the file `suspect` in the run summary (`utils/run_outcome.py` `classify_output`) |
| `summary.notes` | list of str | optional | read at `:792` (1.10.0, owner Q21); printed under the file's row in the run summary and kept in the manifest; changes no state and no exit status |

**Standalone (`--mode`) path** consumes the same dict via `metadata_manager.save_master_metadata`.
**Pattern to copy** when adding a pipeline: `transformers`/`qwen` `process()`; for a pipeline that
runs an external program, `pipelines/parakeet_crispasr_pipeline.py` with the shared `whisperjav/crispasr/` package.

---

## C2 — ASR engine return shape (NOT uniform — know which kind)

Two divergent return contracts (`whisperjav/modules/AGENTS.md` has detail):

- **Dict-returning** — `WhisperProASR.transcribe()` (`whisper_pro_asr.py:339`),
  `FasterWhisperProASR.transcribe()` (`faster_whisper_pro_asr.py:553`):
  `{"segments": [{"start": float, "end": float, "text": str, "avg_logprob": float}], "text": str, "language": str}`
- **WhisperResult-returning** — `StableTSASR.transcribe()` (`stable_ts_asr.py:477`),
  `QwenASR.transcribe()`: a `stable_whisper.WhisperResult` object (segments via `.segments`).

A consumer must not assume one shape. Adding an engine = pick a shape and match an existing peer.

---

## C3 — Pipeline registration (two paths — do not mix)

- **Registry:** add `"name": NamePipeline` to `PIPELINE_CLASSES` (`whisperjav/ensemble/pass_worker.py:38-49`).
- **CLI choices:** add to `--mode`, `--pass1-pipeline`, `--pass2-pipeline` lists (`main.py:366, 394, 450`).
  An **Ensemble-only** pipeline (`parakeet-crispasr`, `cohere-crispasr`, 1.10.0) goes in the two pass
  lists only, with no `--mode` value.
- **LEGACY path** (`balanced/fast/faster/fidelity/kotoba-faster-whisper`): in `LEGACY_PIPELINES`
  (`whisperjav/config/legacy.py:96`); constructed via `resolve_legacy_pipeline()`.
- **DEDICATED path** (`transformers/qwen`, and the Ensemble-only `parakeet-crispasr/cohere-crispasr`):
  **NOT** in `LEGACY_PIPELINES`; constructed via a dedicated `_build_pipeline` block (`pass_worker.py`:
  transformers `:1104`, qwen `:1163`, the two CrispASR pipelines `:1437`) **before**
  `resolve_legacy_pipeline` is reached (`:1487`), + a `resolved_config = None` guard in `main.py`
  (`--mode transformers/qwen` `:2677, 2683`; every `--ensemble` run `:2637`).
  Calling `resolve_legacy_pipeline("parakeet-crispasr")` **raises** (`legacy.py:306-308`) — that's why
  the guards exist.
- **Removed names** stop before argparse runs: the 1.9.x `crispasr` value and the `--crispasr-*` flags
  (removed in 1.10.0) end the run with exit status 2 and a message (`main.py:41-46`,
  `whisperjav/crispasr/legacy.py`).

**Invariant:** a dedicated-CLI pipeline (own `--x-*` args) must NEVER be added to `LEGACY_PIPELINES`.

---

## C4 — Config precedence (CURRENT, v1.8.9+)

**Authority:** `docs/en/architecture/CONFIG_SOURCES_HIERARCHY.md` (updated 2026-03-16).

⚠️ **STALE-DOC WARNING:** `CLAUDE.md` and `memory/MEMORY.md` still describe a 4-source
hierarchy ("Pydantic > asr_config.json > v4 YAML > module defaults"). That model was
**eliminated in v1.8.9**. Treat the lines below as current; the orchestrator should correct
those two docs (tracked as a cleanup item).

- **Legacy modes** (`balanced/faster/fast/fidelity/kotoba`):
  `resolve_legacy_pipeline(mode, sensitivity)` → `resolve_config_v3()` →
  **Pydantic component presets are the SINGLE source of truth**
  (`whisperjav/config/components/{asr,vad,features}/*.py`). CLI args override **only if non-None**.
- **Modern modes** (`transformers/qwen`, and the Ensemble passes `parakeet-crispasr/cohere-crispasr`):
  **direct CLI args → pipeline**. No legacy resolution, no Pydantic. (`main.py` sets `resolved_config = None`.)
  A CrispASR pass takes three pass settings -- `--passN-model` (a short name from
  `whisperjav/crispasr/manifest.py`), `--passN-speech-segmenter` (default whisperseg) and `--language`
  (Japanese only) -- checked in `whisperjav/crispasr/pass_settings.py`; the other pass settings are
  not used and are named in one log line by the pass worker.
  Qwen sensitivity is resolved separately via the **v4** `ConfigManager.get_tool_config()`
  (`pass_worker.py:1200-1277`) — v4 YAML applies to qwen/tools, not to legacy modes.
- `asr_config.json` holds **only** `version` + `ui_preferences` (console verbosity), read by
  `ConfigManager.get_ui_preferences()`.
- **Probe without running ASR:** `python -m whisperjav.main --dump-params /dev/null --mode … --sensitivity …`

Params NOT controlled by sensitivity (fixed): see the authority doc's table
(`MAX_SAFE_CPS`, segmenter pad samples, hardcoded `silero-v4.0` fallback, …).

---

## C5 — Standard GUI→CLI call chain (what call-chain-verifier traces)

```
webview_gui/assets/app.js (collectConfig/collectFormData)
  → webview_gui/api.py (build_args / _build_*_args)
    → main.py (argparse + per-mode dispatch + resolved_config guards)
      → ensemble/pass_worker.py (_build_pipeline, PIPELINE_CLASSES)
        → pipelines/<x>_pipeline.py (process)
          → modules/<x>_asr.py (transcribe)
```
A new GUI→CLI feature is "wired" only if it survives every hop AND `--help | grep <flag>`
passes AND every `main.py` call site actually passes the value (a missing `stream=args.stream`
at one call site is the same bug as a missing argparse entry — `CLAUDE.md` Rule 7).

---

## Changing a contract

1. Confirm the change is truly necessary (re-read the consumer code).
2. Identify every affected area (grep the consumers).
3. Escalate to the orchestrator with the proposed new contract + affected areas.
4. On approval: update the code, the pinning test, this doc, and the affected dossiers — together.
