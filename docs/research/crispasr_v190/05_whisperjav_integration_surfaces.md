# WhisperJAV Integration Surfaces — Phase 0 Inventory

Local-codebase inventory of the existing integration surfaces that any future CrispASR-ecosystem integration would touch. **Descriptive only. No recommendations. No prescriptions.** All citations are file:line against the local repo at the snapshot below.

---

## 0. Snapshot

- Repo path: `C:\BIN\git\WhisperJav_V1_Minami_Edition`
- Branch: `dev_v1.9.0`
- Latest commit at research time: `225b471` (2026-05-13, "docs(tracker): rev49.9 + rev49.10 — v1.8.14 RELEASED state + 8 ship-notifications posted")
- Python range (per CLAUDE.md / pyproject.toml): 3.10–3.12
- Top-level package: `whisperjav/` with sub-packages `byop/`, `pipelines/`, `ensemble/`, `modules/`, `config/`, `installer/`, `webview_gui/`, `utils/`, `translate/`

---

## 1. BYOP plumbing — the existing precedent for external-ASR-binary integration

This is the highest-value section. BYOP (Bring Your Own Provider) is the production pattern already shipped for integrating an external ASR binary as a second-pass pipeline. CrispASR-the-binary maps cleanly onto the same shape.

### 1.1 `whisperjav/byop/xxl_runner.py` — the runner itself

162 lines, zero imports from `whisperjav.pipelines`, `whisperjav.config`, or `whisperjav.ensemble`. Module docstring at `whisperjav/byop/xxl_runner.py:1-10` explicitly states this isolation as a design choice.

Public surface: a single function `run_xxl(input_file, exe_path, model, language, output_dir, extra_args, task) -> Path` at `whisperjav/byop/xxl_runner.py:21`.

Signature defaults:
- `model="large-v3"` (`xxl_runner.py:24`)
- `language="ja"` (`xxl_runner.py:25`)
- `output_dir=None` → falls back to `exe.parent / "_whisperjav_xxl_output"` (`xxl_runner.py:56-58`)
- `task="transcribe"` (`xxl_runner.py:28`)

Subprocess invocation shape (`xxl_runner.py:64-71`):
```
cmd = [
    str(exe),
    str(input_file),
    "--language", language,
    "--output_dir", str(output_dir),
    "--output_format", "srt",
    "--model", model,
]
```
A regex strips any `--model X` from `extra_args` to prevent silent override (`xxl_runner.py:76-78`, comment cites issue #272). User-supplied extra args are tokenized with `shlex.split` (`xxl_runner.py:85`). A pyvideotrans-style sidecar file `whisperjav_xxl.txt` next to the exe is also appended if present (`xxl_runner.py:87-92`).

Environment hardening (`xxl_runner.py:94`): `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` overlay the parent env.

Execution (`xxl_runner.py:104-113`): `subprocess.run(...)` with `stdout=sys.stdout` (live to parent console), `stderr=subprocess.PIPE` (captured for diagnostics), `text=True`, `encoding="utf-8"`, `errors="replace"`, `cwd=str(exe.parent)`.

Crash-tolerance pattern (`xxl_runner.py:117-138`): SRT existence is checked BEFORE exit code, because ctranslate2 crashes during C++ destructor shutdown on Windows (STATUS_STACK_BUFFER_OVERRUN / 0xC0000409) AFTER the SRT has already been written. If the SRT exists, a warning is logged but the path is returned successfully. Only if no SRT is found AND exit is non-zero is `RuntimeError` raised with the last 2000 chars of stderr.

SRT location helper `_find_srt()` (`xxl_runner.py:145-162`): tries `{output_dir}/{input_stem}.srt` first, then falls back to most-recent non-empty `*.srt` in the directory.

### 1.2 CLI surface in `whisperjav/main.py`

- `--pass2-pipeline` accepts `"xxl"` as a valid choice (`main.py:225`, alongside `balanced fast faster fidelity transformers qwen`). Help text: `"Pipeline for pass 2 (enables pass 2). 'xxl' = BYOP XXL Faster Whisper (requires --xxl-exe)"` (`main.py:226`).
- BYOP argument group declared at `main.py:261`.
- `--xxl-exe` registered at `main.py:262-263`.
- Validation in main flow (`main.py:1699-1715`): if `pass2_pipeline == 'xxl'` and `xxl_exe` is missing or not a file, the run aborts with an actionable error before any work starts.
- The persisted-config helper `_get_xxl_extra_args_from_config()` is defined at `main.py:773-784` and reads from `asr_config.json > ui_preferences.byop.xxl_extra_args`.
- Ensemble config assembly at `main.py:2040-2063` and `main.py:2220-2241` packs `xxl_exe` + `xxl_args` (from the persisted prefs) into the `pass2_config` dict alongside `pipeline`, `sensitivity`, etc.

### 1.3 Ensemble dispatch — `whisperjav/ensemble/pass_worker.py`

The pass worker has an explicit early-branch for XXL at `pass_worker.py:658-673`:
```
# BYOP XXL: External subprocess — bypasses _build_pipeline() entirely.
# Produces FileResult objects in the same format as normal pipelines,
# ...
if pass_config.get("pipeline") == "xxl":
    _run_xxl_pass(
        ...
    )
    # _run_xxl_pass calls _write_dropbox_and_exit → os._exit() → never returns
```

`_run_xxl_pass()` is defined at `pass_worker.py:870-994`:
- Imports `from whisperjav.byop.xxl_runner import run_xxl` at `pass_worker.py:893`
- Requires `xxl_exe` from `pass_config` (`pass_worker.py:896-901`) — if absent, RuntimeError
- Reads `xxl_args`, `xxl_model` (defaulting to `"large-v3"`), and per-file iteration produces SRTs in the same shape `{basename}.{lang}.pass{N}.srt` that normal pipelines produce
- Never returns: terminates the worker via `_write_dropbox_and_exit` (Nuclear Exit), the same os._exit(0) pattern documented in `orchestrator.py:608-612`

This is the cleanest, most directly portable pattern in the codebase for "wrap an external ASR binary as a pass."

### 1.4 GUI wiring — `whisperjav/webview_gui/api.py`

The PyWebView API class exposes four BYOP-relevant methods (verified at `api.py`):
- `select_xxl_exe()` at `api.py:682` — file picker for the executable
- `get_byop_preferences()` at `api.py:704` — reads `asr_config.json > ui_preferences.byop`
- `save_byop_preferences(prefs)` at `api.py:714` — writes back `xxl_exe_path` and `xxl_extra_args`
- (Indirect) `start_ensemble_process()` at `api.py:1485` constructs the CLI args including `--xxl-exe` and `--pass2-pipeline xxl` when BYOP is selected by the frontend

Per `whisperjav/webview_gui/assets/app.js` (only one of the JS files; not exhaustively read), BYOP options are surfaced as part of the ensemble configuration panel, persisted via the `save_byop_preferences` round-trip.

### 1.5 Persistence in `whisperjav/config/asr_config.json`

Current schema for BYOP (verified by reading the file directly):
```json
"ui_preferences": {
    ...
    "byop": {
        "xxl_exe_path": "D:\\...\\faster-whisper-xxl.exe",
        "xxl_extra_args": "--model large-v2 --verbose True --standard_asia"
    }
}
```
The file's header `_architecture_note` (`asr_config.json:3`) states: *"As of v1.8.9, this file is ONLY used for ui_preferences (console verbosity). ALL pipeline parameters (ASR, VAD, scene detection, decoder, transcriber) come from Pydantic presets in config/components/."* — so BYOP preferences are an exception to the v1.8.9 architectural rule that pushed runtime config into Pydantic.

### 1.6 BYOP end-to-end summary

1. User picks an XXL executable + extra args in the GUI → `save_byop_preferences()` → `asr_config.json`
2. User selects ensemble with `pass2-pipeline=xxl` → frontend sends options to `start_ensemble_process()`
3. `start_ensemble_process()` builds CLI args including `--xxl-exe <path>` and `--pass2-pipeline xxl`
4. `main.py` parses, validates exe existence, packs into `pass2_config` dict
5. `EnsembleOrchestrator.process_batch()` (in `orchestrator.py`) calls `_run_pass_in_subprocess(pass_number=2, ...)`, which spawns a `mp.Process` running `run_pass_worker(payload, result_file)`
6. Inside the worker, `pass_worker.py:662` detects `pipeline=='xxl'` and routes to `_run_xxl_pass()`
7. `_run_xxl_pass()` calls `whisperjav.byop.xxl_runner.run_xxl()` per file
8. `run_xxl()` subprocess-executes the external binary; finds the resulting SRT; returns the path
9. Worker writes the result pickle (Drop-Box) and exits via `os._exit(0)` (Nuclear Exit)
10. Orchestrator reads the Drop-Box, merges pass1+pass2 SRTs via `MergeEngine.merge()`

The whole chain is subprocess-isolation by design (Drop-Box + Nuclear Exit), specifically to prevent C++ destructor crashes (ctranslate2 on Windows) from killing the parent.

---

## 2. Pipeline base class — `whisperjav/pipelines/base_pipeline.py`

`BasePipeline(ABC)` at `base_pipeline.py:13`.

Constructor (`base_pipeline.py:16-34`):
- `output_dir: str = "./output"`, `temp_dir: str = "./temp"`, `keep_temp_files: bool = False`, `save_metadata_json: bool = False`, `**kwargs` (extras ignored with debug log)
- Attaches `MetadataManager(self.temp_dir, self.output_dir)` (`base_pipeline.py:27`)

Abstract methods:
- `get_mode_name() -> str` (`base_pipeline.py:36-39`)
- `process(media_info: Dict) -> Dict` (`base_pipeline.py:41-53`) — expects a media-info dict with `path`, `basename`, `type`, `duration`, etc.; returns a metadata dict

Concrete lifecycle:
- `cleanup_temp_files(media_basename)` (`base_pipeline.py:55-105`) — file-by-file deletion, scene-aware, respects `save_metadata_json`
- `cleanup()` (`base_pipeline.py:107-170`) — cascades to `scene_detector.cleanup()`, `speech_enhancer.cleanup()`, `asr.cleanup()`, then conditional CUDA cache clear (skipped when `WHISPERJAV_SUBPROCESS_WORKER=1` — known Windows crash class)
- `__enter__`/`__exit__` (`base_pipeline.py:172-196`) — context-manager wrapper guaranteeing cleanup on exception

Concrete pipelines registered in `whisperjav/ensemble/pass_worker.py:17-39` via `PIPELINE_CLASSES` dict:
- `balanced` → `BalancedPipeline`
- `fast` → `FastPipeline`
- `faster` → `FasterPipeline`
- `fidelity` → `FidelityPipeline`
- `transformers` → `TransformersPipeline`
- `qwen` → `QwenPipeline` (ADR-004; the Qwen3-ASR + alignment pipeline)
- Plus `kotoba_faster_whisper_pipeline.py` and `decoupled_pipeline.py` present in `whisperjav/pipelines/` (not in the PIPELINE_CLASSES map at the lines surveyed)

---

## 3. Reference concrete pipeline (high-level)

`whisperjav/pipelines/balanced_pipeline.py` is the default fallback per CLAUDE.md and is the canonical pattern. Per CLAUDE.md: balanced = Faster-Whisper (CTranslate2) + Auditok + Silero VAD. The full body was not read in this Phase 0 pass — it should be read on demand by future iterations evaluating in-process integration vectors.

Other concrete pipelines available in `whisperjav/pipelines/`: `fast_pipeline.py`, `faster_pipeline.py`, `fidelity_pipeline.py`, `kotoba_faster_whisper_pipeline.py`, `transformers_pipeline.py`, `qwen_pipeline.py`, `decoupled_pipeline.py`.

---

## 4. Ensemble surface

### 4.1 Files

- `whisperjav/ensemble/orchestrator.py` — `EnsembleOrchestrator` class, batch + serial flows
- `whisperjav/ensemble/pass_worker.py` — per-pass subprocess worker, `run_pass_worker`, `_run_xxl_pass`, `_build_pipeline`
- `whisperjav/ensemble/merge.py` — `MergeEngine` and `MergeStrategy` enum
- `whisperjav/ensemble/safety_caps.py` — `conditional_sensitivity_cap` logic
- `whisperjav/ensemble/utils.py` — language-code resolution etc.
- `whisperjav/ensemble/__init__.py`

### 4.2 Two-Pass Ensemble dispatch (`orchestrator.py`)

`process_batch()` (`orchestrator.py:100-273`):
- Serializes media dicts to picklable form (`orchestrator.py:118`, `_serialize_media_files` at `orchestrator.py:720-738`)
- Resolves language codes per pass (`orchestrator.py:125-129`)
- Pass1 dispatched first (`orchestrator.py:159-187`); pass2 only if `pass2_config` is provided (`orchestrator.py:189-237`)
- Both passes go through `_run_pass_in_subprocess()` (`orchestrator.py:597-718`)
- Merge step iterates `_process_single_file_merge()` (`orchestrator.py:275-457`)

### 4.3 The Drop-Box + Nuclear Exit pattern

`_run_pass_in_subprocess()` at `orchestrator.py:597-718` uses `mp.get_context('spawn')` explicitly (`orchestrator.py:642`), with rationale comment at `orchestrator.py:636-641`: Linux's default fork breaks CUDA in child processes. The worker process writes its result to `temp_dir/worker_result_pass{N}.pkl` (Drop-Box) and calls `os._exit(0)` (Nuclear Exit) to skip Python interpreter shutdown. The parent reads the pickle, deletes it, and proceeds.

### 4.4 Serial vs batch processing

`serial_file_processing` kwarg (`orchestrator.py:33,69`) controls whether each file completes Pass1 → Pass2 → Merge before the next file begins (`_process_batch_serial` at `orchestrator.py:459-595`). Default is batch; serial is opt-in for incremental result delivery and to prevent model reloads from compounding.

### 4.5 Merge strategies (`merge.py`)

`MergeStrategy` enum at `merge.py:12-20`:
- `FULL_MERGE`, `PASS1_PRIMARY`, `PASS2_PRIMARY`, `PASS1_OVERLAP`, `PASS2_OVERLAP`, `SMART_MERGE`, `LONGEST` — 7 strategies total
- `MergeEngine.__init__` (`merge.py:42-51`) maps the strategy strings to internal `_merge_*` methods
- `MergeEngine.merge(srt1_path, srt2_path, output_path, strategy='smart_merge')` (`merge.py:54-...`) is the public surface called by `orchestrator.py:392-397`
- Helpers: `_deduplicate_consecutive` (`merge.py:114`), `_parse_srt` (`merge.py:133`), `_write_srt` (`merge.py:183`), `_calculate_overlap` (`merge.py:218`), `_coverage_ratio` (`merge.py:230`)

### 4.6 VRAM coexistence

Subprocess-per-pass with `os._exit(0)` between passes is the existing safeguard. Pass2 only starts after Pass1's worker has fully terminated and its Drop-Box has been read — there is no point where both passes hold context simultaneously inside the same process. This is structural serialization, not opt-in.

### 4.7 Safety caps

`safety_caps.py` exists and contains `conditional_sensitivity_cap` logic (referenced in CLAUDE.md / MEMORY as the v1.8.14 catastrophe mitigation). Module not read line-by-line in this Phase 0 pass; flagged in §11.

---

## 5. Config layering

### 5.1 Architecture per CLAUDE.md priority chain

Highest → lowest priority for runtime ASR/VAD/scene/decoder parameters (per CLAUDE.md):
1. `whisperjav/config/components/vad/silero.py` and similar Pydantic presets (**HIGHEST — often overlooked**)
2. `whisperjav/config/asr_config.json` (post v1.8.9, ONLY `ui_preferences` per the file's own header note)
3. `whisperjav/config/v4/ecosystems/tools/*.yaml`
4. Module defaults under `whisperjav/modules/`

### 5.2 v4 YAML system — `whisperjav/config/v4/`

Files inventoried via Glob:
- `manager.py` — central `ConfigManager`
- `gui_api.py` — frontend integration
- `errors.py`
- `loaders/yaml_loader.py`, `loaders/merger.py`
- `registries/base_registry.py`, `model_registry.py`, `tool_registry.py`, `ecosystem_registry.py`, `preset_registry.py`
- `schemas/base.py`, `model.py`, `ecosystem.py`, `tool.py`, `preset.py`
- YAML data lives under `whisperjav/config/v4/ecosystems/` (per CLAUDE.md, e.g., `ecosystems/transformers/hf_models_registry.yaml`)

This is the "patchable without code changes" layer for adding new models per CLAUDE.md's "Adding New Models (v1.7.0+)" section.

### 5.3 Pydantic component presets — `whisperjav/config/components/`

- `components/vad/silero.py` — Silero VAD presets (per CLAUDE.md, *HIGHEST priority for VAD*)
- `components/asr/faster_whisper.py`, `openai_whisper.py`, `stable_ts.py`, `kotoba_faster_whisper.py` — per-backend ASR presets
- `components/features/scene_detection.py` — scene detection presets
- `components/base.py` — base classes

### 5.4 Legacy/v3 config

- `config/manager.py` (legacy ConfigManager)
- `config/asr_config.json` (now `ui_preferences`-only post v1.8.9)
- `config/legacy.py` (`LEGACY_PIPELINES`, `list_legacy_pipelines`, `resolve_legacy_pipeline` — referenced from `pass_worker.py:17` and `api.py:863`)
- `config/resolver_v3.py`
- `config/sanitization_config.py`, `sanitization_constants.py`, `errors.py`, `persistence.py`, `registry.py`, `introspection.py`, `configurator_gui_ds.py`

### 5.5 Schemas

`whisperjav/config/schemas/` — Pydantic schemas for `base, decoder, engine, features, jav, metrics, model, pipeline, presets, transcriber, ui, vad`. Plus v4-specific schemas under `config/v4/schemas/`.

---

## 6. Forced-alignment / decoupled-alignment surfaces

Files matching `ForcedAligner|forced_aligner|chronos|Qwen3Forced`:
- `whisperjav/modules/subtitle_pipeline/aligners/qwen3.py` — Qwen3 forced aligner
- `whisperjav/modules/subtitle_pipeline/aligners/factory.py` — aligner factory pattern
- `whisperjav/modules/subtitle_pipeline/generators/cohere.py` — Cohere text generator (deferred per v1.8.14 carryover)
- `whisperjav/modules/subtitle_pipeline/generators/anime_whisper.py` — anime-whisper text generator
- `whisperjav/modules/subtitle_pipeline/orchestrator.py` — the decoupled orchestrator (text-generator + aligner)
- `whisperjav/modules/qwen_asr.py` — Qwen3-ASR module
- `whisperjav/modules/alignment_sentinel.py` — alignment sentinel
- `whisperjav/pipelines/qwen_pipeline.py` — the QwenPipeline class
- `whisperjav/pipelines/transformers_pipeline.py` — Transformers pipeline (uses alignment)
- Referenced from `webview_gui/api.py`, `webview_gui/assets/app.js`, `webview_gui/assets/qwen_guide.html`, `installer/core/registry.py`, `ensemble/pass_worker.py`, `main.py`

This is the existing decoupled-alignment infrastructure (text-generator → forced-aligner → SRT) that any text-only ASR backend (e.g., a hypothetical CrispASR Canary or Voxtral backend without native word timestamps) would route through. The factory pattern at `aligners/factory.py` is the registration point.

---

## 7. Installer extension points

### 7.1 Single source of truth — `whisperjav/installer/core/registry.py`

Per CLAUDE.md's installer-package section: **"THE SINGLE SOURCE OF TRUTH for all package definitions."**

Inventoried via Grep:
- `PackageEntry` dataclass with fields including `import_name: Optional[str]` (registry.py:214)
- `pyproject_spec()` method generates pyproject.toml dep spec (registry.py:219)
- `Extra` enum (CORE plus 11 optional extras) at registry.py:100+
- `generate_pyproject_extras()` (registry.py:1103) — extras dict derived from the registry
- `generate_core_deps()` (registry.py:1130)
- Import-name mapping (e.g., `import_name="cv2"` for `opencv-python`, registry.py:724; `import_name="PySubtrans"` registry.py:536; `import_name="webview"` registry.py:507) — Import Scanner uses this for ghost-dependency detection
- ~80 packages in the registry per CLAUDE.md

### 7.2 Other installer files

- `whisperjav/installer/core/standalone.py` — self-contained utilities (per CLAUDE.md: ZERO imports from `whisperjav.*` because it runs before whisperjav is installed)
- `whisperjav/installer/core/detector.py` — GPU/CUDA detection
- `whisperjav/installer/core/executor.py`
- `whisperjav/installer/core/config.py`
- Conda-constructor build pipeline lives under `installer/` (not `whisperjav/installer/`) — per CLAUDE.md: VERSION-driven template generation produces `installer/generated/` artifacts
- Validation: `python -m whisperjav.installer.validation` per CLAUDE.md

### 7.3 Adding an external binary

There is no explicit existing precedent in `registry.py` for declaring an external binary dependency (everything currently is a pip-installable Python package). The XXL binary is user-supplied via the `--xxl-exe` CLI flag and persisted in `asr_config.json` — not declared in the registry, not bundled by the installer.

---

## 8. GUI — backend-exposure pattern (`whisperjav/webview_gui/api.py`)

Single large API class (~3200+ lines). Key surfaces grep'd:

### 8.1 Backend enumeration methods

- `get_available_components()` at `api.py:811`
- `get_component_schema(component_type, name)` at `api.py:830`
- `get_legacy_pipelines()` at `api.py:855` (delegates to `whisperjav.config.legacy`)
- `get_component_defaults(component_type, name)` at `api.py:875`
- `get_speech_segmenter_backends()` at `api.py:907`
- `get_speech_enhancer_backends()` at `api.py:931`
- `get_segmenter_schema(backend)` at `api.py:1032`
- `get_enhancer_schema(backend)` at `api.py:1105`
- `get_scene_detector_schema(backend)` at `api.py:1268`
- `get_available_backends()` at `api.py:1320`
- `get_transformers_schema()` at `api.py:1702`
- `_load_hf_models_registry()` at `api.py:1650` — loads `whisperjav/config/v4/ecosystems/transformers/hf_models_registry.yaml`

### 8.2 Args-building methods

- `build_args(options)` at `api.py:97` — translates JS frontend options dict to argv list for the transcribe subprocess
- `_build_ensemble_args(options)` at `api.py:1583` — same for ensemble mode; explicitly handles `--xxl-exe`, `--pass2-pipeline`, `--pass2-sensitivity`, `--pass2-qwen-params`, `--pass2-scene-detector`, `--pass2-speech-segmenter`, `--pass2-speech-enhancer`, `--pass2-enhance-for-vad`, `--pass2-model` (api.py:2640-2692)
- `start_process(options)` at `api.py:350` and `start_ensemble_process(options)` at `api.py:1485` — the two entry points the JS frontend calls

### 8.3 File pickers and BYOP

- `select_files()` (api.py:584), `select_folder()` (api.py:630), `select_output_directory()` (api.py:666)
- `select_xxl_exe()` (api.py:682), `get_byop_preferences()` (api.py:704), `save_byop_preferences()` (api.py:714)

### 8.4 Frontend JS

`whisperjav/webview_gui/assets/app.js` is the main frontend JS bundle (not read line-by-line in this pass). HTML lives under `whisperjav/webview_gui/assets/`; per Grep there is also `qwen_guide.html` and other guide pages.

---

## 9. `whisperjav/main.py` flag surface (BYOP + ensemble subset)

Argparse declarations relevant to multi-backend integration (verified by Grep):

- Two-pass argument group, `--pass2-pipeline` at `main.py:224-226` with the `xxl` choice
- BYOP argument group at `main.py:261`
- `--xxl-exe` at `main.py:262-263`
- `_get_xxl_extra_args_from_config()` helper at `main.py:773`
- `pass2_pipeline == 'xxl'` validation gate at `main.py:1699-1715`
- `xxl_exe`/`xxl_args` packed into `pass2_config` dict at `main.py:2040-2063` and `main.py:2220-2241`

Per CLAUDE.md Rule 7 (CLI Flag Completeness — Mandatory Gate): any new flag intended to be reachable from `main.py` must be registered there AND verified end-to-end via:
```
python -m whisperjav.main --help | grep <flag>
python -m whisperjav.main --<flag> --help; echo $?    # must be 0, not 2
```

---

## 10. Speech-enhancement layering — illustrative multi-backend factory pattern

The cleanest recent precedent for a multi-backend feature is the speech-enhancement subsystem (v1.7.3+ per CLAUDE.md).

### 10.1 Registry pattern — `whisperjav/modules/speech_enhancement/factory.py`

- `_BACKEND_REGISTRY: Dict[str, str]` at `factory.py:24-30` maps name → fully-qualified class path (lazy import)
- 5 backends declared: `none`, `ffmpeg-dsp`, `zipenhancer`, `clearvoice`, `bs-roformer`
- `_BACKEND_CACHE: Dict[str, Type]` at `factory.py:33` caches loaded classes per process
- `_BACKEND_DEPENDENCIES` at `factory.py:36-67` declares pip package, install hint, always-available flag, description per backend
- `_DEFAULT_MODELS` at `factory.py:70-75` declares the default model id per backend
- `SpeechEnhancerFactory.create(backend, model=...)` at `factory.py:78+` is the public surface
- Per CLAUDE.md: this registry MUST stay in sync with `pyproject.toml` and `installer/core/registry.py`

### 10.2 Files

- `whisperjav/modules/speech_enhancement/base.py` — `SpeechEnhancer` Protocol + `EnhancementResult` dataclass
- `whisperjav/modules/speech_enhancement/factory.py` — registry + factory (above)
- `whisperjav/modules/speech_enhancement/pipeline_helper.py` — integration helpers used from pipelines
- `whisperjav/modules/speech_enhancement/backends/none.py`, `ffmpeg_dsp.py`, `zipenhancer.py`, `clearvoice.py`, `bs_roformer.py`

### 10.3 Pattern observations

- Lazy loading via fully-qualified-string registry entries plus `importlib`
- Per-backend dependency declarations co-located with the factory
- Backend-name strings flow through GUI → CLI args → factory call — string-keyed throughout

---

## 11. Could not verify / source-code gaps

Items where the inventory above is incomplete or relies on Grep summaries rather than full file reads:

1. **`balanced_pipeline.py` body** — only its existence and class registration were verified. End-to-end audio-segmentation → ASR → output flow was not traced.
2. **`merge.py` strategy semantics** — the 7 strategies are enumerated; the actual merge logic per strategy (overlap thresholds, deduplication rules) was not read line-by-line.
3. **`safety_caps.py` (`conditional_sensitivity_cap`)** — referenced in MEMORY as the v1.8.14 catastrophe mitigation; module body not read in this pass.
4. **`pass_worker.py` lines 1-650** — only the `xxl` early-branch and `_run_xxl_pass()` (lines 658-994) were traced in depth. The `_build_pipeline()` function at line 1015+ was identified by Grep but not read fully; per-pipeline build paths (transformers, qwen, balanced, etc.) inside it were not traced.
5. **`webview_gui/assets/app.js`** — only one Grep was run against it; the full frontend wiring of BYOP fields (which inputs, which events, which validations) was not traced.
6. **`config/v4/manager.py`** body — listed by Glob but not read. The actual ConfigManager priority-chain merge logic per CLAUDE.md's "highest → lowest" claim was not verified against code.
7. **`config/legacy.py`** — `LEGACY_PIPELINES`, `list_legacy_pipelines`, `resolve_legacy_pipeline` referenced from imports but not read.
8. **`installer/core/registry.py`** — only Grep summary captured. Full `PackageEntry` dataclass surface and the 11-extra enum were not read; whether `core/standalone.py` truly has zero `whisperjav.*` imports was asserted (per CLAUDE.md) but not re-verified by reading the file.
9. **Subtitle-pipeline orchestrator** — `whisperjav/modules/subtitle_pipeline/orchestrator.py` exists and is referenced by alignment files but was not read. The exact decoupled-alignment contract (text-generator interface, aligner interface, factory shape) is not captured in this pass.
10. **`modules/subtitle_pipeline/aligners/factory.py` body** — the registration pattern for forced aligners is presumably analogous to the speech-enhancement factory, but this was not verified.
11. **Conda-constructor flow** — `installer/templates/`, `installer/generated/`, post-install hooks, and the conda-constructor `construct.yaml` were not directly inspected in this pass.
12. **`asr_config.json` schema validation** — whether there is a Pydantic/JSON-Schema validator enforcing the `ui_preferences.byop` shape was not searched.

These gaps are intentional: this inventory captures the BYOP-pattern surface in detail because it is the closest precedent to a future external-binary integration. Deeper traces of items 1-12 should be performed on demand by subsequent iterations when specific integration vectors are being evaluated.
