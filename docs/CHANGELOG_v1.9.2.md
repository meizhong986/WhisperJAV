# WhisperJAV v1.9.2 — Development Change Log

> Developer-facing record of what changed on `dev_v1.9.2`, why, and how it was
> verified. `docs/release_notes_v1.9.2.md` is the user-facing text derived from
> this file. Keep both current; this one carries the detail (files, decisions,
> tests, follow-ups) that the release notes and user guide are written from.
>
> Conventions: one entry per landed change, newest first. "Decision" lines
> record who decided what, so a later reader can tell policy from mechanism.

---

## 2026-09-05 — CFF6: FireRedVAD becomes a first-class dependency (installed with `[cli]`)

**Area:** `pyproject.toml` (`[cli]` extra), `uv.lock`, `whisperjav/installer/core/registry.py`,
`whisperjav/installer/validation/imports.py`, `installer/templates/requirements.txt.template`,
`whisperjav/utils/preflight_check.py`, labels in `whisperjav/modules/speech_segmentation/`
(`backends/firered_vad.py`, `factory.py`, `__init__.py`), the tool YAML
`firered-vad-speech-segmentation.yaml`, `whisperjav/ensemble/pass_worker.py` (comments),
`whisperjav/main.py` (`--qwen-segmenter` help), `webview_gui/assets/index.html` (both segmenter
dropdowns), `README.md`; tests `tests/test_dependency_cross_match.py`.

**What changed**
- `fireredvad>=0.0.2` (PyPI, Apache-2.0, Python ≥3.10) is a `[cli]` dependency, so `[all]`,
  `[colab]` and `[kaggle]` inherit it and the conda-constructor `requirements_v1.9.2.txt`
  (generated from pyproject by `installer/build_release.py`) carries it. Registry entry at CLI
  order 49; removed from the import scanner's optional list; fallback template updated.
  `uv lock` added exactly four packages (fireredvad, kaldi-native-fbank, kaldiio, textgrid);
  no existing pin moved (uv re-derived some environment markers). `kaldi-native-fbank` ships
  wheels for win/linux/macOS × cp310–cp313, so no platform marker is needed.
- Every "experimental" label on FireRedVAD removed (display name is now `FireRedVAD`; the YAML
  drops the `experimental` tag). The remaining truthful caveat is kept in words: detection presets
  are upstream-derived, the segment cap was JAV-tuned on 2026-08-14.
- Preflight lists `fireredvad` as an optional dependency with an actionable message (it is the
  Balanced default from CFF3; without it Balanced falls back to another WhisperJAV segmenter).
- Hygiene found by the sync gate while adding the entry: the registry pinned `numba>=0.60.0`
  while pyproject says `>=0.61.0`, and the fallback template disagreed with the registry on
  `numba` and `transformers` — aligned to pyproject/registry so the sync and template tests pass.

**Verification:** `python -m whisperjav.installer.validation` → PASSED (was failing on the numba
mismatch before); `pytest tests/test_installer.py tests/test_installation.py
tests/test_dependency_cross_match.py` → 100 passed, 6 failed, all six identical on HEAD before this
change (WJ env has numpy 1.26 / pip-check conflicts / a stale entry-point test); `uv lock` exit 0;
`installer/build_release.py --dry-run` reports requirements generated from pyproject; YAML parses;
`tests/test_config_v4.py` 33 passed; `tests/test_speech_segmentation.py` 83 passed, 4 failed = the
known stale silero-v6.2 set.

**Decision:** owner (CFF6, 2026-09-05): FireRedVAD "mandatory" in dependency, setup and
installation; no longer experimental. Fallback when the package is absent at runtime: owner D7
(another WhisperJAV segmenter, never faster-whisper's internal VAD) — implemented with CFF3.
Thread owed: #311 (requester was told it needs `pip install fireredvad`).

## 2026-09-04 — ASR telemetry on by default and inside ensemble; version 1.9.2; note corrections

**Area:** `whisperjav/utils/asr_telemetry.py` (new `resolve_telemetry_path`),
`whisperjav/pipelines/balanced_pipeline.py`, `whisperjav/ensemble/pass_worker.py`
(new `_configure_telemetry`), `whisperjav/main.py`, `whisperjav/__version__.py`,
`installer/VERSION`, `docs/release_notes_v1.9.2.md`; tests
`tests/test_asr_telemetry_default.py` (new), `tests/test_run_outcome.py`.

**What changed**
- Telemetry is **on by default**. Default location
  `<output_dir>/raw_subs/<name>.asr_telemetry.jsonl` (the folder users already
  attach to bug reports). `--asr-telemetry PATH` moves it (directory → one
  file per media; file path used as given); new `--no-asr-telemetry` disables.
  `resolve_telemetry_path()` is the one place the rule lives.
- Ensemble: `main.py` puts `asr_telemetry` / `asr_telemetry_enabled` into both
  pass configs; the pass worker calls `_configure_telemetry()` per file before
  `pipeline.process`, pointing the pipeline at
  `<file_output_dir>/raw_subs/` with tag `passN`, so each Balanced pass writes
  `<name>.passN.asr_telemetry.jsonl` beside that file's pass outputs. Needed
  because in `source` mode the orchestrator points the pipeline's own
  `output_dir` at the temp directory and moves only the SRT out.
- Async path: the two values travel in `resolved_config` and
  `BalancedPipeline.__init__` reads them (the sync path sets the attributes
  after construction, as before).
- Version: `__version__.py` and `installer/VERSION` set to 1.9.2 (VERSION had
  been left at 1.9.0 while `__version__.py` said 1.9.1).
- Release notes: #395 moved from Known limitations to Fixed (the patch exists,
  v4-flash only, credit @mcdman); persistence entry no longer credits #298
  (closed 2026-04-23) or #381 (two-pass Customize persistence not built);
  telemetry section rewritten.

**Decision:** owner (2026-09-04, CL1–CL3): take items 1 and 2; set the version
in code; make telemetry reach ensemble; telemetry should be on by default.
Location under `raw_subs/` is my choice (stated rationale: must survive the
run and be where reporters already look); owner to veto if unwanted.

**After adversary review (same day), also:**
- `AsrTelemetry` appends each record to disk as its scene finishes (truncate on
  first record, append + flush per scene; `finalize()` only logs, `write` is an
  alias); `BalancedPipeline` finalizes from its error handler too, so a crash,
  hang or interrupt leaves a partial file. The buffered version wrote only on
  the success path, which contradicted the reason for default-on.
- GUI opt-out: Advanced options checkbox "Keep per-scene ASR telemetry"
  (`asrTelemetry`, on by default) → `--no-asr-telemetry` in every builder;
  persisted (settings count 33 → 34).
- `cuda_used_mb` (device-wide, `torch.cuda.mem_get_info`) added: the
  PyTorch allocator counters cannot see CTranslate2's arena.
- **Pre-existing bug fixed:** `--async-processing --output-dir source` wrote
  every file's outputs beside the first input (`main.py` set one output_dir
  from `media_files[0]`; the processor never overrode it per task). Now each
  media carries `output_dir` and `AsyncPipelineProcessor._process_media`
  applies it per task.
- Notes: #395 qualified to the direct DeepSeek provider (OpenRouter route not
  patched); "#96's first question" attribution dropped; file-path
  `--asr-telemetry` with several inputs documented as overwriting.

**Limits:** only `BalancedPipeline` records telemetry, so the GUI's default
ensemble pairing (anime-whisper + Qwen3-ASR) still records nothing (the CLI's
default `--pass1-pipeline` is balanced). `installer/generated/` is still the
v1.9.0 set until `build_release.py` is run (owner-gated).

---

## 2026-09-03 — GUI follows the run-outcome contract

**Area:** `whisperjav/webview_gui/api.py`, `assets/app.js`, `assets/index.html`,
`whisperjav/settings/gui_settings.py`; tests `tests/test_gui_run_summary.py`,
`tests/test_gui_settings.py`.

**What changed**
- On process exit the API reads `whisperjav_run.json` (`_read_run_summary`),
  using the same `default_manifest_path(output_dir, raw_inputs)` call the CLI
  uses, and rejects a manifest whose `started_at` predates the process.
- Closing console line is `[FINISHED] <tally> (exit status N)`,
  `[FINISHED WITH FAILURES] …`, or `[STOPPED] <note> …`. `[SUCCESS]` is gone
  from the API, including the separate translation runner's closing lines.
- `get_process_status` returns `run_summary` (counts, tally, per-file states,
  note, fails_on, manifest_path). The frontend shows `Finished · <tally>` in
  the status label, lists every non-`done` file with state and reason in the
  console, and prints the manifest path (`ProcessManager.logFileStates`).
- Two Advanced-options checkboxes, `failOnEmpty` / `failOnSuspect` →
  `fail_on_empty` / `fail_on_suspect` → `--fail-on empty[,suspect]`, forwarded
  by every builder (`build_args`, transformers, crispasr, ensemble, two-pass).
  Persisted through `_GUI_SETTINGS_MAP` and `DEFAULT_GUI_SETTINGS` (settings
  count 31 → 33). Default off.
- crispasr builder also gained `--skip-existing`, which it had never forwarded.

**Decision:** owner (2026-09-03, CL1–CL3): GUI must be consistent with the CLI
contract; say "Finished" then the tally; make the GUI state-aware.

**Verification:** 14 new tests (builders on every mode-dropdown value,
manifest freshness, folder input, closing lines); owner ran a real ensemble
file through the GUI (2199 cues, 99.8% span → `done`, `[FINISHED]` line,
manifest beside the source). Adversary review found and led to fixes for:
crispasr builder missing the flag; GUI/CLI manifest-path divergence on folder
inputs; mtime-based staleness; per-file states collected but not shown;
`[SUCCESS]` surviving in the translation runner.

**Follow-ups:** none required. `_build_ensemble_args` is unreachable from the
frontend (JS calls `start_ensemble_twopass` only); left in place.

---

## 2026-09-03 — One per-file vocabulary and one exit-status rule for every path

**Area:** new `whisperjav/utils/run_outcome.py`; rewritten
`whisperjav/utils/output_coverage.py`; `whisperjav/main.py`; tests
`tests/test_run_outcome.py` (new), `tests/test_output_coverage.py` (rewritten).

**What changed**
- States: `done`, `empty`, `suspect`, `failed`, `skipped`; coverage `ok` /
  `low` / `not assessed`. `classify_output()` is the only verdict function;
  `exit_status()` the only exit rule: 1 if any file is `failed` or the run did
  not complete, else 0; `--fail-on empty|suspect` adds states.
- `process_files_sync`, `process_files_async` and the `--ensemble` branch all
  append to one `outcomes` list owned by `main()` and end in `_finish_run()`,
  which prints the RUN SUMMARY table, writes `whisperjav_run.json` next to the
  outputs (output dir, or beside the first raw input in `source` mode; inside
  the folder for a folder input), and returns the status. `KeyboardInterrupt`
  and the outer `except Exception` also finish (note + status 1).
- Removed: the ensemble branch's own `sys.exit(1)`, the three per-path summary
  blocks, `_failed_count`, `CoverageReport.is_failure`, `report_coverage`, the
  0.60 "suspicious" tier. `output_coverage.py` now measures only.
- `--fail-on` (repeatable / comma-separated) and `--min-coverage` (0–1) are
  validated at startup; a bad value exits 2 before any model loads.
- `apply_vtt_conversion` returns the VTT path so the manifest names an
  existing file. Duplicate-outcome guard when a late step raises. Ensemble
  outcomes keyed by input path (basename collisions). Degraded ensemble files
  with `--translate` record translation `skipped`. A promised-but-missing SRT
  is `failed`; zero cues with corroboration or after a pass-2 failure is
  `suspect`, not `empty`. stdout/stderr flushed before the ctranslate2
  `os._exit`.
- **Pre-existing defect fixed:** `--async-processing` never waited for its
  tasks (`AsyncPipelineManager.process_files` submits with `wait=False`); the
  CLI summarised queued tasks, then `shutdown()` cancelled them. It now waits
  (`wait_for_task`, then the Future as the authority).

**Decisions**
- Owner (2026-09-03): agreed section 5 of the "Exit Status Precedents"
  research, with C1 no halt control and C2 no pre-flight. Facts set: f1 zero
  subtitles is not a failure; f2 a batch contract is needed; f3 consistency
  over right/wrong.
- Implementation choices inside that scope (flagged to owner, not objected):
  translation error → `failed` in every mode; "not assessed" is a coverage
  column, not a sixth state; manifest has a fixed name, overwritten per run.
- Reverses the pre-release behaviour announced on #394 (2026-08-30) in which
  a zero-cue file failed the run.

**Verification:** 75 tests in the two suites, including `main()` executed
through sync, async and ensemble with stubs and one test through the real
`AsyncPipelineManager`; real CLI runs (faster/tiny, CPU) on the 15 s / 5 s /
piano clips through all three paths, and a piped run through the ctranslate2
fast exit. Three adversary passes; findings applied (interrupt path, missing
file, corroborated/degraded empty, VTT path, duplicate outcome, basename
collision, docstring drift, double summary, grammar).

**Known limits (stated in the release notes):** corroboration exists only in
Balanced with an external segmenter; the ensemble pass worker does not return
the streak, so ensemble `suspect` = span or pass-2 failure only; an ensemble
worker crash still ends the whole batch (orchestrator behaviour);
`--output-dir source` with inputs in several folders writes one manifest
beside the first.

---

## 2026-09-03 — Pre-review audit of the branch (no code change)

Independent audit of `dev_v1.9.2` against `main` before the work above:
found the ensemble exit-1 regression (`_failed_count` unbound at the old
`main.py:2806`, introduced by `bd98edc`), four release-note over-claims
(#395 "cannot be disabled" stale; #298 closed since April and #381 not
addressed by PR #378; zero-cue failure sync-only; `--asr-telemetry` never
reaching ensemble), and two script-style test files that break a one-session
`pytest tests/`. Report: https://claude.ai/code/artifact/73c5c95d-0a92-49d1-b127-fb23c02029c2

---

## 2026-08-30 — Earlier v1.9.2 work (see the release-notes changelog table)

`--asr-telemetry`; Balanced corroboration signal; artifacts summary fix;
DeepSeek v4-flash thinking switch (#395); post-processing stage logging
(#372); PR #378 merge (#328, #96, #298, #381, #309, #382); Linux/Windows
install-guide fixes (#366, #334); seven small defects (#340, #341, #306, #325,
#323, Ollama fallback list); GUI settings suite greened. The coverage gate
from that batch is superseded by the contract above.

---

## Open items carried on this branch

- **Async + Balanced + more than one file dies natively** (exit 127 on
  Windows, no traceback, no RUN SUMMARY) when the second task's
  `FasterWhisperProASR` initialises after the first task's pipeline was
  cleaned up in `AsyncPipelineProcessor._process_media`'s `finally`. Verified
  2026-09-04 with `--mode balanced --model tiny` on two clips, both in an
  ordinary output dir and in source mode; `--mode faster` with two files
  works; one Balanced file works. Pre-existing (the async path never ran its
  tasks before 2026-09-03), same family as the ctranslate2 destructor crash
  the sync path avoids by never destroying the ASR. Not fixed; stated in the
  release notes. Owner decision: keep async as-is with the limitation, or make
  it reuse one pipeline per mode as the sync path does.
- Telemetry outside `BalancedPipeline` (the default ensemble pairing records
  nothing); a Transcribe-tab segmenter control (owner decision); a default cue
  ceiling for the default ensemble (owner decision); #394 containment.
- Test hygiene: `tests/test_gui_refactor.py`, `tests/test_postprocessing_performance.py`
  (rewrap stdout at import) and `tests/test_tab_spacing.py` (opens Tk)
  prevent a single-session `pytest tests/`.
- #314 installer progress streaming never landed.
