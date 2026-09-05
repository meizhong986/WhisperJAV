# WhisperJAV v1.9.2 — Development Change Log

> Developer-facing record of what changed on `dev_v1.9.2`, why, and how it was
> verified. `docs/release_notes_v1.9.2.md` is the user-facing text derived from
> this file. Keep both current; this one carries the detail (files, decisions,
> tests, follow-ups) that the release notes and user guide are written from.
>
> Conventions: one entry per landed change, newest first. "Decision" lines
> record who decided what, so a later reader can tell policy from mechanism.

---

## 2026-09-05 — research: `clustering_threshold` is not a scene-granularity lever (docs only)

**Area:** new `docs/research/semantic_scene_premise/` (README, `semantic_lever_study.py`,
`lever_study_results.json`). No product file changed.

**What was found (measured on `test_media` audio, features extracted once, only the boundary logic
varied):** the semantic detector's `clustering_threshold` sets the number of global texture clusters
(4029 → 1 on the 65-minute file) but the final scene count stays 43–65 across thresholds 5–200 and
*rises* over the exposed slider range 5–30 (43 → 58), then collapses to one scene above ~500. Owner's
EKAI-023 runs: 119 scenes at 50, 121 at 90. Cause: at 0.48 s resolution 6060 of 6735 label runs are a
single vector, so raw boundaries are label churn; the count is set by `min_duration` (5 → 354 scenes,
60 → 12) and `snap_window` (1 s → 14, 12 s → 70). With a temporal connectivity constraint the same
threshold is a monotone zoom above ~14; a change-point form gives window/magnitude levers. Options
listed for the owner; the CFF2 "Scene Change Threshold" control and help text (dev, unreleased) are
contradicted by the measurements and await the owner's choice.

**Verification:** study re-run twice (second run adds fine sweeps, `snap_window`, label-run statistics);
adversary pass reproduced the first run bit-for-bit and supplied the `snap_window` and label-run
attacks, which the second run confirms; arm A is boundary-identical to `SemanticSegmenter.segment`.

**Literature (same day, owner asked for the science, not a slider fix):** `literature_review.md`.
Sundaram & Chang (2000-2003) define computable audio scenes as long-term consistency of ambient sound
(>= 8 s to establish context; typically 40-50 s; explicitly "not semantic scenes") and detect them by
correlating an attention span (16 s) against a memory (32 s); they warn that cluster-threshold methods
"critically" depend on a threshold that cannot be set from the data. Every temporal method in the field
(Foote novelty, BIC, memory-model, global partition with a continuity term, embeddings + the same
detectors) carries an explicit temporal term plus a magnitude lever. The shipped engine is a global
partition WITHOUT any temporal-continuity term (unconstrained Ward, cut at every label change), which is
why its threshold sets class count, not granularity; the measured rise in scene count over the slider
range is not explained by this reading and stays open. No audio-only scene work on adult video was
found. Adversary pass ran on the note; ten findings folded in (note §8).

**Decision:** owner (2026-09-05) asked for the logic to be researched, not accuracy; no change made.
Owner N1/N2 (2026-09-05, after the literature review): the threshold slider STAYS in 1.9.2 unchanged; the
study is marked for 2.x, where the semantic detector's boundary logic and granularity control are to be
revisited (roadmap §9.3).

## 2026-09-05 — tools/scene_inspector.py: scene-detector statistics and per-scene screenshots

**Area:** new `tools/scene_inspector.py` + `tools/scene_inspector.md`. Developer/user utility, no
product code touched.

**What it does:** runs a WhisperJAV scene backend (auditok default; silero, semantic, none) on a
media file the way the pipelines do (AudioExtractor 16 kHz mono, `SceneDetectorFactory.create` with
the backend's own defaults, `--param KEY=VALUE` overrides) and writes to `<media folder>/scenes_info/`
(or `--output-dir`): `<name>.scenes.csv`, `<name>.scenes.json`, `<name>.summary.md` (count, coverage,
duration percentiles + histogram, scenes over the 29 s window and the 180 s aligner limit, gaps,
scenes per minute, per-scene table with hh:mm:ss.mmm) and `screenshots/` with begin/middle/end frames
per scene named `<name>__scene0001__begin__00_12_03.450.jpg` (`--no-screenshots` to skip; skipped with
a recorded reason for audio-only input). `--list-params BACKEND` prints accepted parameters with
their YAML defaults and presets. Statistics use the extracted audio duration; the container header is
reported beside it (the 293 s test clip carries a 30:55 header).

**Verification (executed):** 293 s mkv default backend → 19 scenes, 57 frames, exit 0; semantic with
`clustering_threshold=10` → 6 scenes, effective parameter recorded; audio-only wav → screenshots
skipped with reason; `--list-params` for auditok/semantic; `--help` exit 0; one frame opened and
checked visually.

**Same day, owner accepted all seven P8 suggestions (i2/i3) and asked for the semantic threshold
to be user-settable (i1):** `--backend a,b,...` runs several backends and writes `<name>.compare.md/.json`
(counts, coverage, percentiles, pairwise boundary agreement within 1 s); per-scene RMS/peak dBFS;
`--speech-ratio [SEGMENTER]` (speech seconds and ratio per scene, default firered-vad);
`<name>.<backend>.contact_sheet_pNN.jpg` (begin frames, labelled, 60 per page);
`<name>.<backend>.chapters.ffmeta` (muxed with ffmpeg and read back by ffprobe); `--sensitivity`
(preset keys only, see below); `--srt FILE` (cues per scene, scenes without cues, cues in gaps);
`--scene-threshold FLOAT` (shortcut for `clustering_threshold`, warns when semantic is not selected).
Outputs now carry the backend in the file name; screenshots go to `screenshots/<backend>/`. Coverage is
the union of scene intervals (padded semantic scenes overlap; 1.3 s of overlap on the test clip is
reported separately).

**Finding for the owner (product, unchanged):** the scene tool YAML `spec` blocks are not the backend
constructor defaults — auditok `pass2_max_duration_s` 1800 vs 28 s, `pass2_max_silence_s` 1.8 vs 0.94,
`pass1_max_silence_s` 2.5 vs 1.8 — and the pipelines pass a third set from the Pydantic preset.
Applying the whole spec produced one 222 s scene on the 293 s clip; the tool therefore applies only
the `presets.<name>` keys for `--sensitivity`.

**Decision:** owner request 2026-09-05 (I2 P1–P7; i1–i3 for the seven additions).

## 2026-09-05 — CFF1: the recogniser is unloaded and reloaded after 20 minutes of scene audio

**Area:** new `whisperjav/utils/model_refresh.py` (policy), new `whisperjav/modules/asr_worker_proxy.py`
(child-process recogniser for Balanced); `whisperjav/pipelines/balanced_pipeline.py` (`_ensure_asr`
hook, per-scene accounting, `cleanup` override, metadata); `whisperjav/pipelines/fidelity_pipeline.py`
(in-process `_release_asr` + `_load_fresh_asr`); `whisperjav/utils/asr_telemetry.py` (`model_epoch`); `whisperjav/main.py`
(`--model-refresh-audio-minutes`, validation, sync/async/ensemble plumbing, dump echo); GUI
`index.html` / `app.js` / `api.py` / `settings/gui_settings.py`; tests `tests/test_model_refresh_v192.py`
(new, 16), `tests/fake_asr_for_proxy.py` (new), `tests/test_gui_settings.py` (count 34 → 35).

**What changed**
- `--model-refresh-audio-minutes MINUTES` (default **20**, `0` = never; negative → exit 2 at startup).
  The budget is the sum of the durations of the scenes handed to one recogniser instance (owner D2:
  scene granularity, nothing finer; a scene that failed was still handed over and counts). It is
  counted per instance: a sync Balanced batch shares one instance across its files, while Fidelity
  (per-file ASR) and `--async-processing` (per-file pipeline) start a new instance, and therefore a
  new count, with each file. When it is spent, the instance is replaced *between* two scenes. Same flag for sync, `--async-processing` (via
  `resolved_config`) and `--ensemble` (via `worker_kwargs`; Balanced/Fidelity passes honour it,
  other pipelines ignore it). GUI: one Advanced-options field on the Transcription tab, read by both
  the Transcription and Ensemble runs (like the source-language control), persisted.
- **Balanced (owner D5):** with a non-zero budget the CTranslate2 model lives in a spawned worker
  process behind `RemoteFasterWhisperASR`, which exposes the seven members the scene loop already
  used on `FasterWhisperProASR` plus `record_audio` and `shutdown`. A refresh = ask the worker to
  `os._exit(0)` (no destructor) and start a fresh one; the int8 fallback learned by one generation
  is carried into the next. The worker serves every file of the batch until the budget is spent, so
  the model reuse across files is kept. Telemetry, the #394 streak, the progress bar and metadata stay
  in the parent (one telemetry writer per file). A worker that dies natively fails only the scene it
  was on (its completed scenes keep counting in the filter statistics); the next scene starts a fresh
  worker, which is a new telemetry generation (`model_epoch` +1) but not a refresh. After three
  consecutive deaths or failed restarts the proxy stops restarting for the rest of that file: each
  remaining scene is marked failed and the file is then classified by its output like any other
  (usually `empty`, `suspect` if the segmenter kept detecting speech) — whether repeated recogniser
  death should classify the file as `failed` is an owner decision on the exit-status contract, not
  made here. The counter resets at the start of each file (the second adversary pass caught the
  first version, where it did not, so one bad file could have blanked every later file of a batch).
  With `--model-refresh-audio-minutes 0` the pre-v1.9.2 in-process immortal instance is used unchanged.
- **Fidelity:** in-process, in two steps so the old model is really gone before the new one loads:
  `_release_asr` (fold filter statistics, clean up the external segmenter the ASR owns, `cleanup()`),
  then the loop drops its own reference (`asr = None`) — a reference held anywhere keeps the model
  resident — then `_load_fresh_asr` (`gc.collect`, `empty_cache`, fresh `WhisperProASR`, which also
  rebuilds its segmenter). A reload failure raises and fails the file, as a failed initial load would
  (there is no instance left to continue on). The adversary pass caught the first version of this,
  where `del asr` inside the helper deleted only a local name. Measured after the fix on the RTX 3060
  with large-v2: `torch.cuda.memory_allocated()` around a release/load pair (see verification) — the
  discriminating variable; the whole-run peak device memory (11 704 MiB with three refreshes vs
  11 735 MiB with refresh off) is consistent but, being the caching allocator's reserved peak, not
  by itself proof.
- Records: per-scene `model_epoch` in `scenes_detected` (both pipelines) and in the telemetry JSONL
  (Balanced only — Fidelity has no telemetry); `model_refreshes` (this file's share; the Balanced
  proxy counts across the batch and the pipeline reports the per-file difference) and
  `model_refresh_audio_minutes` in the run summary metadata. `model_epoch` is context for reading a
  trend, not evidence of the cause (the `probe_failed` re-transcribe experiment remains unwired).

**Verification (executed, CPU/GPU auto, `--model tiny`):**
- 293 s clip, Balanced, budget 1 min: 14 scenes, refreshes after scenes 6/9/12 (worker PIDs 26344 →
  2248 → 30416 → 13912), telemetry epochs `[1×6, 2×3, 3×3, 4×2]` in ONE file, 54 cues, exit 0.
- Same clip, Fidelity, budget 1 min: 3 in-process refreshes ("Loading Whisper model" ×4), 38 cues, exit 0.
- `--ensemble` with two Balanced passes, budget 1 min: each pass worker spawned its own recogniser
  worker (grandchild) and refreshed three times; merged 55 cues, exit 0.
- `--async-processing` with two Balanced files (the case that died natively with exit 127 on
  2026-09-04): both files done, RUN SUMMARY, exit 0 — the parent never destroys a CT2 model now.
- Kill test: `taskkill /F` on the worker during scene 7/14 → "Scene 7/14 failed: ASR worker process
  died (exit code 1)", fresh worker for scene 8, file finished with 46 cues, exit 0.
- Budget 0 → "Initializing ASR model (exclusive VRAM block)" (in-process path), exit 0.
- GPU (RTX 3060): Fidelity large-v2, budget 1 min → 3 refreshes, 41 cues, exit 0; whole-run peak
  device VRAM 11 704 MiB vs 11 735 MiB with refresh off (control). Direct measurement of the
  release/load pair with `torch.cuda.memory_allocated()` (large-v2, fp32, same config path):
  0 → 6 018 MiB after load A → 0 after `_release_asr` + `asr = None` + gc + empty_cache → 6 018 MiB
  after load B (a resident second model would read ~12 036). Balanced large-v2, budget 1 min → 3
  refreshes of 13–14 s each, 53 cues, 154 s wall, exit 0.
- Two-file sync batch (293 s + 15 s clips, Balanced tiny, budget 1 min): one worker served both
  files (4 "ASR worker ready" = 1 start + 3 refreshes); per-file metadata `model_refreshes` 3 and
  0; the second file's scene carries `model_epoch` 4 (the instance continued across files).
- `pytest tests/test_model_refresh_v192.py` 16 passed (policy; proxy against a fake recogniser in a
  real spawned worker: handshake, refresh → new PID, statistics across generations and across a
  worker death, learned compute type, ordinary error vs native death, give-up rule, shutdown, bad
  model; CLI parse/validation/echo). The pipeline hook sites (`_ensure_asr` branch, the loop's
  `record_audio`, `cleanup`, the Fidelity release/load pair) are covered by the real runs above, not
  by unit tests.
  `test_gui_settings` + `test_gui_run_summary` + `test_asr_telemetry_default` + `test_run_outcome`
  149 passed. `--help` gate + exit 0. `node --check app.js`.

**Cost measured (RTX 3060, this session):** one interpreter start + model load per Balanced refresh =
13–14 s with large-v2 (three refreshes timed from "Model refresh" to "ASR worker ready": 14, 13, 13 s;
about 8 s with `tiny`). At the 20-minute default a two-hour film refreshes at most about six times
(the budget counts scene audio, which is less than the film length), i.e. at most roughly 80 s — a
few percent of a Balanced run. Fidelity: one `whisper.load_model` plus its segmenter per
refresh. Balanced peak VRAM with the worker: 4 447 MiB total on the device during the large-v2 run.

**Decision:** owner (CFF1, 2026-09-05): mechanism on by default at 20 minutes, user-adjustable, at
scene boundaries; D2 budget in scene-audio minutes; D5 child-process recogniser for Balanced. No
over-engineering (owner): budget counted at scene granularity only; no per-scene timeouts; no
re-transcribe probe. Not a fix for the #394 root cause — containment.

## 2026-09-05 — CFF5: Qwen lone-line filter also drops 「はい。」 and 「うん。」

**Area:** `whisperjav/modules/subtitle_pipeline/cleaners/nonverbal_line_filter.py`,
`whisperjav/main.py` (help text of `--[no-]qwen-drop-nonverbal-lines`); tests
`tests/test_nonverbal_line_filter.py`.

**What changed**
- `NONVERBAL_TOKENS` gains `はい` and `うん`. The predicate is unchanged: a line is dropped only
  when its whole stripped text is exactly one token plus an optional single `。`, so `はいはい。`,
  `うんうん。`, `ううん。`, `あ、うん。` and any sentence stay (tests added for each). Runs in Phase 8
  for all three Qwen backends (qwen3 / anime-whisper / cohere) before the nonlinguistic filter,
  which lists the two words as keep-evidence — no conflict, it only sees surviving lines.
- Docstring rewritten (the v1.9.0 text stated the opposite rule) with the owner's evidence:
  two SRT screenshots of lone `はい。` (0.065–2.720 s) and `うん。` (0.300–2.900 s) cues; "almost
  90% … refer to human moans during intimate scenes".
- Off switch unchanged: `--no-qwen-drop-nonverbal-lines` (single-pass). Ensemble Qwen passes inherit
  the default and have no switch — unchanged by owner decision D8.
- Not covered on purpose: the Balanced pipeline (legacy sanitizer path; `filter_list_v08.json`
  already holds both words as exact-match hallucination entries).

**Verification (executed):** `pytest tests/test_nonverbal_line_filter.py` (positives incl. the two
new tokens with/without `。`/whitespace; 13 negatives; SRT drop/renumber 8 → 2); `--help` shows the
tokens; `--no-qwen-drop-nonverbal-lines --help` exit 0.

**Decision:** owner (CFF5, 2026-09-05) + D8 (token list only; no ensemble off-switch, no GUI
checkbox). Thread owed: #254 (owner's v1.9.0 reply lists the old token set).

## 2026-09-05 — CFF2: semantic scene-change threshold exposed on the CLI and in Customize

**Area:** `whisperjav/main.py` (two flags, features injection, qwen kwargs, decoupled kwargs,
`--dump-params` echo); `whisperjav/pipelines/qwen_pipeline.py`, `whisperjav/pipelines/decoupled_pipeline.py`
(ctor param → `clustering_threshold` kwarg); `whisperjav/ensemble/pass_worker.py` (`prepare_qwen_params`
mapping + lift); `whisperjav/webview_gui/api.py` (Qwen Audio-tab schema); `webview_gui/assets/app.js`
(Qwen "Custom Scene Bounds" slider, `QwenManager.defaults`); the semantic tool YAML (slider label /
description); tests `tests/test_scene_clustering_threshold_v192.py` (new, 8 tests).

**What changed**
- The semantic detector's `clustering_threshold` (Ward distance, default 18, YAML presets 10
  aggressive / 22 conservative) was a constructor kwarg all the way to the vendored engine, but
  nothing above the factory set it. Now: `--scene-clustering-threshold FLOAT` for the legacy modes
  (written into `features["scene_detection"]`, so every legacy pipeline passes it to the factory;
  auditok/silero accept and ignore it, with a WARNING when the effective method is not semantic) and
  `--qwen-scene-clustering-threshold FLOAT` for `--mode qwen` (ctor param, applied independently of
  safe chunking); `--pipeline decoupled` takes the legacy flag. Both echoed in `--dump-params` `cli_args`
  (`scene_clustering_threshold`, `qwen_scene_clustering_threshold`; the Qwen echo was missing in the
  first version — caught by the adversary pass, added, and re-measured: `--mode qwen --dump-params
  --qwen-scene-clustering-threshold 10` → `10.0`).
- Ensemble: legacy passes already accepted `clustering_threshold` in `--passN-params`; Qwen passes
  now accept `scene_clustering_threshold` in `--passN-qwen-params` (mapped to
  `qwen_scene_clustering_threshold`, lifted only when set).
- GUI (Ensemble tab Customize): the legacy modal's Scene tab already rendered the slider from the
  YAML hints (label now "Scene Change Threshold"); the Qwen/anime-whisper/cohere modal gains the same
  slider under Audio → Custom Scene Bounds (5–30, default 18), collected by the shared collector.
  Transformers passes are method-only (`--hf-scene`) and unchanged — noted, not extended.
- Wording: the presets call 10 "more, shorter segments" and 22 "fewer, longer"; on the 293 s clip
  the final count went 7 → 6 scenes when lowering 18 → 10 because the detector's min-duration merge
  runs after clustering, so help text says "tends to", not "=".

**Verification (executed):** `--help` lists both flags; each parses with exit 0. `--dump-params
--mode balanced --scene-detection-method semantic --scene-clustering-threshold 10` →
`features.scene_detection = {method: semantic, clustering_threshold: 10.0}`; without `semantic` the
WARNING names the effective method. Real CPU runs on the 293 s Netflix clip (balanced, tiny,
`--debug`) at 18 and 10: the factory log shows `clustering_threshold: 18.0` / `10.0`, 7 vs 6 scenes,
48 vs 50 cues, both exit 0. `inspect.signature` confirms the Qwen and Decoupled ctor params;
`prepare_qwen_params({"qwen_params": {"scene_clustering_threshold": 12}})` → 12.
`node --check app.js`. New test file 8 passed.

**Decision:** owner (CFF2, 2026-09-05): expose to CLI and the GUI Customize parameters tab.
Customize lives on the Ensemble tab only (D4).

## 2026-09-05 — CFF3: Balanced defaults to an external speech segmenter (FireRedVAD), single-pass presets resolved

**Area:** new `whisperjav/config/segmenter_presets.py`; `whisperjav/main.py` (default block,
routing guard, preset merge, `--speech-segmenter` help); `whisperjav/ensemble/pass_worker.py`
(re-exports, balanced default); `whisperjav/webview_gui/api.py` (`get_pipeline_defaults`);
`whisperjav/webview_gui/assets/app.js` (`applyPipelinePresets`, `pickBalancedDefaultSegmenter`);
`whisperjav/modules/faster_whisper_pro_asr.py` (stale comment); `tools/ct2_degradation_probe.py`
(help only); `README.md`; tests `tests/test_balanced_defaults_v192.py` (new, 21 tests).

**What changed**
- `--mode balanced` with no `--speech-segmenter` now runs a WhisperJAV external segmenter:
  `firered-vad`, or `ten` if the fireredvad package is missing, or `silero-v3.1` if both are
  (WARNING with the pip command when FireRedVAD is skipped). The same chain is applied by the
  ensemble pass worker for a balanced pass without `--passN-speech-segmenter`, and by the GUI's
  Ensemble-tab preset when the pipeline is switched to balanced (availability-aware).
  `--speech-segmenter faster-whisper` restores the v1.9.0/v1.9.1 native-VAD behaviour.
- The single-pass path now resolves the backend's per-sensitivity YAML preset (the ensemble
  path always did). Order: backend → YAML preset → Test-D grouping overlay → explicit CLI
  overrides. Before this, `--sensitivity` was inert for a non-silero segmenter on
  `--mode balanced` — the guard's stated reason for downgrading such choices.
- Routing guard: `firered-vad` and `ten` are exempt on `--mode balanced` (their presets now
  flow); every other non-silero backend keeps the downgrade to silero-v3.1 (fidelity/fast/faster
  untouched; wider unification is PR #375's scope). Warning text updated.
- `SEGMENTER_PARAMS`, the backend→YAML tool map and `resolve_qwen_sensitivity` moved verbatim
  from `pass_worker.py` (imports every pipeline) to the light module
  `config/segmenter_presets.py`; `pass_worker` re-exports the old names, so existing imports and
  tests are unchanged.
- GUI Customize panel (`get_pipeline_defaults`): for a non-silero external backend the returned
  `vad` block is the segmenter's effective parameters (YAML preset + Test-D) instead of the
  resolver's silero values that the ASR firewall discards at run time.
- Probe tool default left at `faster-whisper` so new runs stay comparable with the reporter
  datasets already collected; its help says how to mirror the v1.9.2 default.

**Consequences stated for users (release notes):** Balanced is slower than in v1.9.0/v1.9.1
(one recognizer call per VAD group instead of per scene); the #394 corroboration counter is
active on balanced by default, so a file with zero cues while speech kept being detected is
`suspect` rather than `empty` (exit code changes only under `--fail-on suspect`); Test-D
grouping (9.0 s / 0.1 s) overrides the FireRedVAD YAML `max_group_duration_s` (7/6/5) and
`chunk_threshold_s` (1.0) — the Segmenter tab in Customize shows the YAML values for those two
keys, not the Test-D ones (pre-existing display gap for every external segmenter on balanced).

**Verification (executed):** `--dump-params --mode balanced` × conservative/balanced/aggressive →
backend firered-vad, threshold .5/.4/.3, max_speech 7/6/5, end_pad 250/150/100, max_group 9.0,
chunk 0.1, `vad` cleared + `_dump_note`; `--speech-segmenter faster-whisper` → native preset kept;
`ten` → exempt, preset + Test-D; `whisperseg` → downgrade warning; `silero-v6.2`, `--mode
fidelity`, `--mode fast` unchanged; `--vad-threshold 0.6 --max-group-duration 6` win. Real CPU
run (`--mode balanced --model tiny`, 15 s clip): "Speech segmenter set to: firered-vad",
Test-D line, "Speech Segmenter initialized: firered-vad", 3 cues, RUN SUMMARY exit 0.
`--help` lists firered-vad; `--speech-segmenter firered-vad --help` exit 0. `get_pipeline_defaults`
called directly for firered-vad / faster-whisper / silero-v3.1. `node --check app.js`.
Suites: new file 21 passed; `test_qwen_sensitivity` + `test_ensemble_params` +
`test_gui_custom_params_simulation` 104 passed / 8 failed — identical to the baseline before
this change (stale silero-v6.2 set); `test_gui_settings` + `test_gui_run_summary` 76 passed.

**Decision:** owner (CFF3, 2026-09-05): default = FireRedVAD, "external" = a WhisperJAV segmenter
as opposed to faster-whisper's native VAD; D3 Test-D stays; D4 no Transcribe-tab dropdown; D6
`suspect` verdicts on balanced accepted after explanation; D7 fallback to another WhisperJAV
segmenter, never native. Guard exemption limited to the two chain members (owner: no
over-engineering). Thread owed: #311.

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
  drops the `experimental` tag; `speech_segmentation/backends/__init__.py` was missed at first and
  fixed after the adversary pass). The remaining truthful caveat is kept in words: detection presets
  are upstream-derived, the segment cap was JAV-tuned on 2026-08-14.
- Preflight lists `fireredvad` as an optional dependency with an actionable message (it is the
  Balanced default from CFF3; without it Balanced falls back to another WhisperJAV segmenter).
- Hygiene found by the sync gate while adding the entry: the registry pinned `numba>=0.60.0`
  while pyproject says `>=0.61.0`, and the fallback template disagreed with the registry on
  `numba` and `transformers`. Aligned the registry and the fallback template to what pyproject
  already ships (pyproject is what pip/uv and the conda-constructor requirements use, so no user
  install changes); without this the sync gate could not confirm the new dependency. **Owner may
  revert** — it is not traceable to CFF1–CFF6.

**Verification:** `python -m whisperjav.installer.validation` → PASSED (was failing on the numba
mismatch before); `pytest tests/test_installer.py tests/test_installation.py
tests/test_dependency_cross_match.py` → 100 passed, 6 failed, all six identical on HEAD before this
change (WJ env has numpy 1.26 / pip-check conflicts / a stale entry-point test); `uv lock` exit 0;
the real generator `build_release.generate_requirements_from_pyproject()` executed against the current
pyproject emits `fireredvad>=0.0.2` (a `--dry-run` alone prints no content, so it proves nothing);
YAML parses;
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
  Windows) when the second task's `FasterWhisperProASR` initialises after the
  first task's pipeline was cleaned up — verified 2026-09-04. **2026-09-05:**
  with the CFF1 default (recogniser in a worker process) the same two-clip run
  completes with both SRTs and a RUN SUMMARY, because the parent never
  destroys a CT2 model. The limitation still applies with
  `--model-refresh-audio-minutes 0` (in-process instance); stated as such in
  the release notes.
- Telemetry outside `BalancedPipeline` (the default ensemble pairing records
  nothing); a Transcribe-tab segmenter control (owner decision); a default cue
  ceiling for the default ensemble (owner decision); #394 containment.
- Test hygiene: `tests/test_gui_refactor.py`, `tests/test_postprocessing_performance.py`
  (rewrap stdout at import) and `tests/test_tab_spacing.py` (opens Tk)
  prevent a single-session `pytest tests/`.
- #314 installer progress streaming never landed.
