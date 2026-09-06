# WhisperJAV v1.9.2 — Release Notes

> **Status: in development, not yet released.** This document is updated as work
> lands on `dev_v1.9.2`, so it is complete up to the last entry in the changelog
> at the foot of the page. Anything listed under *Planned* or *Known limitations*
> has not shipped.

A maintenance release. The theme is **failures that stayed quiet** — crashes on
particular drives, translation aborts on small models, settings that were ignored
without saying so, and a documented install step that silently disabled GPU
acceleration. Nothing here changes how transcription sounds; it changes how often
you find out something went wrong.

---

## Fixed

### Runs that crashed or aborted

- **A graphics card this build cannot use stops the run at start-up instead of after an
  empty one.** One user with a GTX 1060 saw PyTorch warn that the card is not supported by
  the installed build, then watched a twenty-minute run produce no subtitles and report
  success. WhisperJAV now stops before doing anything, names the card, its compute
  capability and the ones the build supports, and gives the two ways out: install a
  matching PyTorch build, or pass `--accept-cpu-mode` to run on the CPU. `whisperjav
  --check` reports the same fact and exits with an error on such a card. Two things to
  know: with `--accept-cpu-mode` the ChronosJAV pipelines (anime-whisper, Qwen3, Cohere)
  and the speech enhancers still choose the GPU on their own and may fail there, so on
  such a card prefer the Whisper pipelines; and if the Balanced pipeline's CTranslate2
  recognizer was working on such a card, it now stops at the gate too, since there is no
  evidence it was. Machines with no GPU at all behave as before. (#411, #326, #333)

- **Subtitle lines that are only punctuation are removed on the ChronosJAV pipelines.**
  The Qwen3-ASR pass can emit a cue that is nothing but 「。」 for a stretch of sound
  without words; in one user's file a quarter of the second pass was such lines. An entry
  whose text is only punctuation once whitespace and punctuation are stripped is now
  dropped whole, the same way a line of pure breathing is. Punctuation inside text is
  untouched: anime-whisper's ellipses are part of how that model writes and are kept as
  they are. A lone 「?」 or 「!」, which the anime-whisper pass used to keep, now goes with the
  rest. The Whisper pipelines are unchanged; their sanitizer already handled this. (#413)

- **Downloaded models no longer wait on huggingface.co.** Every model load used to check
  huggingface.co for a newer version first. With the site unreachable (no VPN, a blocked
  network), each file was retried five times before the cached copy was used: in one user's
  log the WhisperSeg speech segmenter alone cost about ten minutes per run. WhisperSeg and
  anime-whisper now load from the local cache first and contact the site only when a file
  is genuinely missing. The trade-off: once a model is downloaded, these two loaders keep
  that copy and will not pick up a later re-upload of the same model name on their own.
  The Qwen3-ASR pass is a separate case: its loader makes one online query that has no
  cached fallback, so with the site unreachable that pass still fails unless you switch on
  *Offline mode* (below), which is what makes it work. (#415)

- **Files on cloud-mounted drives no longer abort the run.** A video on a
  CloudDrive2-mapped drive would fail immediately with
  `OSError: [WinError 1005]`, before any processing started, even though the same
  file played normally and worked in other subtitle tools. WhisperJAV was asking
  Windows to canonicalise the path purely to avoid processing the same file
  twice; that question has no answer on such a volume. It now falls back
  gracefully instead of stopping. (#340)

- **Translation no longer fails outright on small-context models.** Using an
  Ollama model with a 2K or 4K context — `qwen2.5:3b` and similar — every
  translation aborted with
  `min_batch_size must be less than max_batch_size`. WhisperJAV was shrinking the
  maximum batch to fit the model's context window without lowering the matching
  minimum, so the two crossed and the translator refused to start. The error also
  named a setting most people had never touched: the *Max Batch Size* value in
  the GUI was already being overridden. Both bounds are now resolved together.
  (#341)

### Settings that were ignored without telling you

- **A mistyped speech-enhancer name is now rejected immediately.** Passing
  `--pass2-speech-enhancer zipenhance` (one letter short of `zipenhancer`) was
  accepted, quietly downgraded to no enhancement at all, and the run continued —
  so you could spend hours transcribing with enhancement you believed was
  active. The name is now checked when the command is parsed, and an unknown one
  fails straight away, listing the valid options. Genuine cases — a backend whose
  optional dependencies are not installed — still fall back with a warning, as
  before. (#306)

- **The Ollama model list no longer recommends a model known to break
  subtitles.** When the curated model list could not be read, the built-in
  fallback offered `shisa-v2.1-qwen3-8b` first. That is a *thinking* model, the
  class removed from the curated list in v1.8.11 precisely because they write
  their reasoning into the subtitle file. The fallback now mirrors the curated,
  instruct-only list.

- **DeepSeek v4-flash no longer reasons before translating.** DeepSeek changed
  its server-side default so that v4 models think before answering, which is
  slow, burns rate limit, and can leak reasoning text into the subtitle file.
  WhisperJAV now asks for reasoning to be turned off on `deepseek-v4-flash`
  when the DeepSeek provider is used directly; `deepseek-v4-pro` *is* the
  reasoning model and is left alone, as are models outside the v4 family. The
  OpenRouter route to the same model is not covered, because it goes through a
  different client. Diagnosed and prototyped by @mcdman. (#395)

- **OpenRouter no longer defaults to a retired model.** The DeepSeek route was
  pinned to `deepseek/deepseek-chat`, which DeepSeek retired on 2026-07-24. It
  now uses `deepseek/deepseek-v4-flash`, matching the direct-API default.
  (#325)

### Messages that misled

- **The speech-segmenter fallback warning no longer says the release is
  broken.** Choosing WhisperSeg or TEN outside ensemble mode printed a warning
  describing a *"known v1.9.0 routing bug (catastrophic empty output on JAV
  moaning content)"*. It is describing an intentional guard against a routing
  limitation, not a defect in the build you are running. The wording now says
  that, and points at the two configurations that do work: `--ensemble` for full
  segmenter support, or `silero-v6.2` for single-pass. (#323)

### Documentation that broke installs

- **The Windows manual-install guide no longer replaces your GPU PyTorch with
  the CPU build.** One step installed ClearerVoice without `--no-deps`, so pip
  pulled an unpinned PyTorch from PyPI over the CUDA build installed moments
  earlier — silently disabling GPU acceleration for the entire installation. The
  standalone installer has always used `--no-deps` here for exactly this reason.
  The guide now matches it, and installs ClearerVoice's own dependencies
  explicitly beforehand so nothing is lost. (#334)

- **The Linux GUI instructions now work on current distributions.** The guide
  named `libwebkit2gtk-4.0`, which Ubuntu 24.04 does not package — it ships the
  4.1 series only. Following the guide exactly still ended in
  `ModuleNotFoundError: No module named 'gi'`, because the system packages it
  installs land outside the virtual environment. Both WebKit series are now
  documented, along with the missing step: pywebview needs a backend installed
  *into the venv*, most simply `pip install "pywebview[qt]"`. This is Linux-only
  — on Windows and macOS pywebview installs its backend automatically, and the
  guide now says so rather than leaving people to wonder. (#366)

### Convenience

- **Batch runs can skip files that already have subtitles.** A *Skip
  already-subtitled files* checkbox now appears in the GUI, and it works in
  ensemble mode as well as single-pass. `--skip-existing` already existed on the
  command line but had never been exposed in the interface, and did not cover
  ensemble runs. (#328)

- **The GUI can remember your settings between launches, if you ask it to.**
  A *Remember settings* checkbox stores the Transcription tab's fields and
  restores them next time. It is **off by default**, so the deliberate
  start-from-defaults behaviour is unchanged unless you opt in. This answers
  the settings-reset question in #96; the two-pass Customize parameters asked
  for in #381 still reset on launch. (#96, in part)

- **Saving presets no longer fails on relocated user profiles.** Where
  `%APPDATA%` is a junction — common when the profile has been moved to another
  drive — saving a preset could fail and the preset then could not be found.
  Writes are now atomic and resolve the real location first. (#309)

- **The GUI's DeepSeek model lists are current.** Both dropdowns offered
  `deepseek-chat` and `deepseek-coder`, retired upstream on 2026-07-24. They now
  offer `deepseek-v4-flash` and `deepseek-v4-pro`, and the OpenRouter list
  matches the backend default instead of contradicting it. (#325, #382)

With thanks to **@Mimic-me**, who contributed these as a reviewable batch.

### Convenience, continued

- **Offline mode.** A new checkbox in the Transcription tab's Advanced options, *Offline
  mode (downloaded Hugging Face models only)*, and the matching `--offline` flag, tell the
  Hugging Face libraries to use only models that are already downloaded and to make no
  requests to huggingface.co at all. It applies to Transcription and Ensemble runs and to
  their worker processes. A model that was never downloaded then fails immediately instead
  of retrying for minutes; for WhisperSeg and anime-whisper the message names the model and
  the cache folder, for other models it is the Hugging Face library's own message, which
  still mentions the connection. It is off by default, so a first run can still download
  what it needs. Under the hood it sets `HF_HUB_OFFLINE=1`, which you can
  also set yourself as an environment variable if you prefer. Not covered, because they
  download through other channels: Silero via torch.hub, openai-whisper weights (Fidelity),
  ModelScope speech enhancers and NeMo configs. (#415)

### Tools

- **A scene inspector, for looking before you transcribe.** `tools/scene_inspector.py`
  runs WhisperJAV's scene detectors (auditok, silero, semantic) on a media file
  exactly as the pipelines do and writes, in a `scenes_info` folder next to the
  file: a per-scene table, statistics about the split, loudness and speech
  content per scene, how an existing SRT lands on the scenes, three screenshots
  per scene plus a contact sheet, FFmetadata chapters you can load in a player,
  and a side-by-side comparison when several detectors are run. Nothing is
  transcribed and no setting is changed. Use it to judge a detector or a
  parameter (`--sensitivity`, or `--scene-threshold` for the semantic detector)
  before spending a full run. Guide: `tools/scene_inspector.md`.

### Failures that used to pass silently

- **The sanitization summary inside the artifacts file told the truth again.**
  When WhisperJAV writes its `.artifacts.srt` — the file people attach to bug
  reports — it opens with a `[SANITIZATION SUMMARY]` block. That block was being
  built from a counter only one internal code path maintained, so on ordinary
  runs it reported `Hallucinations modified/removed: 0` and `Final subtitles: 0`
  while the very same file went on to list the removals and the run produced
  plenty of subtitles. It now counts what was actually removed and how many
  subtitles were actually written. This was worth fixing beyond tidiness: it was
  misinforming the diagnosis of other bugs. Found while investigating #324.

- **Every run now ends with a per-file state table, a manifest, and an exit
  status that means the same thing in every mode.** WhisperJAV could finish,
  print `[SUCCESS]`, and hand back an empty or drastically incomplete subtitle
  file — an 8,766-second video returning output that stopped at 377 seconds was
  reported as a success, and in one case a 0-byte file was written while the
  console declared the run complete. Nothing compared the output against the
  input, and each execution path decided "success" on its own: the normal path
  returned nothing to the caller, the async path hard-coded zero failures, the
  ensemble path kept a separate failure list, and for balanced, fast and faster
  the exit status was hardcoded to 0.

  One vocabulary now describes every file, in the console, in the manifest and
  here:

  | State | Meaning | Exit status |
  |---|---|---|
  | `done` | a subtitle file with at least one cue was written | 0 |
  | `empty` | the run completed and produced no cues, and nothing contradicts that reading | 0 (reported; `--fail-on empty` makes it 1) |
  | `suspect` | something does not add up: the output spans less than `--min-coverage` of the media, the recognizer returned nothing for consecutive scenes while speech was still detected (Balanced with an external segmenter only), or in ensemble pass 2 failed and the output is pass 1 alone. A zero-cue file with any of that evidence is `suspect`, not `empty` | 0 (reported; `--fail-on suspect` makes it 1) |
  | `failed` | an error: an exception, a crash, a translation that raised, or a subtitle file the pipeline reported writing that does not exist | 1 |
  | `skipped` | nothing was attempted because the output already existed | 0 |

  Zero subtitles is an observation, not a failure: silence, music and speech the
  recognizer could not use all end there, and the run says so instead of
  guessing. Warnings never change the exit status. Scripts that want a stricter
  contract opt in with `--fail-on empty`, `--fail-on suspect`, or both.
  Coverage is shown next to the state as `ok`, `low` or `not assessed`
  (unknown duration, media under two minutes), so a check that could not run is
  visible rather than silent.

  A normal or async batch continues through every file; the exit status
  reflects the worst file. (An ensemble batch is processed by one worker per
  pass, so a worker crash still ends the whole batch, as before.) An
  interrupted or crashed run prints the table for the files that finished and
  exits 1. A `whisperjav_run.json` manifest is written next to the outputs
  (into the output directory, or beside the first input when
  `--output-dir source`) with the same states, so a script can read per-file
  results instead of parsing the console.

  Found while wiring this: `--async-processing` never waited for its tasks.
  It submitted them, summarised them while they were still queued, and then
  cancelled whatever had not started when it shut down, so an async run
  reported "Task cancelled before processing started" and its summary never
  described real results. It now waits for each task and reports it like any
  other file.

  **The GUI follows the same contract.** When a transcription or ensemble run
  ends, the GUI no longer prints `[SUCCESS]` on exit status 0. It reads the
  manifest the run wrote and closes with the tally in the same five words:
  `[FINISHED] done 2 · empty 1 · suspect 0 · failed 0 · skipped 0 (exit status
  0)`, or `[FINISHED WITH FAILURES] …`, or `[STOPPED] …` for an interrupted
  run. The status line shows `Finished · <tally>`, the console lists every
  file that did not end `done` with its state and reason, and the manifest
  path is printed. Two checkboxes under Advanced options, *Treat 'empty' files
  as failures* and *Treat 'suspect' files as failures*, are the GUI's form of
  `--fail-on`; they apply to every Transcription-tab mode and to Ensemble
  runs, and are off by default. The GUI never re-derives the exit status; it
  repeats the one the run returned. (The separate translation runner's closing
  lines use the same `[FINISHED]` wording.)

  This replaces the pre-release behaviour, announced on #394, in which a file
  with no subtitles failed the run: that made a normal outcome fatal by
  default, and it did not apply to ensemble or async runs at all. A translation
  error now marks the file `failed` in every mode; previously the ensemble path
  exited 1 on it and the others exited 0. (#394, #263)

---

## Changed defaults and installation

- **The recognizer is reloaded fresh every 20 minutes of audio.** Two controlled observations in
  the #394 investigation point at the recognizer instance rather than the audio: in #302 four minutes
  that returned nothing inside a long run transcribed normally as a separate job, and the diagnostic
  probe saw one instance transcribe identical audio 111 times and then return nothing for the rest of
  the run. Whether cumulative use is the trigger is not established; bounding it is containment.
  Balanced and Fidelity now count the scene audio each
  recognizer instance has been given and, once it passes 20 minutes, unload it and load a fresh
  instance at the next scene boundary. Balanced keeps its recognizer in a separate worker process for
  this (the CTranslate2 model cannot be safely destroyed in-process on Windows); Fidelity reloads in
  place. Cost: one model load per refresh — measured at 13–14 s for large-v2 on an RTX 3060, so
  roughly 80 s over a two-hour film at the default, a few percent of a Balanced run. Adjust with
  `--model-refresh-audio-minutes` (0 = never; the GUI has a matching Advanced-options field that
  applies to Transcription and Ensemble runs). Balanced's per-scene telemetry records which instance
  generation decoded each scene. This is containment, not a fix for the root cause, which is still
  under investigation.
- **Balanced keeps faster-whisper's built-in VAD as its default speech segmenter**, as in
  v1.9.0/v1.9.1 (one recognizer call per scene). A WhisperJAV segmenter is one flag away:
  `--speech-segmenter firered-vad` (or `ten`, `silero-v3.1`, …); in the GUI, the Ensemble tab's
  segmenter dropdown. Under the built-in VAD there is no independent speech detector, so the
  run-outcome check cannot mark a zero-cue Balanced file `suspect`; it is reported `empty`.
- **`--sensitivity` now applies to FireRedVAD and TEN on plain `--mode balanced`.** The
  single-pass path previously never loaded a non-Silero segmenter's sensitivity preset (which is
  why it used to downgrade those choices to Silero). It does now, in the same order as ensemble:
  preset, then the fine-grained grouping overlay, then your explicit flags. Other segmenters
  (WhisperSeg, NeMo, whisper-vad) are still routed through `--ensemble` only.
- **Balanced runs with a WhisperJAV segmenter can be reported `suspect`.** The "speech kept being
  detected but nothing came back" counter that corroborates a #394-style stall only works with a
  real speech detector, so it is inert under the default built-in VAD. With `--speech-segmenter
  firered-vad` (or any other WhisperJAV segmenter) it is active: a Balanced file with zero cues
  while speech was detected for several scenes in a row is classified `suspect` instead of
  `empty`. The exit code changes only if you use `--fail-on suspect` (CLI or the GUI checkbox).
- **FireRedVAD is installed with WhisperJAV.** The `fireredvad` package is now part of the
  standard install (every extra that includes `cli`, the Windows installer, Colab and Kaggle).
  It is no longer marked experimental in the CLI, the GUI or the docs. Its detection presets are
  the upstream defaults; the segment-length cap was tuned on JAV field tests in v1.9.0. The
  ~2 MB model weights still download from HuggingFace on first use. (#311)

- **The semantic scene detector's sensitivity is adjustable.** `--scene-clustering-threshold`
  (legacy modes and `--pipeline decoupled`) and `--qwen-scene-clustering-threshold` (`--mode qwen`)
  set the clustering distance that separates one scene from the next: 18 by default, 10 for more
  and shorter scenes, 22 for fewer and longer. In the GUI it is the "Scene Change Threshold" slider
  in the Ensemble tab's Customize Parameters dialog (Scene tab for Whisper pipelines, Audio → Custom
  Scene Bounds for the ChronosJAV pipelines). It only affects the semantic detector; the CLI warns
  if you set it with auditok or silero. **Treat it as experimental:** measurements made while
  preparing this release (`docs/research/semantic_scene_premise/`) show the scene count does not
  follow this setting closely — on a 65-minute file it stayed between 43 and 65 scenes across the
  whole range and even rose slightly as the value went up, because the number is a clustering
  distance rather than a scene-length control. The control stays in this release as-is; the
  detector's boundary logic is scheduled for a 2.x revision.
- **ChronosJAV output drops lone 「はい。」 and 「うん。」 lines.** The Qwen pipelines' lone-line
  filter (v1.9.0) removed single-character artefacts such as 「あ。」 and 「は。」 but deliberately kept
  「はい。」 and 「うん。」 as backchannel. In JAV material almost all of those lone lines are moans, not
  agreement, so they are now dropped too. Only a line that is exactly that one word goes; 「はいはい。」,
  「ううん。」, 「あ、うん。」 and full sentences are untouched. Disable with
  `--no-qwen-drop-nonverbal-lines`. (#254)

---

## Documentation

- The README's **Mix-and-match strategies** section gained a naming note for
  command-line users: *ChronosJAV* is one pipeline with interchangeable
  recognizers, so `qwen` and `anime-whisper` are two backends of it rather than
  two separate pipelines. The GUI lists them side by side; on the CLI both live
  under `qwen`, with the recognizer chosen inside `--pass1-qwen-params`. The
  exact invocation is now written out.
- The merge-strategy guidance previously covered three of the seven available
  strategies. All seven are now listed with what each one does.

---

## Already live, no update required

- **The `pornify` translation instructions were eight months out of date for
  every user.** Instruction files are fetched from a Gist at run time, and that
  copy had never received the January fix that removed hardcoded English. The
  served version told the model to imitate an "American adult movie", gave
  worked examples whose translations were all in English, and instructed it that
  *every* line must be sexualised with no plain dialogue allowed — which explains
  both the reports of English output when another target language was selected,
  and of greetings being rewritten as explicit content. The Gist was corrected on
  2026-08-29. Because instructions are fetched on each run, **this reached
  everyone immediately** — no update or reinstall needed. Diagnosed by
  @yhxkry on #397. (#397, #339, #347, #305)

---

## For diagnosing #394

- **Every Balanced run now keeps a per-scene telemetry file, on by default.**
  One JSON record per scene: how long the recognizer took, how many segments
  it returned, whether the decoder had to retry at a higher temperature, the
  confidence and compression figures behind those retries, and GPU and process
  memory at that moment. It is written to `raw_subs/<name>.asr_telemetry.jsonl`
  next to the outputs, the folder that already holds the artifacts people
  attach to bug reports, so a run that later turns out to be a #394 case has
  its record without anyone having known to ask. Inside an ensemble run each
  Balanced pass writes its own file (`<name>.pass1.asr_telemetry.jsonl`,
  `…pass2…`), beside that file's pass outputs, in `source` mode too.
  Each scene's record is appended the moment the scene finishes, so a run that
  crashes, hangs or is stopped still leaves everything up to that point on
  disk. `--asr-telemetry PATH` moves it (a directory gets one file per media;
  a file path is for a single input, later inputs overwrite it);
  `--no-asr-telemetry` switches it off, and the GUI has the same switch under
  Advanced options (*Keep per-scene ASR telemetry*, on by default). Pipelines
  other than Balanced do not record it yet, so the GUI's default ensemble
  pairing (anime-whisper + Qwen3-ASR) still produces no telemetry. The memory
  figures include a device-wide CUDA reading, since the recognizer allocates
  outside PyTorch's own counters.

  Found while wiring this: with `--async-processing` and `--output-dir
  source`, every file's outputs were written beside the *first* file in the
  batch. Each file's outputs now go beside that file, as in the normal path.

  It exists because every record we had described the *aftermath* of the #394
  failure and none described the approach to it. The most useful unexplained
  detail in that issue is that the collapse is preceded by a slowdown —
  @daoran9 measured the median call going from 1.05s to 31.65s beforehand — and
  a model that is merely stuck returns nothing *quickly*. Whether that slowdown
  comes from repeated decoder retries or from memory growth points at very
  different causes, and until now nothing recorded either.

  At the end of a run it also prints a one-line before/after comparison, so a
  reporter can paste a single line rather than be persuaded to attach a file.

---

## Under the hood

Not user-visible, but worth recording:

- The GUI settings test suite was failing on `main` because four expectations
  were left behind by v1.9.0's ensemble default change. The test that was
  supposed to catch HTML/backend drift never actually read the markup — it
  compared against a hand-copied list, so it could only detect drift from
  whatever someone last typed into it. It now parses `index.html` directly, and
  both of its failure paths were verified by deliberately perturbing the markup.
- Regression tests were added for every fix above, including the two real-world
  failures reported on #394 encoded as explicit cases.
- Two research notes under `docs/research/semantic_scene_premise/` record why the
  semantic scene detector's threshold sets the number of sound classes rather than
  scene granularity (measurements, re-runnable) and what the audio-segmentation
  literature establishes about the "same sound = same scene" premise. They inform
  the 2.x work on the detector; nothing in this release changes because of them.

---

## Planned for this release, not yet landed

- **Corroboration beyond Balanced with a WhisperJAV segmenter.** Balanced reports the
  speech-positive empty-scene signal only when a WhisperJAV segmenter is selected
  (`--speech-segmenter firered-vad`, `ten`, …); under its default built-in VAD, and in
  Fidelity, Fast, Faster and the ChronosJAV pipelines, `suspect` can only come from the
  span check (or, in ensemble, from a pass-2 failure).

---

## Known limitations

- **Offline mode covers Hugging Face downloads only.** Silero (torch.hub, see #263),
  openai-whisper weights, ModelScope enhancers and NeMo configs still reach their own
  servers when a model is missing. Models must have been downloaded once while online.
- **`--async-processing` with Balanced mode, more than one file, and model refresh
  switched off (`--model-refresh-audio-minutes 0`) ends the process without a
  summary.** In that configuration the second file's recognizer initialising after
  the first file's pipeline was torn down kills the process natively, the same way
  the ctranslate2 destructor crash the normal path deliberately avoids. With the
  default refresh setting the recognizer lives in a worker process and the same
  two-file run completes normally (verified on two clips). The normal (non-async)
  path is unaffected either way.
- **The root cause behind #394 is still open.** The recognizer can enter a state
  where it returns nothing for the rest of a run, and the work above detects the
  *result* rather than preventing it. Investigation continues, with useful
  evidence contributed by @AlanZ-Git and @daoran9.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-09-06 | A GPU the PyTorch build has no kernels for stops the run at start-up with the reason; `--accept-cpu-mode` is the way through; `--check` exits 1 on such a card (#411, #326, #333) |
| 2026-09-06 | Subtitle entries that are only punctuation (a lone 「。」 or 「、」, an ellipsis alone) are dropped on the ChronosJAV pipelines; inline punctuation untouched (#413) |
| 2026-09-06 | Cached Hugging Face models load without a hub round-trip (WhisperSeg, anime-whisper); `--offline` flag and GUI "Offline mode" checkbox set `HF_HUB_OFFLINE=1` for the run and its workers, off by default; a missing model fails at once (#415) |
| 2026-09-05 | `tools/scene_inspector.py` added: scene-detector statistics, per-scene screenshots and contact sheet, loudness and speech ratio, SRT overlay, chapters, multi-detector comparison, `--sensitivity` / `--scene-threshold` presets (`tools/scene_inspector.md`) |
| 2026-09-05 | Research notes on the semantic scene detector's threshold (not a granularity lever; slider kept, detector revision scheduled for 2.x) |
| 2026-09-05 | Recognizer refreshed after 20 minutes of scene audio (`--model-refresh-audio-minutes`, GUI field): Balanced hosts its CTranslate2 model in a worker process and replaces it, Fidelity reloads in place; telemetry gains `model_epoch`; async + Balanced + several files now completes with refresh on |
| 2026-09-05 | Qwen lone-line filter also drops 「はい。」 and 「うん。」 (exact lone token only; `--no-qwen-drop-nonverbal-lines` to keep) (#254) |
| 2026-09-05 | Semantic scene-change threshold exposed: `--scene-clustering-threshold`, `--qwen-scene-clustering-threshold`, `--passN-qwen-params scene_clustering_threshold`, and a slider in both Customize modals |
| 2026-09-06 | Balanced default segmenter stays faster-whisper's built-in VAD (owner reversed the 2026-09-05 FireRedVAD default before release); `--sensitivity` presets resolved for FireRedVAD/TEN on single-pass balanced; Balanced runs with a WhisperJAV segmenter can be `suspect` |
| 2026-09-05 | `fireredvad` becomes a standard dependency (`[cli]`, installer, Colab/Kaggle); all "experimental" labels on FireRedVAD removed; registry/template pins aligned with pyproject (#311) |
| 2026-09-04 | ASR telemetry on by default, written to `raw_subs/` next to the outputs; reaches Balanced passes inside ensemble runs (per-pass files, source mode included); `--no-asr-telemetry` added. Version set to 1.9.2 in code. Release-note corrections: #395 is fixed, not a limitation; persistence entry no longer credits #298 (closed) or #381 |
| 2026-09-03 | GUI speaks the same contract: reads `whisperjav_run.json` on exit, closes with `[FINISHED] <tally>` instead of `[SUCCESS]`, shows the tally in the status line and dialog, and offers the two "treat as failure" checkboxes (`--fail-on`) for Transcription and Ensemble runs |
| 2026-09-03 | One per-file vocabulary (`done` / `empty` / `suspect` / `failed` / `skipped`), one exit-status rule for sync, async and ensemble, `--fail-on`, the RUN SUMMARY table and the `whisperjav_run.json` manifest. Replaces the pre-release gate that failed a run on empty output and fixes the pre-release defect where every successful ensemble run exited 1 (#394, #263) |
| 2026-08-30 | `--asr-telemetry`: per-scene decode and memory record, plus a trend summary, to capture what precedes a #394 collapse rather than only its aftermath |
| 2026-08-30 | Corroborating signal instrumented in Balanced: consecutive scenes with detected speech but no output now corroborate a low-coverage failure (#394) |
| 2026-08-30 | Artifacts `[SANITIZATION SUMMARY]` now counts real removals and the real final subtitle count instead of a counter the active workflow never updated |
| 2026-08-30 | Output coverage gate wired: empty output fails the run and exits non-zero; short output warns. Nuclear exit no longer hardcoded to 0 (#394, #263) |
| 2026-08-30 | DeepSeek reasoning disabled for v4-flash (#395, @mcdman); post-processing stage logging so a stall names the operation that hung (#372) |
| 2026-08-30 | PR #378 merged (@Mimic-me): skip-existing in the GUI and ensemble (#328), opt-in settings persistence (#96, #298, #381), junction-safe preset saves (#309), DeepSeek v4 names in the GUI (#325, #382); OpenRouter dropdown synced to the backend default |
| 2026-08-30 | Linux install guide: GUI-backend step moved to a shared section covering every distribution; Fedora WebKit package corrected to the 4.1 series (#366) |
| 2026-08-30 | Seven small defects fixed: #340, #341, #306, #325, #323, #334, #366, plus the Ollama fallback model list |
| 2026-08-30 | Output coverage assessment added with tests — inert, pending a decision on enforcement (#394) |
| 2026-08-30 | GUI settings suite greened; drift between `index.html` and the backend defaults is now detected automatically |
| 2026-08-29 | README: ChronosJAV naming note for CLI users, and the complete seven-strategy merge table |
| 2026-08-29 | `pornify` instruction Gist synced to the repository version — delivered to users immediately, outside the release (#397) |
