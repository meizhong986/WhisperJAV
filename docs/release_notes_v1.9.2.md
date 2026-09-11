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
  capability and the ones the build supports, and asks, in a boxed prompt you cannot miss,
  whether to continue on the CPU or abort; it waits for your answer and never continues on
  its own. A machine with no GPU at all is asked the same question; the old thirty-second
  countdown that then continued by itself is gone. In the GUI, where the
  worker cannot ask, the run aborts and tells you to tick *Accept CPU-only mode* if you
  want it to proceed on the CPU (the box now applies to Ensemble runs as well); on the
  command line, `--accept-cpu-mode` or an explicit `--device cpu` answers in advance, and
  a "yes" is remembered for the rest of that run. `whisperjav --check` reports the same
  fact and exits with an error on such a card. Two things to know. With
  `--accept-cpu-mode` the ChronosJAV pipelines (anime-whisper, Qwen3, Cohere) and the
  speech enhancers still choose the GPU on their own and may fail there, so on such a card
  prefer the Whisper pipelines. And this check judges the card by what the installed
  PyTorch build can run; the Balanced, Fast and Faster pipelines transcribe through
  CTranslate2, which has its own GPU kernels and for which WhisperJAV has carried a
  Pascal-specific setting since issue #123, so on a GTX 10-series card those pipelines may
  have been working on the GPU and now stop at this check unless you answer yes, after
  which they run on the CPU. If you script WhisperJAV on a CPU-only machine, pass
  `--accept-cpu-mode` (or `--device cpu`); without it the run now stops where it used to
  continue after the countdown. `--dump-params` and the installer's own import check are
  not asked. (#411, #333; probably #326)

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
  segmenter support, or `silero-v6.2` for single-pass on Fidelity. On Balanced no
  external segmenter can be chosen at all (see "Changed defaults"). (#323)

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
  | `suspect` | something does not add up: the output spans less than `--min-coverage` of the media, the recognizer returned nothing for consecutive scenes while speech was still detected (needs a separate speech segmenter, so not on Balanced — see "Changed defaults"), or in ensemble pass 2 failed and the output is pass 1 alone. A zero-cue file with any of that evidence is `suspect`, not `empty` | 0 (reported; `--fail-on suspect` makes it 1) |
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

- **A false alarm is gone from long scenes, and the diagnostics for them now work.** On any scene of
  eight minutes or more, Balanced used to print `Speech segmentation produced insufficient coverage
  ... Falling back to full-clip transcription`. Nothing had failed. Balanced hands speech detection to
  faster-whisper itself, and a placeholder standing in for the old external detector was being
  misread as a detector failure — on a threshold that was an accident of an unrelated setting rather
  than a chosen value. On one 3-hour test film, seven scenes produced that warning and it accounted
  for every warning the run printed, so a real problem would have been indistinguishable from it.
  Balanced now goes straight from scene detection to the recogniser with no such check, the message
  is gone, and the per-scene diagnostics — which used to record nothing at all for those long scenes,
  even when they produced subtitles — record them properly. Subtitles are unchanged: on the same
  inputs the output files are byte-for-byte identical.

- **The terminal now says what it is doing to each scene, and what it got back.** Each scene reports
  its length, the detection method actually in use, how many subtitles it produced and how long it
  took, and a scene that produced nothing is marked `NO OUTPUT` — for example
  `Scene 4/10 (426s, Internal FW Silero VAD): 7 subtitle(s) in 7s`. A scene that fails now prints a
  line too, where before it only appeared in a transient status message.

- **One diagnostic field was removed because it was never true.** The per-scene record carried
  `speech_detected`, derived from that same placeholder, so on Balanced it was always `true` — in one
  test run all 26 scenes reported speech detected, including the 20 that produced nothing.
  faster-whisper does not report the regions its internal detector used, so there is nothing truthful
  to put in its place; what remains is measured. The "consecutive empty scenes" warning was removed
  for the same reason: it could not tell a recogniser that had stopped working from a scene with no
  intelligible speech in it, which on this material is a perfectly normal thing to find.

- **On Fidelity, a rescue behaviour was removed.** If Fidelity's speech detector reported almost no
  speech in a scene, that scene used to be transcribed from end to end anyway on the assumption the
  detector had malfunctioned. It no longer is: a detector that reports no speech is believed. This
  removes a source of long, expensive passes over scenes that hold no dialogue, and it does mean a
  scene whose detector genuinely fails now yields nothing instead of being salvaged. Balanced is
  unaffected — it has no external detector.


- **Aggressive is retuned so it cannot run away with your time.** On a three-hour film an
  Aggressive run took more than twice as long as Balanced and had to be abandoned before it
  finished. The cause was not one setting but a stack of them, and the largest was that
  Aggressive was allowed to decode a passage a second time at a higher temperature whenever a
  quality check tripped — which on continuous, repetitive audio is often. Aggressive now decodes
  once, at temperature 0, with a narrower beam (2 instead of 3), standard beam termination
  (patience 1.0) and a stricter repetition check on paper (compression ratio 2.2 instead of 2.6 —
  but see the next entry: with a single decode that limit no longer rejects anything), and it
  penalises immediate token repetition more firmly (1.5 instead of 1.3). It still admits quiet
  audio into the recognizer as before — the no-speech and log-probability gates are unchanged on
  Balanced. The same retune is applied to Fidelity, except the repetition penalty, which the
  OpenAI Whisper engine Fidelity uses does not support. On a 25-minute test file Aggressive now
  costs the same wall-clock time as Balanced. Expect it to be faster and more repeatable; expect
  it to explore slightly fewer alternative readings of a difficult passage.

- **The Whisper pipelines now decode at temperature 0 and never retry at a higher one.** Previously
  Fast, Faster and the Kotoba pipeline fell back through as many as four temperatures. That
  fallback is what made a run's duration unpredictable — a bad stretch could quietly cost four
  decodes instead of one. Balanced, Fast, Faster, Fidelity and Kotoba now decode once, at all three
  sensitivities. Output is reproducible run to run and worst-case run time is bounded. The
  ChronosJAV pipelines (anime-whisper, Qwen3, CrispASR) have their own decoders and are unaffected.

  One consequence is worth knowing, because the setting is still in the presets and still has a
  number beside it. **The compression-ratio limit no longer rejects anything on its own.** In both
  recognisers it only ever decided whether to decode the passage again at a higher temperature; it
  never dropped a result. With a single decode there is nothing to retry, so a passage that trips
  the limit is kept. Repeated-character loops — a line of two hundred ん, a stock phrase repeating —
  are removed afterwards by the subtitle cleaner instead, and are listed with their reason in the
  `.artifacts.srt` file beside your subtitles. On a 116-minute test film the cleaner removed all 66
  of them. Lowering the number in the presets would not change what you get.

- **Scene detection uses one sensitivity threshold instead of three.** The semantic scene detector
  had a different clustering threshold and silence-snap window for each sensitivity (22/18/10 and
  6/5/2 seconds). Measurement during v1.9.2 showed the clustering threshold is not the lever its
  name suggests — scene count barely moves across most of its range, and the scene-length bounds
  are what actually decide granularity. Carrying three values implied a control that was not
  there. All sensitivities now use 22.0 and a 6-second snap window. The scene-length bounds still
  vary by sensitivity on Fast; on Balanced and Fidelity they are 28 seconds to 4 minutes (see the
  next entry).

- **Scenes are between 28 seconds and 4 minutes on Balanced and Fidelity.** The ceiling was 20 minutes on
  Balanced and 7 minutes (3 at Aggressive) on Fidelity. On a 3-hour test film the 20-minute ceiling
  left 60% of the running time inside scenes longer than 4 minutes, and each such scene is decoded
  as one long chain of 30-second recogniser windows, so one bad stretch could cost up to 20 minutes
  of audio and there was no progress line for the whole of it. At 4 minutes the same film splits
  into 89 scenes instead of 67, the longest exactly 240 seconds, with the same share of cuts landing
  in silence (95%). What you will see: more, shorter per-scene status lines, and any scene that
  produces nothing costs at most 4 minutes of audio. Whether a whole run finishes faster is not
  yet measured.

  **The minimum is 28 seconds on both, at every sensitivity.** Balanced already had it; Fidelity
  used the detector's own 30, 20 and 10 seconds for Conservative, Balanced and Aggressive. On a
  116-minute test film, Fidelity at Aggressive cut 252 scenes, 187 of them shorter than 30 seconds,
  and a short scene is expensive: the recogniser pads it out to a whole 30-second window, so a scene
  under 30 seconds cost three times as much time per second of audio as a scene over two minutes.
  Forty-seven per cent of the audio took sixty-four per cent of the pass. What you will see on
  Fidelity at Aggressive: fewer, longer scenes, and the pass should finish sooner. Nothing is lost
  at a scene boundary — the semantic detector merges a short piece into its neighbour rather than
  discarding it.

  Fast, and the auditok and silero scene detectors on Balanced and Fidelity, are unchanged: on
  auditok a minimum duration discards the shorter region instead of merging it, so the same floor
  there would lose speech. If the detector ever leaves a scene longer than the ceiling
  (possible only when one unbroken stretch of identical texture exceeds it, never seen so far), a
  warning now appears in the log — previously that warning went only to the console.

- **Subtitle cues are shorter by default on Balanced.** The speech detector's ceiling on a single
  unbroken speech chunk drops from 20/15/9 seconds to 7/6/6 seconds across Conservative, Balanced and
  Aggressive, and Aggressive no longer keeps speech shorter than 80 milliseconds (it was 30). Longer
  chunks are split at an internal silence, so this affects how lines are divided rather than how much
  speech is found; it is a readability change, not a speed change. Two caveats. On Fidelity the same
  ceilings are set but the Silero v3.1 and v4.0 library does not enforce them, so only the 80
  millisecond floor takes effect there. And if you explicitly select the Silero v6.2 segmenter, which
  does enforce the ceiling, Aggressive moves the other way — from 4 seconds to 6 — so cues get longer,
  not shorter.


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
- **Balanced now runs faster-whisper's built-in VAD and nothing else, and you choose which Silero
  build it uses.** Two things drove this. Running an external speech segmenter on Balanced was slow
  — it splits every scene into many small groups and calls the recognizer once per group, and each
  of those calls is padded to a 30-second window — which is why Balanced could fall to around
  real-time on a feature-length file. And the model faster-whisper bundles is conservative on this
  material: on one 293-second test clip it treated 21.8 % of the audio as speech, where Silero 3.1
  treated 31.2 % and Silero 4.0 treated 44.8 %. Speech the VAD does not find is speech the
  recognizer never sees, which is the mechanism behind output that skips long stretches.

  So Balanced keeps the built-in VAD (one recognizer call per scene, the fast path), and
  `--vad-version` selects the Silero build that VAD runs: **3.1** (the default), **4.0** or **6.2**.
  In the Ensemble tab, a pass whose Pipeline is Balanced shows those three in the Speech Segmenter
  column; the Transcription tab uses the default. For an ensemble pass on the command line the flags
  are `--pass1-vad-version` and `--pass2-vad-version`. All three models ship inside WhisperJAV
  (about 4.7 MB), so nothing is downloaded and nothing needs the network. The detection thresholds
  are 0.5 conservative / 0.4 balanced / 0.3 aggressive, the same for every build.

  **Which build suits your material is worth trying.** The numbers above are a single clip and are
  not a recommendation; 3.1 is the default because it is the most conservative of the three about
  what it calls speech.

  **This breaks scripts that set a speech segmenter for Balanced, deliberately.**
  `--speech-segmenter` is no longer accepted with `--mode balanced`, and
  `--pass1/2-speech-segmenter` is no longer accepted for a Balanced pass. So are
  `--max-group-duration` and `--chunk-threshold` on Balanced, which set how an external speech
  segmenter groups what it found. All of them now stop the run with a message rather than quietly
  giving you a different pipeline or doing nothing. Fidelity, Qwen, Anime-Whisper and any
  non-Balanced ensemble pass keep their speech segmenters unchanged.

  One consequence: the check that reports a file as `suspect` — speech kept being detected but no
  subtitles came back — needs a separate speech detector to compare against, and Faster-Whisper's own
  voice detection does not provide one. On Balanced a file with no subtitles is now always reported
  `empty`, and choosing an external segmenter is no longer a way around that.

- **An ensemble that runs Fidelity first and Balanced second still lowers a typed "aggressive" to
  "balanced" for the second pass.** That exact combination produced empty or badly truncated
  second-pass subtitles in about two thirds of the runs that led to this rule; running the second
  pass at balanced instead stopped it. The rule is unchanged in v1.9.2 and it announces itself in
  the log when it fires, so you can see that the sensitivity you chose was not the one used. Half of
  the original reason has gone — the higher-temperature retry it also blamed no longer exists, since
  every sensitivity now decodes once at temperature 0 — but the failure it prevents was measured and
  nothing has yet measured that removing the rule is safe. If you want a pass at aggressive, run it
  on its own rather than as the second pass after Fidelity. No other combination of passes is
  touched.

  If the version you picked cannot be loaded, WhisperJAV falls back to the Silero model
  Faster-Whisper ships with and says so in the log. If that one cannot be loaded either, the run
  stops rather than transcribing with no voice detection at all.
- **`--no-vad` is removed.** It only ever set the speech segmenter to "none", which
  `--speech-segmenter none` already does on the modes that still take a segmenter. Balanced no
  longer takes one at all.
- **Faster-Whisper and CTranslate2 are fixed to exact versions.** CTranslate2 is `4.8.1` — 4.6.2 is
  the version that reproduces the crash on exit reported in #125.

  Faster-Whisper is taken from a specific commit on its development branch rather than from the 1.2.1
  release, because the three commits made after that release are the ones WhisperJAV needs: the
  bundled Silero voice-detection weights were updated to version 6.2, new voice-detection settings
  were added, and a deprecated download option was removed. The 1.2.1 release still carries the older
  weights. Taking a fixed commit rather than "the latest" means every install gets the same code,
  and it is the same build WhisperJAV is developed and tested against.
- **FireRedVAD is installed with WhisperJAV.** The `fireredvad` package is now part of the
  standard install (every extra that includes `cli`, the Windows installer, Colab and Kaggle).
  It is no longer marked experimental in the CLI, the GUI or the docs. It is available on Fidelity
  and on the ChronosJAV pipelines; Balanced no longer takes an external segmenter (above). Its detection presets are
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

## CPU-only users

Two situations look alike from the outside and behave differently.

**No GPU at all.** At start-up WhisperJAV stops and asks, in a boxed prompt, whether to
continue on the CPU. In the GUI tick *Accept CPU-only mode* on the Advanced tab (the GUI
cannot ask, so without the box the run aborts and tells you to tick it); on the command
line pass `--accept-cpu-mode` or `--device cpu`. Everything then runs on the CPU, because
no component can find a card: the Whisper-family recognisers, the ChronosJAV models
(Qwen3-ASR, anime-whisper, Cohere) and the transformers pipeline ask PyTorch; WhisperSeg
asks ONNX Runtime; NeMo and ClearVoice leave the choice to their own libraries; FireRedVAD,
TEN-VAD, Silero and the ffmpeg-dsp enhancer are CPU components by design. The
Whisper-family pipelines (Balanced, Faster, Fast, Fidelity) are the practical choice; the
README's rough figure of 30-60 minutes per hour of video is for them and was not
re-measured for this release. The ChronosJAV models, the transformers pipeline, NeMo,
WhisperVAD, WhisperSeg and the neural speech enhancers (ZipEnhancer, ClearVoice) were not
timed on the CPU at all; expect them to be far slower and to need more memory.

**A GPU that this PyTorch build cannot use**, such as a GTX 10-series card with the
current installer (#411). WhisperJAV now recognises the card as unusable and asks the same
question. Answering yes puts the Whisper-family pipelines on the CPU: their recognisers take
their device from WhisperJAV's own detector, and their default speech segmenters
(faster-whisper's built-in VAD, Silero) never leave the CPU. The components below do not
consult the detector. They ask PyTorch, or their own library, whether a CUDA card exists,
get yes because the card is present, and try to use it:

- **Qwen3-ASR, anime-whisper and Cohere** fail there with a CUDA error unless their own
  device setting says CPU. `--device cpu` does not reach them; it only answers the
  start-up question. On the command line use `--qwen-device cpu` for a single pass, or
  `--pass1-qwen-params '{"device": "cpu"}'` (and `--pass2-qwen-params`) in an ensemble.
  In the GUI, open the pass's *Customize Parameters*, then *Model*, then *Hardware*, and set
  *Device* to CPU.
- **The transformers pipeline** likewise: `--hf-device cpu`, or `--pass1-hf-params
  '{"device": "cpu"}'` in an ensemble. In the GUI this is reachable only through the
  Ensemble tab's Customize panel; the Transcribe tab always sends auto.
- **The NeMo speech segmenter, ZipEnhancer and ClearVoice** try the card and fail, and no
  setting overrides that in this release. The GUI shows a Device control for the speech
  enhancers, but it has no effect; that is recorded as a defect for a later release.
- **WhisperSeg**, the default segmenter of the ChronosJAV pipelines, asks ONNX Runtime.
  The standard install ships ONNX Runtime's CPU build, so WhisperSeg stays on the CPU; if
  you installed the GPU build yourself, it will try the card.
- **WhisperVAD** runs through CTranslate2, which has its own GPU kernels and supports these
  cards, so it will use the card and is expected to work there.

So on such a card either use Balanced, Faster, Fast or Fidelity with the default speech
segmenter and speech enhancement off, or use a ChronosJAV pipeline with its Device set to
CPU as above and no neural enhancer. The lasting fix is a PyTorch build with kernels for
the card (https://pytorch.org/get-started/locally/). Making every component ask
WhisperJAV's detector is not part of this release; this section is the documented
behaviour instead.

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
- **A Balanced file that produced no subtitles is flagged `suspect` only by its span.**
  On Balanced the only check is whether the subtitles cover less than `--min-coverage`
  of the video. The second check, "speech was detected but nothing came back", needs a
  separate speech segmenter, and Balanced no longer has one. In ensemble runs a failed
  pass 2 also makes the file `suspect`.
- **The Transcription tab always uses Silero 3.1; only the Ensemble tab lets you choose.**
  `--vad-version 3.1|4.0|6.2` works on the command line, and in the Ensemble tab a pass whose
  pipeline is Balanced offers the three builds in its Speech Segmenter column. The Transcription
  tab has no such control, so a single-pass Balanced run started from it gets the default, 3.1. If
  you want 4.0 or 6.2 for a single-pass run, use the command line.
- **Kaggle support (#329, #330) has moved to a release after this one.** It was expected in
  September; it is not in v1.9.2. Colab is unaffected.
- **The root cause behind #394 is still open.** The recognizer can enter a state
  where it returns nothing for the rest of a run, and the work above detects the
  *result* rather than preventing it. Investigation continues, with useful
  evidence contributed by @AlanZ-Git and @daoran9.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-09-09 | Balanced runs the Internal FW Silero VAD and nothing else: `--vad-version 3.1|4.0|6.2` (default 3.1) picks the Silero build, all three models ship inside WhisperJAV, and `--speech-segmenter`, `--pass1/2-speech-segmenter`, `--max-group-duration` and `--chunk-threshold` now stop the run on Balanced instead of being accepted. `--no-vad` removed. Detection thresholds 0.5 / 0.4 / 0.3. Faster-Whisper fixed to SYSTRAN master @ ed9a06c (three commits past 1.2.1, for the Silero v6.2 weights) and `ctranslate2==4.8.1` |
| 2026-09-11 | Semantic scene bounds are 28 s to 240 s on Balanced and Fidelity, at every sensitivity. The ceiling was 20 min on Balanced and 7 min (3 at aggressive) on Fidelity; Fidelity's floor was 30/20/10 s. Fast, auditok and silero untouched. The engine's overlong-scene warning now reaches the log |
| 2026-09-09 | Semantic is the default scene detector, and each scene detector finally receives its own parameter names — every semantic run since v1.8.11 had silently used the engine's built-in 20 s/420 s. Balanced resolved 28 s minimum / 20 min maximum at the time (superseded by the 240 s ceiling, 2026-09-11) |
| 2026-09-06 | A GPU the PyTorch build has no kernels for stops the run at start-up and asks whether to continue on the CPU or abort (GUI: abort, tick "Accept CPU-only mode" to proceed); `--check` exits 1 on such a card (#411, #326, #333) |
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
