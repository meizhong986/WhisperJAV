# WhisperJAV 1.9.2

This release is mostly about **failures that used to stay quiet**.

If you have ever started a run, come back half an hour later and found an empty subtitle
file with a message saying everything went fine — that is what most of the work went into.
WhisperJAV could finish, report success and hand you nothing; it could quietly ignore a
setting you had chosen; and on Windows it could finish installing something that could not
actually transcribe. Those are the things that changed.

One thing has genuinely got **faster**, there is one **new thing on screen**, and a
long-standing cause of empty subtitle files has been **found and fixed**. Otherwise the
subtitles you get are much the same as before. That is deliberate.

---

## Contents

- [At a glance](#at-a-glance) · [If you are new here](#if-you-are-new-here)
- [Installing and upgrading](#installing-and-upgrading)
- [New](#new) · [Fixed](#fixed) · [Changes you will notice](#changes-you-will-notice)
- [If you use the command line or scripts](#if-you-use-the-command-line-or-scripts)
- [Known limitations](#known-limitations) · [Which reports this release touches](#which-reports-this-release-touches) · [Thanks](#thanks)

---

## At a glance

- **Empty subtitle files are fixed** at their root cause. WhisperJAV was telling the
  recogniser that the first subtitle in each chunk had to begin at exactly zero seconds. On
  our test clips that produced **zero subtitles, every time**.
- **Aggressive is much faster on Balanced** — one three-hour film went from a run that had
  to be abandoned to one that finished. The same settings were applied to Fidelity, but we
  have not timed it there.
- **RTX 50 / Blackwell cards** no longer get a precision setting that was garbling their
  output. (Every Blackwell part, including the Ti and mobile variants — the check is on the
  chip generation, not the model name.)
- **A failed Windows install now says so** instead of finishing and leaving you a shortcut
  to something broken. It was also not installing one essential package at all.
- **New:** after working out the scenes, WhisperJAV tells you if your audio looks difficult,
  and roughly where.
- **Every run now ends with a summary table** saying, per file, whether it worked — and the
  exit status finally means the same thing in every mode.
- **A graphics card this build cannot use stops the run at the start** and asks you, instead
  of running for twenty minutes and producing nothing.

---

## If you are new here

WhisperJAV turns the speech in a video into a Japanese subtitle file. You point it at a
file, it thinks for a while, and it writes an `.srt` next to it.

**Which mode should I use?** Start with **Balanced**. It is the default and the one that
gets the most attention. The sensitivity setting (Conservative, Balanced, Aggressive)
controls how eagerly it picks up quiet or unclear speech — Aggressive finds more and invents
more. If you are unsure, leave both alone.

**A realistic expectation.** This is adult video audio: a lot of it is not speech, and long
stretches with no subtitles are normal, not a fault. For a sense of scale: a three-hour film
on an RTX 3060 took 41 minutes for a two-pass run. A single pass is quicker; your card and
your audio both matter.

---

## Installing and upgrading

### Windows, the easy way

Download **`WhisperJAV-1.9.2-Windows-x86_64.exe`** from the release page and run it.

- It does **not** need administrator rights and installs into your own user folder.
- Budget **10 to 20 minutes**. It downloads a lot, and how long depends entirely on your
  internet connection.
- The first time you actually transcribe something it downloads the AI model as well, about
  **3 GB**, which adds another 5 to 10 minutes. After that models are cached and start-up is
  quick.
- When it finishes you get a desktop shortcut.

**If something goes wrong during installation you will now be told, and no shortcut is
created.** That is new in this release, and it exists because people were getting a shortcut
to an installation that could not transcribe. You will get a dialog naming what failed, and
a file called `INSTALLATION_FAILED_v1.9.2.txt` in the installation folder with the details.
**Running the installer again usually fixes it**, because the usual cause is a download that
did not finish. If it happens twice, that file and `install_log_v1.9.2.txt` are what to
attach to a bug report.

### Upgrading from an earlier version

- **Windows installer:** download and run the new `.exe`. It installs over your existing copy.
- **Installed from source:** run `whisperjav-upgrade`.
- **Google Colab:** nothing to do; the notebook picks this up once the release is published.

> **⚠ Important if you upgrade by hand with pip.** This release deliberately changes two
> underlying libraries (faster-whisper and CTranslate2). If you normally use
> `pip install --no-deps`, **do not this time** — you would keep the old ones, and the RTX 50
> fix and the Silero 6.2 option would not reach you.

### After installing, if you ever need to file a bug report

There is a new script that prints what your installation actually contains — package
versions, your graphics card and driver, CUDA, and which precision settings your card
supports. Most bug reports arrive without any of this, and it is usually what decides the
answer.

It is **not installed with WhisperJAV** — download it from the repository, under `tools/`:
`whisperjav_env_report.py`, plus `whisperjav_env_report.ps1` which Windows users can
right-click and run. Run the `.py` with the same Python that runs WhisperJAV.

---

## New

### It tells you when your audio is difficult

After WhisperJAV has worked out the scenes in your file — before it starts the long part —
it now prints something like this:

```
------------------------------------------------------------------
  Audio analytics - EXAMPLE-123.mp4
------------------------------------------------------------------
  10 scenes, 25:00 total. Speech detected in 6% of the running time.

  This file appears to be unusually quiet for speech detection.
  Subtitles may be missing or sparse throughout.

  2 scenes appear to be acoustically difficult -
  speech close in level to everything else around it:

     scene 2       3:50 - 7:50     scene 6      18:24 - 21:34

  If the subtitles do look sparse, this may help:
     --vad-threshold 0.15   pick up quieter speech

  Raising the level first also helps, but it is only available on
  two-pass runs (--pass1-speech-enhancer ffmpeg-dsp, or the Speech
  Enhancer column in the GUI's Ensemble tab) and on --mode qwen
  (--qwen-enhancer ffmpeg-dsp).
------------------------------------------------------------------
```

**If your audio looks normal, none of this appears** — you only get the first line.

(Shown without the timestamp each line actually carries on screen. These lines go through the
log, so on your console each is prefixed with something like
`2026-09-12 15:36:42 - whisperjav - INFO - `.)

Why it exists: a 25-minute film produced 25 subtitles and the run reported success. Two
thirds of the dialogue had never been transcribed because the recording was very quiet, and
nothing in the run said so. You would only have found out by watching the film against the
subtitles.

Two honest caveats. **It is a hint, not a verdict** — the wording says "appears to be"
because that is what it means. And the "difficult" measure was calibrated on seven clips
scored as **whole clips** against human Japanese subtitles, where it picked out exactly the
three worst and nothing else; applying it to individual scenes stretches it to a shorter span
than it was measured on. Expect it to be useful, not exact.

It runs on **Balanced**, **Fidelity** and **`--mode qwen`** (with either generator, Qwen3 or
Anime-Whisper), when scene detection is set to `semantic` — which is now the default.

One thing the notice will tell you, which is worth knowing in advance: **raising the audio
level before transcribing is only available on two-pass runs and on `--mode qwen`.** A
single-pass Balanced or Fidelity run has no speech-enhancement control, on the command line
or in the GUI. Lowering `--vad-threshold` works everywhere.

### Every run ends with a summary you can trust

Previously each mode decided "success" in its own way, and one of them simply always said
yes. Now every run ends with a table like this:

```
============================================================
RUN SUMMARY  (exit status 0; a run fails on: failed)
============================================================
STATE    MILEAGE        FILE
done     ok 100%        movie1.mp4 -> movie1.ja.srt (1961 cue(s))
suspect  low 6%         movie2.mp4 -> movie2.ja.srt (82 cue(s)); subtitles stop at 400s of 7000s (5.7% of the file); 82 cue(s) produced
empty    not assessed   movie3.mp4 -> movie3.ja.srt; movie3.ja.srt contains no subtitles
------------------------------------------------------------
done 1  empty 1  suspect 1  failed 0  skipped 0  total 3
Manifest: whisperjav_run.json
============================================================
```

Five words describe every file, and they mean the same thing in every mode:

| Word | What it means |
|---|---|
| `done` | a subtitle file with at least one line was written |
| `empty` | the run finished and produced no lines, and nothing contradicts that |
| `suspect` | something does not add up — the subtitles stop early, or in a two-pass run the second pass failed |
| `failed` | an error: a crash, or a subtitle file that was reported written and is not there |
| `skipped` | nothing was attempted because the subtitle file already existed |

`suspect` is decided by `--min-coverage`, which is the share of the video your subtitles have
to reach before the run stops questioning them (a quarter, by default).

**Zero subtitles is an observation, not a failure.** Silence, music, and speech the
recogniser could not use all end there, and the run says so instead of guessing. If you
would rather those counted as failures — for a script, say — there are two checkboxes on the GUI's
**Transcription Adv. Options** tab (*Treat 'empty' files as failures*, *Treat 'suspect'
files as failures*), both off by default.

**MILEAGE** is how far into the file your subtitles reach: where the last subtitle ends, as
a share of the video's length. It is **not** how much of the video carries subtitles. A
two-hour film whose subtitles stop at seven minutes reads 6%. (It was called COVERAGE in
earlier builds of this release; the number has not changed.)

A `whisperjav_run.json` file is written next to your subtitles with the same information, so
a script can read the result instead of reading the console.

### Smaller new things

- **Offline mode.** A checkbox on the **Transcription Adv. Options** tab, *Offline mode
  (downloaded Hugging Face models only)*, and a matching `--offline` switch. A model that was
  never downloaded then fails immediately instead of retrying for minutes. Off by default.
- **Skip files that already have subtitles.** A *Skip already-subtitled files* checkbox,
  which now also works for two-pass runs.
- **Remember my settings.** A *Remember settings* checkbox keeps the Transcription tab's
  fields between launches. **Off by default**, so nothing changes unless you ask for it.
- **You can pick which Silero voice detector Balanced uses** — 3.1, 4.0 or 6.2 — in the
  **Ensemble Mode** tab's *Speech Segmenter* column for a Balanced pass. All three ship inside
  WhisperJAV, so nothing is downloaded.
- **A scene-change sensitivity slider** ("Scene Change Threshold") in the Customize
  Parameters dialog. Treat it as experimental — our own measurements show the scene count
  does not follow it closely.
- **The terminal says what it is doing to each scene**: its length, how many subtitles it
  produced, how long it took, and `NO OUTPUT` when a scene produced nothing.
- **A scene inspector**, `tools/scene_inspector.py`, which shows you how a file would be cut
  into scenes — with screenshots, a contact sheet and chapter marks — without transcribing
  anything.

---

## Fixed

### Runs that produced nothing, or produced nonsense

**Empty subtitle files — the root cause, found and fixed.** WhisperJAV was telling the
recogniser that the first subtitle in each chunk had to start at exactly zero seconds. That
left the decoder one legal choice, and on some hardware it took a path that produced a long
run of `!` characters, which the recogniser then scored as non-speech and skipped — leaving
you with no subtitles and a run that reported success. Measured on seven test clips: **zero subtitles on all seven,
twice over**, with every window discarded. The setting is now Whisper's own default, letting
the first subtitle start up to a second in; the same clips then produced about 32 subtitles
each with nothing discarded.

- **This affects Balanced.** Measured with `--dump-params` on all four Whisper modes:
  Balanced now resolves `max_initial_timestamp = 1.0`; **Fast and Faster never carried the
  setting at all** and are unchanged by this fix; Fidelity still carries the old `0.0`.
- **On material like the test clips the output is otherwise unchanged** — six of the seven
  came out byte-identical. **On JAV audio your subtitles will differ**, because the decoder
  now has choices it did not have. We have no human-checked Japanese subtitles for JAV
  material, so we cannot tell you they are *better*, only that the total-failure case is
  gone. If you keep old output for comparison, expect differences.
- **Fidelity still carries the old setting.** It uses a different recogniser (OpenAI
  Whisper rather than Faster-Whisper) and whether the same fault happens there has not been
  tested.
- Likely the same fault as several reports of empty output: **#414**, **#287**, and possibly
  **#411** and **#326**.

**RTX 50 cards were being given the wrong precision setting.** These cards were forced to
`float16` to avoid a crash in an underlying library. That crash was fixed upstream, and the
forced setting had itself become the problem — a user measured it producing garbled output
and almost no subtitles on an RTX 5070, where the automatic setting worked cleanly. RTX 50
cards now let the library choose.

**Every other NVIDIA card is deliberately unchanged.** We measured the alternative against
human-checked subtitles: it was no more accurate (the difference was statistically
indistinguishable from zero) while running **14–30% slower** on those clips and **32% slower**
on JAV audio. That is not a trade worth
making for you. If you are short of video memory you can still ask for it with
`--compute-type int8_float16`. (**#414**)

**A card this build cannot use now stops the run at the start.** One user with a GTX 1060
watched a twenty-minute run produce nothing and report success. WhisperJAV now stops before
doing anything, names your card and what the build supports, and asks — in a box you cannot
miss — whether to continue on the CPU or stop. **It waits for your answer**; the old
thirty-second countdown that continued by itself is gone. In the GUI, which cannot ask, the
run stops and tells you to tick *Accept CPU-only mode*. (**#411**, **#333**, probably **#326**)

**Subtitles that were nothing but punctuation.** On `--mode qwen` (either generator), a
stretch of sound with no words could produce a subtitle containing only `。` — in one user's
file, a quarter of the second pass. Those lines are now dropped whole. Punctuation *inside*
real text is untouched, and anime-whisper's ellipses are kept because they are how that model
writes. (**#413**)

### Installs that claimed to work when they had not

**The Windows installer was not installing Faster-Whisper at all.** It is pinned to an exact
build, and the step that writes the package list drops anything pinned that way. Balanced,
Fast and Faster all need it, so a fresh install from the `.exe` would have failed on
the first transcription — and the installer checked for exactly this at the end, printed
`✗ Faster-Whisper: FAILED`, and then finished with "Installation completed successfully!"
and a desktop shortcut.

Both halves are fixed: it is installed now, and **a check that fails stops the installation**
(see [Installing and upgrading](#installing-and-upgrading)).

**What will stop an install:** WhisperJAV itself not importing; `whisperjav.exe` or
`whisperjav-gui.exe` not being created; or PyTorch, OpenAI Whisper, Stable-TS, Faster-Whisper,
PyWebView, SRT or PyYAML failing to load. **What will only warn:** llama-cpp-python,
ClearVoice and Transformers — those are optional features you can add later. A check that
simply runs out of time is also only a warning: on a slow disk the first load of PyTorch can
take minutes, and that is not evidence of anything wrong.

### Runs that crashed

**Files on cloud-mounted drives.** A video on a CloudDrive2-mapped drive failed immediately
with `OSError: [WinError 1005]`, before any processing, even though the same file played
normally. WhisperJAV was asking Windows a question about the path that has no answer on such
a volume; it now carries on instead of stopping. (**#340**)

**Translation on small local models.** With an Ollama model that has a 2K or 4K context —
`qwen2.5:3b` and similar — every translation aborted with `min_batch_size must be less than
max_batch_size`, naming a setting most people had never touched. Both limits are now worked
out together. (**#341**)

### Long waits

**Ten-minute delays when Hugging Face was unreachable.** Every model load checked the site
for a newer version first, and with the site blocked each file was retried five times before
the cached copy was used — in one user's log, about ten minutes per run for one component.
Models now load from your cache first and only go online when a file is genuinely missing.
(**#415**)

### Settings that were ignored without telling you

- **A mistyped speech-enhancer name** (`zipenhance` for `zipenhancer`) was accepted, silently
  downgraded to no enhancement, and the run continued — so you could spend hours transcribing
  with enhancement you believed was on. It is now rejected immediately, listing the valid
  names. (**#306**)
- **The Ollama model list no longer recommends a model known to break subtitles.** When the
  curated list could not be read, the fallback offered a *thinking* model first — the class
  that writes its reasoning into your subtitle file.
- **DeepSeek v4-flash no longer reasons before translating.** DeepSeek changed its
  server-side default; this is slow, burns your rate limit, and can leak reasoning text into
  the subtitles. (`deepseek-v4-pro` *is* the reasoning model and is left alone. The OpenRouter
  route to the same model is not covered.) Diagnosed by **@mcdman**. (**#395**)
- **OpenRouter no longer defaults to a model DeepSeek retired** in July. (**#325**)
- **Saving presets no longer fails on relocated user profiles** — common when your profile
  has been moved to another drive. (**#309**)
- **The GUI's DeepSeek model lists are current**, instead of offering two retired models.
  (**#325**, **#382**)

With thanks to **@Mimic-me**, who contributed several of these as a reviewable batch.

### Messages that misled

- **The speech-segmenter warning no longer says the release is broken.** Choosing WhisperSeg
  or TEN outside two-pass mode printed a warning describing a *"known v1.9.0 routing bug
  (catastrophic empty output on JAV moaning content)"*. It was describing a deliberate guard,
  not a defect in your build. (**#323**)
- **The `[SANITIZATION SUMMARY]` block inside the `.artifacts.srt` file told the truth
  again.** It was reporting `Hallucinations modified/removed: 0` and `Final subtitles: 0` on
  ordinary runs while the same file went on to list the removals. That file is what people
  attach to bug reports, so it was misinforming other diagnoses.

### Documentation that broke installs

- **The Windows manual-install guide no longer replaces your GPU PyTorch with the CPU
  build** — one step silently disabled GPU acceleration for the whole installation. (**#334**)
- **The Linux GUI instructions now work on current distributions**, and include the step that
  was missing entirely. (**#366**)

### Already fixed for everyone, without an update

**The `pornify` translation instructions were eight months out of date.** They are fetched
from a Gist at run time, and that copy had never received the January fix. It told the model
to imitate an "American adult movie", gave worked examples translated into English, and
insisted *every* line be sexualised — which explains both the reports of English output when
another language was selected and of greetings being rewritten. The Gist was corrected on
2026-08-29 and **reached everyone immediately**. Diagnosed by **@yhxkry**, confirmed by
**@SangenBR**. (**#397**, **#339**)

**This is not the whole of the English-output problem.** With a local model through Ollama,
one reporter still measured 167 of 994 lines coming back in English. The cause is different
and is **not fixed in this release**: the running summary is carried into the next batch, so
once one batch answers in English it pulls the following ones with it. That is why the
English comes in blocks. A custom instruction file that names your target language
explicitly is the workaround for now. (**#347**, and probably **#305**)

---

## Changes you will notice

### Faster, and the same answer twice

**Aggressive no longer runs away with your time.** On a three-hour film an Aggressive run
used to take more than twice as long as Balanced, and one had to be abandoned before it
finished.

Three things changed together: Aggressive no longer decodes a passage a second time when a
quality check trips (on continuous, repetitive audio that check trips constantly), a rescue
behaviour was removed, and scenes are now capped at 4 minutes. **The measurement below covers
all three at once — we have not measured how much each contributed.**

The measurement, on a three-hour film (179 minutes) that an earlier Aggressive run had to be
abandoned on: a two-pass run — Balanced at Conservative, then Balanced at Aggressive —
finished in **41 minutes** and produced **1,961 subtitles**, the last of them landing 99.8% of
the way through the film. The Aggressive pass alone decoded at **8.8× real time**, against
**2.4×** in the abandoned run, and **none of its 89 scenes came back empty**, where 20 of 26
had before. One film, one graphics card (an RTX 3060).

**All the Whisper-based modes now decode once rather than retrying.** Run times are
predictable, and the same file gives the same subtitles every time. What you give up:
slightly fewer alternative readings of a difficult passage.

### Scenes and subtitle timing

- **Scene detection now uses the `semantic` detector by default, where 1.9.1 used
  `auditok`.** This is the change most likely to make your subtitles differ from 1.9.1 output:
  a different algorithm decides where the file is cut, so the pieces, their number and the run
  time all change. It has not been proven better on a feature-length film — it is a considered
  default, not a measured win. `--scene-detection-method auditok` puts the old one back.
- **On the semantic detector, a scene is between 28 seconds and 4 minutes on Balanced and
  Fidelity.** 1.9.1's auditok scenes were capped near 29 seconds, so expect **fewer and
  longer** pieces than you are used to, and fewer progress lines. (Inside this release's own
  development the ceiling was briefly 20 minutes; 4 minutes is where it landed.) The auditok
  and silero detectors are unchanged.
- **Subtitle lines are shorter on Balanced.** The longest single unbroken line drops from
  20/15/9 seconds to 7/6/6 across the three sensitivities. Long lines are split at a pause,
  so this changes how lines are divided rather than how much speech is found.
- **Scene detection now uses one sensitivity setting instead of three.** Our measurements
  showed the old three-value setting was not the control its name suggested.

### Speech detection

- **Balanced now uses Faster-Whisper's own voice detection and nothing else**, and you choose
  which Silero build it runs. The old way — an external detector on Balanced — was slow: it
  split every scene into many small pieces and called the recogniser once per piece. **The
  default build is Silero 4.0.** On one test clip it treats 44.8% of the audio as speech
  where 3.1 treats 31.2%, so the recogniser is shown more of your file. `--vad-version 3.1`
  puts the older, more cautious build back.
- **Fidelity now finds speech with FireRedVAD** instead of Silero 3.1. Its model is
  downloaded once — see [Known limitations](#known-limitations).
- **On Fidelity, a rescue behaviour was removed.** If the speech detector reported almost no
  speech in a scene, that scene used to be transcribed end to end anyway. It no longer is.
  This saves a lot of pointless work, and it does mean a scene whose detector genuinely fails
  now yields nothing instead of being salvaged.
- **A false alarm is gone.** `Speech segmentation produced insufficient coverage` on long
  scenes was a false alarm — nothing had failed. On one test film it accounted for *every*
  warning printed, which meant a real problem would have looked identical to it.

### Other behaviour

- **The recogniser is reloaded fresh every 20 minutes of audio.** This is containment for a
  fault where the recogniser stops returning anything part-way through a long run
  (**#394**). It costs one model load per refresh — about 80 seconds over a two-hour film.
  Change it with `--model-refresh-audio-minutes` (0 = never), or the matching field on the
  GUI's **Transcription Adv. Options** tab. It applies to **Balanced and Fidelity**; other
  pipelines ignore it.
- **Lone 「はい。」 and 「うん。」 lines are dropped on `--mode qwen`, with either generator.** In JAV
  material almost all of them are moans rather than agreement. Only a line that is exactly
  that one word goes. Turn it off with `--no-qwen-drop-nonverbal-lines`. (**#254**)
- **A two-pass run of Fidelity then Balanced still lowers a typed "aggressive" to "balanced"
  for the second pass.** That exact combination produced empty or truncated second-pass
  subtitles in about two thirds of the runs that led to the rule. It now says so in the log
  when it happens. If you want a pass at Aggressive, run it on its own.
- **`--speech-pad-ms` no longer does anything on `--mode fidelity`.** Fidelity's new speech
  detector has no such setting — it pads the start and end of each piece separately — and the
  run now tells you the setting is ignored instead of silently dropping it. On Balanced, Fast,
  Faster and with the Silero detectors it works as before.
- **A Japanese run now reports its post-processing steps.** Reading the file, saving the
  original, writing the result and the artifacts file, and the two cleanup filters each print
  a line. Previously the log went quiet after stitching, which looked like a hang. (**#372**)
- **Balanced runs its recogniser in a separate process by default.** This is what makes the
  20-minute refresh above possible on Windows. You will see a second WhisperJAV process in
  Task Manager; that is expected. Set `--model-refresh-audio-minutes 0` to go back to
  one process.
- **On `--mode qwen`, the scene-change threshold default moved from 18 to 22**, which gives
  slightly fewer and longer scenes.
- **`whisperjav --check` gained a line for FireRedVAD's model**, saying whether it is on your
  machine. It reports; it does not download.
- **Some scene-detector warnings moved from the console into the log file**, so a normal run
  prints less.
- **If you explicitly choose the `silero-v6.2` detector on Fidelity at Aggressive, subtitle
  lines get *longer*, not shorter** (4 to 6 seconds) — the opposite direction from the
  Balanced change above.
- **Every Balanced run keeps a small per-scene diagnostic file** next to your subtitles
  (`raw_subs/<name>.asr_telemetry.jsonl`), on by default, so a run that later turns out to be
  a **#394** case has its record. Turn it off with `--no-asr-telemetry` or the GUI switch.

---

## If you use the command line or scripts

Four things will stop a command that used to work. All four fail with a message instead of
quietly doing something else.

- **`--speech-segmenter` is no longer accepted with `--mode balanced`**, and neither are
  `--pass1/2-speech-segmenter` for a Balanced pass, nor `--max-group-duration` and
  `--chunk-threshold` on Balanced. Balanced has no external speech detector any more. Use
  `--vad-version` to choose the Silero build instead. Fidelity, Qwen, anime-whisper and any
  non-Balanced pass are unchanged.
- **`--no-vad` is removed.** It only ever set the speech detector to "none", which
  `--speech-segmenter none` already does on the modes that still take one.
- **On a machine with no usable GPU, a script now needs `--accept-cpu-mode`** (or
  `--device cpu`). Without it the run stops and waits, where it used to continue after a
  countdown.
- **`--mode fidelity` now needs FireRedVAD's model, and stops the run if it cannot get it.**
  Fidelity's speech detector changed, and the new one downloads a small model once. On a
  machine behind a proxy or with no network that has never fetched it, a Fidelity script that
  worked in 1.9.1 now exits 1 before it reads any audio. The previous detector needed no
  download. `--speech-segmenter silero-v3.1` restores the old behaviour; see
  [Known limitations](#known-limitations) for the other ways round it.

New switches you may want: `--offline`, `--fail-on empty`, `--fail-on suspect`,
`--vad-version`, `--model-refresh-audio-minutes`, `--scene-clustering-threshold`,
`--no-asr-telemetry`.

---

## Known limitations

- **Fidelity's first run needs the network, once.** FireRedVAD's model (about 2 MB) is
  downloaded from Hugging Face. The Windows installer now fetches it **during installation**,
  so if you install online you will never notice. If it could not be fetched, the run stops
  at the start and tells you — it does not transcribe and hand you an empty file. On a
  machine that will never have a connection, either use `--speech-segmenter silero-v3.1`
  (which needs no download), or download the model elsewhere with
  `huggingface-cli download FireRedTeam/FireRedVAD --local-dir <folder>` (in China,
  `modelscope download --model xukaituo/FireRedVAD --local_dir <folder>`) and set
  `WHISPERJAV_FIREREDVAD_MODEL_DIR` to that folder.
- **`--offline` covers Hugging Face downloads only.** Some components fetch from their own
  servers. Models must have been downloaded once while online.
- **The Transcription tab always uses Silero 4.0.** The Ensemble tab and the command line let
  you pick 3.1 or 6.2; the Transcription tab has no such control.
- **The empty-subtitle fix has not been applied to Fidelity**, which uses a different
  recogniser where the fault is untested. **Fast and Faster were never affected** — they
  do not carry the setting. If you are seeing empty subtitle files on Fast or Faster, this
  release does not address it and we would like to hear about it.
- **The "difficult audio" notice does not catch every case**, as described above.
- **The root cause behind #394 is still open.** The recogniser can enter a state where it
  returns nothing for the rest of a run. This release **detects the result** and bounds how
  long one recogniser is used; it does not prevent the state. Work continues, with useful
  evidence from **@AlanZ-Git** and **@daoran9**.
- **Kaggle support (#329, #330) has moved to a later release.** Colab is unaffected.
- **`--async-processing` with Balanced, more than one file, and model refresh switched off**
  (`--model-refresh-audio-minutes 0`) ends without a summary. With the default refresh
  setting the same run completes normally. Normal (non-async) runs are unaffected.
- **On a graphics card this build cannot use, answering "continue on the CPU" does not reach
  every component.** The anime-whisper, Qwen3 and Cohere models, the transformers mode, NeMo
  and the neural speech enhancers choose the card on their own and will fail there. On such a
  card use Balanced, Faster, Fast or Fidelity with speech enhancement off.

---

## Which reports this release touches

| Report | What changed |
|---|---|
| **#414** | RTX 50 cards no longer forced to `float16`; likely also the empty-output root cause |
| **#411**, **#333** | A card this build cannot use stops the run at the start and asks |
| **#413** | Punctuation-only subtitle lines dropped on anime-whisper and Qwen3 |
| **#415** | Cached models load without waiting on Hugging Face; new `--offline` mode |
| **#394**, **#263** | Per-file summary, manifest and one exit-status rule; recogniser refreshed every 20 min; per-scene diagnostics. **Root cause still open** |
| **#340** | Files on cloud-mounted drives no longer abort the run |
| **#341** | Translation no longer fails outright on small-context local models |
| **#306** | A mistyped speech-enhancer name is rejected instead of silently ignored. **The subtitle truncation in its title is not addressed** |
| **#325**, **#382** | Retired DeepSeek models replaced in the backend and both GUI lists |
| **#395** | DeepSeek v4-flash no longer reasons before translating |
| **#397** | `pornify` instructions corrected — already delivered, no update needed |
| **#339** | The same correction. **Its reporter has not retested**, so it is not claimed as resolved |
| **#323** | The warning no longer describes the release as broken. Two changes in opposite directions: Fidelity gains `silero-v6.2`, Balanced refuses separate speech detectors outright |
| **#334**, **#366** | Windows and Linux install guides corrected |
| **#328** | Skip already-subtitled files, in the GUI and in two-pass runs |
| **#96** | Optional *Remember settings* for the Transcription tab (the two-pass Customize values in **#381** still reset) |
| **#309** | Preset saving on relocated user profiles |
| **#311** | FireRedVAD installed as standard, no longer marked experimental |
| **#254** | Lone 「はい。」 and 「うん。」 dropped on `--mode qwen` |
| **#419** | The command as written no longer runs (Balanced takes no separate speech detector), and a run with no subtitles is reported `empty` instead of success. **Why it produced nothing is still open** |
| **#416** | Evidence fed the Balanced speech-detection rework. **Not confirmed resolved** |
| **#287**, **#326** | Possibly resolved by the empty-output fix — **not confirmed**, please retest and say |
| **#347**, **#305** | English blocks from local translation models — **not fixed**, cause identified |
| **#329**, **#330** | Kaggle — **moved to a later release** |

---

## Thanks

Almost everything in this release started as somebody's bug report. These are the people
whose reports, logs and measurements it came from.

### Who measured, diagnosed, or sent a fix

- **@yyyanlei** — tested carefully on an RTX 5070, ruled out the explanations that were
  wrong, and found the precision setting that was garbling that card's output. (**#414**)
- **@weifu8435** — sent the full console log behind the Hugging Face stall, and then the raw
  first-pass and second-pass subtitle files that proved where the punctuation-only lines were
  coming from: 209 of 815 lines in one pass. Both are fixed here. (**#415**, **#413**)
- **@skysstst** — retested the translation fault on 1.9.1, measured 167 of 994 lines still
  coming back in English, and found a custom instruction file that brought it to 0 of 997.
  That workaround is the best answer available until the underlying cause is fixed.
  (**#347**)
- **@mcdman** — diagnosed and prototyped the DeepSeek v4-flash fix. (**#395**)
- **@Mimic-me** — contributed a batch of GUI improvements as one reviewable pull request.
  (**#328**, **#96**, **#309**, **#325**, **#382**)
- **@daoran9** and **@AlanZ-Git** — the measurements on the recogniser that stops returning
  anything part-way through a long run. It is not solved, and their evidence is what the work
  continues from. (**#394**)
- **@yhxkry** and **@SangenBR** — confirmed that the translation instructions served to
  everyone were eight months out of date. (**#397**, **#339**)

### Who reported something this release fixes

- **@leonscottkennedy5625-lang** and **@clt002** — a graphics card the build cannot use now
  stops the run at the start instead of producing nothing for twenty minutes.
  (**#411**, **#333**)
- **@david2926121jay** — videos on cloud-mounted drives no longer abort the run. (**#340**)
- **@helsinkii** — translation no longer fails outright on small local models. (**#341**)
- **@ktrankc** — a mistyped speech-enhancer name is rejected instead of silently ignored.
  (**#306**)
- **@TinyRick1489** — the speech-segmenter warning no longer describes the release as broken.
  (**#323**)
- **@ConnoisseurProtege** — the Windows install guide no longer replaces your GPU PyTorch
  with the CPU build. (**#334**)
- **@thumper100** — the Linux GUI instructions now work on current distributions. (**#366**)
- **@Jerry199022** — saving presets no longer fails on relocated user profiles. (**#309**)
- **@unretired1516** — batch runs can skip files that already have subtitles. (**#328**)
- **@kylesskim-sys** — FireRedVAD is now installed as standard, and is Fidelity's speech
  detector. (**#311**)
- **@zoqapopita93** — lone 「はい。」 and 「うん。」 lines are dropped on `--mode qwen`.
  (**#254**)
- **@loggias06row** — DeepSeek v4-flash no longer reasons before translating. (**#395**)
- **@1Dreamer666** and **@Angelholl** — the retired DeepSeek model names are gone from the
  backend and from both GUI lists. (**#325**, **#382**)
- **@sky9639** — the Transcription tab can now remember your settings, if you tick the box.
  (**#96**, in part — the two-pass Customize values still reset.)

### Who shared results that changed what we did, even where nothing is fixed yet

- **@Timi2028** — the run that produced no subtitles and still said `[SUCCESS]`. His command
  no longer runs as written, because Balanced no longer accepts a separate speech detector,
  and a run with no subtitles is now reported as `empty` rather than as success. **Why it
  produced nothing is still open.** (**#419**)
- **@yangming2027** and **@wjingshan** — logs and repeated results on Balanced producing
  little or nothing. Their evidence fed the rework of Balanced's speech detection. Whether it
  resolves their files is the open question. (**#416**)
- **@zoqapopita93** — the subtitles that came out as nothing but `!` characters. The cause of
  that symptom was found and fixed in this release, but you retested a previous fix and told
  us it had not worked, so we are not claiming this one until you say so. (**#287**)

If you reported something that is fixed here and your name is missing, that is our mistake
and worth telling us about.
