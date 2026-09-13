# WhisperJAV 1.9.2

This release is about one thing: making the **Balanced** pipeline usable on a full-length
film. Balanced is the mode most people run — it needs the least video memory, it is the
fastest of the accurate modes, and it is the one that times subtitles word by word. It was
also the mode that most often could not finish.

Everything else in this release is listed underneath, in [Other fixes](#other-fixes).

---

## Contents

- [The Balanced problem, and what changed](#the-balanced-problem-and-what-changed)
- [What was measured](#what-was-measured) · [What is not settled](#what-is-not-settled)
- [Installing](#installing) · [Upgrading](#upgrading)
- [Your subtitles will differ from 1.9.1](#your-subtitles-will-differ-from-191)
- [Other fixes](#other-fixes) · [Command line and scripts](#command-line-and-scripts)
- [Known limitations](#known-limitations) · [Reports addressed](#reports-addressed)
- [Credits](#credits)

---

## The Balanced problem, and what changed

Balanced had two failures, and which one you hit depended on a single setting.

**Problem 1 — too slow to use.** With WhisperJAV finding the speech and handing the
a three-hour film took about three hours, sometimes longer. At the Aggressive setting it could
fail outright: on a three-hour film it never finished its first pass and had to be abandoned
after 73 minutes with no subtitle file at all.

**Problem 2 — most of the film missing.** When Faster-Whisper was left to find the speech
itself, the speed was normal, but the subtitles skipped over large parts of the movie.

### What changed

- **Balanced now always lets Faster-Whisper find the speech itself.** The step that made
  Problem 1 slow is gone from this mode: it split each scene into hundreds of short
  fragments, and every fragment cost as much to transcribe as a full 30-second one.
- **You choose which Silero build finds the speech, and the default changed to 4.0.**
  `--vad-version 3.1 | 4.0 | 6.2`, or the *Speech Segmenter* column of a Balanced pass in the
  GUI's **Ensemble Mode** tab. All three ship inside WhisperJAV; nothing is downloaded.
  The build decides how much of the film is treated as speech, which is Problem 2.
- **A false alarm is removed.** On any scene of eight minutes or more, Balanced decided that
  finding the speech had failed — it had not — printed
  `Speech segmentation produced insufficient coverage`, and transcribed the whole scene from
  end to end instead. It broke the per-scene diagnostics while doing it.
- **Scenes are now between 28 seconds and 4 minutes** on Balanced and Fidelity, where the
  detector used to produce both very short and very long ones.
- **The same file now produces the same subtitles every time**, and a difficult stretch of
  audio can no longer quadruple how long a run takes.
- **Aggressive was retuned** on Balanced and Fidelity: it still picks up quiet speech, but it
  stops sooner when the model starts repeating itself and spends less time on each passage.
- **The transcription model is restarted every 20 minutes of audio** on Balanced and Fidelity, so a
  model that stops producing output part way through a long film costs you 20 minutes of
  the film rather than the rest of it. This bounds a fault; it does not fix it (**#394**).
  Restarting the model costs a little time on each restart. `--model-refresh-audio-minutes 0`
  turns it off.

---

## What was measured

One film, one graphics card — an RTX 3060. **EKAI-023, 179 minutes**, run as two passes:
Balanced at Conservative, then Balanced at Aggressive.

| | Result |
|---|---|
| Exit status | 0 |
| Subtitles written | 1,961 |
| How far into the film the last subtitle lands | 99.8% |
| Total time, both passes | 40.9 minutes |
| Aggressive pass alone | 8.8× real time |
| Scenes that produced nothing, Aggressive pass | 0 of 89 |

The comparison that matters: **Aggressive used to be abandoned after 73 minutes with nothing
written; it now completes.** Both films are from the maintainer's own
collection; it is not established that they are the same film.

---

## What is not settled

Read this before deciding whether the release helps you.

- **There is no measured speed comparison against 1.9.1.** No before-and-after run of the same
  film on the released 1.9.1 exists. The figures above stand on their own.
- **Three changes were made together** — transcribing each passage once, removing the false
  alarm, and the 4-minute scene cap — and the measurement covers all three. How much each
  contributed is unknown.
- **The suspected effect of the CTranslate2 version on speed was never isolated.** It remains a
  suspicion.
- **One film, one card.** Nothing here has been measured on a second machine, a second film, or
  a card other than an RTX 3060.
- **The cause of #394 is still open.** The transcription model can enter a state where it returns nothing
  for the rest of a run. This release limits the damage and records evidence; it does not stop
  it happening.
- **The Silero build default of 4.0 rests on one 293-second clip**, where 4.0 treated 44.8% of
  the audio as speech and 3.1 treated 31.2%, producing 50 subtitles against 32. That is a clip,
  not a film. `--vad-version 3.1` gives you the more cautious behaviour.

---

## Installing

WhisperJAV produces a Japanese subtitle file from the speech in a video. Point it at a file; it
writes an `.srt` alongside.

### Windows

Download `WhisperJAV-1.9.2-Windows-x86_64.exe` from the release page and run it.

- No administrator rights required; it installs into your user folder.
- Allow 10–20 minutes, depending on your connection.
- The first transcription downloads the transcription model, about 3 GB, adding 5–10 minutes. It is cached
  afterwards.
- A desktop shortcut is created on success.

**An installation that fails now stops and says so**, and creates no desktop shortcut to an
installation that cannot transcribe. The reason is named in a dialog and written to
`INSTALLATION_FAILED_v1.9.2.txt` next to `install_log_v1.9.2.txt` in the installation folder.
Run the installer again first — the usual cause is an incomplete download. If it fails twice,
attach both files to a report.

*This failure path has been built but never triggered on a real failing installation. If you
meet it, please say so in a report, whether or not it behaved as described here.*

**What stops an installation:** WhisperJAV failing to import; `whisperjav.exe` or
`whisperjav-gui.exe` not being created; or PyTorch, OpenAI Whisper, Stable-TS, Faster-Whisper,
PyWebView, SRT or PyYAML failing to load. **What only warns:** llama-cpp-python, ClearVoice and
Transformers, which are optional and can be added later, and any check that times out — on a
slow disk the first PyTorch import can take minutes.

### Choosing a mode

Start with **Balanced**, the default. Sensitivity — Conservative, Balanced, Aggressive —
controls how readily quiet or unclear speech is transcribed; Aggressive captures more and also
invents more. Long stretches without subtitles are normal on this material.

---

## Upgrading

- **Windows installer:** download and run the new `.exe`; it installs over the existing copy.
- **Source install:** `whisperjav-upgrade`.
- **Google Colab:** no action required.

> **If you upgrade manually with pip, do not use `--no-deps` this time.** This release changes
> two underlying libraries on purpose — Faster-Whisper is pinned to an exact build, and
> CTranslate2 to 4.8.1. Keeping your existing versions means the RTX 50 change does not reach
> you, and the Balanced work above was measured against these two versions and no others.

### Reporting a problem

`tools/whisperjav_env_report.py` in the repository prints your package versions, graphics card,
driver, CUDA version and supported compute types. It is not installed with WhisperJAV —
download it and run it with the same Python that runs WhisperJAV. On Windows you can
right-click `whisperjav_env_report.ps1`.

---

## Your subtitles will differ from 1.9.1

Three changes alter the output on files that worked before.

- **Scene detection now defaults to `semantic`; 1.9.1 used `auditok`.** This is the single
  biggest reason a file you have transcribed before will come out differently — a different
  method decides where the film is cut, and every later stage follows those cuts. **It has not
  been proved better on a feature-length film.** It is a considered choice, not a measured
  result. `--scene-detection-method auditok` restores 1.9.1's behaviour.
- **Balanced's Silero build defaults to 4.0**, and Balanced no longer takes a separate
  *Speech Segmenter* at all. `--vad-version 3.1` is the cautious option.
- **Subtitle lines are shorter on Balanced.** The longest unbroken line drops from 20/15/9
  seconds to 7/6/6 across Conservative, Balanced and Aggressive. Lines are split at pauses, so
  this changes how speech is divided, not how much is found.

---

## Other fixes

Everything below is outside the Balanced work.

### Empty and incorrect output

- **One cause of empty subtitle files is fixed, on Balanced only.** The transcription model was told the
  first subtitle of every chunk had to start at exactly 0.000 seconds. Where that left it no
  valid choice it emitted a run of `!` characters, which was then discarded as non-speech — an
  empty subtitle file from a run that reported success. On seven test clips this produced zero
  subtitles every time before the change and about 32 each after. **Empty subtitle files can
  still happen for other reasons**, including #394, and are still worth reporting. Checked with
  `--dump-params` across all four Whisper modes: Fast and Faster never carried the setting and
  are unaffected; Fidelity keeps the old value. Probably behind **#414** and **#287**, and
  possibly **#411** and **#326** — none confirmed. On JAV audio the subtitles will change, and
  no human-checked Japanese subtitles exist for JAV material, so nothing is claimed about
  accuracy — only that this total-failure case is gone.
- **RTX 50 (Blackwell) cards no longer have a workaround forced on them.** They were pinned to
  `float16` because CTranslate2 could not run them any other way. CTranslate2 now can; this
  release pins version 4.8.1, and the workaround is removed, so the library chooses for itself.
  **This is the removal of a workaround, not a demonstration that these cards now produce
  correct subtitles** — nobody has scored their output. If you have one of these cards, please
  report what you get. Detection is by chip generation, not by model name. Every other NVIDIA
  card is deliberately unchanged. (**#414**)
- **A graphics card this build cannot use now stops the run at start-up.** One user with a GTX
  1060 ran for twenty minutes, got nothing, and was told the run succeeded. WhisperJAV now
  names the card and the architectures the build supports and asks whether to continue on CPU
  or stop. It waits for an answer; the old thirty-second auto-continue is gone. The GUI cannot
  ask, so it stops and points you at *Accept CPU-only mode*. (**#411**, **#333**)
- **Subtitle lines containing nothing but punctuation** are removed on `--mode qwen` with
  either generator. In one user's file a quarter of the second pass was lines consisting only
  of `。`. Punctuation inside real text is untouched. (**#413**)

### Every run now ends with a summary

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

| State | Meaning |
|---|---|
| `done` | a subtitle file with at least one line was written |
| `empty` | the run finished and produced no lines, with nothing to contradict that |
| `suspect` | the subtitles stop early, or in a two-pass run the second pass failed |
| `failed` | an error, or a subtitle file reported as written that is not there |
| `skipped` | not attempted; the subtitle file already existed |

The run exits **1 if any file failed or the run did not complete**, otherwise 0. No subtitles is
an observation, not a failure — silence and music both produce it — so `empty` and `suspect`
exit 0. `--fail-on empty` or `--fail-on suspect` makes them failures, as do the two checkboxes
on the GUI's **Transcription Adv. Options** tab, both off by default.

**MILEAGE** is how far into the file the subtitles reach — where the last one ends, as a share
of the film's length. It is not the share of the film that carries subtitles: a two-hour film
whose subtitles stop at seven minutes reads 6%. `--min-coverage` sets the point below which a
file is called `suspect`; the default is a quarter.

### Audio analytics

A new report, printed after scene detection and before transcription, on Balanced, Fidelity and
`--mode qwen`. It appears only when scene detection is `semantic`, which is now the default.

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

On ordinary audio only the first line appears. **It does not catch every difficult file.** The
"difficult" measure was calibrated on seven clips scored whole against human Japanese
subtitles, where it picked out the three worst and nothing else; applying it scene by scene
stretches it to a shorter span than it was measured on. Treat it as a hint.

### Crashes and stalls

- **Files on cloud-mounted drives.** A video on a CloudDrive2-mapped drive failed immediately
  with `OSError: [WinError 1005]`. WhisperJAV asked the drive something it does not support; it
  now carries on. (**#340**)
- **Translation with small local models.** An Ollama model with a 2K or 4K context aborted every
  translation with `min_batch_size must be less than max_batch_size`, naming a setting most
  users had never touched. Both limits are now worked out together. (**#341**)
- **Ten-minute stalls when Hugging Face was unreachable.** Every model load checked the site for
  a newer copy first, and with the site blocked each file was retried five times before the
  cached copy was used. Models now load from the local cache first. A new `--offline` switch, or
  the checkbox on **Transcription Adv. Options**, makes that explicit: a model that is not
  already downloaded fails at once instead of retrying for minutes. (**#415**)

### Settings that were quietly ignored

- **A mistyped speech-enhancer name** (`zipenhance` for `zipenhancer`) was accepted and silently
  turned into no enhancement. It is now rejected with the valid names listed. The subtitle
  truncation also reported in that thread is **not** addressed. (**#306**)
- **DeepSeek v4-flash no longer reasons before translating.** DeepSeek changed its server-side
  default; the reasoning is slow, eats your rate limit, and can end up in the subtitle file.
  `deepseek-v4-pro` is the reasoning model and is unchanged. The OpenRouter route is not
  covered. Diagnosed by **@mcdman**. (**#395**)
- **OpenRouter no longer defaults to a model DeepSeek retired** in July. (**#325**)
- **The Ollama fallback list no longer recommends a reasoning model**, which writes its
  reasoning into the subtitle file.
- **The GUI's DeepSeek model lists are current**, replacing two retired models.
  (**#325**, **#382**)
- **Presets save again on relocated user profiles**, common where the profile has been moved to
  another drive. (**#309**)

### Translation instructions — already delivered, no update needed

The `pornify` instructions were eight months out of date. They are fetched at run time, and that
copy never received the January correction: it told the model to imitate an "American adult
movie", gave worked examples in English, and required every line to be sexualised. That
accounts for reports of English output when another language was chosen, and of greetings being
rewritten. Corrected on 2026-08-29 and delivered to everyone immediately — this release is not
required for it. Confirmed by **@yhxkry** and **@SangenBR**. (**#397**, **#339**)

**English output is not fully solved.** With a local model through Ollama, one reporter measured
167 of 994 lines still coming back in English. That has a different cause and is **not fixed
here**: the running summary is carried into the next batch, so one English reply drags the
following batches with it, which is why the English arrives in blocks. A custom instruction file
naming the target language is the workaround for now. (**#347**, probably **#305**)

### Messages that were wrong

- **The speech-segmenter warning no longer calls the release broken.** Choosing WhisperSeg or
  TEN outside two-pass mode printed a warning about a *"known v1.9.0 routing bug (catastrophic
  empty output on JAV moaning content)"*. It was describing a deliberate guard. (**#323**)
- **The `[SANITIZATION SUMMARY]` block in `.artifacts.srt` counts correctly again.** It had been
  reporting `Hallucinations modified/removed: 0` and `Final subtitles: 0` on ordinary runs while
  the same file listed the removals. That file is commonly attached to bug reports.
- **A Japanese run now reports its post-processing steps.** The log used to go silent after
  stitching, which looked like a hang. (**#372**)

### Documentation

- **The Windows manual-install guide no longer replaces GPU PyTorch with the CPU build**, a step
  that silently disabled GPU acceleration for the whole installation. (**#334**)
- **The Linux GUI instructions work on current distributions** and include a missing step.
  (**#366**)

### Smaller additions

| Addition | Detail |
|---|---|
| Skip existing subtitles | *Skip already-subtitled files* now works in two-pass runs too. (**#328**) |
| Settings persistence | *Remember settings* keeps the Transcription tab's fields between launches. Off by default. (**#96**) |
| Scene-change threshold | "Scene Change Threshold" slider in Customize Parameters. Experimental — measurement shows the scene count does not track it closely. |
| Per-scene console output | Each scene reports its length, subtitle count and duration; scenes that produced nothing are marked. |
| Per-scene diagnostic file | Balanced writes `raw_subs/<name>.asr_telemetry.jsonl` next to the subtitles, on by default, so a run that later turns out to be a **#394** case has a record. `--no-asr-telemetry` turns it off. |
| Scene inspector | `tools/scene_inspector.py` shows how a file would be cut into scenes, with screenshots and chapter marks, without transcribing it. |
| FireRedVAD | Installed as standard and no longer marked experimental. `whisperjav --check` reports whether its model is present; it does not download it. (**#311**) |
| Lone 「はい。」 and 「うん。」 | Removed on `--mode qwen` with either generator — in JAV material these are almost always vocalisations, not agreement. Only lines consisting of exactly that word go. `--no-qwen-drop-nonverbal-lines` keeps them. (**#254**) |

### Other changed behaviour

- **Fidelity uses FireRedVAD** to find the speech, instead of Silero 3.1. Its model is
  downloaded once — see [Known limitations](#known-limitations).
- **A Fidelity rescue behaviour was removed.** A scene in which almost no speech was found used
  to be transcribed end to end anyway. It no longer is. That removes a great deal of wasted
  work, and it also means a scene where the search for speech genuinely fails now yields
  nothing at all.
- **With `silero-v6.2` on Fidelity at Aggressive, subtitle lines get longer** (4 seconds to 6) —
  the opposite direction from the Balanced change above.
- **`--speech-pad-ms` has no effect on `--mode fidelity`.** FireRedVAD has no equivalent setting.
  The run now says the value is ignored instead of dropping it silently. Unchanged on Balanced,
  Fast, Faster and with the Silero detectors.
- **A two-pass run of Fidelity then Balanced still reduces a requested Aggressive to Balanced for
  the second pass.** That combination produced empty or truncated second-pass subtitles in about
  two thirds of the runs that led to the rule. It is now stated in the log when it happens. To
  run a pass at Aggressive, run it on its own.
- **Balanced runs its transcription model in a separate process by default**, which is what makes the
  20-minute restart possible on Windows. A second WhisperJAV process in Task Manager is
  expected. `--model-refresh-audio-minutes 0` goes back to one process.
- **Scene detection uses one clustering threshold instead of one per sensitivity.** Measurement
  showed the three-value setting was not the control its name implied.
- **On `--mode qwen` the scene-change threshold default moved from 18 to 22**, giving slightly
  fewer and longer scenes.
- **Some scene-detector warnings moved from the console into the log file.**

---

## Command line and scripts

Four things will stop a command that worked in 1.9.1. All four fail with a message rather than
silently.

- **`--speech-segmenter` is not accepted with `--mode balanced`**, and neither are
  `--pass1-speech-segmenter` or `--pass2-speech-segmenter` for a Balanced pass, nor
  `--max-group-duration` or `--chunk-threshold`. Balanced takes no separate *Speech Segmenter*
  any more. Use `--vad-version` to pick the Silero build. Fidelity, Qwen, anime-whisper and
  non-Balanced passes are unchanged.
- **`--no-vad` is removed.** It only set the *Speech Segmenter* to "none", which
  `--speech-segmenter none` already does on the modes that still take one.
- **A machine with no usable graphics card now needs `--accept-cpu-mode` or `--device cpu`.**
  Without one the run stops and waits, where it used to continue after a countdown.
- **`--mode fidelity` needs FireRedVAD's model and stops if it cannot get it.** Behind a proxy,
  or with no network and no earlier download, a Fidelity script that worked in 1.9.1 exits 1
  before reading any audio. `--speech-segmenter silero-v3.1` restores an option that needs no
  download; see [Known limitations](#known-limitations).

New switches: `--offline`, `--vad-version`, `--fail-on empty`, `--fail-on suspect`,
`--model-refresh-audio-minutes`, `--scene-clustering-threshold`, `--no-asr-telemetry`.

---

## Known limitations

- **Empty subtitle files can still happen.** This release fixes one cause on Balanced and
  reports the result honestly. It does not fix the fault behind **#394**, and it changes nothing
  on Fast and Faster. Keep reporting them.
- **The root cause of #394 is open.** The transcription model can enter a state where it returns nothing
  for the rest of a run. This release limits how long one copy of the model is kept loaded and
  records per-scene evidence; it does not prevent the state. Work continues with evidence from
  **@AlanZ-Git** and **@daoran9**.
- **Fidelity needs one run with a network connection.** FireRedVAD's model, about 2 MB, comes
  from Hugging Face. The Windows installer fetches it during installation, so an online
  installation will not meet this. On a permanently offline machine, either use
  `--speech-segmenter silero-v3.1`, which needs no download, or fetch the model elsewhere with
  `huggingface-cli download FireRedTeam/FireRedVAD --local-dir <folder>` (in China,
  `modelscope download --model xukaituo/FireRedVAD --local_dir <folder>`) and point
  `WHISPERJAV_FIREREDVAD_MODEL_DIR` at that folder.
- **`--offline` covers Hugging Face downloads only.** Other components use their own servers,
  and every model must have been downloaded once while online.
- **The GUI's Transcription tab always uses Silero 4.0.** The Ensemble Mode tab and the command
  line let you choose 3.1 or 6.2; the Transcription tab has no such control.
- **The audio analytics report does not catch every difficult file**, as described above.
- **Kaggle support (#329, #330) has moved to a later release.** Colab is unaffected.
- **`--async-processing` with Balanced, several files, and `--model-refresh-audio-minutes 0`**
  ends without a summary. With the default refresh setting the same run completes normally.
  Runs without `--async-processing` are unaffected.
- **On a graphics card this build cannot use, continuing on CPU does not reach everything.**
  anime-whisper, Qwen3, Cohere, the transformers mode, NeMo and the neural speech enhancers pick
  the card themselves and will fail there. On such a card use Balanced, Faster, Fast, or
  Fidelity with speech enhancement off.
- **The two-pass Customize values in #381 still reset** between launches; *Remember settings*
  covers the Transcription tab only.

---

## Reports addressed

| Report | What changed |
|---|---|
| **#416** | The Balanced speed and coverage rework above was shaped by this evidence. **Not confirmed resolved on the reporters' files** |
| **#414** | RTX 50 cards no longer forced to `float16`; probably also the empty-output cause. **Output on those cards has not been scored** |
| **#411**, **#333** | A card this build cannot use stops the run at start-up and asks |
| **#394**, **#263** | Per-file summary, manifest, one exit-status rule, transcription model restarted every 20 minutes, per-scene diagnostics. **Root cause still open** |
| **#413** | Punctuation-only subtitle lines removed on `--mode qwen` |
| **#415** | Cached models load without contacting Hugging Face; new `--offline` switch |
| **#340** | Files on cloud-mounted drives no longer abort the run |
| **#341** | Translation no longer fails on small-context local models |
| **#306** | A mistyped speech-enhancer name is rejected rather than ignored. **The subtitle truncation in the same thread is not addressed** |
| **#325**, **#382** | Retired DeepSeek models replaced in the backend and in both GUI lists |
| **#395** | DeepSeek v4-flash no longer reasons before translating |
| **#397** | `pornify` instructions corrected — already delivered, no update needed |
| **#339** | The same correction. **Its reporter has not retested**, so it is not claimed as resolved |
| **#323** | The warning no longer describes the release as broken |
| **#334**, **#366** | Windows and Linux install guides corrected |
| **#372** | A Japanese run reports its post-processing steps instead of going silent |
| **#328** | Skip already-subtitled files, in the GUI and in two-pass runs |
| **#96** | Optional *Remember settings* for the Transcription tab. The two-pass values in **#381** still reset |
| **#309** | Presets save on relocated user profiles |
| **#311** | FireRedVAD installed as standard and no longer marked experimental |
| **#254** | Lone 「はい。」 and 「うん。」 removed on `--mode qwen` |
| **#419** | The command as written no longer runs, and a run with no subtitles is reported `empty` rather than as a success. **Why it produced nothing is still open** |
| **#287**, **#326** | Possibly resolved by the empty-output fix. **Not confirmed** — please retest and report |
| **#347**, **#305** | English blocks from local translation models. **Not fixed**; the cause is identified |
| **#329**, **#330** | Kaggle. **Moved to a later release** |

---

## Credits

Most of this release started with a user report. With thanks to:

**For measurement, diagnosis or a supplied fix**

- **@yyyanlei** — identified the precision fault on an RTX 5070 and ruled out the wrong
  explanations. (**#414**)
- **@weifu8435** — supplied the console log behind the Hugging Face stall, and the raw first-
  and second-pass subtitle files that located the punctuation-only lines: 209 of 815 lines in
  one pass. (**#415**, **#413**)
- **@skysstst** — retested the translation fault on 1.9.1, measured 167 of 994 lines still
  returned in English, and produced a custom instruction file that cut it to 0 of 997.
  (**#347**)
- **@mcdman** — diagnosed and prototyped the DeepSeek v4-flash fix. (**#395**)
- **@Mimic-me** — contributed a batch of GUI improvements as a single reviewable pull request.
  (**#328**, **#96**, **#309**, **#325**, **#382**)
- **@daoran9** and **@AlanZ-Git** — measurements on the transcription model that stops returning output
  during long runs. Not resolved; their evidence is the basis for continuing work. (**#394**)
- **@yhxkry** and **@SangenBR** — established that the translation instructions served to every
  user were eight months out of date. (**#397**, **#339**)

**For reports resolved in this release**

**@leonscottkennedy5625-lang** and **@clt002** (#411, #333) · **@david2926121jay** (#340) ·
**@helsinkii** (#341) · **@ktrankc** (#306) · **@TinyRick1489** (#323) ·
**@ConnoisseurProtege** (#334) · **@thumper100** (#366) · **@Jerry199022** (#309) ·
**@unretired1516** (#328) · **@kylesskim-sys** (#311) · **@zoqapopita93** (#254) ·
**@loggias06row** (#395) · **@1Dreamer666** and **@Angelholl** (#325, #382) ·
**@sky9639** (#96, in part)

**For results that shaped decisions where nothing is yet fixed**

- **@yangming2027** and **@wjingshan** — logs and repeated results on Balanced producing little
  or no output. This is the reporting behind the Balanced rework that is the subject of this
  release. Whether it resolves their files is not yet known. (**#416**)
- **@Timi2028** — a run that produced no subtitles and reported `[SUCCESS]`. The command no
  longer runs as written, and such a run is now reported `empty`. Why it produced nothing is
  still open. (**#419**)
- **@zoqapopita93** — subtitles consisting only of `!` characters. The cause of that symptom is
  fixed on Balanced in this release, but an earlier fix for the same thread was retested and did
  not work, so this one is not claimed until confirmed. (**#287**)

If your report is fixed here and your name is missing, please tell us.
