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
  A *Remember settings* checkbox stores the first tab's fields and restores them
  next time. It is **off by default**, so the deliberate start-from-defaults
  behaviour is unchanged unless you opt in. (#96, #298, #381)

- **Saving presets no longer fails on relocated user profiles.** Where
  `%APPDATA%` is a junction — common when the profile has been moved to another
  drive — saving a preset could fail and the preset then could not be found.
  Writes are now atomic and resolve the real location first. (#309)

- **The GUI's DeepSeek model lists are current.** Both dropdowns offered
  `deepseek-chat` and `deepseek-coder`, retired upstream on 2026-07-24. They now
  offer `deepseek-v4-flash` and `deepseek-v4-pro`, and the OpenRouter list
  matches the backend default instead of contradicting it. (#325, #382)

With thanks to **@Mimic-me**, who contributed these as a reviewable batch.

### Failures that used to pass silently

- **A run that produces no usable subtitles now says so, and exits non-zero.**
  WhisperJAV could finish, print `[SUCCESS]`, and hand back an empty or
  drastically incomplete subtitle file — an 8,766-second video returning output
  that stopped at 377 seconds was reported as a success, and in one case a
  0-byte file was written while the console declared the run complete. Two things
  were wrong: nothing compared the output against the input, and the process
  returned "success" to the operating system regardless — for balanced, fast and
  faster it was hardcoded to do so. Both are fixed. A file with no subtitles at
  all now fails the run.

  Short-but-present output is treated more cautiously: it produces a prominent
  warning and does **not** fail the run, because speech can legitimately stop
  early and a wrong failure would be worse than the silence it replaces. Use
  `--min-coverage` to adjust the threshold, or `--min-coverage 0` to switch the
  span check off entirely.

  The policy here was set by the people who reported the problem rather than by
  us — see *Known limitations* for what is still missing. (#394, #263)

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

---

## Planned for this release, not yet landed

- **Corroborated failure detection for #394.** The check above fails a run only
  when there are no subtitles at all. Catching the *partial* cases — where output
  stops part-way while dialogue continues — needs the recogniser to report
  consecutive empty results, which is not yet instrumented. Until it is, those
  runs warn rather than fail.
- **Coverage checking in async mode.** Wired for normal processing only.

---

## Known limitations

- **The root cause behind #394 is still open.** The recognizer can enter a state
  where it returns nothing for the rest of a run, and the work above detects the
  *result* rather than preventing it. Investigation continues, with useful
  evidence contributed by @AlanZ-Git and @daoran9.
- **DeepSeek's thinking mode cannot be disabled from WhisperJAV** (#395). The
  translation library builds a fixed request and offers no way to pass the
  option DeepSeek requires, so this needs an upstream change first.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-08-30 | Output coverage gate wired: empty output fails the run and exits non-zero; short output warns. Nuclear exit no longer hardcoded to 0 (#394, #263) |
| 2026-08-30 | DeepSeek reasoning disabled for v4-flash (#395, @mcdman); post-processing stage logging so a stall names the operation that hung (#372) |
| 2026-08-30 | PR #378 merged (@Mimic-me): skip-existing in the GUI and ensemble (#328), opt-in settings persistence (#96, #298, #381), junction-safe preset saves (#309), DeepSeek v4 names in the GUI (#325, #382); OpenRouter dropdown synced to the backend default |
| 2026-08-30 | Linux install guide: GUI-backend step moved to a shared section covering every distribution; Fedora WebKit package corrected to the 4.1 series (#366) |
| 2026-08-30 | Seven small defects fixed: #340, #341, #306, #325, #323, #334, #366, plus the Ollama fallback model list |
| 2026-08-30 | Output coverage assessment added with tests — inert, pending a decision on enforcement (#394) |
| 2026-08-30 | GUI settings suite greened; drift between `index.html` and the backend defaults is now detected automatically |
| 2026-08-29 | README: ChronosJAV naming note for CLI users, and the complete seven-strategy merge table |
| 2026-08-29 | `pornify` instruction Gist synced to the repository version — delivered to users immediately, outside the release (#397) |
