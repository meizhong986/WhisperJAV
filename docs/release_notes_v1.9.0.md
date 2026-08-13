# WhisperJAV v1.9.0 — New Ensemble Defaults, Better Subtitle Timing, New Japanese Models

This is a larger release than usual — it carries everything since v1.8.14, including
the v1.8.15 work that was prepared but never published. The theme is subtitle
*quality you can see*: better timing, cleaner output, and more Japanese-specialised
models to choose from. As always: these are incremental improvements, not miracles —
results still depend on your source audio.

**Highlights:**

- New two-pass ensemble defaults: anime-whisper + Qwen3-ASR out of the box
- Subtitle timing overhaul for anime-whisper: far fewer subtitles spanning pauses,
  starts that land on the actual speech
- Cleaner output: sound-only lines removed, absurdly long subtitle durations repaired
- Three new selectable Japanese models, and one new experimental voice detector
- Balanced mode is faster (native VAD by default)
- Qwen pipeline uses less VRAM by default (~1 GB saved)

---

## Two-pass ensemble: new defaults

The Ensemble tab now starts with a tested pairing instead of a generic one:

- **Pass 1:** anime-whisper · semantic scenes · WhisperSeg VAD · aggressive
- **Pass 2:** Qwen3-ASR · semantic scenes · TEN VAD

The idea: two different recognizers *and* two different voice detectors, so the
passes miss different things and the merge covers both. Everything remains
changeable per pass — the README's new "Mix-and-match strategies" section walks
through every option with plain-language strengths and weaknesses.

Note for existing users: the app restores your previous dropdown selections, so
you may need to re-select pipelines once to pick up the new defaults.

## Subtitle timing overhaul (anime-whisper)

Three changes that together address the most common timing complaints:

- **Fewer subtitles spanning pauses.** The voice detector now decides segment
  boundaries with a new offline analysis pass instead of a purely live one. On our
  ground-truth benchmark, subtitles that incorrectly bridged a ≥0.4s silence
  dropped from 11 of 35 to 2 of 36.
- **Starts land on speech.** Subtitle start times previously could sit up to half a
  second before the first spoken word (the detector's wide capture net). Display
  starts now snap to the first confident speech frame — median start error went
  from roughly −0.5s to near zero on the benchmark — while the recognizer still
  hears the full captured audio, so no words are lost.
- **Per-sensitivity tuning.** Conservative / balanced / aggressive now each have
  their own tuned voice-detector profile for anime-whisper, instead of one shared
  setting. Aggressive is the tuning target for JAV content.

For tinkerers, four new levers appear in the Customize dialog (decoder mode,
growth floor, gap merge, max speech duration). The previous behavior remains
selectable.

## Cleaner subtitle output

- **Sound-only lines are removed.** Lines consisting purely of moan/breathing kana
  (e.g. あぁん、はぁはぁ) are dropped from the final SRT across all pipelines. An
  evidence check protects real dialogue — anything containing kanji, particles,
  or recognizable words is always kept.
- **Single-token artifact lines are removed** (Qwen pipelines). Lone one-character
  lines like `あ。` `は。` `え。` that the sensitive decoding produces are dropped.
  Disable with `--no-qwen-drop-nonverbal-lines` if you want them.
- **Absurd durations are repaired** (Qwen pipelines). When a subtitle's duration is
  far too long for its text — a few characters stretched over ten seconds — the
  start is pulled in to match a normal reading speed while the end stays put. The
  console reports how many lines were retimed, so nothing happens silently.
- **Scene-boundary overlaps are resolved** (Qwen pipelines). Duplicate fragments
  and overlapping timestamps at scene joins are cleaned up automatically.

## New models

All are selectable from the ensemble Model dropdown (and via CLI); weights
download from Hugging Face on first use.

- **whisper-ja-1.5B (CT2)** — a Japanese Whisper fine-tune, used with the Balanced
  pipeline. The strongest results in our scene-length benchmarks; word timestamps
  intact. Community model — occasional repetition, which the cleanup filters
  mostly catch. Not compatible with the Fidelity pipeline.
- **JA Anime-Galgame 1.7B** — a Qwen3-ASR fine-tune trained on galgame speech,
  with published gains on anime-style dialogue (about 27% relative error
  reduction vs. base). It recovers lines the base model drops, at the cost of
  slightly more junk — which the new filters are well suited to remove.
- **JA-tuned 1.7B (neosophie)** — a Qwen3-ASR fine-tune aimed at proper nouns and
  kanji-heavy phrasing. No published benchmarks; included after positive manual
  testing.

## New experimental voice detector: FireRedVAD

A tiny (~2 MB) multilingual voice-activity model, added as an **experimental**
speech-segmentation option for early feedback. It is not installed by default —
select it in the GUI and it will tell you to run `pip install fireredvad`. Its
presets are first-pass values, not yet tuned on JAV ground truth like WhisperSeg's.
Feedback welcome on the issue tracker.

## Qwen pipeline: VAD-first timestamps by default

Subtitle timing in the Qwen/ChronosJAV pipelines now comes from the voice
detector's speech frames by default, instead of loading a separate forced-aligner
model. Practical effects: about 1 GB less VRAM used, slightly faster runs, and
timing that benefits directly from the detector improvements above. The forced
aligner is still there — pick an aligner mode in Customize → Alignment if you
want word-level alignment (the Cohere preview model keeps the aligner, as it has
no native timing).

## Balanced mode is faster

From the unpublished v1.8.15 work: Balanced now uses faster-whisper's built-in
voice detection by default — the fastest configuration in A/B testing on a
reference clip. Selecting any external segmenter still works and now uses finer
grouping defaults. Also fixed: the anime-whisper Frame Gap / Max Group sliders
previously had no effect (they were silently overridden inside the pipeline) —
they now work, along with retuned Qwen VAD defaults and fewer temporary files
during processing.

## CrispASR — still an experimental preview

The opt-in CrispASR external pipeline remains a preview in this release. Japanese
output quality is still being tuned; rough output is expected for now.

## Documentation

The README was rewritten for end users: the science behind the approach up front,
then a plain-language tour of the processing chain, and a new **Mix-and-match
strategies** section — one table per component (pipeline, scene detection, audio
pre-processing, speech segmentation, ASR model) with honest strengths and
weaknesses, plus known-good two-pass recipes.

---

## Known limitations

- Returning users' saved form values can shadow the new ensemble defaults until
  the dropdowns are touched once (see above).
- FireRedVAD presets are not yet JAV-tuned — treat it as an experiment.
- whisper-ja-1.5B (CT2) cannot be used with the Fidelity pipeline (different
  engine format); the GUI notes this.
- CrispASR remains preview quality for Japanese.

## Version

- 1.8.14 → 1.9.0. (v1.8.15 was prepared but never published; its changes are
  included here.)
