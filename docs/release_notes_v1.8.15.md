# WhisperJAV v1.8.15 — Faster Balanced VAD + Qwen/ChronosJAV Cleanup + CrispASR Preview

Performance and quality update for the speech-segmentation front-end and the Qwen (ChronosJAV) pipeline, plus an early experimental preview of CrispASR for enthusiast users.


BALANCED PIPELINE:
- Default VAD is now native faster-whisper VAD — the fastest option in owner A/B testing on a 293-second JAV clip (RTX 3060)
- Choosing any external segmenter on `balanced` now uses finer "Test-D" grouping by default (max group 9.0s, chunk threshold 0.1s)
- New CLI knobs: `--max-group-duration`, `--chunk-threshold`
- Simple / ensemble / customize paths now share one defaults helper, so the GUI Customize panel shows the real defaults
- Reversible: `--speech-segmenter silero-v3.1` (or any external segmenter) restores external segmentation; `fidelity` is unchanged

QWEN / CHRONOSJAV:
- anime-whisper grouping bug fixed: the Frame Gap / Max Group sliders, CLI flags, and defaults were being overridden inside the pipeline and had no effect — they now take effect
- VAD defaults retuned for qwen3 + anime-whisper to JAV-tuned values: frame gap 300ms, max group 3.0s, VAD threshold 0.25, padding 100/100ms
- In-memory frame audio: anime-whisper and qwen3 no longer write one temporary WAV per VAD group (fewer temp files, less disk I/O)
- New nonverbal line filter: drops lone single-token artifact lines (`あ。` `は。` `え。` `切。` …) that the sensitive VAD/decoding captures — all three Qwen backends, on by default, `--no-qwen-drop-nonverbal-lines` to disable

EXPERIMENTAL PREVIEW:
- CrispASR: an early experimental preview of an upcoming pipeline, for enthusiast users. Opt-in only; Japanese output quality is still being tuned — not recommended for production use yet


---


## Balanced: Native Faster-Whisper VAD Default

- **What changed** — On `balanced`, the default speech segmentation is now faster-whisper's built-in VAD instead of an external Silero pass. In owner A/B testing across five configurations on a 293-second JAV reference clip (RTX 3060), native VAD was the fastest end-to-end.

- **External segmenters still available, now finer** — If you opt into any external segmenter on `balanced` (e.g. `--speech-segmenter silero-v3.1`), it now defaults to "Test-D" grouping — max group 9.0s, chunk threshold 0.1s — which produced the finest subtitle granularity in testing. Pass `--max-group-duration` / `--chunk-threshold` to override.

- **Consistency fix** — The simple, ensemble, and GUI-customize paths now resolve their balanced VAD defaults through one shared helper, so the values you see in the GUI Customize panel are the values actually used.

- **Reversible** — Native VAD is a default, not a lock-in. Select any external segmenter to go back to external segmentation. `fidelity` is untouched and continues to use WhisperSeg.


## anime-whisper Grouping Fix

- **Symptom** — On the anime-whisper generator, adjusting the "Frame Gap Threshold" and "Max Group Duration" sliders (or the equivalent CLI flags) had no effect on the output.

- **Cause** — The Qwen pipeline constructor unconditionally reset those two values to a fixed 0.5s / 5.0s for anime-whisper, after the incoming slider/CLI/default values had already been applied — silently discarding them. The internal grouping stage inherited the same fixed values.

- **Fix** — That override was removed. Frame Gap and Max Group now flow through from the slider, CLI flag, or default. A regression test locks this in.


## Qwen VAD Default Retune (qwen3 + anime-whisper)

- **New defaults** — Frame gap 300ms, max group 3.0s, VAD threshold 0.25, padding 100ms before / 100ms after. These are JAV-tuned values validated on a real clip. Cohere is unchanged.

- **All controls live** — With the grouping fix above, every Qwen VAD control (Frame Gap, Max Group, Threshold, padding) is now adjustable from the GUI Customize panel and the CLI.


## In-Memory Frame Audio (anime-whisper + qwen3)

- **What changed** — The subtitle pipeline no longer writes one temporary WAV file per VAD group. For the anime-whisper and qwen3 generators, audio slices are now passed in memory.

- **Effect** — Fewer temporary files and less disk I/O per run, with no change to the subtitles produced. Cohere continues to use the temp-WAV path.


## Nonverbal Line Filter (all Qwen backends)

- **Why** — Qwen3-ASR is run with deliberately sensitive VAD/decoding to capture detail, which also surfaces moan onsets, breaths, and ASR truncations as lone one-token subtitle lines. Too many of these clutter the subtitle.

- **What it does** — A new Phase-8 filter drops SRT entries whose entire line is a single curated artifact token (`つ ふ ふっ 切 は え あ ん`, optionally followed by `。`). Whole-line exact match, so real dialogue and multi-token moans are never touched. Validated on a real qwen3 output (removed 302 of 2414 lines, all artifacts, zero dialogue hits).

- **Scope + control** — Applies to all three Qwen backends (qwen3 / cohere / anime-whisper), on by default. Disable with `--no-qwen-drop-nonverbal-lines`.


## CrispASR — Experimental Preview

- **What it is** — An early, opt-in preview of an upcoming standalone pipeline, included for enthusiast users who want to experiment ahead of the full feature.

- **Status** — Experimental. Japanese output quality is still being tuned and the model-selection flow is not final. Not recommended for production subtitles yet — please treat rough output as expected for a preview, not as a regression.


## Version

- Bumped 1.8.14 → 1.8.15.
