# faster-whisper library A/B harness

`tools/fw_library_ab_harness.py`

## What it answers

Whether swapping the faster-whisper library underneath the balanced pipeline changes
speed, speech coverage, or word-timestamp availability, and which **call pattern** or
**parameter recipe** is responsible for what. It was written to evaluate the
`BBC-Esq/faster-whisper2` fork as a candidate, but it is library-agnostic: the same
file runs unchanged in every environment below, and its manifest fingerprints what
was actually loaded so result folders can be compared without guessing.

The script imports **nothing from whisperjav**. It needs only `faster_whisper`,
`numpy` and the standard library. `torch`, `pynvml` and `psutil` are used when present.

## Environments

| Label | Library | CTranslate2 | Silero VAD asset | Python | What it represents |
|---|---|---|---|---|---|
| `WJ` | faster-whisper 1.2.1 | 4.6.2 | v6 | conda `WJ` | WhisperJAV dev environment (torch 2.9.1+cu130 present) |
| `WJ-clean` | faster-whisper 1.2.1 | 4.6.2 | v6 | `D:\Git\faster-whisper2\venv_fw121` | the same upstream library in a venv built exactly like `FW2` (no torch): the cleaner "A" arm |
| `FW2` | faster-whisper2 2.1.1 | 4.6.2 | v6 (identical file) | `D:\Git\faster-whisper2\venv` | the candidate fork, CT2 held equal |
| `FW2-ct2-4.8.1` | faster-whisper2 2.1.1 | 4.8.1 | v6 | same venv, CT2 pin swapped for the run | the candidate with the CT2 that `uv.lock` ships to users |
| `V07B` | faster-whisper 1.0.2 | 4.4.0 | **v4** | `D:\Git\faster-whisper2\venv_v07b` | the v0.7b community notebook stack |

All four venvs exist and have been exercised (2026-09-08). Their recipes are in
`docs/research/faster_whisper2_candidate/README.md`, section "Install guide".
**Run the harness from the WhisperJAV repo root, not from inside `D:\Git\faster-whisper2`**,
otherwise the clone's source folder shadows the installed package.

## Arms

Each arm is one call pattern combined with one parameter recipe. All arms transcribe the
same 16 kHz mono audio decoded by the library's own `decode_audio` (verified bit-identical
across the environments).

| Arm | Pattern | Recipe | Question it isolates |
|---|---|---|---|
| `wj_native_scene` | one `transcribe(vad_filter=True)` per scene (≤ 29 s) | `--recipe` | the current balanced "internal VAD" path as the pipeline calls it (Problem 2 configuration) |
| `wj_native_file` | one call for the whole file | `--recipe` | does the per-scene call pattern itself cost speed or coverage? |
| `wj_external_group` | one `transcribe(vad_filter=False)` per Silero group (≤ 9 s, 0.1 s gap) | `--recipe` | the per-group call pattern of the external-segmenter path (P1H1) |
| `lib_default_file` | whole file | library defaults, `language=ja`, `word_timestamps=True` | does WhisperJAV's recipe hurt compared with leaving the library alone? |
| `v07b_file` | whole file | the v0.7b notebook's exact call | the reference users compare against |
| `wj_batched_file` | `BatchedInferencePipeline`, whole file, batch 8 | `--recipe`, minus what the batched path overrides (see `recipe_keys_not_honoured`) | the batched path the fork claims to have fixed (`skipped` on 1.0.2, which has no such class) |

### Recipes: `--recipe`

The four `wj_*` arms are **call patterns**; `--recipe` chooses the decoding parameters they
carry. `lib_default_file` and `v07b_file` have their own fixed decoding parameters and ignore
the flag (they record `recipe = fixed`) — but see the compute-type warning below: they do
**not** escape the recipe's model precision.

A recipe is frozen once added, never edited, so nothing this flag does re-interprets an
earlier run. Runs from 2026-09-09 onwards are self-identifying: the recipe id is in the
run-folder name (`<label>_<recipe>_<timestamp>`), in `manifest.json` and in every row of
`summary.csv`. Folders written **before** that date have none of those three fields and are
named `<label>_<timestamp>`; every one of them is `wj_v192` at `float16`.

| Recipe | Decoding parameters | Native VadOptions | `compute_type` |
|---|---|---|---|
| `wj_v192` (default) | WhisperJAV v1.9.2 balanced, per `--sensitivity` | the pipeline's preset | `float16` |
| `fwdefaults` | faster-whisper's own defaults (below) | the library's defaults | `auto` |

`fwdefaults` is every faster-whisper 1.2.1 default, key for key, with two deliberate exceptions:

| Parameter | Value | |
|---|---|---|
| `beam_size` | 5 | library default |
| `patience` | 1.0 | library default |
| `temperature` | 0.0 | **not** the default (`[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`) |
| `condition_on_previous_text` | True | library default |
| `compression_ratio_threshold` | 2.4 | library default |
| `log_prob_threshold` | -1.0 | library default (the CLI-facing name is `logprob_threshold`) |
| `no_speech_threshold` | 0.6 | library default |
| `repetition_penalty` | 1.0 | library default, i.e. no penalty |
| `no_repeat_ngram_size` | 0 | library default, i.e. disabled |
| `suppress_blank` | True | library default |
| `hallucination_silence_threshold` | not passed | library default `None`, i.e. disabled |
| `word_timestamps` | True | **not** the default (`False`); the harness needs it for the `words_*` columns |

`language="ja"` and `task="transcribe"` are set so no per-call language detection runs; they are
not decoding parameters. Everything else is left to the library (`best_of=5`,
`max_initial_timestamp=1.0`, `without_timestamps=False`, `prompt_reset_on_temperature=0.5`) —
**on the plain `transcribe` path only**. `BatchedInferencePipeline` accepts several of these
arguments and then overrides them in its body: in faster-whisper 1.2.1 it hardcodes
`condition_on_previous_text=False`, `hallucination_silence_threshold=None`,
`max_initial_timestamp=0.0`, `prompt_reset_on_temperature=0.5`, and uses only the first
temperature. `kwargs_dropped_by_library` cannot see this, because it inspects the signature,
not the body. The harness reads the override list out of the installed library and reports it
per arm in the new `recipe_keys_not_honoured` column — so `wj_batched_file` under `fwdefaults`
runs with `condition_on_previous_text=False` and says so, and `wj_batched_file` under
`wj_v192 --sensitivity aggressive` reports that its `(0.0, 0.2)` ladder is cut to `0.0`.

Three consequences worth holding in mind when reading a `fwdefaults` run:

- Pinning `temperature=0.0` **removes the temperature fallback ladder**, so
  `compression_ratio_threshold` and `log_prob_threshold` have nothing to fall back to: a window
  that trips them is **kept as it is**. The counters do not go quiet — faster-whisper logs the
  threshold miss inside the loop whether or not another temperature follows — so under this
  recipe `compression_ratio_fallbacks` and `log_prob_fallbacks` count *failed windows that were
  kept*, not re-decodes. Measured on the 40 s clip: `wj_external_group` recorded 3, with lines
  such as `Compression ratio threshold is not met with temperature 0.0 (55.625000 > 2.400000)`
  — a repetition loop the `wj_v192` recipe would have re-decoded at a higher temperature.
- `compute_type = auto` is **not** `float16`, and it is **not** the library's default either
  (that is `"default"`, which resolves to `float16` here). `auto` asks CTranslate2 for the
  fastest supported type, which on this GPU with CT2 4.8.1 is `int8_float16` — an int8-quantised
  model (measured 2026-09-09, `faster-whisper-tiny` and `large-v2` alike). It is in this recipe
  because the parameter set was specified that way.
- `compute_type` is a **model-load** setting, so it is a property of the run, not of an arm.
  One `WhisperModel` serves every arm, which means a `fwdefaults` run puts `lib_default_file`
  and `v07b_file` on the int8 model too, even though they ignore `--recipe` for their decoding
  parameters. Speed and VRAM from a `fwdefaults` run therefore cannot be compared with a
  `wj_v192` run at all, and a coverage or CER difference between the two carries the
  quantisation with it. **Pass `--compute-type float16` to A/B the decoding parameters alone.**
  The harness prints this warning at start-up whenever the compute type is left to CTranslate2.

`--recipe` moves the **decoding** parameters only. The Silero parameters that split the audio
for `wj_native_scene` and `wj_external_group` are WhisperJAV's preset for `--sensitivity` in
every recipe, because those arms exist to emulate WhisperJAV's segmentation: holding the split
constant keeps the spans comparable, so a difference between two recipes is a decoding
difference. With `fwdefaults`, `--sensitivity` therefore affects only the split — and if none
of the selected arms splits the audio (`wj_native_file`, `wj_batched_file`, `lib_default_file`,
`v07b_file`), it has **no effect at all** on the run while `summary.csv` still records it. The
harness warns at start-up when that is the case.

The `wj_v192` recipe is the resolved output of
`python -m whisperjav.main --dump-params <file> --mode balanced --sensitivity balanced|aggressive`
taken on 2026-09-08, after the ASR module's own renaming step (independently re-checked key for key
by the adversary pass). The three WhisperJAV-side post filters (`logprob_margin`,
`drop_nonverbal_vocals`, `post_model_filter_enabled`) are not library parameters and are deliberately
absent, so the harness measures the library, not the pipeline's post-processing. Keys a given library
version does not accept (1.0.2 lacks `multilingual` and `log_progress`) are dropped per call and listed
in `kwargs_dropped_by_library`.

### Segmentation: two arms emulate, one column tells you how much

- `wj_native_scene` packs the library's own Silero speech chunks into scenes of at most 29 s,
  splitting at silences of 2.5 s or more. WhisperJAV's detector (`scene_detection.py:558-566`)
  runs auditok with `max_silence` 2.5 s and `drop_trailing_silence`, so it also hugs activity rather
  than tiling the file, but its energy gate (32 dB) keeps more audio than Silero does. The emulation
  is therefore a **stand-in**, and any ground-truth speech outside the emulated scenes can never be
  covered. The column `gt_speech_in_spans_pct` reports that ceiling for every arm (100 for
  whole-file arms). Read `gt_covered_pct` against it.
- To use the **real** boundaries, pass `--scenes-json` with the master metadata WhisperJAV writes to
  `<temp-dir>/<basename>_master.json` (its `scenes_detected` list is read directly), or any JSON list
  of `{start,end}` / `{start_time,end_time}` / `{start_abs,end_abs}` / `[s,e]` entries in seconds. One
  file per invocation. `scene_source` in the summary says which was used.
- `wj_external_group` merges Silero chunks closer than 0.1 s and caps groups at 9 s (the "Test-D"
  grouping). It reproduces the per-group **call pattern**, not the exact boundaries of WhisperJAV's
  silero-v3.1 segmenter, which uses a different Silero model and thresholds.

## Running

```bash
# WJ environment (from the repo root)
C:/Users/MK/anaconda3/envs/WJ/python.exe tools/fw_library_ab_harness.py --label WJ --repeat 2 --shuffle-arms \
    --files "test_media/Ground_Truths/Netflix/*.mkv" --out test_media/ab_results

# FW2 environment
D:/Git/faster-whisper2/venv/Scripts/python.exe tools/fw_library_ab_harness.py --label FW2 --repeat 2 --shuffle-arms \
    --files "test_media/Ground_Truths/Netflix/*.mkv" --out test_media/ab_results

# Call pattern under faster-whisper's own parameters, on one long file
C:/Users/MK/anaconda3/envs/WJ/python.exe tools/fw_library_ab_harness.py --label WJ --recipe fwdefaults \
    --repeat 1 --shuffle-arms --arms "wj_native_file,wj_external_group" \
    --files "F:/MEDIA_DLNA/EKAI-023/EKAI-023.mp4" --out "F:/MEDIA_DLNA/EKAI-023/ab_results_WJ"
```

| Flag | Default | Notes |
|---|---|---|
| `--model` | `Systran/faster-whisper-large-v2` | the repo id WhisperJAV's `large-v2` resolves to; identical files in every environment. faster-whisper2 has **dropped** the bare `large-v2` alias (and `large`, `large-v1`, `distil-large-v2`); if you pass one anyway the harness retries with the Systran repo id and records `model_alias_fallback: true` |
| `--recipe` | `wj_v192` | decoding parameters for the `wj_*` arms: `wj_v192` (the original behaviour) or `fwdefaults`. Also supplies the default `--compute-type`, and lands in the run-folder name |
| `--compute-type` | from `--recipe` | `float16` for `wj_v192`, `auto` for `fwdefaults`. Passing the flag overrides the recipe; `manifest.json` records what CTranslate2 resolved it to |
| `--sensitivity` | `balanced` | or `aggressive`. With `fwdefaults` it selects only the Silero split parameters, not the decoding parameters |
| `--arms` | `all` | comma list, e.g. `wj_native_scene,wj_native_file` |
| `--repeat` | 1 | run every arm N times; rows carry a `rep` column and per-arm files get a `.repN` suffix. GPU fp16 beam search is not bit-reproducible (the same 10 s of audio gave 1, 4 and 5 segments on `tiny` in three runs), so a difference smaller than the spread between reps is noise |
| `--shuffle-arms` | off | randomise arm order within each repetition; use it with `--repeat` so a first-arm effect cannot masquerade as a library effect |
| `--gt-dir` | media folder | folder holding `<stem>.srt` or `<stem>.ja.srt`; without ground truth the coverage columns stay empty and the timeline still records output speech per bin |
| `--scenes-json` | none | real scene boundaries for `wj_native_scene` (single file only) |
| `--max-seconds` | none | truncate audio for a quick smoke test |
| `--no-warmup` | off | skip the 8 s warm-up call (the warm-up keeps model JIT out of the first arm's timing) |

Run each environment **alone** on the GPU. A second process on the same GPU contaminates
both the timing and the VRAM columns.

The script keeps the model referenced until it terminates its own process after flushing
(`TerminateProcess` on Windows, `os._exit` elsewhere). With CTranslate2 4.6.2, destroying the CUDA
model kills the process with 0xC0000409 (WhisperJAV issue #125; Git Bash shows it as 127), and that
happened the moment `main()` released its locals. CTranslate2 4.8.1 and 4.8.2 do not have the crash.
Every result file is written and closed before that point (research README, section 7b), so the exit
code you see reflects the run, not the destructor.

## Outputs

`<out>/<label>_<recipe>_<timestamp>/`

| File | Content |
|---|---|
| `manifest.json` | Python, library version and file path, installed distributions (faster-whisper / faster-whisper2, ctranslate2, onnxruntime, numpy, av, torch), CT2 supported compute types, GPU name and driver, CUDA DLL folders registered, Silero asset name, size and MD5, model path, `config.json` MD5 and `model.bin` size and partial MD5, effective compute type, model load time, device VRAM before and after load, every arm's exact kwargs |
| `summary.csv` | one row per file × arm × rep (columns below) |
| `<stem>/<arm>.segments.jsonl` | every segment: start, end, text, `avg_logprob`, `no_speech_prob`, `compression_ratio`, `temperature`, `seek`, word count, words with timestamps and probabilities |
| `<stem>/<arm>.srt` | the arm's output for eyeballing against the ground truth |
| `<stem>/<arm>.timeline.csv` | 30 s bins: ground-truth speech seconds, output speech seconds, covered seconds |
| `<stem>/<arm>.calls.jsonl` | one line per `transcribe()` call: span, wall time, segments produced, `duration_after_vad`, error |
| `<stem>/<arm>.log` | the `faster_whisper` DEBUG log captured during the arm (VAD kept/removed lines, no-speech skips, threshold fallbacks) |

### summary.csv columns

| Column | Meaning |
|---|---|
| `recipe`, `recipe_version` | the recipe this arm used (`fixed` for `lib_default_file` and `v07b_file`) and its frozen version string |
| `recipe_keys_not_honoured` | recipe keys the library accepted and then overrode in its body, with the value it substituted. Empty for every plain-`transcribe` arm; populated for `wj_batched_file`, whose overrides `kwargs_dropped_by_library` cannot see |
| `compute_type`, `compute_type_effective` | what was requested at model load, and what CTranslate2 actually used (`auto` resolves to `int8_float16` here) |
| `status` / `error` | `ok`, `partial(N call errors)`, `failed`, `skipped` |
| `kwargs_dropped_by_library` | recipe keys this library version's `transcribe()` does not accept |
| `n_calls`, `span_total_s`, `split_s` | number of `transcribe()` calls, seconds of audio handed over, time spent segmenting before the calls |
| `decode_wall_s`, `wall_s`, `x_realtime` | decode time alone; decode plus segmentation; audio-seconds per wall-second computed on the **total**, so per-scene and per-group arms pay for their Silero pass the way whole-file arms pay for theirs inside `transcribe()` (**speed**, benefit BB) |
| `vram_device_baseline_mib`, `vram_device_peak_mib`, `vram_device_delta_mib` | whole-device `memory.used` sampled every 0.25 s, baseline taken immediately before the arm. CTranslate2 keeps its allocations between arms, so the delta is meaningful for the **first arm after load** and near zero afterwards. To measure one arm's VRAM, run it alone (`--arms X`) and add `vram_device_after_load_mib − vram_device_idle_before_load_mib` from the manifest. Per-process accounting (`vram_process_*`) is filled only where the driver exposes it; on Windows WDDM it is empty (benefit BA) |
| `duration_after_vad_s` | what the library's VAD kept, summed over calls |
| `words_with_timestamps`, `words_nonempty`, `words_empty` | words carrying timestamps; `words_empty` counts entries whose text is empty, which is what a punctuation merge leaves behind, so a library that merges Japanese words differently shows up here even when the total does not move (benefit BC) |
| `mean_no_speech_prob`, `mean_avg_logprob`, `max_compression_ratio` | decoder health per arm |
| `out_segments`, `out_speech_s`, `out_mean_seg_s`, `out_chars` | output quantity |
| `gt_cues`, `gt_speech_s` | the ground truth for this file |
| `gt_speech_in_spans_pct` | share of ground-truth speech time inside the spans handed to the recogniser: the ceiling for the next column |
| `gt_covered_pct` | share of ground-truth speech time overlapped by output cues (**coverage**, Problem 2) |
| `gt_cues_hit_pct` | share of ground-truth cues touched by at least one output cue |
| `out_outside_gt_pct` | share of output time that lies outside ground-truth speech (over-extension or hallucination) |
| `cer` | character error rate against the concatenated ground-truth text, NFKC-normalised with spaces and punctuation removed |
| `longest_uncovered_speech_bins`, `longest_uncovered_span_s` | longest run of consecutive **speech-bearing** 30 s bins (≥ 3 s of ground truth) with zero covered seconds, and the elapsed seconds from the first to the last bin of that run; a cascade signature |
| `no_speech_skips` | count of `No speech threshold is met` in the captured log: windows the decoder discarded |
| `compression_ratio_fallbacks`, `log_prob_fallbacks`, `prompt_resets` | threshold events from the log |
| `windows_processed` | count of `Processing segment at`: encoder windows decoded (not logged by the batched path) |
| `silero_chunks`, `scene_source` | how the per-scene / per-group arms were split |

## Reading the results

Compare rows with the same `file` and `arm` across labels, always against the spread between reps:

1. **Library effect**: `WJ-clean` vs `FW2` (same venv recipe, no torch), then `WJ` vs `WJ-clean` to see
   whether the conda environment itself matters. With the same model files, CT2 and Silero asset, any
   difference is the fork's Python code. One such difference is known before measuring: the fork's
   `merge_punctuations` merges any Japanese word that begins with `。、！？` into the preceding word
   (research README, section 3), which moves word and segment boundaries; watch `words_empty`,
   `out_mean_seg_s` and `gt_covered_pct` together.
2. **CT2 effect** (P1H2): `FW2` vs `FW2-ct2-4.8.1`.
3. **VAD generation effect** (P2H1): `V07B` vs `WJ-clean` on `v07b_file`, the only arm whose recipe 1.0.2
   accepts unchanged.
4. **Call-pattern effect** (P1H1): within one label, `wj_external_group` vs `wj_native_file`
   (`n_calls` and `x_realtime`), and `wj_native_scene` vs `wj_native_file` (per-scene cost, read
   against `gt_speech_in_spans_pct`).
5. **Recipe effect** (M3): within one label, `wj_native_file` vs `lib_default_file` vs `v07b_file` on
   `gt_covered_pct`, `no_speech_skips` and `cer`.
6. **Recipe x call pattern** (M3 crossed with P1H1): the same arm across two runs that differ only in
   `--recipe`. Run them back to back on the same file and machine, and read `out_chars` and `cer`
   together with `gt_covered_pct`: `fwdefaults` removes WhisperJAV's repetition guards
   (`repetition_penalty=1.5`, `no_repeat_ngram_size=3`) and turns `condition_on_previous_text` on, so
   over-generation shows up as `out_chars` and `cer` rising while coverage barely moves. Note that the
   two runs also differ in `compute_type` unless you pass `--compute-type` explicitly.

The Netflix set is short (40 s to 397 s per clip, 23.6 minutes in total) and speech-dense (82 % of the
40 s clip is dialogue, against 22–41 % of the subtitled span in the recorded JAV baseline). It measures
coverage and speed per call pattern well; its coverage percentages cannot be read against the JAV
baseline, and it cannot show a long-run collapse. For that, point `--files` at a multi-hour file
without ground truth and read `timeline.csv` and `calls.jsonl`.

## Verified 2026-09-09 (recipes)

- `--recipe wj_v192` reproduces the pre-recipe harness exactly: `build_arm` output compared key
  for key against the previous revision of the file for all six arms x both sensitivities —
  0 differences, `ARM_ORDER` unchanged.
- `--recipe fwdefaults` run end to end in the `WJ` env on
  `test_media/Ground_Truths/Netflix/The.Naked.Director.S01E04.Scene.1.mkv` with
  `--arms wj_native_file,wj_external_group --shuffle-arms`: both arms `ok`, no kwargs dropped,
  `compute_type auto -> int8_float16`, recipe columns populated, run folder
  `SMOKE_fwdefaults_20260909_024107`.
- Default path re-run (`--arms wj_external_group`, no `--recipe`): `compute_type_source
  recipe:wj_v192`, requested and effective `float16`, folder `SMOKE2_wj_v192_20260909_024336`.
- Every remaining arm of the new flag executed rather than left to inference:
  `A_fwdefaults_20260909_025427` (`wj_native_scene` + `wj_batched_file` under `fwdefaults`;
  the batched arm reported `condition_on_previous_text: asked True -> forced False;
  max_initial_timestamp: left to the library (1.0) -> forced 0.0`),
  `B_wj_v192_20260909_025516` (`wj_batched_file`, `--sensitivity aggressive`; reported
  `temperature: asked (0.0, 0.2) -> only 0.0 is used` — a pre-existing property of the batched
  arm that no earlier run had surfaced), and `C_fwdefaults_20260909_025535` (a whole-file arm
  plus a `fixed` arm; the "`--sensitivity` has NO effect on this run" warning fired).
- The owner's table checked against `WhisperModel.transcribe`'s signature in faster-whisper
  1.2.1: every listed **decoding** value is the library default except `temperature` and
  `word_timestamps`. `compute_type` is a third departure: `WhisperModel.__init__`'s default is
  `"default"`, not `"auto"`.
- Reviewed by the `assessment-adversary` subagent (CLAUDE.md rule A6), which refuted three
  statements in an earlier draft of this file — the claim that old result folders carry a recipe
  id, the claim that the fallback counters read 0 under a pinned temperature, and the claim that
  the batched arm leaves everything else to the library. All three are corrected above; the
  batched override is now detected in code rather than described in prose.

## Verified 2026-09-08

- Every arm and every conditional path executed: all six arms in `WJ` and `FW2` with large-v2
  `float16` on the 40 s clip (identical model and Silero fingerprints in both manifests); the alias
  fallback (`--model large-v2` in FW2, `model_alias_fallback: true`); `--repeat 2`; `--sensitivity
  aggressive` with `--shuffle-arms`; the no-ground-truth path; `--scenes-json`; and all arms in `V07B`
  with the batched arm `skipped` and `log_progress;multilingual` reported as dropped.
- Result folders: `test_media/ab_results/WJ_20260908_135041`, `FW2_20260908_135427`,
  `FW2-alias_20260908_135516` (pre-fix column names), smoke runs in the session scratchpad.
