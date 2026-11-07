# CrispASR viability suite

## Purpose

Measures a candidate for one stage of WhisperJAV's chain (owner W581 V3, W582, W584). It runs Parakeet-CrispASR exactly as
WhisperJAV ships it and swaps one stage for the candidate; everything else is WhisperJAV's unchanged code, imported read-only.
Each run is scored with subeval against the reference subtitles (the seven Netflix clips by default).

Phase A swaps:
- **A1, the speech enhancer:** CrispASR separators (htdemucs, Mel-Band-RoFormer, Mini-BS-RoFormer) and audio-to-audio speech
  models (Sidon, VoxCPM2 AudioVAE), against no enhancer and WhisperJAV's own four enhancers.
- **A2, the speech segmenter:** CrispASR VAD engines (Silero v6.2, FireRedVAD, the Japanese-ASMR Whisper VAD, MarbleNet,
  WebRTC), against TEN VAD (the pipeline's default) and WhisperJAV's own Silero v6.2, FireRedVAD and WhisperSeg.

It changes nothing in WhisperJAV. Areas B-D (text processing, translation, direct to a language) add their own swap point
later.

## Files

- `tools/crispasr_viability.py`: the suite.
- `tools/crispasr_viability.json`: the candidates. Keys:

| Key | Meaning |
|---|---|
| `area` | `A1`, `A2`; `A` for the baseline; `check` for the self-check entries |
| `kind` | `wj` (WhisperJAV's own engine, no swap), `crisp-separator`, `crisp-s2s` (audio to audio), `crisp-vad`, and the self-check kinds `wj-enhancer-swap`, `wj-segmenter-swap` |
| `enhancer`, `segmenter` | for `wj`: the names the pipeline is built with |
| `repo`, `file`, `revision` | the model file on Hugging Face, pinned by commit; `path` for WebRTC (no file); `same_file_as` to share another entry's file |
| `backend`, `stem` | the CrispASR backend to open, and the separator's stem kept (`vocals`) |
| `twin_segmenter`, `settings`, `model_threshold` | A2: whose settings WhisperJAV resolves (`twin` uses them; `own` uses CrispASR's defaults with TEN's grouping); `model_threshold` leaves the threshold to the model |
| `twin` | the run a CrispASR candidate is compared with: the same model through WhisperJAV's own engine |
| `one_file_per_process` | one child process per clip (for ZipEnhancer's known fault F2) |
| `simplicity` | facts for the simplicity measure: what it would replace, the packages no longer needed, why |

- `tests/test_crispasr_viability.py`: the helpers on made-up data, without CrispASR or a model.

Needs: the `crispasr` package, ffmpeg and ffprobe, huggingface_hub, nvidia-smi for the GPU figures; `subeval/` and
`tools/crispasr_model_test.py` (its scoring, Hugging Face and GPU helpers, used unchanged);
`tools/vad_groundtruth_analyser/` (its frame metrics, unchanged).

## How a stage is swapped

All in a child process, so a crash of CrispASR is recorded and the parent's results survive (exit status 5).

- The chain is `ParakeetCrispASRPipeline`, built as the Ensemble pass worker builds a CrispASR pass that names only its
  pipeline, speech enhancer and speech segmenter (`ensemble/pass_worker.py`, `crispasr_kwargs`); `--debug`'s per-group record
  is on.
- **A1:** the pipeline gets the candidate's name as its speech enhancer (WhisperJAV then extracts at 48 kHz, as for any
  enhancer); `whisperjav.modules.speech_enhancement.create_enhancer_direct` answers that one name with the suite's
  `CrispEnhancer`. WhisperJAV's own `enhance_scenes()` runs it scene by scene, resamples to 16 kHz, and uses a scene's
  original audio when its enhancement fails, as always.
  - Separators: the mono scene duplicated into 44.1 kHz stereo (as WhisperJAV's htdemucs does), `Session.separate`, the
    `vocals` stem averaged to mono.
  - Audio-to-audio: `Session.speech_to_speech` on 16 kHz mono; the output rate is read from the lengths (48 kHz).
  - `--track vad`: enhance for VAD (the enhanced audio to the speech segmenter, the original to the ASR model).
- **A2:** the pipeline is built with the twin's name, so WhisperJAV resolves the twin's settings for Parakeet-CrispASR
  (`crispasr/pass_settings.py: segmenter_settings`), then the backend is renamed to the candidate, and
  `SpeechSegmenterFactory.create` answers that name with the suite's `CrispSegmenter`: `crispasr.vad_slices` finds the speech,
  the suite pads it (start and end pads of the twin; padded spans that would overlap meet in the middle of their gap),
  WhisperJAV's `group_segments()` makes the VAD groups (longest 6 s, the twin's gap), and WhisperJAV's `VadGroupedFramer`
  applies its 0.1 s / 0.2 s rules.
- Measuring only: the time of every speech segmenter call (for all segmenters alike) and the wall-clock window of the
  enhancer, segmenter and ASR steps, for the GPU samples.

## Commands

```bash
python tools/crispasr_viability.py fetch --all --out ROOT          # or: fetch NAME [NAME ...]
python tools/crispasr_viability.py run NAME --out ROOT [--track vad] [--run-name X] [--only A,B] [--clips DIR]
python tools/crispasr_viability.py bench NAME --out ROOT [--seconds 120] [--piece-seconds 30]
python tools/crispasr_viability.py rnnoise --out ROOT
python tools/crispasr_viability.py table --out ROOT
python tools/crispasr_viability.py selfcheck --out ROOT [--clip S01E04.Scene.1] [--crisp crisp-silero,crisp-htdemucs-f16]
```

- **fetch:** downloads into the Hugging Face cache at the pinned commit; checks the size and SHA-256 against Hugging Face's
  listing; writes `ROOT/_fetch/<name>.json`.
- **run:** writes `ROOT/<run>/`: `<clip>.ja.whisperjav.srt`, `raw_subs/` (the per-group record, the raw copy, the
  artifacts), `_scores/<clip>/` (subeval), `events.jsonl`, `run.log`, `result.json`, `result.md`. The run's folder is the
  candidate's name, `_vadtrack` added for `--track vad`; `--run-name` gives another (the baseline's repeat:
  `--run-name baseline-repeat`). Recorded: pooled CER with wrong / missing / extra, missed lines, suspected hallucination
  lines, lines, CER plain; per clip; for every run the VAD groups against the reference lines (frame precision, recall, F1 at
  10 ms; reference subtitle times are reading times, not speech times); the swapped stage's own record (scenes, failures,
  settings).
- **Each component (owner W586):** every step of the chain (model check, audio extraction, scene detection, speech
  enhancer, speech segmenter, ASR, subtitle building, clean-up) is timed per call. Every half second the parent samples the
  card's memory in use (nvidia-smi) and the RAM of the run's process tree (psutil, resident memory of the child and its
  descendants), kept in `samples.json`. Per step: calls, seconds, x real time, GPU memory's peak over the use before the
  run and its largest rise within one call, RAM peak and its largest rise within one call (a negative rise: memory was
  freed). Only samples inside a call's window count, so a step shorter than half a second shows no memory figure. A step's
  first call includes its model's load.
- **bench:** the candidate's stage alone on the first `--seconds` (default 120) of the longest clip on the GPU, and the
  first `--cpu-seconds` (default 30) on the CPU, in pieces of `--piece-seconds` (scene-sized): A1 on the GPU and the CPU
  (CrispASR: `use_gpu`; WhisperJAV: `device`); A2 CrispASR as built (`vad_slices` takes no GPU switch), WhisperJAV's VAD on
  the CPU. Records the stage's seconds and x real time, the GPU memory's peak over the use before, and the child's RAM
  peak and its rise over the process before the model. Writes `ROOT/_bench/<name>/`.
- **rnnoise:** calls `enhance_audio_rnnoise` on one second of silence and records the answer (`ROOT/_rnnoise.json`).
- **table:** `ROOT/table.md` and `table.json`: every run, the margin (the larger of 0.5 percentage points and twice the
  difference between `baseline` and `baseline-repeat`), the verdict against the baseline and against the twin (better, keeps
  quality, worse; charter 1.6, proposed), the bench figures, the simplicity facts.
- **selfcheck:** (1) the suite's baseline against WhisperJAV's own command line run on one clip, final subtitles text and
  times; (2) WhisperJAV's TEN through the A2 swap point against the baseline, and WhisperJAV's ffmpeg-dsp through the A1 swap
  point against ffmpeg-dsp without it (ffmpeg-dsp because htdemucs's output is not repeatable, known fault F4); (3) one CrispASR
  candidate per area end to end. Writes `ROOT/_selfcheck/selfcheck.md`.

## Limits

- The installed CrispASR build only (here the CUDA build of 0.8.50); Vulkan and Apple are not measured; the CPU figures come
  from the CUDA build with the GPU off.
- One fixed chain, Parakeet-CrispASR.
- The Netflix clips are drama, not JAV.
- Known faults of WhisperJAV's own enhancers (speech enhancer guidance): BS-RoFormer cannot start (F1), ZipEnhancer cannot
  start on a run's second file (F2, worked around with one file per process), htdemucs is not repeatable (F4).

## Exit status

0 done; 1 a self-check failed; 2 a setup or input problem; 5 a child process ended abnormally or a file failed.
