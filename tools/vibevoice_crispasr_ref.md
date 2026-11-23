# VibeVoice through CrispASR: reference tool

## Purpose

Runs Microsoft's VibeVoice-ASR models through CrispASR's Python binding (the `crispasr` package WhisperJAV uses) on a
media file, with chosen settings, and measures what comes back: the text, the model's own times and speaker labels,
the speed and the GPU memory. Each run can be scored with subeval against a reference SRT. Several variants of
settings run one after another with the model loaded once.

It is a reference way of calling VibeVoice, written to try its models and settings before a WhisperJAV pipeline for it
is designed (owner W549, W550, 2026-11-06). It changes nothing in WhisperJAV and imports nothing from it, except
WhisperJAV's own speech segmenter when `--vad` is given (owner W554, W555).

## Script

`tools/vibevoice_crispasr_ref.py`. Needs: the `crispasr` package, numpy, ffmpeg (on PATH or `--ffmpeg`), nvidia-smi
for the GPU figures, huggingface_hub for `models` and `download`, psutil for finding leftover processes. Scoring needs
the `subeval/` folder of this repository.

## Commands

```bash
python tools/vibevoice_crispasr_ref.py gpu
python tools/vibevoice_crispasr_ref.py models
python tools/vibevoice_crispasr_ref.py download q8_0
python tools/vibevoice_crispasr_ref.py run --model q8_0 --input CLIP.mkv --out OUT --piece-seconds 60 --reference CLIP.ja.srt
python tools/vibevoice_crispasr_ref.py run --model q8_0 --input CLIP.mkv --out OUT --variants variants.json --crisp-cutting off
python tools/vibevoice_crispasr_ref.py run --model q8_0 --input CLIP.mkv --out OUT --segments groups.jsonl
```

- `gpu`: the card's memory (in all, in use, free), the programs nvidia-smi lists on it, and any leftover Python or
  WhisperJAV process on it.
- `models`: the known VibeVoice files, their size on Hugging Face, and whether they are downloaded.
- `download NAME`: downloads one file into the Hugging Face cache at the repository's current commit and checks its
  size and SHA-256 against Hugging Face's listing. Nothing is downloaded by `run`.
- `run`: loads the model once, runs each variant, closes the model, and writes the results.

## Models

| Short name | Hugging Face file | Size | Model |
|---|---|---|---|
| `q4_k` | `cstr/vibevoice-asr-GGUF` `vibevoice-asr-q4_k.gguf` | 4.81 GB | VibeVoice-ASR 7B, Q4_K |
| `q8_0` | `cstr/vibevoice-asr-GGUF` `vibevoice-asr-q8_0.gguf` | 9.45 GB | VibeVoice-ASR 7B, Q8_0 |
| `f16` | `cstr/vibevoice-asr-GGUF` `vibevoice-asr-f16.gguf` | 16.66 GB | VibeVoice-ASR 7B, F16 |
| `streaming-1.5b-q4_k` | `cstr/vibevoice-asr-streaming-1.5b-GGUF` `vibevoice-asr-streaming-1.5b-q4_k.gguf` | 1.86 GB | VibeVoice-ASR-Streaming 1.5B |
| `bitnet-tq2` | `cstr/vibevoice-asr-bitnet-GGUF` `vibevoice-asr-bitnet-tq2.gguf` | 1.62 GB | VibeVoice-ASR-BitNet |

Sizes as listed on Hugging Face on 2026-11-06. Any other file: a local `.gguf` path, or `hf:<owner>/<repo>/<file>`
(downloaded first with `download hf:...`). GGUF files made for other programs (hipEngine, audio.cpp, vibevoice.cpp)
are not expected to open in CrispASR.

## How the audio is cut

One call to the binding (`Session.transcribe`) per piece:
- default: the whole audio window (`--start`, `--duration`) in one call;
- `--piece-seconds N`: pieces of at most N seconds, each cut at the quietest 100 ms of its last quarter;
- `--segments FILE`: one call per `{start, end}` entry (seconds of the media file), `.jsonl` or `.json`, for example
  the VAD groups of a WhisperJAV `--debug` run.

**With a VAD (`--vad firered-vad`; owner W554, W555).** WhisperJAV's own speech segmenter finds the speech, and each
VAD group (or, with `--vad-unit segments`, each speech segment) is one call. It runs as the CrispASR pipelines run it:
settings from `whisperjav/crispasr/pass_settings.py` `segmenter_settings()` for the pipeline named by `--vad-like`,
groups from `whisperjav/modules/subtitle_pipeline/framers/vad_grouped.py` `VadGroupedFramer`; a group under 0.2 s is
not sent (`modules/model_audio.py`). It runs over each scene: the pieces of `--piece-seconds`, the `--segments`, or
the whole audio window. The VAD runs before VibeVoice loads and is released; FireRedVAD runs on the CPU with all three
settings. The groups are saved as `vad_groups.jsonl` in the variant's folder (usable again with `--segments`).

| `--vad-like` | Longest VAD group | New group at a silence over | Sensitivity | Pads (start / end) |
|---|---|---|---|---|
| `cohere-crispasr` (default, W555) | 3.0 s | 0.3 s | balanced | 100 / 100 ms |
| `moss-crispasr` | 240 s | 5.0 s | balanced | 50 / 150 ms |
| `parakeet-crispasr` | 6.0 s | 1.0 s | aggressive | 50 / 100 ms |

(FireRedVAD values as `segmenter_settings()` gave them on 2026-11-06.) `--max-group-seconds`, `--group-gap-seconds`,
`--vad-sensitivity` and `--vad-threshold` change single values. With `--variants`, these options apply to every variant
that does not set its own keys (`vad`, `vad_like`, `vad_sensitivity`, `vad_threshold`, `max_group_seconds`,
`group_gap_seconds`, `vad_unit`). A single speech segment longer than the longest group stays one call (its longest is
the preset's `max_speech_duration_s` plus the pads: 4.18 s with the cohere-crispasr settings).

**CrispASR also cuts, on its own.** CrispASR 0.8.50 cuts a call's audio longer than 30 s into pieces of at most 30 s at
quiet points and decodes each piece on its own (for every backend but parakeet, reazonspeech and index-echo). So a
60 s piece reaches VibeVoice as two pieces unless `--crisp-cutting off` is given (`CRISPASR_SESSION_AUTOCHUNK=0`).
`--crisp-cutting N` sets CrispASR's window to N seconds instead. This is set once per run, before CrispASR loads.

The audio is extracted as WhisperJAV extracts it (ffmpeg, `aresample=async=1`, 16 kHz mono, 16-bit) and passed at
16 kHz; CrispASR converts it to the 24 kHz VibeVoice takes (linear interpolation, CrispASR 0.8.50
`crispasr_c_api.cpp`, the vibevoice branch of the session's transcribe).

## Settings

A single variant takes its settings from the options; `--variants FILE` takes a JSON list of objects, each with a
`name` and any of the keys below (and `piece_seconds`, `segments` or nothing for the whole audio):

```json
[
 {"name": "pieces_60s", "piece_seconds": 60},
 {"name": "pieces_60s_context", "piece_seconds": 60, "context": "日本語の会話 (Japanese conversation)", "max_new_tokens": 2048}
]
```

| Key / option | Binding call | Applied to VibeVoice by CrispASR 0.8.50? |
|---|---|---|
| `context` / `--context` | `set_hotwords` | Yes: put into VibeVoice's prompt ("with extra info:"). Cleared by a variant without it. |
| `max_new_tokens` / `--max-new-tokens` | `set_max_new_tokens` | Yes (`vibevoice_set_max_new_tokens`). 0 or absent = CrispASR's default: the larger of 512 and 8 per second of the call's audio (`vibevoice_resolve_max_new_tokens`). |
| `language` / `--language` | the call's `language` | Passed (default `ja`); the vibevoice branch does not read it. |
| `temperature`, `seed`, `do_sample`, `top_k`, `top_p`, `beam_size`, `best_of`, `frequency_penalty`, `repetition_penalty`, `ask` | the binding's setters | Not found applied to VibeVoice's transcription in the 0.8.50 source; VibeVoice decodes greedily. The tool calls them and records "applied (no error)", which means only that the call did not fail. The binding cannot clear them, so a later variant may still carry them (the summary says so). |
| `--no-flash-attn`, `--threads`, `--cpu` | how the model is opened | Yes. |
| `--raw-transcript` | switch `CRISPASR_VIBEVOICE_RAW_TRANSCRIPT=1` | Yes: the model's whole answer (its JSON) comes back as one text, not split into utterances. For seeing exactly what the model wrote. |
| `--encoder-window S` | switch `CRISPASR_VIBEVOICE_ENCODER_CHUNK_SECONDS` | Yes: the audio encoder takes S seconds at once (default 300); bounds its GPU memory; the model still writes the whole call's text. |
| `--gpu-layers N` | switch `CRISPASR_N_GPU_LAYERS` | Yes: N of the 28 language-model layers on the GPU, the rest on the CPU. The binding's own `n_gpu_layers=` argument does not reach VibeVoice (measured: same memory as all layers). |

Which settings reach VibeVoice was read from CrispASR 0.8.50's C source on 2026-11-06; a newer CrispASR may differ.
Compare a variant's output with and without a setting to see whether it acts.

## GPU memory

Before anything is loaded, `run` compares the model file's size plus `--margin-mib` (default 512 MiB) with:
- the budget: the card's memory less `--reserve-mib` (default 2048 MiB, what the system and other programs usually
  need); a model that does not fit the budget is not loaded;
- the memory free now; if it is short, the tool lists what is on the GPU and stops (exit 3).

`--end-leftover` then offers to end leftover Python or WhisperJAV processes on the card (not this tool and not its
parents), asking y/n for each one; it ends nothing without a typed `y`, and never offers any other program. On Windows
nvidia-smi reports each program's memory as N/A, so the tool cannot say which program holds how much; closing other
programs is the user's choice. `--force` loads the model anyway.

During the run the card's used memory is sampled every half second. When it comes within 200 MiB of the card's total,
a warning is printed: the Windows NVIDIA driver may then move memory into ordinary RAM instead of failing, and the run
becomes slow. The tool runs one model at a time and closes it at the end; the figures before, after opening, at the
peak and after closing are in `results.md`. Two CrispASR switches lower the GPU memory: `--encoder-window` (long
calls) and `--gpu-layers` (the model itself; slower).

## Output

In `--out`:
- `results.md`: the run's facts (model file, SHA-256 against Hugging Face, CrispASR version, GPU figures), one table
  of calls (speed, longest call, utterances, characters), and with `--reference` two score tables;
- `results.json`: the same, every number;
- one folder per variant: `pieces.jsonl` (each call's result as the binding returned it: text, start, end, speaker,
  word and token counts, mean token confidence; every token with `--keep-tokens`), `model_times.srt` (one subtitle
  per utterance at the model's own start and end; `--srt-speaker` puts the speaker label in front),
  `piece_edges.srt` (each piece's text as one subtitle at the piece's edges), `summary.json`, and with `--reference`
  `subeval_model_times/` and `subeval_piece_edges/`.

The two scores: `piece_edges.srt` scores the text alone, whatever the model's times; `model_times.srt` scores the
model's own times as well. Early and late: a start or end more than `--timing-threshold` seconds (default 0.2) before
or after the reference line's, counted only where subeval measures that edge. CER and the line counts are defined in
`subeval/METHOD.md`.

What CrispASR returns for VibeVoice (0.8.50): one result entry per utterance the model wrote, with its start and end
(seconds within the call's audio) and a speaker label "(Speaker N) "; speaker numbers are counted per call, not across
calls. Entries CrispASR judges to be non-speech markers are dropped by CrispASR. Words and tokens carry no times.

## Seen in the first runs (2026-11-06, CrispASR 0.8.50, RTX 3060 12 GB)

One clip (The Naked Director S01E04 scene 4, 294.5 s, the Netflix reference); results in a private measurement
record of 2026-11-06 (`step4_report.md`, `tags_removed_report.md`).
- GPU memory over the card's use before loading: q8_0 about 9.0 GB once open, 9.5 GB at its peak with pieces of up to
  30 s; with pieces of 60 s the card was full (12,088 of 12,288 MiB). q4_k: 4.7 GB open, 6.0 GB at its peak with
  30 s pieces.
- Long pieces fail with CrispASR's default encoder window (300 s): the audio encoder's GPU working memory for the
  whole piece cannot be reserved (log: "failed to allocate CUDA0 buffer of size 2962368128 ... vibevoice: at_enc graph
  alloc failed" for a 104 s piece with q8_0; 8.37 GB asked for 294.5 s). With `--encoder-window 30` both q8_0
  (peak 11,982 MiB) and q4_k ran 104 s pieces and the whole 294.5 s clip in one call. On the whole clip both models
  stopped writing at 81 s (9 utterances).
- `--gpu-layers 24` with q8_0: 8,158 MiB for the model once open instead of 9,100 (the binding's `n_gpu_layers=24`
  argument: 9,100, no change); the same text; 0.5 times real time instead of 2.8 on 60 s pieces.
- Sound tags such as `[Environmental Sounds]`, `[Human Sounds]`, `[Music]`, `[Sigh]` came back as utterances
  without a speaker label. CrispASR drops only "silence", "blank_audio", "blank audio", "no speech" and "inaudible"
  (`core/vibevoice_transcript.h`, `is_non_speech_marker`).
- Repetition loops: VibeVoice decodes greedily and sometimes repeats a word or phrase until the token limit
  ("let's go let's go …", "うわ、うわ、…"). The utterance that loops is then never closed in the model's JSON, and
  CrispASR's reader drops an unclosed utterance whole, with any right text before the loop (seen with
  `--raw-transcript`); when it is the first utterance of the call, the raw JSON comes back as the text. CrispASR's
  loop clean-up (`fix_loops`) runs only where CrispASR itself cut audio over 30 s; the frequency and repetition
  penalties are not used by VibeVoice's decoder.
- CrispASR printed twice "transcript is not in time order".
- FireRedVAD with the cohere-crispasr settings on S01E04 Scene 4 (whole clip as one scene): 81 VAD groups, 196.2 s
  of speech, longest 4.18 s, in 2.0 s; the first groups equal those of Cohere-CrispASR's own run of 2026-10-29. q4_k
  on them: CER 5.22, 50 of 68 lines missed (tags kept); on the near-identical October groups 3.42.

## Exit status

0 done; 2 a setup or input problem; 3 not enough GPU memory (nothing loaded); 4 CrispASR could not open the model.
