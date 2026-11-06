# GGUF models through CrispASR: test tool

## Purpose

Tests any GGUF model that CrispASR can run, before any WhisperJAV code is written for it (owner W568, W569,
2026-11-06). It shows what CrispASR can run, downloads and checks a model file, opens it, checks that the object
CrispASR returns is consistent, measures the model on clips against reference subtitles, prepares files for a hand
test, and writes one report with a draft of what WhisperJAV would need if the owner decides to take the model.

The same tool serves three purposes:
- **a candidate model**: run every stage, compare with the baseline model;
- **a new CrispASR release**: run `catalog --compare` and the open, payload and measure stages again for the shipped
  models, and compare with the saved results before the lock is refreshed;
- **research on settings**: the measure stage with other VAD groups, settings or switches.

It changes nothing in WhisperJAV. The stages catalog to payload import nothing from WhisperJAV (catalog reads
`whisperjav/crispasr/manifest.py` as text). The measure stage imports, unchanged and read only (owner W569), WhisperJAV's
speech segmenter (only with `--vad`), `crispasr/engine.py` `segments_from_binding`, the line rules of
`crispasr/lines.py` and the clean-up of `crispasr/cleanup.py`, and records the commit it ran against.

`tools/vibevoice_crispasr_ref.py` stays as it was, so its measurements of 2026-11-06 can be repeated.

## Files

- `tools/crispasr_model_test.py`: the tool.
- `tools/crispasr_models.json`: the models, one entry per GGUF file:

| Key | Meaning |
|---|---|
| `label` | The model's name in messages, with its quantization. |
| `repo`, `file`, `revision` | The Hugging Face repository and file. With `revision`, fetch takes that commit; without it, the repository's current commit (recorded in `fetch.json`). |
| `backend` | What CrispASR must report when it opens the file (`Session.backend`). |
| `licence`, `base_model`, `note`, `shipped_as` | For the report. |
| `row` | The WhisperJAV pipeline row the measure stage builds subtitles with: `line_rule` (`tokens`, `group`, `segments`; `crispasr/lines.py`), `onsets`, `cut_at_pauses`, `cleanup` (`srt` = `cleanup.SRT_STAGES`, `moss` = `MOSS_STAGES`, `none`). As `EngineRow` in `pipelines/crispasr_pipeline.py`. |
| `baseline` | The model a candidate is compared with in the report. |
| `settings`, `env` | Session settings (`max_new_tokens`, `context`, `temperature` ...) and CrispASR switches (`CRISPASR_...`) for every stage. |

  The file also holds the list of registry names that are not speech-to-text models, for the catalog. It was sorted
  out by hand from their names; a name not on it is shown as a candidate.
- `tests/test_crispasr_model_test.py`: the checks and helpers on made-up results, without CrispASR.

Needs: the `crispasr` package, numpy, ffmpeg, huggingface_hub; nvidia-smi for the GPU figures; psutil for the
leftover processes; the `subeval/` folder for scoring.

## Layout of the results

Every command takes `--out ROOT` and writes under `ROOT/<model>/`: `fetch.json`, `open/`, `payload/`,
`measure/` (or `--name`), `report.md`. The catalog writes `ROOT/catalog_crispasr-<version>.json`.

The model work of open, payload and measure runs in a child process (`_worker`), which writes one JSON line per
event as it goes (`events.jsonl`: the opening, each call's result as CrispASR returned it, the closing). If CrispASR
crashes, the child process's exit code and the call it stopped at are recorded (`run.json`, the stage's report), and
everything written before is kept. Exit status 5.

## Stages

### 1. catalog

```bash
python tools/crispasr_model_test.py catalog --out ROOT [--compare ROOT/catalog_crispasr-0.8.50.json]
```

The installed CrispASR's backends (`Session.available_backends()`), every name in its registry
(`crispasr.list_known_models()`, `registry_lookup()`: file, address, size), whether it is a speech-to-text model by
the data file's list, and whether WhisperJAV ships that file (by file name, from `manifest.py`). With `--compare` an
earlier catalog: the names and backends that are new or gone. The registry gives one file per name; other
quantizations of the same model are not listed.

### 2. fetch

```bash
python tools/crispasr_model_test.py fetch MODEL --out ROOT
```

Downloads the file into the Hugging Face cache at the entry's `revision` (or the current commit) and checks its
size and SHA-256 against Hugging Face's listing. It reads the licence from the repository's card and the registry
entry for the file. `fetch.json` holds the facts and a draft `ModelFile(...)` line for `manifest.py`.

### 3. open

```bash
python tools/crispasr_model_test.py open --model MODEL --out ROOT [--no-cpu]
```

Opens the model on the GPU and then on the CPU, twice each, with one 5 s call of silence after the first opening.
It records the backend (and whether it is the expected one), the time to open, and GPU memory: before, after
opening, the peak (sampled every half second), and after each closing. If memory does not come back to where it was
after closing, that is a leak. GPU memory is the card's use over its use just before, never total minus free
(`docs/plans/design/2026-11-06-vibevoice-crispasr-first-measurements.md` §8).

### 4. payload: the returned object

The stage's name is document B's "payload disassembly": taking the model's result apart to see which field carries
usable times. (Document B's "contract" is the code interface between the chain and the ASR call; this stage tests what
comes through it.)

```bash
python tools/crispasr_model_test.py payload --model MODEL --input CLIP --reference CLIP.srt --out ROOT [--noise-at S] [--no-cpu]
```

The fixed test audio, from one clip:
- 0.2 s, 0.5 s and 5 s of silence (zeros);
- 3 s and 6 s of speech starting 0.2 s before each of the reference's first three lines (VAD-group sized);
- 30, 60 and 120 s from the clip's start;
- with `--noise-at`, 8 s of music or noise.

Three child processes:
- **A:** every probe twice, then the speech probes once more with no language.
- **B:** the long probes with CrispASR's own cutting off (`CRISPASR_SESSION_AUTOCHUNK=0`).
- **C:** the speech probes and the 30 s probe on the CPU.

The checks on each call:
- each segment's start and end within the call's audio (0.05 s allowed);
- segments in order, with their largest overlap;
- words and tokens: how many, how many with a start, how many with an end, any outside the audio, any going back in
  time within a segment;
- the tokens' text, rebuilt from their bytes, against the segment's text;
- token confidence missing (the binding's -1);
- word confidence all 1.0 (the binding's value when it has none);
- speaker labels and no-speech probabilities.

The checks between calls:
- text on silence;
- the same call twice;
- no language against `ja`;
- cutting off against the default;
- CPU against GPU (text equal, similarity, largest token-start difference).

From the times it proposes the pipeline row by WhisperJAV's line rules:

| What comes back | Proposed row |
|---|---|
| Tokens with starts and ends | `tokens`, as Parakeet, Kotoba and Whisper |
| Starts only | `tokens` with `onsets`, as ReazonSpeech |
| Only words timed | `tokens`, read from the words (check the subtitles) |
| Several timed segments a call | `segments`, as MOSS |
| Nothing usable | `group` with the `wav2vec2-ja` aligner, as Cohere |

The results are in `payload.md` and `payload.json`. A failed check is a candidate for a report to CrispASR
(`triage/upstream.md`, after the owner's word).

### 5. measure

```bash
python tools/crispasr_model_test.py measure --model MODEL --clips test_media/Ground_Truths/Netflix \
    --groups-from ROOT/pipeline_parakeet [--compare-pipeline ROOT/pipeline_parakeet] --out ROOT
```

The clips: `--clips FOLDER`, each media file with `<stem>.ja.srt` or `<stem>.srt` beside it (a `.wav` beside a
`.mkv` of the same name is skipped); or `--input FILE [--reference SRT]`; `--only A,B` keeps the clips whose names
contain A or B.

The audio of each call, in order of preference:
1. `--groups-from DIR`: the VAD groups of a WhisperJAV run with `--debug`
   (`DIR/raw_subs/<clip>.<pipeline>.groups.jsonl`, pipeline `--groups-pipeline`, default `parakeet-crispasr`). This is
   the same audio the pipeline sent, for a fair comparison of models.
2. `--vad ten --vad-like parakeet-crispasr`: WhisperJAV's speech segmenter with that pipeline's settings, over the
   whole clip.
3. Pieces of at most `--piece-seconds` (30).

The model is loaded once for all clips. Per clip:
- `pieces.jsonl`: every call as returned;
- `model_times.srt`: each segment at the model's own times;
- `group_edges.srt`: each group's text at its edges;
- `<clip>.whisperjav.srt`: the subtitles WhisperJAV's unchanged code builds with the row. That is `lines.build()` per
  group, the scenes in order, then `cleanup.SrtCleanup` with the row's stages, whose trace files go to `raw_subs/`.

All three are scored with subeval against the reference. The pooled CER is all edits over all reference characters.

**The self-check (`--compare-pipeline DIR`).** It compares each group's text, and every segment, word and token
start and end, with what the pipeline received (its `--debug` record). It compares the tool's subtitles with the
pipeline's final SRT (`--pipeline-srt`, default `{stem}.ja.pass1.srt`): the counts, then each subtitle's text and
times in order ("SRT identical" only when all agree). When the tool reproduces the shipped pipeline on the same audio, its figures for a new model
can be trusted. Small differences can come from the audio: the pipeline cuts each group from its scene's WAV, the
tool from the whole clip, at times rounded to 1 ms.

`measure.md` and `measure.json` hold the pooled figures, each clip, the speed (audio seconds over ASR seconds), GPU
memory, and the self-check table.

### 6. listen

```bash
python tools/crispasr_model_test.py listen --model MODEL --baseline BASELINE --out ROOT
```

Copies each clip's `<clip>.whisperjav.srt` of the model and of the baseline into `ROOT/<model>/measure/listen/` as
`<clip>.<model>.srt`, to load in PotPlayer beside the clip. `review.md` lists, per clip, the reference lines that
either model missed (time, text, missed or got by each) and the model's suspected hallucination lines (subeval's
definitions).

### 7. report

```bash
python tools/crispasr_model_test.py report --model MODEL --out ROOT
```

`ROOT/<model>/report.md`:
- each stage's facts;
- the pooled figures of the model and its baseline;
- a draft `ModelFile(...)` entry, with `vram_mib` taken from the measuring stage's peak over the use before;
- a draft `EngineRow(...)` from the proposed row.

Whether a model goes into WhisperJAV is the owner's decision. After it, the draft is the checklist for the code
change: manifest, pipeline class, speech segmenter default, GUI, CLI, `--help`, documents, tests.

## Other commands

- `gpu`: the card's memory and the programs on it.
- `models`: the data file's models and whether they are downloaded.

## Not yet tested (as of 2026-11-06)

These parts of the tool have not run on a real case:
- recording a crash of the child process (exit 5);
- a model CrispASR cannot open (exit 4);
- the GPU memory check stopping a run (exit 3);
- `catalog --compare`;
- `measure --vad` (WhisperJAV's speech segmenter over the whole clip as one scene, without scene detection: not the
  pipeline's groups);
- `measure` with pieces (`--piece-seconds`), and `--input` without a reference.

For a like-for-like test on new media, first run the WhisperJAV pipeline the model is compared with, with `--debug`,
then measure with `--groups-from` its output folder.

## Limits

- The installed CrispASR build only (here the CUDA build; the Vulkan and CPU builds are not tested, apart from the
  CPU run of the CUDA build in the open and payload stages).
- One GPU.
- The Netflix clips are drama, not JAV.
- The registry lists one file per name.

## Exit status

- 0: done.
- 2: a setup or input problem.
- 3: not enough GPU memory (nothing loaded).
- 4: CrispASR could not open the model.
- 5: the child process ended abnormally.
