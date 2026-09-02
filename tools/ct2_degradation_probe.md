# CTranslate2 degradation probe — issue #394

A small, self-contained tool for the failure where the recognizer stops producing
output part-way through a long file and never recovers, while the run still
reports success.

It exists to answer **one question**: does that failure belong to WhisperJAV, or
to faster-whisper / CTranslate2 underneath it? Everything else about #394 depends
on the answer, and nobody has established it yet.

Credit for the underlying idea goes to **@AlanZ-Git**, who found that feeding a
known-good clip to the *already-loaded* model returns nothing, while a fresh
process transcribes the same clip correctly. This tool generalises that so the
same experiment can be run with and without WhisperJAV in the path.

---

## What you need

- A **problem file** — audio that has actually shown the failure.
- A **reference clip** — 5–15 seconds of clear speech that you have already
  confirmed transcribes correctly, ideally from the same kind of material.

You supply the reference clip yourself. Nothing is uploaded; everything stays on
your machine, and the tool only ever reads your files.

> **Fixed 2026-09-02 — please re-run if you tried this before.** The
> `--engine whisperjav` arm could not start: it called the ASR class with a
> constructor signature that has never existed, and failed with
> `TypeError: FasterWhisperProASR.__init__() got an unexpected keyword argument
> 'model_name'`. It now builds its configuration through WhisperJAV's own
> resolver, so it cannot drift from the pipeline again. Two further defects were
> fixed at the same time: the tool crashed with `UnicodeEncodeError` when
> printing a Japanese transcript on a Windows console using a legacy code page,
> and the `bare` arm ran with its own decode defaults rather than the ones
> WhisperJAV ships. Thank you to the people who reported the first one — twice.

## Running it

```bash
python tools/ct2_degradation_probe.py \
    --audio  /path/to/problem_file.wav \
    --reference /path/to/known_good_clip.wav \
    --engine bare \
    --out ct2_probe_bare.jsonl

python tools/ct2_degradation_probe.py \
    --audio  /path/to/problem_file.wav \
    --reference /path/to/known_good_clip.wav \
    --engine whisperjav \
    --out ct2_probe_whisperjav.jsonl
```

Please run **both**. One arm on its own cannot answer the question — see the
table below.

If you installed WhisperJAV with the Windows standalone installer, use the Python
that came with it:

```
%LOCALAPPDATA%\WhisperJAV\python.exe tools\ct2_degradation_probe.py --audio ... --reference ...
```

### Decode settings — you should not need to set any

By default (`--profile shipped`) **both arms resolve WhisperJAV's own balanced
configuration from your installed copy** and run with it: the same model, beam
size, temperature ladder, repetition penalty, word timestamps and VAD parameters
the pipeline uses. That is the point — the comparison is then about the engine,
not about settings.

Adjust only if your failing run differed: `--sensitivity` (conservative /
balanced / aggressive) and `--speech-segmenter` (default `faster-whisper`, the
v1.9.0 balanced native VAD; pass `silero-v3.1` if you use the external one).
`--model` overrides the resolved model name.

`--profile raw` restores the tool's own defaults — VAD off, beam 5, the full
temperature ladder, no repetition penalty, no word timestamps. **Results
collected before 2026-09-02 were all effectively `raw`**, which is a
configuration WhisperJAV never runs, so a clean `bare` result from those runs
says less than it appears to.

## What it does

One model instance is loaded and kept alive for the whole run. Your reference
clip is transcribed once to establish a baseline. Then, repeatedly: a chunk of
your problem audio is transcribed, and the reference clip is transcribed again.

When the reference stops matching its own baseline, the loaded instance has
degraded, and the tool reports the chunk at which that happened.

## The two arms, and why both matter

```
--engine bare        faster-whisper directly, no WhisperJAV code involved   (default)
--engine whisperjav  the same experiment through WhisperJAV's ASR module
```

Same audio, same model, same decode parameters, same probe. **The difference
between the two runs is the answer.** Running both from one harness means nobody
has to wonder whether the two configurations really matched — and under
`--profile shipped` both are pinned to the configuration WhisperJAV actually
runs, so that question is settled by construction rather than by care.

| Result | What it means | What happens next |
|---|---|---|
| Degrades on `bare` | Upstream, in faster-whisper / CTranslate2 | We report it there with this reproduction, and ship mitigation on our side |
| Clean on `bare`, degrades on `whisperjav` | Ours | We bisect our own parameters — temperature fallback, repetition penalty, VAD segment sizing |
| Clean on both | Not reproducible on demand | We fall back to `--asr-telemetry` from real failing runs |

This table was written **before** any results came in, deliberately, so the
conclusion cannot be fitted to whatever we happen to see.

Note the second row: **a clean `bare` result on its own matches no row in this
table.** It is only informative next to the `whisperjav` arm on the same audio
and the same reference clip. If you can run only one, run `bare` — it needs
nothing but faster-whisper — but please come back for the other when you can.

## Output

A JSONL file, one record per chunk:

| Field | Meaning |
|---|---|
| `work_wall_s`, `work_rtf` | How long that chunk took. A slowdown *preceding* the failure is one of the strongest clues we have — @daoran9 measured the median call going from 1.05 s to 31.65 s beforehand |
| `work_max_temperature`, `work_fallback_segments` | faster-whisper retries at a higher temperature when a decode trips its guards, so these show whether decoder fallback is involved |
| `probe_similarity` | How closely the reference clip still matches its own baseline. This is the degradation signal |
| `probe_degraded` | Whether that chunk crossed the line |
| `cuda_allocated_mb`, `cuda_reserved_mb`, `rss_mb` | Memory. Monotonic growth up to the failure would be close to conclusive |

**Please attach the JSONL to [issue #394](https://github.com/meizhong986/WhisperJAV/issues/394).**
The `probe_text` field contains only the transcription of the short reference
clip you chose, so review it before posting if that matters to you.

## Notes

- Read-only with respect to your installation.
- Imports WhisperJAV's *config* layer for `--profile shipped` (no GPU stack) and
  its ASR module only for `--engine whisperjav`. With `--profile raw --engine
  bare` it imports nothing from WhisperJAV at all and runs against any version.
- Every JSONL now opens with a `"record": "config"` line naming the engine,
  profile, model, segmenter and every decode option in force, so a result file
  can never be read out of context.
- Needs `faster-whisper`, `numpy` and `soundfile`; `librosa` only if your audio
  is not already 16 kHz; `psutil` and `torch` only for the memory columns, and it
  degrades quietly without them.
- If the reference clip fails to transcribe even on a freshly loaded model, the
  tool stops and asks for a different clip rather than reporting a false result.
