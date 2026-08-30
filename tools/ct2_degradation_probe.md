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

## Running it

```bash
python tools/ct2_degradation_probe.py \
    --audio  /path/to/problem_file.wav \
    --reference /path/to/known_good_clip.wav \
    --model large-v2 \
    --out ct2_probe.jsonl
```

If you installed WhisperJAV with the Windows standalone installer, use the Python
that came with it:

```
%LOCALAPPDATA%\WhisperJAV\python.exe tools\ct2_degradation_probe.py --audio ... --reference ...
```

Match the model and decode settings to the run that failed for you — `--model`,
`--beam-size`, `--best-of`, `--temperature`, `--language`, and `--vad-filter` if
you use faster-whisper's internal VAD.

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
has to wonder whether the two configurations really matched.

| Result | What it means | What happens next |
|---|---|---|
| Degrades on `bare` | Upstream, in faster-whisper / CTranslate2 | We report it there with this reproduction, and ship mitigation on our side |
| Clean on `bare`, degrades on `whisperjav` | Ours | We bisect our own parameters — temperature fallback, repetition penalty, VAD segment sizing |
| Clean on both | Not reproducible on demand | We fall back to `--asr-telemetry` from real failing runs |

This table was written **before** any results came in, deliberately, so the
conclusion cannot be fitted to whatever we happen to see.

Start with `--engine bare`. It is the decisive one, and it needs nothing except
faster-whisper.

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
- Imports nothing from WhisperJAV unless `--engine whisperjav` is used, so it
  runs against any version — including one older than the branch it ships on.
- Needs `faster-whisper`, `numpy` and `soundfile`; `librosa` only if your audio
  is not already 16 kHz; `psutil` and `torch` only for the memory columns, and it
  degrades quietly without them.
- If the reference clip fails to transcribe even on a freshly loaded model, the
  tool stops and asks for a different clip rather than reporting a false result.
