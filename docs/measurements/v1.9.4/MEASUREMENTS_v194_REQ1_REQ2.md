# 1.9.4 measurements — REQ1 (audio integrity) and REQ2 (ChronosJAV timing), 2026-10-05

A record of every measurement run on 2026-10-05, so that later work can repeat it and compare, and so that release
notes or replies can quote numbers from one place. Facts and numbers only; choices are listed in §5 as open.
All runs were made in this session; commands, scripts and data locations are in §4.

Code state for the runs: branch `sprint-20260927` at `f4a4c43`. The REQ2 runs used the product code unchanged. The
REQ1 checks used `whisperjav/modules/audio_integrity.py` (new, written that day) called from scratch scripts; the
pipelines were not involved.

---

## 1. REQ1 — audio integrity

### 1.1 The damage found: hidden holes in the audio track (Film A, 2 h 2 min)
Five audio packets (at 35:52, 47:52, 59:52, 71:52, 83:52) state a length of 2.027 s while holding 0.021 s of sound
(AAC, 1024 samples at 48 kHz). The next packet starts where the stated length ends, so the timeline has a 2.006 s
hole at each. A player waits through it; WhisperJAV's extraction (`ffmpeg -i <film> -vn -acodec pcm_s16le -ar 16000
-ac 1`) joins the sound across it.

Where audio cut from the film by timestamp (`ffmpeg -ss t`, 8 s) is found in the extracted WAV (normalised cross-
correlation; peak 0.975–1.000 everywhere):

| Film time t | Today's extraction: offset | With `-af aresample=async=1:first_pts=0`: offset |
|---|---|---|
| 0:30 | 0.000 s | 0.000 s |
| 16:40 | 0.000 s | 0.000 s |
| 35:00 | 0.000 s | 0.000 s |
| 36:40 | −2.005 s | 0.000 s |
| 48:20 | −4.011 s | 0.000 s |
| 61:40 | −6.016 s | 0.000 s |
| 73:20 | −8.021 s | 0.000 s |
| 85:00 | −10.027 s | 0.000 s |
| 108:20 | −10.027 s | 0.000 s |
| 121:40 | −10.027 s | 0.000 s |

WAV length: today 7335.829 s, with the filter 7345.856 s (= the audio track). Extraction time (file in memory):
5.5 s today, 5.1 s with the filter. Negative offset = the audio sits earlier in the WAV than in the film, so every
subtitle made from it is early by that amount.

### 1.1b Second damaged film, and the cost of the filter (owner asked before adopting it)
Film B (five 2.0 s holes): today 0.000 / −2.005 / −4.011 / −5.995 / −8.000 / −10.005 / −10.005 s at 10:00, 38:20,
50:00, 61:40, 73:20, 86:40, 116:40; with the filter 0.000 s at all seven. WAV 7388.288 → 7398.293 s (= the track).

Extraction time, same film read from memory, three alternating runs each (one warm-up read first):
| Film | Today | With the filter |
|---|---|---|
| Film B (2 h, damaged) | 5.05 5.02 5.23 (median 5.05 s) | 4.91 4.98 4.91 (median 4.91 s) |
| Film G (2 h, clean) | 4.61 4.75 4.30 (median 4.61 s) | 4.64 4.52 4.39 (median 4.52 s) |
| MKV film 2 (MKV) | 3.25 3.27 3.17 (median 3.25 s) | 3.20 3.17 3.30 (median 3.20 s) |
Same single FFmpeg pass; no video re-encoding; no extra file. **Adopted for 1.9.4 (owner, 2026-10-05).**

After adoption, through WhisperJAV's own `AudioExtractor`: Film A 0.000 s at 7 points (WAV 7345.856 s); the 3-minute
damaged clip 0.000 s at 5 points. End to end, Qwen3-ASR on that clip: lines before the 0:53 hole unchanged (6 lines,
0.0 s); lines after it 1.81 / 1.98 / 2.02 s later than before the fill (one more at 0.88 s, re-segmented). Balanced's
text after the hole changed between runs, so its lines could not be paired; no conclusion from that comparison.

### 1.2 The gap-filling filter on films without damage
| File | Container | WAV length today / with filter | Offsets at 4 points (today and with filter) |
|---|---|---|---|
| Film E | MP4 | 8107.038 / 8107.038 s | all 0.000 s |
| Film F | MP4 | 8058.137 / 8058.137 s | all 0.000 s |
| MKV film 1.au.college.1979 | MKV | 3929.728 / 3929.728 s | all 0.000 s |

### 1.3 Audio that starts later than the video (a reference clip remuxed with `-itsoffset`, stream copy)
| Remux | Stream starts | Today | With filter |
|---|---|---|---|
| audio +1.5 s | video 0.000, audio 1.511 | every point 1.511 s early | 0.000 s |
| audio −1.5 s | video 1.489, audio 0.000 (the muxer renormalised) | matches the player's timeline | same as today |
Not tested: a file whose audio timestamps are negative inside the file.

### 1.4 Library sweep — the owner's media library, every video file over 300 MB (43 files)
42 run through the check (extraction as today + `check_audio_integrity`); Film F only through §1.2.
| Result | Files |
|---|---|
| Damaged: five hidden 2.0 s holes, extracted audio 10.0 s short | **Film A** (holes 35:53–1:23:53, every 12 min), **Film C** (37:07–1:25:07, 12 min), **Film B** (37:17–1:25:17, 12 min), **Film D** (47:43–1:51:43, every 16 min) |
| Does not open (FFmpeg "Invalid data found when processing input"; WhisperJAV fails it at extraction today) | two files |
| No findings (36) | 36 files: MP4, MKV, AVI and TS; JAV and other films (titles kept off this record) |
Four damaged out of 40 that open and were checked, all four in the same form. Films I and L, from the same label as Films A and C, are clean.
The four files checked before the "discard packets" rule (§1.6) was added — Film E, Film G, Film I, Film H —
had no findings without it; the rule can only remove packets a player never plays.

### 1.5 Cost of the check on 2-hour films
| What | Time |
|---|---|
| Stream list (ffprobe header) | 0.3–0.5 s |
| Packet list, read right after the extraction, film still in memory | 0.3–6.5 s (most 2–5 s) |
| Packet list, read after the extraction, film no longer in memory | 46.4 s (Film D), 61.8 s (Film K), 86.9 s (Film J) |
| Packet list read alongside the extraction (Film I, 5.39 GB) | both finished at 36.9 s: no added time |
| Packet list, cold, before any extraction (Film A, 5.86 GB) | 97 s (2.8 s on a second read) |
| Extraction itself, cold, 2-hour films | 23–98 s (disk-bound; F: reads about 60 MB/s) |
The wired version reads the packet list alongside the extraction.

### 1.6 Other check results
- 60 s stream-copied cut of Film I with 40 × 200 bytes overwritten: 182 FFmpeg `[error]` lines, extracted audio 1.8 s
  short of the track → reported. FFmpeg prints two `[error]` lines per damaged packet.
- The clean 60 s cut itself: before the rule, reported 1.0 s short (47 lead-in packets flagged `D`, discard, at
  negative times); with discard-flagged packets ignored, no findings.
- MKV files report no per-stream duration (`N/A`); the length checks fall back to the packet timeline and the
  container duration.
- 3-minute stream-copied cut of Film A from 35:00 (end-to-end test file): one 2.0 s hole at 0:53, extracted audio
  2.0 s short → reported.

### 1.7 End-to-end runs of the wired check (3-minute stream-copied clips of Film A; 2026-10-05)
Damaged clip: one 2.0 s hidden hole at 0:53. Clean clip: from 16:40, no hole.
| Run | Exit | Result in the run summary and manifest |
|---|---|---|
| Qwen3-ASR (`--mode qwen`), damaged | 0 | `suspect`: "audio integrity: 1 gap(s) in the audio track, 2.0 s in all (2.0 s at 0:00:53); 17 cue(s) …" |
| Qwen3-ASR, clean | 0 | console "Audio check: no problems found (1.8 s)." |
| Balanced, damaged | 0 | `suspect` with the same reason |
| Fidelity `--model turbo`, damaged | 0 | `suspect` with the same reason |
| Ensemble (anime-whisper pass 1 + Qwen3-ASR pass 2), damaged | 0 | `suspect`, reason given once ("pass 1: audio integrity: …") |
| Balanced `--async-processing`, damaged | 0 | `suspect` with the same reason |
| Qwen3-ASR `--fail-on suspect`, damaged | 1 | `failed`: "stopped before transcription: audio integrity: …", after 9 s, no traceback |
| Ensemble `--fail-on suspect`, damaged | 1 | both passes stop; `failed` with the same reason |
Two faults found by these runs and fixed the same day, then re-run: the console line also printed a "extracted audio
2.0 s shorter" clause that the gap already explains (now left out when the gaps account for it); the ensemble's run
summary said only "ensemble pass failed" for a stopped file (now names the reason). The ensemble prints the console
warning once per pass.

---

## 2. REQ2 — ChronosJAV subtitle timing

### 2.1 Set-up
- **Clips:** 7 Netflix scenes ("The Naked Director", S01E03–S02E05) in `test_media/Ground_Truths/Netflix`, with
  Japanese ground-truth subtitles: 305 lines in all. These are TV-drama scenes, not JAV.
- **Runs:** the command the GUI Ensemble tab builds for a pass-1-only run, default timestamp mode (no aligner),
  semantic scene detection, WhisperSeg speech segmenter (`measure-scripts/measure/run_chronosjav_reference_runs.py`):
  - **Qwen3-ASR:** `Qwen/Qwen3-ASR-1.7B`, sensitivity balanced (hysteresis decoder, onset 0.25, end level 0.10,
    longest segment 5 s, pads 100/100 ms, group gap 0.3 s, group cap 3.0 s).
  - **anime-whisper:** `litagin/anime-whisper`, sensitivity aggressive = the GUI Ensemble default (offline decoder,
    threshold 0.15, grow floor 0.05, gap merge 350 ms, longest segment 4 s, pads 0/30 ms, group gap 0.2 s, cap 2.0 s).
- **One setting changed per run**, through `--pass1-qwen-params` (pads, grow floor, gap merge, longest segment) or
  `--pass1-params` (end threshold); the recorder confirmed the value reached the segmenter.
- **Scoring** (`whisperjav.bench.timing`): each ground-truth line is paired with at most one of our lines (time
  overlap + text similarity). Start error = our start − theirs, end error = our end − theirs; positive = late.
  "Common" columns compare two runs on the ground-truth lines both matched. **Character error rate (CER):** the whole
  text of each clip against the ground truth's, after NFKC and keeping letters and digits only, sound notes in
  brackets removed; pooled over the clips. Netflix text is not word-for-word, so CER is for comparing runs only.
- **Repeatability:** baseline run 1 and run 2 gave identical results for both models (same lines matched, same
  errors to the millisecond). One run per setting is therefore used.

### 2.2 Baseline (the 1.9.3 defaults, before the 1.9.4 change)
| | Our lines | GT lines matched | Start error, median | End error, median | End, average lean | Ends within 0.5 s | CER |
|---|---|---|---|---|---|---|---|
| Qwen3-ASR | 220 | 169 | 0.342 s | 1.062 s | +1.114 s | 50 | 0.390 |
| anime-whisper (GUI default) | 227 | 188 | 0.178 s | 0.708 s | +0.589 s | 75 | 0.392 |

Why ends are late (matched lines with end error over 0.5 s):
| | Our line also covers the next GT line | Late tail only | Our median line length | GT median line length |
|---|---|---|---|---|
| Qwen3-ASR | 75 (median 2.20 s late) | 33 (median 0.93 s) | 3.66 s | 2.25 s |
| anime-whisper | 49 (median 1.61 s late) | 47 (median 0.84 s) | 3.04 s | 2.25 s |

What the segmenter produced (recorded during the runs):
| | Segments | Groups | Segment length median / p90 | Segments cut by the length limit, not at a pause |
|---|---|---|---|---|
| Qwen3-ASR | 266 | 262 | 3.44 / 4.92 s | 127 (48 %) |
| anime-whisper | 385 | 379 | 2.62 / 3.88 s | 258 (67 %) |
Almost every segment is its own line; the group settings rarely join two. The speech probability stays above the end
level through long stretches of these scenes, so the length limit, not a pause, ends most lines.

### 2.3 One setting at a time
A = on the ground-truth lines both runs matched (baseline → setting). B = each run on its own.

| Model | Setting (default) | Common / lost | Start median A | End median A | End lean A | Lines | Matched | Ends ≤ 0.5 s | Median line | CER | Cut by limit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-ASR | baseline | 169 / 0 | 0.342 | 1.062 | +1.114 | 220 | 169 | 50 | 3.58 | 0.390 | 127 / 266 |
| | end threshold 0.20 (0.10) | 161 / 8 | 0.388→0.316 | 1.062→1.005 | +1.086→+1.053 | 217 | 166 | 52 | 3.52 | 0.387 | 128 / 280 |
| | end pad 0 ms (100) | 168 / 1 | 0.361→0.361 | 1.066→1.061 | +1.126→+1.081 | 221 | 168 | 51 | 3.54 | 0.389 | 126 / 266 |
| | longest segment 4 s (5) | 156 / 13 | 0.328→0.311 | 1.057→0.677 | +1.073→+0.652 | 245 | 186 | 79 | 3.28 | 0.394 | 170 / 309 |
| | longest segment 3 s | 159 / 10 | 0.328→0.348 | 1.052→0.507 | +1.064→+0.132 | 307 | 209 | 105 | 2.60 | 0.435 | 250 / 389 |
| | **longest segment 2.5 s** | 160 / 9 | 0.361→0.327 | 0.999→0.489 | +1.057→−0.130 | 333 | 213 | **113** | 2.32 | 0.428 | 309 / 448 |
| | longest segment 2 s | 159 / 10 | 0.328→0.508 | 1.062→0.555 | +1.082→−0.156 | 358 | 212 | 99 | 1.98 | 0.431 | 415 / 554 |
| anime-whisper | baseline | 188 / 0 | 0.178 | 0.708 | +0.589 | 227 | 188 | 75 | 2.96 | 0.392 | 258 / 385 |
| | end pad 0 ms (30) | 187 / 1 | 0.176→0.176 | 0.695→0.675 | +0.588→+0.584 | 227 | 187 | 76 | 2.94 | 0.391 | 258 / 385 |
| | grow floor 0.10 (0.05) | 184 / 4 | 0.181→0.178 | 0.695→0.606 | +0.584→+0.570 | 222 | 184 | 82 | 2.95 | 0.394 | 234 / 382 |
| | grow floor 0.15 | 184 / 4 | 0.181→0.178 | 0.695→0.596 | +0.584→+0.562 | 221 | 184 | 82 | 2.96 | 0.392 | 230 / 377 |
| | gap merge 150 ms (350) | 188 / 0 | 0.178→0.181 | 0.708→0.694 | +0.589→+0.580 | 228 | 188 | 77 | 2.93 | 0.391 | 255 / 395 |
| | **longest segment 3 s** (4) | 180 / 8 | 0.175→0.191 | 0.708→0.428 | +0.594→+0.183 | 260 | 194 | 106 | 2.45 | 0.399 | 364 / 491 |
| | longest segment 2.5 s | 179 / 9 | 0.170→0.332 | 0.695→0.437 | +0.582→−0.056 | 278 | 199 | 113 | 2.11 | 0.419 | 452 / 579 |
| | longest segment 2 s | 179 / 9 | 0.170→0.352 | 0.695→0.476 | +0.582→−0.360 | 309 | 200 | 106 | 1.66 | 0.453 | 560 / 687 |
| | **longest segment 3 s + grow floor 0.15** | 178 / 10 | 0.175→0.196 | 0.708→0.364 | +0.593→+0.151 | 258 | 192 | 112 | 2.42 | 0.392 | 332 / 479 |
Times in seconds. "Lost" = ground-truth lines the baseline matched and the setting did not.

CER split (characters against 3,115 ground-truth characters):
| Run | Our characters | Substituted | Missing | Extra |
|---|---|---|---|---|
| Qwen3-ASR baseline | 2788 | 709 (0.228) | 416 (0.134) | 89 (0.029) |
| Qwen3-ASR longest segment 3 s | 2818 | 782 (0.251) | 435 (0.140) | 138 (0.044) |
| anime-whisper baseline | 2522 | 493 (0.158) | 660 (0.212) | 67 (0.022) |

### 2.4 What the tables show (observations, not choices)
- The longest-segment limit is the only setting that moves timing substantially; end level, grow floor, gap merge and
  end pad move the end median by at most 0.10 s.
- Qwen3-ASR: 2.5 s doubles the ends within 0.5 s (50 → 113) and halves the end median (1.06 → 0.49 s) with no start
  loss; every limit of 3 s or less raises CER from 0.390 to 0.428–0.435. 4 s is the step with no text cost: ends
  within 0.5 s 50 → 79, end median 1.06 → 0.68 s, CER 0.390 → 0.394, start median 0.33 → 0.31 s.
- anime-whisper: 3 s raises ends within 0.5 s from 75 to 106 and lowers the end median from 0.71 to 0.43 s, for
  +0.007 CER and +0.016 s start median; 2.5 s and 2 s double the start error and raise CER more. 3 s together with
  grow floor 0.15 is the best measured: ends within 0.5 s 75 → 112, end median 0.71 → 0.36 s, CER unchanged (0.392),
  start median 0.18 → 0.20 s, 10 ground-truth lines lost against 8 for 3 s alone.
- Shorter limits write more lines (Qwen3-ASR 220 → 333 at 2.5 s against 305 in the ground truth) and match more
  ground-truth lines (169 → 213).

### 2.5 Limits of these measurements
- Seven drama scenes, 305 lines; no JAV scene with moans and action sound (the users' complaint, S2/S4 in the research
  report) is in the set. The direction of the effect on JAV audio is not measured.
- The Netflix lines follow subtitling rules (out-time at least 0.5 s after speech, lines linked with short gaps); part
  of every "error" is that convention.
- CER against non-verbatim text; only the differences between runs mean anything.
- One run per setting, justified by the baseline's identical repeat; not repeated for the settings themselves.
- One sensitivity per model was measured at first (Qwen3-ASR balanced, anime-whisper aggressive). The
  adversary review (2026-10-05) noted anime-whisper conservative and balanced run the hysteresis decoder and were
  unmeasured; the confirming runs (§2.6) then measured them: timing gains as on aggressive, with a text cost
  (CER +4 % and +5.5 % relative). Qwen3-ASR conservative (6 → 4 s) remains unmeasured.

### 2.6 Character-error trace (owner's request 2026-10-05: keep the trace so a preset can be tuned or reverted)

Every REQ2 run, with the settings it actually used (read from the segmenter recorder, not from notes) and its text
measures, pooled over the 7 clips. CER = (substituted + missing + extra characters) / 3,115 ground-truth characters
after normalisation (sound notes in brackets removed, NFKC, letters and digits only). Netflix text is not
word-for-word, so CER compares runs; it is not an absolute accuracy. "End level None" = WhisperSeg's derived end
level (threshold − 0.15). "End error med" here counts each run's own matched lines; §2.3 compares runs on the lines
both matched. Produced by `measure-scripts/cer_trace.py`.

| Run | Model | Sensitivity | Longest segment (s) | Decoder | Grow floor | Gap merge (ms) | End level | End pad (ms) | Group cap / gap (s) | Lines | Matched | Ends ≤ 0.5 s | End error med (s) | CER | Substituted | Missing | Extra | Our chars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3_whisperseg | Qwen3-ASR | balanced | 5.0 | hysteresis | - | - | None | 100 | 3.0 / 0.3 | 220 | 169 | 50 | 1.06 | 0.390 | 0.228 | 0.134 | 0.029 | 2788 |
| q_neg020 | Qwen3-ASR | balanced | 5.0 | hysteresis | - | - | 0.2 | 100 | 3.0 / 0.3 | 217 | 166 | 52 | 1.02 | 0.387 | 0.212 | 0.144 | 0.031 | 2764 |
| q_endpad0 | Qwen3-ASR | balanced | 5.0 | hysteresis | - | - | None | 0 | 3.0 / 0.3 | 221 | 168 | 51 | 1.06 | 0.389 | 0.228 | 0.132 | 0.029 | 2795 |
| q_maxseg4 | Qwen3-ASR | balanced | 4.0 | hysteresis | - | - | None | 100 | 3.0 / 0.3 | 245 | 186 | 79 | 0.72 | 0.394 | 0.194 | 0.159 | 0.041 | 2747 |
| q_maxseg3 | Qwen3-ASR | balanced | 3.0 | hysteresis | - | - | None | 100 | 3.0 / 0.3 | 307 | 209 | 105 | 0.50 | 0.435 | 0.251 | 0.140 | 0.044 | 2818 |
| q_maxseg25 | Qwen3-ASR | balanced | 2.5 | hysteresis | - | - | None | 100 | 3.0 / 0.3 | 333 | 213 | 113 | 0.47 | 0.428 | 0.224 | 0.156 | 0.048 | 2776 |
| q_maxseg2 | Qwen3-ASR | balanced | 2.0 | hysteresis | - | - | None | 100 | 3.0 / 0.3 | 358 | 212 | 99 | 0.54 | 0.431 | 0.231 | 0.160 | 0.039 | 2738 |
| anime_whisperseg | anime-whisper | aggressive | 4.0 | offline | 0.05 | 350 | - | 30 | 2.0 / 0.2 | 227 | 188 | 75 | 0.71 | 0.392 | 0.158 | 0.212 | 0.022 | 2522 |
| a_endpad0 | anime-whisper | aggressive | 4.0 | offline | 0.05 | 350 | - | 0 | 2.0 / 0.2 | 227 | 187 | 76 | 0.68 | 0.391 | 0.158 | 0.212 | 0.022 | 2523 |
| a_floor010 | anime-whisper | aggressive | 4.0 | offline | 0.1 | 350 | - | 30 | 2.0 / 0.2 | 222 | 184 | 82 | 0.61 | 0.394 | 0.158 | 0.216 | 0.021 | 2508 |
| a_floor015 | anime-whisper | aggressive | 4.0 | offline | 0.15 | 350 | - | 30 | 2.0 / 0.2 | 221 | 184 | 82 | 0.60 | 0.392 | 0.154 | 0.218 | 0.021 | 2501 |
| a_gap150 | anime-whisper | aggressive | 4.0 | offline | 0.05 | 150 | - | 30 | 2.0 / 0.2 | 228 | 188 | 77 | 0.69 | 0.391 | 0.159 | 0.211 | 0.022 | 2525 |
| a_maxseg3 | anime-whisper | aggressive | 3.0 | offline | 0.05 | 350 | - | 30 | 2.0 / 0.2 | 260 | 194 | 106 | 0.43 | 0.399 | 0.144 | 0.241 | 0.013 | 2406 |
| a_maxseg3_floor015 | anime-whisper | aggressive | 3.0 | offline | 0.15 | 350 | - | 30 | 2.0 / 0.2 | 258 | 192 | 112 | 0.37 | 0.392 | 0.138 | 0.241 | 0.013 | 2405 |
| a_maxseg25 | anime-whisper | aggressive | 2.5 | offline | 0.05 | 350 | - | 30 | 2.0 / 0.2 | 278 | 199 | 113 | 0.42 | 0.419 | 0.141 | 0.262 | 0.017 | 2352 |
| a_maxseg2 | anime-whisper | aggressive | 2.0 | offline | 0.05 | 350 | - | 30 | 2.0 / 0.2 | 309 | 200 | 106 | 0.47 | 0.453 | 0.141 | 0.291 | 0.021 | 2274 |
| a_cons_old6 | anime-whisper | conservative | 6.0 | hysteresis | - | - | None | 100 | 3.0 / 0.3 | 169 | 142 | 46 | 1.14 | 0.397 | 0.163 | 0.223 | 0.011 | 2453 |
| a_cons_new3 | anime-whisper | conservative | 3.0 | hysteresis | - | - | None | 100 | 3.0 / 0.3 | 236 | 193 | 97 | 0.50 | 0.413 | 0.140 | 0.257 | 0.016 | 2366 |
| a_bal_old5 | anime-whisper | balanced | 5.0 | hysteresis | - | - | None | 50 | 2.5 / 0.25 | 177 | 154 | 51 | 1.13 | 0.400 | 0.175 | 0.208 | 0.017 | 2518 |
| a_bal_new3 | anime-whisper | balanced | 3.0 | hysteresis | - | - | None | 50 | 2.5 / 0.25 | 233 | 190 | 95 | 0.50 | 0.422 | 0.146 | 0.263 | 0.013 | 2339 |
| a_cons_4 | anime-whisper | conservative | 4.0 | hysteresis | - | - | None | 100 | 3.0 / 0.3 | 201 | 168 | 74 | 0.60 | 0.399 | 0.161 | 0.219 | 0.019 | 2492 |
| a_bal_4 | anime-whisper | balanced | 4.0 | hysteresis | - | - | None | 50 | 2.5 / 0.25 | 200 | 170 | 74 | 0.61 | 0.402 | 0.165 | 0.217 | 0.020 | 2500 |

The rows `a_cons_*` and `a_bal_*` are the confirming runs for anime-whisper conservative and balanced (hysteresis
decoder; 1.9.3 value passed explicitly vs the 1.9.4 default). On the ground-truth lines both runs matched
(`measure-scripts/compare_pairs.py`):

| Base -> candidate | Common lines | Lost | Start median (s) | End median (s) | End mean (s) |
|---|---|---|---|---|---|
| a_cons_old6 -> a_cons_new3 | 135 | 7 | 0.225 -> 0.178 | 1.030 -> 0.495 | +1.361 -> +0.079 |
| a_bal_old5 -> a_bal_new3 | 147 | 7 | 0.282 -> 0.219 | 1.094 -> 0.507 | +1.065 -> +0.091 |
| a_cons_old6 -> a_cons_4 | 135 | 7 | 0.225 -> 0.195 | 1.131 -> 0.663 | +1.390 -> +0.609 |
| a_bal_old5 -> a_bal_4 | 143 | 11 | 0.300 -> 0.252 | 1.052 -> 0.576 | +1.069 -> +0.586 |

**What the trace shows**
- A shorter longest segment always moves text from "substituted" to "missing": fewer wrong characters, more text not
  written at all. The total CER can stay flat while text is lost (anime-whisper aggressive 3 s + grow floor 0.15:
  CER 0.392 → 0.392, missing 0.212 → 0.241, 4.6 % fewer characters written).
- At the 1.9.4 defaults, relative CER change against 1.9.3: Qwen3-ASR balanced +1 % (0.390 → 0.394); anime-whisper
  aggressive 0 %; anime-whisper conservative **+4 %** (0.397 → 0.413); anime-whisper balanced **+5.5 %**
  (0.400 → 0.422). The +10 % figures belong to Qwen3-ASR at 3 s and below (0.428–0.435), which were not adopted.
- For the same rows, timing: lines ending within 0.5 s of the ground truth roughly double (46 → 97, 51 → 95) and the
  median end error halves (about 1.1 → 0.5 s).

- The middle value, 4 s, for anime-whisper conservative and balanced (owner's request, same day): text as
  in 1.9.3 (CER 0.397 → 0.399 and 0.400 → 0.402; characters written +1.6 % and −0.7 %), timing about two thirds of
  the 3 s gain (ends within 0.5 s 46 → 74 and 51 → 74; median end error about 1.1 → 0.6 s). Aggressive at 4 s with
  grow floor 0.15 (`a_floor015`): CER 0.392, characters −0.8 %, ends within 0.5 s 75 → 82. Qwen3-ASR at 4 s: CER
  +1 %. So 4 s is, for every measured model and sensitivity, the step with no material text cost.

**Where each value lives (to tune or revert)**
| Model, sensitivity | Longest segment now (1.9.3) | Where | Other measured lever |
|---|---|---|---|
| Qwen3-ASR, all | 4.0 s (6 / 5 / 4 s) | `whisperjav/config/qwen3_whisperseg_vad.py`, `QWEN3_WHISPERSEG_DEFAULTS["max_speech_duration_s"]` | — |
| anime-whisper conservative | 3.0 s (6 s) | `whisperjav/config/anime_whisper_vad.py`, row "conservative", `max_speech_duration_s` | — |
| anime-whisper balanced | 3.0 s (5 s) | same file, row "balanced" | — |
| anime-whisper aggressive | 3.0 s (4 s) | same file, row "aggressive" | `grow_floor` 0.15 (0.05) |
Removing a row's `max_speech_duration_s` returns it to the WhisperSeg YAML preset
(`whisperjav/config/v4/ecosystems/tools/whisperseg-speech-segmentation.yaml`: 6 / 5 / 4 s), which other pipelines
and Cohere also use; change the YAML only with that in mind. All values reach every entry point through
`whisperjav/config/chronosjav_vad.py`. Not measured: values between the tested ones (e.g. 4 s for anime-whisper
conservative and balanced).

### 2.7 Patterns in the errors (owner's questions, 2026-10-05; `measure-scripts/error_patterns.py`)

Six runs: Qwen3-ASR 5 s / 4 s, anime-whisper aggressive 4 s / 3 s + grow floor 0.15, anime-whisper balanced
5 s / 3 s. Method: each clip's whole ground-truth text is aligned with our text character by character; every
missing character is traced to its ground-truth line, its place in the line, and an estimated time (in proportion
to the line's span). Patterns, not exact counts: the ground truth is not word-for-word.

- **Missing text is mostly whole lines.** 59–69 % of missing characters are in ground-truth lines we did not match
  at all. Going shorter adds loss both ways (anime aggressive 4 → 3 s: +91 missing, 57 in whole lines, 34 inside
  matched lines; anime balanced 5 → 3 s: +169, 72 and 97).
- **What goes missing.** The most frequent single gaps are interjections and short replies (う, あ, っ, ん, はい,
  うん, ああ, あっ) and particles (は, を, も, の) — about one tenth of the missing characters; part of that is the
  deliberate はい / うん / あ line filter. The rest is ordinary speech.
- **Start or end of a line.** Missing characters are slightly over-represented in the first quarter of a line
  (25–28 % against 23.6 % of all characters; more so at shorter limits), and for Qwen3-ASR also the last quarter
  (27–28 %). They do not cluster at our line edges: within 0.3 s of an edge, 15–20 % of missing characters
  against 13–22 % of all characters. Shorter windows make the models drop speech inside the window; the cut
  points do not slice words.
- **Timing by line length.** Short ground-truth lines (1–2 s) have the worst ends (median 0.9–1.6 s at 1.9.3
  values: our line runs into the next one), and gain most from a shorter limit (2–3 s lines: 0.66–0.97 →
  0.27–0.45 s at 3 s). Long ones (3–5 s) have the worst starts (about 0.5 s) because we split them; for anime
  aggressive 3 s makes these starts worse (0.48 → 0.95 s). Lines cut by the length limit end far worse at 1.9.3
  values (Qwen3-ASR 1.54 vs 0.78 s; anime balanced 1.68 vs 0.66 s); at shorter limits the gap mostly closes.
- **End error vs text errors.** Per-line Spearman correlation between end error and missing or wrong characters:
  −0.03 to +0.14 in all six runs — essentially none. For anime-whisper, lines ending more than 0.5 s late carry
  somewhat more missing text (0.12–0.17 vs 0.09–0.13 per character). Timing and text loss are largely separate.

### 2.8 Line and character accounting (owner's questions W1–W5, 2026-10-05; `measure-scripts/line_accounting.py`)

Base: all 305 ground-truth lines and 3,115 characters. Each clip's whole text is aligned character by character;
every ground-truth character is exactly one of correct / wrong / missing. A **missing line** has none of its
characters in our text. "Produced lines" are all other ground-truth lines; their characters are split into correct /
wrong / missing. Positions: first quarter, middle half, last quarter of the line (expected 24 % / 52 % / 24 %).

| Run | Our lines | Missing lines | Chars in missing lines | Produced lines: correct | wrong | missing | Extra chars | Missing in first / last quarter | Wrong in first / last quarter | Expected first / last |
|---|---|---|---|---|---|---|---|---|---|---|
| a_cons_old6 | 169 | 15.4% | 8.0% | 66.7% | 17.7% | 15.6% | 34 | 30% / 25% | 28% / 20% | 24% / 24% |
| a_cons_4 | 201 | 17.4% | 9.0% | 68.2% | 17.6% | 14.2% | 60 | 28% / 26% | 29% / 20% | 24% / 24% |
| a_cons_new3 | 236 | 18.0% | 10.5% | 67.4% | 15.7% | 16.9% | 51 | 30% / 25% | 29% / 20% | 24% / 24% |
| a_bal_old5 | 177 | 14.8% | 6.4% | 65.9% | 18.7% | 15.4% | 52 | 29% / 25% | 27% / 22% | 24% / 24% |
| a_bal_4 | 200 | 17.7% | 9.0% | 67.8% | 18.2% | 14.0% | 61 | 28% / 24% | 29% / 23% | 24% / 24% |
| a_bal_new3 | 233 | 18.7% | 10.9% | 66.4% | 16.4% | 17.2% | 42 | 32% / 24% | 28% / 22% | 24% / 24% |
| anime_whisperseg | 227 | 14.8% | 7.0% | 67.7% | 17.0% | 15.2% | 67 | 29% / 22% | 29% / 22% | 24% / 24% |
| a_floor015 | 221 | 15.4% | 7.2% | 67.7% | 16.6% | 15.7% | 64 | 29% / 24% | 29% / 21% | 24% / 24% |
| a_maxseg3_floor015 | 258 | 17.7% | 9.0% | 68.2% | 15.2% | 16.6% | 41 | 31% / 24% | 30% / 20% | 24% / 24% |
| qwen3_whisperseg | 220 | 11.1% | 5.1% | 67.3% | 24.0% | 8.7% | 89 | 27% / 33% | 26% / 22% | 24% / 24% |
| q_maxseg4 | 245 | 13.1% | 6.5% | 69.2% | 20.7% | 10.0% | 128 | 29% / 30% | 26% / 21% | 24% / 24% |
| q_maxseg3 | 307 | 10.2% | 4.0% | 63.5% | 26.1% | 10.4% | 138 | 28% / 30% | 26% / 22% | 24% / 24% |

- 5–6 s keeps the most text (fewest missing lines, about 15 %) but packs it into long lines that span two ground-
  truth lines (late ends). 3 s misses more whole lines (18–19 %) and more characters inside produced lines (about
  17 %), leaning to line starts. 4 s misses about as many whole lines as 3 s but writes the most complete lines
  (anime-whisper: missing 14 %, correct 68 %); its total error rate equals 1.9.3's. Qwen3-ASR at 3 s misses fewer
  lines but gets more characters wrong (26 %): it guesses where anime-whisper drops.
- Inside produced lines, missing and wrong characters lean to the first quarter (28–32 % and 26–30 %); the line's
  very first character is wrong about twice as often as an average character.

### 2.9 Why line starts go wrong (character-accuracy step 1; `measure-scripts/onset_analysis.py`)

Runs at the option-B values (anime-whisper conservative / balanced / aggressive at 4 s, Qwen3-ASR at 4 s). Each
matched pair is aligned on its own; the audio window the model received is rebuilt from the recording.
- The effect is real and it is "wrong", not "missing": first character wrong 24–33 % against about 14 % for all
  characters; first-character missing about average. Line ends are fine (last character wrong 8–11 %).
- About a third of first-character substitutions are spellings of names (お→美, ケ→賢, み→ミ, ニ→二). Others look
  like a lost first sound: 君→ミ (ki-mi → mi), 黒→ロ (ku-ro → ro), 俺→レ (o-re → re), 弁→ン (be-n → n).
- Lines that start at a forced split: first character wrong or missing 47–54 %, against 28–39 % after a pause.
- Window start minus ground-truth start: more than 0.3 s late → first character wrong or missing 83–91 % (24–33
  lines per run); 0–0.3 s early → 16–32 % (best); more than 0.3 s early → 39–42 %.
- Of the late-window lines, 55–65 % start at a forced split; in 6–11 per run the opening is at the end of our
  previous line (the split moved it); in 16–21 per run (about one matched line in ten) it is not found: not heard.
- Ends: a window ending more than 0.3 s before the ground-truth end loses the last character 56–72 % (16–21 lines).
- Step 2 (owner's go, same day) tests start pads, added silence and the forced-split rule (`run_step2.py`,
  `experiment_hooks/`).

### 2.10 Character-accuracy step 2: start pads, added silence, split rule (`measure-scripts/run_step2.py`, `score_step2.py`)

Each experiment against its option-B baseline (same 7 clips). Pads are set through the normal settings; added
silence and the split rule through `measure-scripts/experiment_hooks/` (an experiment-only module; no product code
changed). Every run's recording confirms the intended settings. "Late lines" = matched lines whose audio window
starts more than 0.3 s after the ground-truth line. Start / end = medians on the lines both runs matched.

| Model | Run | CER | Characters written | Missing lines | First char wrong | Late lines | Ends within 0.5 s | Start (s) | End (s) |
|---|---|---|---|---|---|---|---|---|---|
| anime aggressive | baseline (pad 0 / 30 ms) | 0.392 | 2,501 | 15.4 % | 31.5 % | 29 | 82 | 0.175 | 0.596 |
| | start pad 100 ms | 0.390 | 2,518 | 15.1 % | 31.9 % | 28 | 82 | 0.178 | 0.596 |
| | start pad 200 ms | 0.391 | 2,514 | 14.8 % | 31.9 % | 29 | 82 | 0.176 | 0.596 |
| | start pad 300 ms | 0.388 | 2,521 | 14.4 % | 32.3 % | 29 | 81 | 0.176 | 0.596 |
| | 200 ms silence before | 0.387 | 2,472 | 15.7 % | 28.3 % | 29 | 82 | 0.172 | 0.579 |
| | 200 ms silence before and after | identical to "before" (Whisper pads its input to 30 s with silence anyway) | | | | | | | |
| Qwen3-ASR | baseline (pad 100 / 100 ms) | 0.394 | 2,747 | 13.1 % | 24.2 % | 33 | 79 | 0.291 | 0.695 |
| | start pad 0 ms | 0.394 | 2,742 | 10.8 % | 25.5 % | 34 | 79 | 0.286 | 0.695 |
| | start pad 200 ms | 0.387 | 2,724 | 12.1 % | 25.1 % | 33 | 81 | 0.314 (was 0.300) | 0.695 |
| | start pad 300 ms | 0.389 | 2,763 | 11.1 % | 25.7 % | 33 | 80 | 0.384 (was 0.311) | 0.695 |
| | 200 ms silence before | 0.409 | 2,848 | 11.5 % | 23.0 % | 32 | 79 | 0.286 | 0.677 |
| | 200 ms silence before and after | 0.401 | 2,822 | 11.1 % | 25.5 % | 33 | 79 | 0.290 | 0.680 |
| | split: dip search from 30 %, any dip | 0.418 | 2,790 | 11.8 % | 21.8 % | 30 | 102 | 0.303 (was 0.290) | 0.477 (was 0.655) |
| anime balanced | baseline (4 s) | 0.402 | 2,500 | 17.7 % | 32.9 % | 26 | 74 | 0.232 | 0.614 |
| | split: dip search from 30 %, any dip | 0.405 | 2,513 | 17.4 % | 32.1 % | 27 | 101 | 0.238 | 0.450 |

**What it shows**
- Start pads: small text gains (CER −0.5 to −1.8 % relative; fewer missing lines), but the late lines do not move
  (28–34 per run in every variant). They mostly start at a forced split, with no gap before them, so a start pad
  cannot reach back into the previous segment. For Qwen3-ASR a 300 ms start pad also makes starts later on screen
  (0.311 → 0.384 s), because today the pad is part of the displayed time.
- Added silence: anime-whisper gains a little (CER −1.3 %, first character wrong 31.5 → 28.3 %, fewer extra
  characters); silence after the window changes nothing. Qwen3-ASR gets worse (CER +3.8 %; more text written, more
  of it wrong).
- Split rule: for anime-whisper balanced it gives most of the 3 s timing gain (ends within 0.5 s 74 → 101, end
  error 0.61 → 0.45 s) at almost no text cost (CER +0.7 %, characters +0.5 %) — where 3 s cost +5.5 %. For
  Qwen3-ASR the same rule gains timing (79 → 102) but costs text (CER +6 %): shorter windows again.
- Not fixed by any of these: the 16–21 lines per run whose opening is never heard. A window that may reach back
  into the previous segment for the model only (Clar3: the model's window separate from the displayed time) is the
  remaining candidate; it needs a code change.
- Effects of 1–2 % relative are measured on 7 clips; runs are repeatable (identical on repeat), but small
  differences may not hold on other material.

---

## 3. Numbers for release notes or replies (drafts of facts; wording and use are the owner's decision)
- "In one library, 4 of the 40 video files checked had the same hidden fault: five 2-second holes in the audio track. In such a
  file every subtitle after the first hole came out early, by up to 10 seconds."
- "The audio check reads the file alongside the audio extraction; on 2-hour films it added no measurable time."
- "Of 36 undamaged files (MP4, MKV, AVI, TS) none was reported."
- Timing figures: the 1.9.4 defaults chosen from them are in §5; nothing is released yet.

## 4. How to repeat (`measure-scripts/` = docs/measurements/v1.9.4/scripts; others are in the repository)
| What | Command / script | Data |
|---|---|---|
| Source-audio probe with timings | `measure-scripts/measure/probe_source_audio_faults.py <film> --work <dir> --ffmpeg-dir <bin>` | — |
| Offset of extracted audio vs film time | `measure-scripts/offset_check.py`, `fix_test.py <film> <dir> t1 t2 …` | — |
| Late-starting audio | `measure-scripts/offset_start_test.py` | — |
| Check on real files / sweep | `measure-scripts/integrity_real.py <work> <files…>` | sweep outputs: `measure-scripts/sweep_out.txt`, `sweep2_out.txt` |
| Packet-list cost after / alongside extraction | `measure-scripts/scan_cost.py <film> after|parallel <dir>` | — |
| REQ2 baseline (2 runs each) | `measure-scripts/measure/run_chronosjav_reference_runs.py --python <WJ python> --media-dir test_media/Ground_Truths/Netflix --out <measure folder>/req2_baseline --runs 2 --pairs qwen3_whisperseg anime_whisperseg --record` | `<measure folder>\req2_baseline` |
| REQ2 settings | `measure-scripts/run_levers.py <measure folder>/req2_levers [names]` (each run's exact command is in its `command.json`) | `<measure folder>\req2_levers` |
| Scoring | `python -m whisperjav.bench.timing_cli --ref-dir test_media/Ground_Truths/Netflix --base <run>`; `measure-scripts/measure/score_reference_runs.py`; `measure-scripts/score_levers.py`, `late_ends.py` | — |
The scripts in `measure-scripts/` were written for one machine. Repository paths are found from the script's own location and ffmpeg / ffprobe / python are taken from PATH; the media library, the working folder and the measurement output folder are placeholders (`<media library>`, `<work folder>`, `<measure folder>`) to fill in before re-running. Films are labelled A, B, …; the label-to-title key is kept off the repository.

## 5. Open for the owner
1. ~~The extraction fix~~ — adopted 2026-10-05 (§1.1b).
2. ~~REQ2 defaults~~ — DECIDED 2026-10-05 (owner) and committed (77d2078 … a943545): with WhisperSeg, longest
   segment Qwen3-ASR 4.0 s and anime-whisper 3.0 s at every sensitivity; anime-whisper aggressive grow floor
   0.15; group cap and group gap unchanged (grouping joined only 4–6 pairs per run, so it was not the cause of
   joined sentences). Verified on all 14 entry-point runs with the segmenter recorder (CLI and ensemble, both
   models, three sensitivities, plus a user value of 5.5 s that wins). Same day: the GUI Customize dialog's
   decoder default fixed (it showed "offline" where hysteresis runs), and the qwen_guide.html segmenter entry
   rewritten.
3. ~~Where the measurements live~~ — docs/measurements/v1.9.4 (owner, 2026-10-05).
