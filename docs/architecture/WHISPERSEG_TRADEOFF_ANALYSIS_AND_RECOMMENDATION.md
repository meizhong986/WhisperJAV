# WhisperSeg VAD Tradeoff — Analysis and Recommendation

**Companion to:** `WHISPERSEG_TRADEOFF_INFO_PACK.md` (commit `d41571e`)
**Date:** 2026-07-30
**Status:** R1 (wiring) + R2 (aggressive row) IMPLEMENTED per owner instruction
(2026-07-30, "go ahead with your R1, R2, and R3"). R3 verification results in
§10. Awaiting owner E2E quality assessment.

---

## 1. Executive Summary

**The central finding invalidates the premise of the E2E comparison: Fix 1
(`neg_threshold=0.10`) never reached the segmenter.** The value sits in the
anime table, the `WhisperSegSpeechSegmenter.__init__` signature, and the
factory schema — but no entry point lifts it from the table into
`segmenter_config`, and the `SEGMENTER_PARAMS` filter that every qwen
segmenter config passes through does not contain `"neg_threshold"`. Both E2E
runs therefore executed with **dead hysteresis** (derived
`neg_threshold = max(0.15 − 0.15, 0.01) = 0.01`).

Consequences:

1. The observed regression (worse transcription accuracy, fewer captured
   dialogues) was **not** caused by restored hysteresis. It can only have come
   from the two changes that *did* bind: `max_speech` 4.0 → 3.5 and the smart
   force-split. Section 4 gives the mechanism.
2. The intended live-hysteresis behavior has **never been observed E2E**. The
   "core tension" in the info pack (§4.3) is between the old code and a
   configuration that never ran.
3. Separately, synthetic tests against the real state machine show that
   `neg_threshold=0.10` — had it bound — would truncate quiet tails and drop
   quiet replies (Section 5). The intended value was too high anyway.

**Recommendation (Section 6):** wire `neg_threshold` end-to-end; set the
aggressive row to `neg_threshold=0.05`, `min_silence_duration_ms=250`,
`start_pad_ms=100`, `end_pad_ms=150` (threshold 0.15, max_speech 3.5,
grouping values unchanged); re-run E2E with verification that the values
actually bind.

---

## 2. Verified Finding: the neg_threshold Call Chain Is Broken

Chain for the ensemble path the owner tested (standalone `main.py` path is
broken identically):

| Hop | File / lines | What happens to `neg_threshold` |
|-----|--------------|-------------------------------|
| 1. Anime table | `config/anime_whisper_vad.py:64` | Present: `0.10` in aggressive row |
| 2. Override lift | `ensemble/pass_worker.py:1200-1211` | **NOT lifted.** Only `threshold` and `max_speech_duration_s` are copied from the table into `user_segmenter_overrides` |
| 3. Sensitivity resolve | `ensemble/pass_worker.py:1215-1217` → `resolve_qwen_sensitivity()` (`:491-542`) | Resolved config is filtered to `SEGMENTER_PARAMS` keys (`:535`) |
| 4. The filter | `ensemble/pass_worker.py:89-105` | **`"neg_threshold"` is not in `SEGMENTER_PARAMS`** — stripped even if a user/GUI override supplied it |
| 5. YAML preset | `config/v4/.../whisperseg-speech-segmentation.yaml:57-59` | Does not define it either ("auto-derived … Not user-tunable") |
| 6. Backend | `backends/whisperseg.py:457-460` | `self.neg_threshold is None` → formula → **0.01** |

Standalone path: `main.py:1197-1201` lifts only `threshold` +
`max_speech_duration_s`, then the same filtered resolve at `main.py:1207-1209`.

The factory schema entry (`factory.py:151`) and the `__init__` parameter are
real but unreachable from any user-facing entry point. This is exactly the
dead-flag failure class the CLAUDE.md call-chain mandate targets; the prior
session's 7 synthetic tests exercised the backend directly and could not catch
it.

### Corrected effective parameters for the owner's E2E runs

Both runs: `threshold=0.15`, **`neg_threshold=0.01` (dead)**,
`min_speech=80 ms`, `min_silence=80 ms` (YAML aggressive preset — the info
pack's §8 values of 100/100 were the backend defaults, which the YAML
overrides), `start_pad=0`, `end_pad=30`, `chunk_threshold=0.2`,
`max_group=2.0`.

Only differences between "old" and "new" runs:

| | Old run | New run |
|---|---|---|
| `max_speech_duration_s` | 4.0 | 3.5 |
| Force-split | Naive chop at exact interval; `triggered=False` after; 20 ms gap | Split at lowest-prob frame in last 40%; `triggered` stays True; contiguous |

---

## 3. Why the Old Code Transcribed Well (confirmed mechanism)

With dead hysteresis, a segment ends only at force-split. The naive chop set
`triggered=False`, and re-onset requires `prob ≥ 0.15`. Two properties follow:

- **Every frame begins at a strong onset** (a ≥ 0.15 probability event) and
  contains up to 4 s of following audio. Quiet speech (prob 0.01–0.15) rides
  inside these frames with generous acoustic context → good recall, good
  per-character accuracy.
- **Noise-only stretches are pruned**: after a chop lands in a long non-speech
  stretch, the machine stays untriggered until the next ≥ 0.15 event, so pure
  noise/music frames are mostly not sent to the ASR.

The cost was meaningless boundaries (all 4.020 s, starts ~1 s early).

## 4. Why the New Run Regressed (hypotheses, consistent with all four symptoms)

> The following are hypotheses — mechanically verified against the code, but
> not yet confirmed on the owner's audio.

**H1 — Loss of onset anchoring (primary).** The smart split keeps
`triggered=True` forever. Under dead hysteresis nothing else ends a segment,
so from the first onset in a scene *every* frame goes to the ASR in ~3.5 s
contiguous chunks — including pure noise/music chunks, and chunks that start
mid-noise rather than at a speech onset. Quiet dialogue is no longer packaged
with a strong-onset anchor; a frame that is mostly noise plus faint dialogue
gets an empty/garbage transcription → "dialogue in quiet or noisy areas missed
entirely". Noise-only frames produce empty output and are dropped, so the
*surviving* subs have boundaries at probability minima → "timestamps align
more naturally" — the two observations are the same mechanism seen from both
sides.

**H2 — Split-at-dip targets the quietest syllables.** When the 40% window
contains no true silence, `argmin` lands on the quietest *speech* frame
(devoiced/whispered syllables — common in JAV). The split cuts exactly there,
and because split segments are contiguous, the overlap-prevention clamp
(`whisperseg.py:557-575`) reduces effective padding at split boundaries to
**zero**. Characters at those boundaries are clipped on both sides →
"characters missing at beginning/end".

**H3 — 3.5 s vs 4.0 s context (minor).** ~12% less audio context per frame.

Note the corollary: **wiring live hysteresis fixes H1 at the root** — segments
end in real silence again, re-onset requires ≥ 0.15, noise stretches are
excluded, frames re-anchor at onsets, and the smart split only fires for
genuinely long continuous speech (its intended role).

---

## 5. Synthetic Evidence: what live hysteresis will do (and why 0.10 is wrong)

Six synthetic probability tests run against the real `_probs_to_segments`
(same method as the prior session's 7 tests; script:
`scratchpad/test_tradeoff_mechanisms.py`, results 2026-07-30). Frame = 20 ms.

1. **Tail truncation at neg=0.10.** Loud speech, one 20 ms dip to 0.08, then
   0.5 s of quiet trailing speech at 0.12: end is stamped at the *dip*
   (`end=1.02s`), quiet tail lost. At neg=0.05: tail kept (`end=1.54s`).
   Root cause in code: `current["end"] = temp_end` (first sub-neg frame), and
   `temp_end` only resets when prob climbs back above **threshold** (0.15),
   not neg (`whisperseg.py:525-543`). Quiet speech in the 0.10–0.15 band after
   any momentary dip is cut off wholesale.
2. **Recall loss at neg=0.10.** Loud sentence, 200 ms pause at 0.07, quiet
   reply at 0.12: segment ends at the pause and the reply — which never
   reaches the 0.15 onset — is **lost entirely**. At neg=0.05 the pause
   doesn't end the segment and the reply is captured. This is precisely the
   mechanism by which dead hysteresis won its recall: the effective in-segment
   capture threshold was 0.01. The hysteresis floor must sit *below* the
   quiet-speech band (~0.10–0.15) that `threshold=0.15` exists to capture.
3. **min_silence robustness.** A 150 ms hard dip mid-speech splits the segment
   at `min_silence=100` but survives at 250.
4. **min_silence does not move end timestamps.** Clean speech→silence: end is
   identical at 100 vs 250 (end = silence start + end_pad, because the end is
   stamped at `temp_end`, not at confirmation time). Raising it costs nothing
   in timestamp accuracy.
5. **Generous pads are safe.** start_pad=100/end_pad=150 on two segments
   200 ms apart: no overlap (clamped by design).
6. **Composite scene.** Current row (neg .10, sil 100, pads 0/30): loses the
   quiet tail *and* the entire quiet reply — 1.4 s of dialogue. Proposed row
   (neg .05, sil 250, pads 100/150): captures everything; boundaries within
   ~0.15 s of ground truth.

Honest limitation: **isolated** quiet utterances (never ≥ 0.15, not within
`min_silence` of a triggered segment) are lost under *any* live hysteresis.
The old code caught them only as riders inside noise frames. If E2E shows this
recall loss matters, the options are lowering the aggressive onset below 0.15
(owner decision) or a recall second pass — not more hysteresis tuning.

---

## 6. Recommendation

### R1 — Wire `neg_threshold` end-to-end (bug fix, required regardless of values)

1. `ensemble/pass_worker.py:89-105` — add `"neg_threshold"` to
   `SEGMENTER_PARAMS`. (Leak risk to other backends is nil in the qwen paths:
   it is only injected for anime-whisper below, the YAMLs don't define it, and
   the factory sanitizer drops keys absent from a backend's schema.)
2. `ensemble/pass_worker.py` anime branch (~line 1208) — lift it from the
   table exactly like `max_speech_duration_s`:
   `if "neg_threshold" not in overrides and table.get("neg_threshold") is not None: inject`.
3. `main.py` anime branch (~line 1198) —
   `_user_vad_overrides.setdefault("neg_threshold", _aw["neg_threshold"])`
   guarded by presence in the table row.
4. Same lift for **`min_silence_duration_ms`** (already in `SEGMENTER_PARAMS`
   and the factory schema — only the table→overrides lift is missing; without
   it the YAML's 80 ms wins over any table value).

### R2 — Aggressive row (values informed by Section 5)

```python
"aggressive": {
    "chunk_threshold_s": 0.2,        # keep
    "max_group_duration_s": 2.0,     # keep
    "threshold": 0.15,               # keep (non-negotiable)
    "neg_threshold": 0.05,           # was 0.10 (never bound); floor below the quiet-speech band
    "min_silence_duration_ms": 250,  # new; YAML preset 80ms is far too twitchy; free (test 4)
    "start_pad_ms": 100,             # was 0; protects onset-ramp characters
    "end_pad_ms": 150,               # was 30; protects trailing characters
    "max_speech_duration_s": 3.5,    # keep
}
```

How each goal is served: **accuracy** — pads protect both edges (the slicer
cuts exactly at frame boundaries, `orchestrator.py:405-409`, so pad = ASR
context); neg=0.05 + sil=250 stop ends from landing inside quiet speech.
**Capture** — the effective in-segment threshold drops to 0.05, recovering
most of what dead hysteresis caught (test 2/6). **Timestamps** — ends stamp at
silence start (+150 ms), starts at onset (−100 ms); worst case ~0.25 s of
slack vs the old ~1 s, and slightly-early subtitle starts are desirable.

### R3 — Verification gate before E2E

`neg_threshold` appears in `_get_parameters()` (`whisperseg.py:718`), which is
written into the debug metadata JSON and the `[VadGroupedFramer]` result
metadata. Run one file with `--debug` and confirm the segmenter params show
`neg_threshold: 0.05` and `min_silence_duration_ms: 250` **before** judging
subtitle quality. If they still read `null`/80, the wiring is not done.

### R4 — E2E watchpoints and fallback knobs (one change at a time)

| Symptom after R1+R2 | Knob |
|---|---|
| Endings still truncated | `end_pad_ms` 150→200, then `neg_threshold` 0.05→0.03 |
| Quiet dialogues still missed | `min_silence_duration_ms` 250→350 (extends the capture reach around triggered segments) |
| Two-way dialogue blending returns | `min_silence_duration_ms` 250→180, or `chunk_threshold_s` 0.2→0.15 |
| Segments run long through music/noise | `neg_threshold` 0.05→0.08 (accuracy tradeoff — prefer the max_speech smart split handle it) |

---

## 7. Info-Pack Directions Evaluated (§5 of the pack)

1. **Lower neg further (0.05)** — adopted; evidence in tests 1/2/6.
2. **Longer min_silence** — adopted (250 ms); test 4 shows it is free w.r.t.
   timestamps. This knob was misdiagnosed as 100 ms; it was actually 80.
3. **Two-pass** — not needed now; the E2E that reopened the question tested a
   configuration that never ran. Revisit only if R1+R2 fails on recall.
4. **Post-VAD merging of close segments** — `group_segments` already does this
   (`chunk_threshold_s=0.2`); frames span merged segments including inner gaps.
   No change needed.
5. **ChickenRice post-VAD grouping** — we already have the equivalent
   (VAD → segments → gap/duration grouping → frames). What we lack is its
   *gap-midpoint* split behavior; see Stage-2 below.
6. **Adaptive neg near max_speech** — unnecessary complexity once the smart
   split exists; the split already finds the best boundary when the ceiling is
   hit.

## 8. Stage-2 Options (only if R1+R2 E2E still shows edge clipping)

- **Silence-run split refinement.** The smart split currently hands both sides
  a shared boundary frame at the argmin; if the dip is a real silence run, end
  segment 1 at the run's start and begin segment 2 at the run's end (the true
  analog of ChickenRice's gap-midpoint split). This restores speech-edge
  boundaries *and* creates room for pads at split points (today the contiguity
  clamps effective pad to zero there — H2).
- **Context-margin slicing.** Decouple ASR audio from subtitle timestamps in
  `_step1_frame_and_slice`: slice `frame ± ~250 ms` clamped to neighboring
  frames, keep `frame.start/end` for timing. This is the architectural answer
  to "dead-hysteresis accuracy with live-hysteresis timestamps", but it
  touches all qwen backends — hold unless needed.

## 9. Implementation (2026-07-30) and the GUI Ensemble Path (owner C1)

R1/R2 were implemented with a structural change beyond the minimal lifts: the
table→segmenter_config lift now lives in ONE place —
`anime_whisper_vad.py::apply_anime_segmenter_defaults()` +
`SEGMENTER_CONFIG_KEYS` — called by both entry points. The original bug was
two entry points each hand-copying keys; the helper removes that failure mode.

Changed files:

| File | Change |
|------|--------|
| `config/anime_whisper_vad.py` | Aggressive row → neg 0.05, min_silence 250, pads 100/150; new `SEGMENTER_CONFIG_KEYS` + `apply_anime_segmenter_defaults()` |
| `ensemble/pass_worker.py` | `"neg_threshold"` added to `SEGMENTER_PARAMS`; anime branch uses the helper (was per-key copies) |
| `main.py` | Anime branch uses the helper (was per-key copies) |
| `config/v4/.../whisperseg-speech-segmentation.yaml` | Stale "Not user-tunable" comment corrected (no behavioral change) |

**GUI Ensemble tab path (scrutinized per owner C1).** The Ensemble tab spawns
`pass_worker`; there were TWO hops dropping `neg_threshold` in that path —
the Customize-dialog param extraction (`pass_worker.py:1171-1175`, filtered by
`SEGMENTER_PARAMS` at collection time) and the resolve filter. Both are opened
by the single `SEGMENTER_PARAMS` addition. Precedence is preserved: GUI
Customize params and sliders are collected into `user_segmenter_overrides`
*before* the helper runs, and the helper only `setdefault`s — user values win.
The Customize dialog's five displayed VAD fields (api.py:1954-1961) read the
table directly, so the dialog now shows start_pad=100 / end_pad=150 for
aggressive automatically. `neg_threshold` / `min_silence_duration_ms` are not
displayed in the dialog (segmenter_config-only keys); adding widgets for them
is possible follow-up work if the owner wants them tunable from the GUI.
The lift is gated to `generator_backend == "anime-whisper"` — cohere (which
also defaults to whisperseg) and qwen3 are untouched, and the
balanced/conservative anime rows carry none of the new keys, so only
aggressive changes behavior.

## 10. What Was Verified and How

- **Read end-to-end:** `anime_whisper_vad.py`, `whisperseg.py` (full),
  `factory.py` schema, `ten.py::group_segments`, `vad_grouped.py` (full),
  `orchestrator.py` slicing, `pass_worker.py:75-120/491-542/1120-1300`,
  `main.py:1160-1320`, `api.py` qwen schema + Customize flow,
  `whisperseg-speech-segmentation.yaml`, plus a repo-wide grep for every
  `neg_threshold` occurrence.
- **Executed (analysis):** 6 synthetic probability tests against the real
  `_probs_to_segments` in the WJ env (all behaved as predicted; §5).
- **Executed (post-implementation, R3):**
  - `py_compile` on the three edited Python files — pass.
  - `python -m whisperjav.main --help` — exit 0.
  - 19-check chain-simulation test (`scratchpad/test_wiring_chain.py`): parses
    the REAL `SEGMENTER_PARAMS` set out of `pass_worker.py` source via `ast`
    (the module cannot be imported lightly), then runs the actual
    `ConfigManager.get_tool_config` layering, the actual filter, and the
    actual `SpeechSegmenterFactory.create("whisperseg", ...)`. Confirms:
    backend receives threshold 0.15, neg_threshold 0.05, min_silence 250,
    max_speech 3.5, pads 100/150; user overrides win over table values;
    balanced/conservative resolve with NO neg_threshold (formula fallback);
    `SEGMENTER_CONFIG_KEYS ⊆ SEGMENTER_PARAMS`. All pass.
  - Real E2E CLI run on the Netflix benchmark
    (`test_media/Ground_Truths/Netflix/293sec-The.Naked.Director...mkv`,
    standalone `--mode qwen --qwen-generator anime-whisper
    --qwen-sensitivity aggressive --debug`) with debug-metadata inspection of
    the bound segmenter parameters — result recorded below when the run
    completed.
- **Not verified by me:** the GUI Ensemble tab E2E (cannot launch the GUI
  here — the pass_worker hops were verified by the chain simulation + code
  trace; owner should confirm one ensemble run shows the new values in its
  debug output), and subtitle *quality* vs ground truth, which is the owner's
  E2E judgment.

## 11. Ground-Truth Benchmark (2026-07-30, owner C2)

All runs: standalone CLI, anime-whisper aggressive, Netflix benchmark
`test_media/Ground_Truths/Netflix/293sec-The.Naked.Director.S01E04.Scene4.mkv`
scored with `tools/vad_hypothesis_suite/score.py` against the sanitized GT
(68 subs). Runs are deterministic. The **pre-i2 old code** was run from a
detached git worktree at `0921b97` (naive force-split, max_speech 4.0, dead
neg 0.01, pads 0/30 — the exact state the owner's original E2E praised for
transcription accuracy; includes the i1 filter, so the output pipeline is
identical to current).

| Config | CER↓ | sub↓ | del↓ | char_recall↑ | time_recall↑ | time_prec↑ | mean_dur | segs |
|---|---|---|---|---|---|---|---|---|
| **PRE-I2 old code** (naive split, max 4.0, dead neg) | **0.539** | 0.258 | **0.277** | **0.723** | **0.740** | 0.750 | 3.88s | 35 |
| PRE-SESSION (smart split, max 3.5, dead neg, sil 80, pads 0/30) | 0.586 | 0.244 | 0.334 | 0.666 | 0.622 | 0.751 | 2.93s | 39 |
| VAR neg=.02, sil 250, pads 100/150, max 3.5 | 0.586 | 0.239 | 0.339 | 0.661 | 0.620 | 0.758 | 2.89s | 39 |
| VAR neg=.03, sil 250, pads 100/150, max 3.5 | 0.583 | 0.193 | 0.384 | 0.617 | 0.609 | 0.778 | 2.84s | 38 |
| **R2 as implemented** (neg=.05, sil 250, pads 100/150, max 3.5) | 0.583 | **0.192** | 0.387 | 0.613 | 0.596 | **0.790** | 2.81s | 37 |
| VAR sil=350 (else R2) | 0.583 | 0.192 | 0.387 | 0.613 | 0.600 | 0.787 | 2.84s | 37 |
| **VAR max=4.0 (else R2)** | 0.563 | 0.222 | 0.338 | 0.662 | 0.615 | 0.766 | 3.35s | 33 |

### Findings

1. **The old code's E2E-observed capture advantage is real and quantified:**
   char_recall 0.723 vs 0.613 for R2 (+11 points), deletions 0.277 vs 0.387.
   Its regional wins are exactly the quiet/noisy stretches (4:00–4:20 region
   recall 0.52 vs 0.20; 1:40–2:00 and 2:40–3:00 similar). Its cost: worst
   substitution rate (0.258), mean sub length 3.88s ≈ the mechanical 4s
   chunks (the original complaint), and no time-precision advantage (0.750).
2. **Capture attribution is two roughly additive effects (~5 points each):**
   (a) max_speech 3.5 vs 4.0 — costs ~5 points capture even with live
   hysteresis (compare R2 0.613 vs max=4.0 0.662); (b) live hysteresis
   (neg ≥ 0.03) vs dead — costs ~5 points (compare neg .02 0.661 vs
   neg .05 0.613). The remaining ~0.06 vs old code is naive-vs-smart split
   mechanics. min_silence and neg fine-tuning (0.03 vs 0.05) barely move it.
3. **The frontier is monotonic — no config dominates.** Capture trades
   against substitution accuracy, time precision, and sub length at every
   step. R2 is the max-precision end; old code is the max-capture end;
   R2-with-max=4.0 is the midpoint (0.662 recall / 0.222 sub / 0.766 prec /
   3.35s subs).
4. **The two hard regions (4:20–4:40 ≈ 0.10, 1:20–1:40 = 0.15) are bad in
   every config including old code** — content the VAD/ASR cannot get under
   any post-processing setting; not a tuning target.
5. **Owner decision point:** if the capture loss matters more than sub length,
   raising the row's `max_speech_duration_s` back to 4.0 is the single
   biggest recoverable lever (+5 points capture, subs lengthen 2.81→3.35s
   mean, still no 4.020s mechanical wall). Beyond that, capture parity with
   old code requires giving the ASR wide audio while keeping tight subtitle
   times — the Stage-2 context-margin slicing (§8), which this benchmark now
   supports with data.

## 12. The "3a" Refined Display Start — Implemented and Benchmarked (2026-07-30)

Owner-approved package ("I agree"): wide capture for content + probability-
derived display starts for timing, decoupled. The RAW VAD boundary still feeds
the ASR audio slice; the subtitle's displayed start moves to the first frame
whose probability reaches `speech_start_threshold` (unchanged if the segment
already starts at/above it; fallback to raw when never reached). Empirical
basis: sweeping refinement thresholds over the pre-i2 output showed T=0.30
moves median start error −0.489s → −0.026s; dynamic thresholds (Otsu, p95)
overshoot (+0.211s late), so a fixed 0.30 ships.

### Implementation

| File | Change |
|---|---|
| `backends/whisperseg.py` | `speech_start_threshold` param; first-crossing computed at all three segment-append sites (natural end / force-split / end-of-audio); emitted as `metadata["speech_start_sec"]`; in `_get_parameters()` |
| `factory.py` | schema: `speech_start_threshold: (float, None, True)` |
| `framers/vad_grouped.py` | per-frame refined start (first segment in group with one) → `metadata["speech_starts"]` |
| `orchestrator.py` | Step-1 returns `frame_speech_starts`; Step-9 applies them to subtitle starts in **Branch B only, `vad_only` only** (aligner paths untouched); starts only move later → cannot create overlaps; diag gains `frame_timing` |
| `types.py` | `SceneDiagnostics.frame_timing`: per-frame `{vad_start, vad_end, speech_start, internal_breaks}` — the owner's two-timestamp JSON; `internal_breaks` reserved for the future "3b" splitter |
| `anime_whisper_vad.py` | aggressive row → wide-capture package: `neg 0.05`, `max_speech 4.0`, `speech_start_threshold 0.30` (sil 250 / pads 100/150 kept); key added to `SEGMENTER_CONFIG_KEYS` |
| `pass_worker.py` | `speech_start_threshold` added to `SEGMENTER_PARAMS` |

Verified: 14-check synthetic unit suite (ramp onset, loud onset, never-reached
fallback, disabled, force-split per-side starts, padding independence, framer
grouping) — all pass; 26-check wiring chain — all pass; E2E on the Netflix
benchmark with sidecar inspection — 40/68 frames refined, median shift 0.16s,
raw + refined both recorded in `frame_timing`.

### Benchmark (Netflix GT; start-error = signed vs max-overlap GT sub)

| Config | CER↓ | sub↓ | char_recall↑ | start err (median) | early>0.5s | mean_dur |
|---|---|---|---|---|---|---|
| PRE-I2 old code | **0.539** | 0.258 | **0.723** | −0.489s | 17 | 3.88s |
| R2 (neg .05, max 3.5) | 0.583 | **0.192** | 0.613 | +0.045s | **5** | 2.81s |
| PKG neg=.02, max 4.0, sst .30 | 0.563 | 0.198 | 0.639 | +0.133s | 9 | 3.22s |
| **SHIPPED: neg=.05, max 4.0, sst .30** | 0.563 | 0.222 | 0.662 | **+0.063s** | 11 | 3.18s |

The shipped row is the best all-around point measured: +5 capture points over
R2, substitutions well below old code, and start timing transformed from half
a second early to GT-aligned. Notes:

- **neg=0.02 anomaly:** near-dead hysteresis LOST capture vs 0.05 at
  max_speech 4.0 (0.639 vs 0.662) — opposite of the max 3.5 interaction.
  Hypothesis (unverified): with no natural ends, the no-re-onset smart split
  floods frames with noise (the §4 H1 dilution); pre-i2's naive chop pruned
  noise via its `triggered=False` re-onset requirement. A future "hybrid
  split" (smart dip boundary + re-anchor) is the candidate lever for the
  remaining ~6-point capture gap to pre-i2 — alongside the 3b splitter.
- Residual start-error tail (11 subs >0.5s early) = merged multi-turn frames;
  that is the 3b splitting problem, out of scope for 3a by design.
- All effect sizes are single-clip (Netflix drama); owner will judge on JAV
  content E2E. The GUI Ensemble path inherits everything (shared orchestrator
  + `SEGMENTER_PARAMS` lift); owner should eyeball one ensemble run.

## 13. Correction (2026-07-31): Faithful Pre-i2 Capture Restored — CURRENT CANDIDATE

The owner's design was "pre-i2 for content + start adjustment ONLY", but the
§12 candidate paired 3a with a *different* capture engine (smart split, live
neg 0.05) — hence char_recall 0.662 instead of pre-i2's 0.723. The owner
correctly flagged this. Fixed by restoring the exact pre-i2 capture path:

- **`force_split_mode` parameter** on WhisperSeg: `"dip"` (default; v1.9.0
  smart split, unchanged for other presets) vs `"chop"` — the vendor-faithful
  pre-i2 behavior copied verbatim from `0921b97`: end at exactly
  `start + max_speech`, full state reset (`triggered=False`), re-onset
  ≥ threshold required. Wired through factory schema, `SEGMENTER_CONFIG_KEYS`,
  `SEGMENTER_PARAMS`.
- **Aggressive row (current candidate, uncommitted):** threshold .15,
  neg .01 (explicit; the value the pre-i2 formula produced), sil 80,
  pads 0/30, max_speech 4.0, `force_split_mode="chop"`,
  `speech_start_threshold=0.30`. Capture = exact pre-i2; only display
  starts differ.

### E2E acceptance results (Netflix GT)

- **Text identity: PASS** — 35 subs, output text byte-identical to the pre-i2
  benchmark run; all ends unchanged; 14/35 starts moved later. Content
  metrics equal pre-i2 exactly: CER 0.539, sub 0.258, del 0.277,
  char_recall **0.723**.
- **Start errors:** median −0.489s → **−0.026s** — matching the offline 3a
  sweep prediction to the millisecond (deterministic pipeline). mean|e|
  1.092→1.087; early>0.5s 17→15 (the surviving tail is the merged-turn
  problem = 3b scope; a within-sub split mechanism is required to fix it, no
  start-threshold can).
- time_recall 0.740→0.716 (removed leading-silence span coverage — intended),
  time_precision 0.750.

This is the owner's design delivered literally: pre-i2 content, ends
untouched, starts refined. The §12 configuration (smart split + neg .05,
char_recall 0.662 but a cleaner sub-rate and shorter subs) remains available
by flipping `force_split_mode` back to `"dip"` — a one-key table change —
if the owner ever prefers that operating point after JAV E2E.

## 14. The Offline Two-Level Decoder — NEW DEFAULT (2026-07-31, owner-approved)

Owner findings drove a decoder replacement: (a) ≥400ms dialog gaps were never
cut (measured: 0/21 GT gaps under any hysteresis value; 11/35 subs spanned a
gap); (b) frame-level separability is fundamentally poor (WhisperSeg AUC
0.791, speech p10 = 0.04 < gap p50 = 0.11 — no single threshold can work;
Silero 0.673 and RMS 0.622 are WORSE); (c) the TEN backend demonstrated the
offline pipeline decoder shape in-repo. The state machine's `neg_threshold`
is WhisperJAV post-processing, not part of the model — fully replaceable.

**Design** (`whisperseg.py::_probs_to_segments_offline`,
`segmentation_decoder="offline"`): SEED segments at prob ≥ `threshold`
(0.15); GROW edges while prob ≥ `grow_floor` (0.05) — attached quiet speech
rides along, floor-runs without a seed are dropped as noise; MERGE gaps
< `gap_merge_ms` (350) — longer gaps become dialog cuts; DROP micro-segments;
SPLIT overlong segments at minima of the smoothed probability curve
(`split_smooth_ms`=120, near-tie → latest, 0.6s sliver guard, even-split
fallback). Padding/conversion shared with the hysteresis decoder
(`_pad_and_convert`); 3a `speech_start` computed per final part. `grow_floor`
is the capture-vs-cut dial; `gap_merge_ms` is the pause-length knob.

**Second option preserved:** `segmentation_decoder="hysteresis"` keeps the
vendor/ChickenRice state machine intact; the aggressive row retains
`neg .01` / `chop` / `sil 80` so flipping back reproduces exact pre-i2
capture. New keys wired through factory schema, `SEGMENTER_CONFIG_KEYS`,
`SEGMENTER_PARAMS`, AND the GUI: four new Customize-modal fields
(VAD Decoder, Grow Floor, Gap Cut ms, Max Speech Duration) →
`prepare_qwen_params` mapping → segmenter-override lift (user wins over
table). Modal collection is schema-generic (app.js:4517), sensitivity-aware
defaults via `get_qwen_schema`.

**Netflix GT benchmark (all four timing/content dimensions):**

| Config | char_recall↑ | CER↓ | start err median | early>0.5s | **subs spanning ≥0.4s gap** | mean_dur |
|---|---|---|---|---|---|---|
| Pre-i2 + 3a (§13) | **0.723** | **0.539** | −0.026s | 15 | 11/35 | 3.76s |
| **OFFLINE default** | 0.699 | 0.569 | +0.147s | 9 | **2/36** | 2.90s |

The offline decoder gives up 2.4 points of capture vs pre-i2 (0.699 — far
above every previous live-cutting config: R2 0.613, neg.02 0.639, max4.0
0.662) and in exchange: **the spanned-gap symptom drops 11 → 2**, subs
starting >0.5s early drop 15 → 9, mean|start error| is the best measured
(0.665s), and mean sub length moves toward GT (2.90s vs 3.76s). Verified by
18-check unit suite, 33-check wiring chain, and E2E with sidecar inspection.
Owner judgment on JAV content pending; the R4-style fallback: raise
`grow_floor` to cut more/capture less, lower it (or raise `gap_merge_ms`)
for the reverse — all from the GUI.
