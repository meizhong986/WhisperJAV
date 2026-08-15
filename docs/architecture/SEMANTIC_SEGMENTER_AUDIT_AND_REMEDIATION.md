# Semantic Scene Segmenter — Cut & Padding Audit and Remediation Plan

- **Status:** Investigation complete. Remediation **proposed, NOT implemented**. Candidate to **schedule for the next release** (foundational / high priority).
- **Date:** 2026-07-15
- **Owner area:** scene detection (semantic) → feeds the qwen / ChronosJAV (anime-whisper, qwen3, cohere) pipeline.
- **Files under audit:**
  - `whisperjav/vendor/semantic_audio_clustering.py` (vendored engine — the core logic)
  - `whisperjav/modules/scene_detection_backends/semantic_adapter.py`
  - `whisperjav/modules/scene_detection_backends/semantic_backend.py`
- **Trigger:** anime-whisper subtitle **timestamp misalignment** reports (subs ending ~0.7 s early; next sub starting while the previous line is still spoken).
- **Method:** two **independent** investigations that were **not** cross-informed:
  1. an **empirical acoustic measurement** of cut safety on 4 ground-truthed reference clips, and
  2. an **independent senior-architect code audit** (instructed to ignore all comments/docstrings and reason only from executable code).
  Both converged on the same core diagnosis; the audit additionally surfaced several correctness bugs.

---

## 1. Executive summary

The semantic segmenter's two headline guarantees — **(a) "cuts land in silence"** and **(b) "segments longer than `max_duration` are split"** — **are not implemented as the code's own comments claim.**

- Cut placement (`_snap_to_silence`) selects the **`argmin` of smoothed RMS in a ±5 s window** with **no threshold and no minimum-width check** — i.e. the *least-loud* frame, which equals *silence* only when silence happens to exist in range.
- A **fixed 0.35 s pad** is applied to each segment edge, producing a **constant 0.70 s overlap** between every pair of adjacent scenes, **regardless** of whether the cut is actually silent.
- **No max-duration split exists at all**; `max_duration` is used only as a *merge ceiling*.

**Empirical result (26 cuts, 4 clips):** 35 % of cuts land in outright speech; **81 % have the 0.35 s pad overshooting into speech** on ≥1 side; median silence half-width at a cut is **~0.26 s < 0.35 s pad**.

**Consequence (the reported symptom):** on the majority of cuts the padding reaches into neighbouring speech → adjacent scenes transcribe the same audio → overlapping/duplicated subtitles. The downstream `SceneOverlapResolver` (shipped v1.9.0, `9290917`) then removes the overlap by **clipping the earlier subtitle's end back to the next subtitle's start**, which manifests as **subtitles that end ~0.7 s early and next-subs that begin while the previous line is still being spoken.**

**Good news for remediation:** the fixes for the dominant symptom are **low effort** — a calibrated silence threshold (`rms_base`) is *already computed each run but discarded*, and padding can be clamped to the measured silence extent.

---

## 2. Background: how a subtitle timestamp is produced (anime-whisper path)

For **anime-whisper** (ChronosJAV) with default settings the timestamp chain is (verified in prior tracing):

1. **Semantic scene detection** cuts the audio into scenes and returns **padded (`asr_processing`) timestamps** (`semantic_adapter.py:401-436`). Scenes overlap by ~0.7 s.
2. The qwen pipeline extracts each scene's audio with those padded bounds and uses the **padded scene start as the stitch offset**.
3. anime-whisper runs `timestamp_mode = vad_only`, `regroup_mode = off` → orchestrator **Branch B frame-native**: each VAD group becomes one subtitle with `start = frame.start`, `end = frame.end` (`orchestrator.py:881-893`; `reconstruction.py:reconstruct_frame_native`). VAD hardening for `VAD_ONLY` is a **no-op** (`hardening.py:_apply_vad_only_timestamps` → `return`).
4. **Net:** a subtitle's timestamps are the **VAD group boundaries plus the padded scene offset** — used raw, with no word-level alignment.
5. **Phase 8** `SceneOverlapResolver` clips residual scene-boundary overlaps (`cleaners/scene_overlap_resolver.py`).

Because step 1 introduces a ~0.7 s overlap and step 5 resolves it by clipping, **any inaccuracy in where the semantic cut lands surfaces directly as a subtitle-timing error at scene boundaries.** That is why the segmenter is foundational and why this audit matters.

---

## 3. Reconstructed algorithm (code-only)

Entry `process_movie_v7` (`semantic_audio_clustering.py:693`): extract → calibrate → `SemanticSegmenter.segment` → classify → write JSON with padded timestamps.

**Feature extraction** `StreamFeatureExtractor.extract` (`:268-388`): audio streamed in 60 s blocks, mono-mixed, resampled per-block to 16 kHz; librosa features at `hop_length=512` (frame rate ≈ 31.25 fps); 36 feature rows incl. `RMS` at index 26 (`FeatureRegistry.RMS`, `:180`). Global time axis = `frames_to_time(arange(n_frames), sr=16000, hop_length=512)` (`:379`).

**Segmentation** `SemanticSegmenter.segment` (`:404-441`):
1. `median_filter(features, size=(1,15))` smoothing (`:409`).
2. Subsample `step = int(fps*0.5) = 15` → clustering resolution ≈ 0.48 s (`:411-413`).
3. `AgglomerativeClustering(distance_threshold=18.0, linkage='ward')` on scaled features (`:416-419`).
4. Raw boundaries at each cluster-label change, bracketed by `[0, duration]` (`:422-426`).
5. `_snap_to_silence` (`:443-468`).
6. `_smart_merge` (`:470-520`).
7. `_forced_cleanup` (`:522-540`).
8. `_ensure_timeline_coverage` (`:542-572`).

**`_snap_to_silence`** (`:443-468`): for each interior boundary, search **±`snap_window`(=5) s** and set the boundary to `times[argmin(rms_smooth[window])]` (`:456-463`). `return sorted(set(refined))` (`:468`).

**`_smart_merge`** (`:470-520`): merge any `dur < min_dur(20)` segment into the more cosine-similar neighbour **iff** the merged `dur <= max_dur(420)` (`:494-517`).

**`_forced_cleanup`** (`:522-540`): merge any remaining `< min_dur` segment forward/back **with no `max_dur` check** (`:528-538`).

**`_ensure_timeline_coverage`** (`:542-572`): prepend/append edge fillers, fill inter-segment gaps `>0.001` with zero-vector fillers, forward-clamp overlaps (`:559-561`).

**Padding / output** `Segment.to_dict` (`:194-226`): fixed `pad = 0.35` (`:198`); `safe_start = max(0, start-0.35)`, `safe_end = end+0.35` (`:199-202`, `safe_end` **not** clamped to duration). Emits both **strict** (`timestamps`) and **padded** (`asr_processing`) sets.

**Adapter** `semantic_adapter.py:_transform_and_split` (`:358-455`): reads **padded** `asr_processing` timestamps (`:401-407`), clamps to sample length (`:414-419`), and returns the **padded** `start_sec/end_sec` as the canonical scene bounds (`:436`) → `SceneInfo` (`semantic_backend.py:169-182`).

---

## 4. Findings (consolidated, evidence-based)

### F1 — `_snap_to_silence` does not verify silence (severity: HIGH)
It optimizes `argmin` of smoothed RMS (`:461-462`) with **no comparison to any threshold** and **no minimum-width requirement**. "Least-loud frame ≠ silent frame." A calibrated floor exists (`AdaptiveClassifier.calibrate` → `stats["rms_base"] = percentile(rms,20)`, `:595-601`) but is **never passed to the segmenter** (see F6). Additionally, the 5-frame (~160 ms) RMS median smoothing (`:449`) erases short true pauses.

### F2 — Empirical: the cut is unsafe on the majority of real cuts (severity: HIGH)
Measured on the 4 reference clips using the vendor's own extractor/segmenter/threshold (`tools/vad_hypothesis_suite/measure_cuts.py`):

| Metric | Value |
|---|---|
| Cuts **below** the silence threshold | 65 % (⇒ **35 % land in outright speech**) |
| Cuts with ≥0.35 s silence on **both** sides (pad-safe) | **19 %** |
| Cuts where **0.35 s pad overshoots speech** on ≥1 side | **81 %** |
| Cuts with total silence < 0.70 s | 58 % |
| **Median silence half-width** at a cut | **L 0.26 s / R 0.27 s** |
| Median `rms/threshold` at cut | 0.44 |

Worst case is loud/continuous content: on `S01E04` (chaotic filming scene), **7 of 11 cuts land in full speech** (`rms/threshold` up to 6.4, 0 s silence either side). Calmer clips still show narrow pauses (0.06–0.29 s) on at least one side. → the artifact is **content-dependent** ("sometimes"), worst where there is no true silence to snap to.

### F3 — Fixed 0.70 s adjacent overlap, unconditional (severity: HIGH)
Padding is fixed & symmetric (`:198-202`); strict adjacent segments are contiguous (`_ensure_timeline_coverage`), so padded ranges are `[…, T+0.35]` and `[T-0.35, …]` → **constant 0.70 s overlap at every boundary irrespective of silence.** When `T` is mid-speech, the same speech goes into **both** scenes' audio → duplicate transcription + Whisper edge-hallucination on the truncated word; a 0.35 s buffer is not guaranteed to contain a whole Japanese word/particle.

### F4 — No `max_duration` split exists; docstring is false (severity: MEDIUM-HIGH) — *verified*
Nothing splits long segments. `max_dur` is used **only** as a merge ceiling (`:498-504`); `_forced_cleanup` merges with **no** `max_dur` check (`:528-538`) and can even *produce* over-long segments. The docstring "Segments longer than this will be split" (`:138`) is **contradicted by the code**. Impact: the qwen pipeline's "12–48 s safe chunking" enforces only the **minimum** (via merge); the **maximum is never enforced**, so a homogeneous cluster can yield an arbitrarily long scene — a latent risk for aligner paths (qwen3/cohere ForcedAligner ~180 s limit) and long inputs.

### F5 — Cumulative time-axis drift on multi-chunk inputs (severity: MEDIUM on features / LOW on short clips) — *review-identified, mechanism verified*
Features are framed **per 60 s chunk** with `center=True` and concatenated, but the time axis is a single continuous `arange*hop` (`:378-379`). Each chunk contributes ~1 extra frame, so the axis drifts **~32 ms per 60 s chunk** (≈ our clips: ~0.1–0.16 s; a 2 h film: ~4 s). The adapter converts these drifted seconds to real samples (`semantic_adapter.py:414`), so on long files the extracted cut lands at a progressively **wrong real position**, defeating the snap. *Empirical magnitude on our short clips is small; recommend confirming on a long input before prioritizing.*

### F6 — Calibrated silence floor computed but discarded (severity: informative — makes the fix cheap)
`rms_base` (a usable silence threshold) is computed every run (`:595-601`) but lives in `AdaptiveClassifier` and is **never plumbed into `SemanticSegmenter`**. Wiring this existing value into `_snap_to_silence` is the low-effort enabler for F1's fix.

### F7 — Secondary fragilities (severity: LOW)
- `sorted(set(refined))` (`:468`) silently **collapses** boundaries that snap to the same minimum → non-deterministic scene-count reduction.
- `_smart_merge` skips **empty-mask** intervals (`:474-475`); the resulting gap is later filled by a zero-vector filler (`:557`), classified as SILENCE → spurious tiny "silence" scenes; their padded ranges can overlap **both** neighbours (triple overlap).
- `fps=31` hardcoded vs true 31.25 (`:157`,`:408`); per-block independent `librosa.resample` adds filter-edge artifacts at 60 s seams (`:341`); last-segment padded `end` exceeds `duration` in the JSON/returned tuple (`:202`, adapter clamps at sample level so over-read is safe).

### F8 — Downstream manifestation (the reported symptom)
F1+F2+F3 → overlapping subs at scene boundaries → `SceneOverlapResolver` Rule 2 clips `prev.end = next.start − 1 ms` → **sub N ends ~0.7 s early; sub N+1 starts early while line N is still spoken.** The resolver is a **downstream band-aid**; the root cause is the cut/padding logic here.

---

## 5. Root-cause analysis

1. **Acoustic-boundary-as-subtitle-boundary.** In the anime-whisper (vad_only, frame-native) path, subtitle timing is the VAD group edge plus the *padded* scene offset — no word alignment corrects it.
2. **Unvalidated silence snap.** The cut is placed at a windowed energy minimum with no proof it is silent or wide enough (F1/F2).
3. **Padding/cut-safety mismatch.** A fixed 0.35 s pad is applied everywhere, but the actual silence at a cut is usually narrower (median 0.26 s) or absent — so the pad overshoots into speech (F3), creating the overlaps that become early-ends.
4. **The overlap resolver treats the symptom, not the cause.**

---

## 6. Impact assessment

- **Affected paths:** semantic is the **default** scene detector for the qwen / ChronosJAV pipeline (anime-whisper, qwen3, cohere). Legacy pipelines (balanced/fast/faster/fidelity) use other scene detectors and are **not** affected by this document.
- **Timestamp accuracy (all qwen backends):** early-end/early-start at scene boundaries (≤ ~0.7 s), content-dependent.
- **Content duplication / hallucination:** mid-speech cuts place the same audio in two scenes (F3).
- **Aligner paths (qwen3/cohere):** F4 (no max-split) risks scenes exceeding the ForcedAligner limit on some inputs.
- **Long inputs (features vs short clips):** F5 drift degrades cut accuracy as file length grows.

---

## 7. Remediation plan (prioritized)

> **Vendor caveat:** `semantic_audio_clustering.py` is a **vendored** module; edits diverge from upstream. Prefer changes that are (i) minimal and well-marked, or (ii) applied in the **adapter** layer where feasible. Decide per-item below.

### Phase 1 — Kill the early-end artifact for the majority (LOW effort, HIGH value)

**R1. Silence-aware snap** *(fixes F1; enabled by F6)*
- Plumb `AdaptiveClassifier`'s `rms_base` (or recompute the 20th-percentile floor) into `SemanticSegmenter._snap_to_silence`.
- Accept a snap point only if `rms_smooth[idx] <= rms_base * k` (reuse `silence_threshold_multiplier=1.5`) **and** the sub-threshold region has a **minimum width** (e.g. ≥ target pad on each side). If no qualifying point exists in the window, fall back deterministically (keep the raw boundary, or widen the search once) and **flag the cut as non-silent** in metadata.
- Files: `semantic_audio_clustering.py` (`segment`/`_snap_to_silence`), thread `rms_base` from `calibrate`.
- Risk: could shift some boundaries; validate scene-count stability.

**R2. Adaptive / silence-clamped padding** *(fixes F3 for cuts that have silence)*
- Replace the fixed `pad=0.35` in `Segment.to_dict` with **per-side padding clamped to the measured silence half-width** at that boundary: `pad_side = min(0.35, silence_half_width_side)`. Requires carrying the per-boundary silence extent (computed during snap) into `Segment`.
- Effect: padding never reaches into speech → boundary overlap **disappears** for the ~65 % of cuts with any silence, so the downstream resolver rarely has to clip.
- Files: `semantic_audio_clustering.py` (`Segment`, boundary metadata).
- Risk: less ASR context at some cuts; measure CER/recall (should be neutral — pad only ever shrinks *into what was speech anyway*).

**R5. Hygiene** *(fixes F7 low-risk items)*
- Guard the `set()` boundary collapse (F7) to keep deterministic scene count; handle empty-mask intervals so they don't become spurious SILENCE fillers.

**Metric (validation enabler):** add a **boundary-error metric** to `tools/vad_hypothesis_suite/score.py`: for GT-matched subtitles, `mean|sub_start − gt_start|`, `mean|sub_end − gt_end|`, and a "premature-cut rate" (subs ending > X ms before their GT end). Turns "timing feels off" into a tracked number.

### Phase 2 — Correctness (MEDIUM effort)

**R3. Implement real max-duration splitting** *(fixes F4)*
- After merge/cleanup, split any segment `> max_duration` at its lowest-energy interior point(s) (reuse the snap machinery), so "safe chunking" actually bounds scene length.
- Files: `semantic_audio_clustering.py` (`segment` pipeline).

**R4. Fix time-axis drift** *(fixes F5)*
- Compute frame times **per chunk with the correct cumulative sample offset** (or drop `center` seam duplication) so `times` matches real audio position on multi-chunk files.
- Files: `StreamFeatureExtractor.extract`.

### Phase 3 — Structural (LARGER effort, optional)

**R6. Strict-boundary output assignment** *(addresses the residual ~35 % truly-mid-speech cuts)*
- Keep padded audio for **ASR context**, but assign each output subtitle to **exactly one** scene using the **strict** (non-overlap) boundary — realizing the "strict timestamps for output, buffered for ASR" split the vendor *claims* but the pipeline does not honor (adapter returns padded bounds as canonical, `:436`).
- This makes the output robust even when a cut is unavoidably in speech, and lets us reduce reliance on `SceneOverlapResolver`.

**Relationship to `SceneOverlapResolver`:** keep it as a **safety net**, but after Phase 1 it should rarely fire. Consider changing its Rule 2 from "clip earlier end to later start" to a **midpoint** split (less biased) as a cheap independent improvement.

---

## 8. Validation strategy

Use the existing benchmark harness (`tools/vad_hypothesis_suite/` + `test_media/reference_benchmarks/`, 4 GT clips):
1. **Cut-safety regression:** `measure_cuts.py` — track the F2 table (target: pad-overshoot % ↓, mid-speech-cut % unchanged since it's content-bound).
2. **Timing:** the new boundary-error metric (Phase 1) — before/after each remediation.
3. **Content quality guard:** CER / `del_rate` / `time_recall` must **not regress** while timing improves (R2 only ever removes padding that was over speech, so quality should be neutral).
4. **A/B protocol:** one variable at a time, on all 4 clips, aggregate + per-clip (as already established for the VAD sweep).

---

## 9. Risks, open questions, non-goals

- **Vendored code divergence** — mark all edits; consider whether to fork-document upstream.
- **The ~35 % no-silence cuts cannot be made silent** — only R6 (strict-boundary) or accepting hard cuts addresses them.
- **F5 magnitude** — confirm on a long (feature-length) input before prioritizing R4.
- **Interaction with the shipped resolver** — ensure Phase 1 + resolver don't double-correct.
- **Non-goals:** word-level alignment for anime-whisper (aligner enablement) is a *separate* lever tracked elsewhere; it would also fix timing but at VRAM/time cost. This document is scoped to the **scene-cut/padding** foundation.

---

## 10. Recommended scope for the next release

- **Must (Phase 1):** R1 (silence-aware snap) + R2 (adaptive padding) + R5 (hygiene) + boundary-error metric. Low effort, directly kills the reported early-end artifact for the majority of cuts.
- **Should (Phase 2):** R3 (max-split) + R4 (drift) — real correctness bugs.
- **Optional (Phase 3):** R6 (strict-boundary output) if residual mid-speech cases remain material after Phase 1.

---

## Appendix A — Per-cut measurement (silence at each semantic cut)

`silence_thresh = percentile(RMS,20) × 1.5`; `L/R` = silence half-width; flagged when the 0.35 s pad overshoots speech.

```
S01E04_scene4  dur=294s  11 cuts  thr=0.0026
  0:21.31 rms/thr=0.11  L=0.26 R=0.83  *overshoot
  0:41.92 rms/thr=0.15  L=0.58 R=0.58
  1:17.63 rms/thr=0.18  L=0.51 R=0.13  *overshoot
  1:58.85 rms/thr=0.17  L=0.35 R=0.42
  2:15.65 rms/thr=1.46  L=0.00 R=0.00  *overshoot (mid-speech)
  2:44.70 rms/thr=2.79  L=0.00 R=0.00  *overshoot (mid-speech)
  3:20.45 rms/thr=3.73  L=0.00 R=0.00  *overshoot (mid-speech)
  3:41.57 rms/thr=5.61  L=0.00 R=0.00  *overshoot (mid-speech)
  4:04.13 rms/thr=6.42  L=0.00 R=0.00  *overshoot (mid-speech)
  4:19.33 rms/thr=3.13  L=0.00 R=0.00  *overshoot (mid-speech)
  4:34.75 rms/thr=1.19  L=0.00 R=0.00  *overshoot (mid-speech)
S02E02_more    dur=138s   2 cuts  thr=0.0109
  0:37.57 rms/thr=0.48  L=0.26 R=0.32  *overshoot
  1:25.54 rms/thr=0.68  L=0.06 R=0.10  *overshoot
S02E04_dream   dur=210s   5 cuts  thr=0.0015
  0:46.75 rms/thr=0.22  L=0.26 R=0.06  *overshoot
  1:31.78 rms/thr=0.23  L=0.29 R=0.22  *overshoot
  2:12.67 rms/thr=0.23  L=0.26 R=0.51  *overshoot
  2:37.06 rms/thr=0.21  L=0.26 R=0.90  *overshoot
  2:57.41 rms/thr=0.39  L=1.28 R=3.55
S02E05_bubble  dur=228s   8 cuts  thr=0.0019
  0:23.04 rms/thr=1.19  L=0.00 R=0.00  *overshoot (mid-speech)
  0:38.94 rms/thr=0.23  L=0.29 R=2.34  *overshoot
  1:05.47 rms/thr=0.16  L=0.22 R=2.62  *overshoot
  1:30.53 rms/thr=0.15  L=0.64 R=6.02
  1:54.24 rms/thr=0.64  L=0.64 R=0.32  *overshoot
  2:39.17 rms/thr=0.64  L=0.10 R=0.32  *overshoot
  2:57.70 rms/thr=0.40  L=1.57 R=7.30
  3:13.34 rms/thr=1.15  L=0.00 R=0.00  *overshoot (mid-speech)

SUMMARY (26 cuts): below-thresh 65% | pad-safe both sides 19% | pad overshoot ≥1 side 81% |
total-silence<0.70s 58% | median half-width L=0.26 R=0.27
```

## Appendix B — Key code citations

| Concern | Location |
|---|---|
| Snap = argmin, no threshold/width | `semantic_audio_clustering.py:443-468` (esp. 461-463) |
| Silence floor computed, unused | `:588-601` (`calibrate`) vs `:443-468` |
| Fixed 0.35 s pad | `:194-226` (esp. 198-202) |
| No max-duration split | `:470-520` (merge ceiling only), `:522-540` (no cap) |
| Per-chunk framing vs continuous time axis (drift) | `:354-379` |
| `set()` boundary collapse | `:468` |
| Empty-mask skip → filler | `:474-475`, `:557` |
| Adapter returns padded bounds as canonical | `semantic_adapter.py:401-436` |

## Appendix C — Method & provenance

- **Empirical measurement:** `tools/vad_hypothesis_suite/measure_cuts.py` (uses the vendor's own `StreamFeatureExtractor`/`SemanticSegmenter`/`AdaptiveClassifier`; `SegmentationConfig(min_duration=12, max_duration=48)` to match qwen safe-chunking).
- **Independent code audit:** a senior-architect review agent instructed to disregard comments/docstrings and reason only from executable code; its findings on F4 (no split) were re-verified by direct code reading before inclusion.
- Both methods were performed **without knowledge of each other's conclusions** and converged.
