# `clustering_threshold` sets the number of global texture clusters, not scene granularity: the shipped scene count is set by `min_duration` and `snap_window`

Research note, 2026-09-05. Owner questions C1–C6 (session of 2026-09-05): *how can the detector
find larger clusters; the user expects a zoom-in/zoom-out lever; is the algorithm's logic sound;
if the science is sound, why does the implementation not deliver?* Accuracy against a scene
reference is **out of scope** (owner C5); this note is about the logic and about which knobs move the
result. No product file was changed for it.

Files: `semantic_lever_study.py` (the experiment), `lever_study_results.json` (every number below).
Re-run from the repo root: `python -X utf8 docs/research/semantic_scene_premise/semantic_lever_study.py
<wav> [<wav>] OUT.json` (the output path is the **last** argument; `-X utf8` because the engine prints a
non-cp1252 arrow). Results here: sklearn 1.6.1, numpy 1.26.4, scipy 1.17.0. An independent adversary
pass re-ran the study and reproduced every figure of the first version; its corrections are folded in.

## 1. The premise (owner C4)

"If the audio characteristics are similar, we are in the same scene. If the characteristics change and
stay changed for a period of time, a new scene has started." Two quantities define a boundary in that
sentence: the **size of the change** (a big change versus a small drift) and its **persistence** (for
a period of time). A granularity lever must act on at least one of them.

In signal-processing terms this is *temporal segmentation*: partition the timeline into contiguous
stretches that are internally homogeneous. The audio-segmentation literature does it with novelty
detection on a self-similarity matrix
([Foote 2000](https://www.researchgate.net/publication/3863771_Automatic_audio_segmentation_using_a_measure_of_audio_novelty),
[AudioLabs FMP notebook](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html)),
with model-selection change detection between adjacent windows
([Chen & Gopalakrishnan 1998, BIC](https://www.semanticscholar.org/paper/Speaker,-Environment-and-Channel-Change-Detection-Chen-Gopalakrishnan/84f343209a5072509b93d16db408d4e9ad88a8a6)),
or with hierarchical clustering **constrained to contiguous samples**
([scikit-learn: structured vs unstructured Ward](https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html),
[`connectivity` parameter](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)).
In each, the user-facing knob is a change magnitude and/or a persistence window. (Whether the final
segment count is monotone in that knob is measured below for the clustering family; it is not asserted
for the others.)

## 2. The algorithm as implemented (`whisperjav/vendor/semantic_audio_clustering.py`, read this session)

1. Features: 36 per frame — 13 MFCC, 13 deltas, RMS, ZCR, 7 spectral-contrast bands, chroma std
   (`FeatureRegistry` `:202-210`), hop 512 samples at 16 kHz.
2. Median smoothing over 15 frames (`:544`), sub-sampled every 15 hops = **0.48 s** per vector
   (`:546`; 8130 vectors for the 65-minute file), standardised (`:551-552`).
3. **`AgglomerativeClustering(n_clusters=None, distance_threshold=clustering_threshold, linkage="ward")`
   over all vectors of the whole file, with no connectivity constraint** (`:553`). Every vector gets one
   of *k* global texture labels; *k* falls as the threshold rises.
4. A raw boundary wherever consecutive vectors carry different labels (`:557-560`).
5. Move each boundary to the nearest silence within ±`snap_window` (default 5 s), widening to ±2× if
   none is found; duplicates collapse (`_snap_to_silence` `:578-672`).
6. Merge every segment shorter than `min_duration` (20 s) into a neighbour, under the `max_duration`
   ceiling (`_smart_merge` `:674`, `_forced_cleanup` `:726`, `_ensure_timeline_coverage` `:746`).
7. Classify each final segment from its own feature means and calibrated percentiles (`classify`
   `:807-830`) and build an ASR prompt. The classifier does **not** use the cluster labels; the labels
   are consumed only at step 4.

## 3. Hypotheses

- **H1** the exposed `clustering_threshold` is a granularity lever: the final scene count decreases
  monotonically as it rises.
- **H2** the final count is set mainly by the downstream repair stages (`min_duration`, `snap_window`).
- **H3** with the *same features and the same threshold*, Ward constrained to adjacent samples yields a
  count that decreases monotonically with the threshold.
- **H4** a change-point detector on the same features gives a window (persistence) and a magnitude
  lever that each move the count gradually.

## 4. Method

Features are extracted once per file with the engine's own extractor and classifier calibration
(same silence floor as `process_movie_v7`). Only the boundary logic is varied; every arm then runs the
engine's own snap → merge → cleanup → coverage code. Arm A was checked to be boundary-identical to
`SemanticSegmenter.segment` on the 293 s clip (same 7 starts) and its count matches the scene
inspector's real run (7). Inputs: `test_media/293sec-S01E04-scene4_extracted.wav` (293 s) and
`test_media/3630sec-HODV-22019.wav` (65 min of JAV). Nothing ran on the owner's F: drive; the owner's
own EKAI-023 runs are quoted from their result files.

- **A** shipped logic (unconstrained Ward, label flips).
- **B** Ward with a chain connectivity matrix (sample *i* may merge only with *i±1*); every cluster is a
  contiguous interval, so raw boundaries = segments − 1 in every row (a self-check that held).
- **C** change-point: cosine distance between the mean vector of the *W* seconds before and after each
  point; boundary where it exceeds mean + *k*·std of that file's distances and is a local maximum
  within ±*W*. Note *k* is relative to the file's own distribution, and *W* changes that distribution,
  so the two levers are not independent by construction.

## 5. Results

### 5.1 The exposed threshold under the shipped logic (A) — H1

65-minute file, `min_duration` 20, `snap_window` 5:

| threshold | global clusters | raw boundaries | **final scenes** |
|---|---|---|---|
| 5 | 4029 | 7965 | 43 |
| 10 | 885 | 7460 | 43 |
| 18 (default) | 181 | 6734 | 46 |
| 30 | 52 | 5791 | 58 |
| 50 | 14 | 4733 | 60 |
| 90 | 5 | 3187 | 65 |
| 200 | 2 | 1475 | 59 |
| 500 and above | 1 | 0 | 1 |

At one-unit resolution over the range the GUI slider exposes (5–30, `api.py:2251-2258` on
`dev_v1.9.2`, unreleased): 43 for 5–13, 45–46 for 14–23, 50 for 24–29, 58 at 30. On the 293 s clip:
6 for 5–11, 7 for 12–24, 8 for 25–27, 7 for 28–40, then 1 from threshold 90.

The owner's EKAI-023 runs (`F:\MEDIA_DLNA\EKAI-023\*\EKAI-023.semantic.scenes.json`): **119 scenes at
threshold 50, 121 at 90**, both `min_duration` 20.

**Reading.** The threshold does what it says on *clusters* (4029 → 1) but the scene count does not
follow: across the entire exposed slider range the count **rises** (43 → 58; 6 → 8) — the opposite of
the CFF2 help text "lower tends to give more, shorter scenes" — and it stays within 43–65 until the
threshold is high enough to leave a single cluster, when everything collapses to one scene. There is
no zoom: a plateau, an inversion, and a cliff. **H1 is refuted.**

### 5.2 Why: the raw boundaries are per-vector label churn

Established from the label sequences (65-minute file): at threshold 18 the 8130 vectors form 6735
label runs, of which **6060 are a single 0.48 s vector** (median run 1, longest 21); even at threshold
90 with only 5 clusters, 1801 of 3188 runs are single vectors. The raw boundary list is therefore not a
list of texture transitions; it is sampling noise at half-second resolution between a handful of large
global clusters. What survives to become a scene is decided entirely by steps 5–6 — which is why the
count barely responds to the threshold (H1) and why the next two knobs dominate.

### 5.3 What actually sets the count today — H2

Shipped logic, 65-minute file, threshold 18 unless stated:

| `min_duration` (s) | 5 | 10 | 20 | 30 | 60 |
|---|---|---|---|---|---|
| scenes | 354 | 130 | 46 | 22 | 12 |

| `snap_window` (s) | 1 | 2 | 5 (default) | 8 | 12 |
|---|---|---|---|---|---|
| scenes at threshold 18 | 14 | 22 | 46 | 55 | 70 |
| scenes at threshold 50 | 28 | 32 | 60 | 64 | 78 |

293 s clip: `min_duration` 5/10/20/30/60 → 27/15/7/4/2; `snap_window` 1/2/5/8/12 → 4/5/7/7/8 at
threshold 18.

**H2 holds, with two levers, not one.** `min_duration` is the persistence criterion of the premise and
behaves like one (30× range). `snap_window` — an exposed GUI slider ("Snap Window", YAML `:137-146`)
that every preset changes (conservative 6, aggressive 2, dialogue 3, music 2, action 8) — spans a
5× range on its own: a small window collapses many raw boundaries onto the same silence run, a large
one spreads them over distinct runs. The first version of this note held `snap_window` fixed and
declared it neutral; the adversary pass measured it, and these are my confirming numbers. Nobody
should read the presets as "threshold presets": their effect comes mostly from these two keys.
`max_duration` is not a factor (raising it to 100 000 changed 60 → 59 at threshold 50 and nothing
elsewhere; adversary measurement).

### 5.4 Same features, temporally constrained clustering (B) — H3

65-minute file, `min_duration` 20, `snap_window` 5, one-unit steps:

| threshold | 5 | 8 | 10 | 12 | 13 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| contiguous segments | 5446 | – | 1237 | – | – | – | – | 132 | – | – | – | – | – | 26 | – | – |
| **final scenes** | 52 | 72 | 82 | 92 | 93 | 85 | 79 | 65 | 59 | 50 | 44 | 34 | 27 | 25 | 21 | 14 |

Coarse: 50 → 8, 90 → 5, 200 → 1. 293 s clip: 6, 7, 7, 8, 8, 8 for 5–10, flat at 7 for 11–20, then
5, 4, 3 down to threshold 40.

**H3 holds above about threshold 14 and fails below it.** The *cluster* count is monotone everywhere
(no inversions found), but below ~14 the constrained logic produces thousands of sub-20 s segments and
`min_duration` merging makes the final count rise to a peak at 13 before it falls. Above 14 the
threshold is a clean zoom on the 65-minute file (93 → 14 across 13–40) but has a dead zone on the
short clip (flat at 7 for 11–20). Structured Ward therefore offers a monotone control over roughly
the upper two thirds of the currently exposed range, not all of it, and a shipped range would have to
be chosen from measurements like these.

### 5.5 The premise stated directly (C) — H4

65-minute file, `min_duration` 20, final scenes:

| window W | k = 1.0 | k = 1.5 | k = 2.0 | k = 3.0 |
|---|---|---|---|---|
| 5 s | 77 | 57 | 42 | 23 |
| 10 s | 52 | 42 | 32 | 12 |
| 20 s | 37 | 29 | 24 | 8 |
| 30 s | 30 | 22 | 15 | 6 |

293 s clip: W = 5 s → 8, 8, 7, **1**; W = 10 s → 8, 6, 1, 1; W = 20 s → 5, 2, 2, 1.

**H4 holds on the long file and not on the short one.** On 65 minutes both levers move the count
gradually in the expected direction. On 293 s the magnitude lever cliffs (7 → 1 between k = 2 and 3)
because *k* is a multiple of that file's own distance spread, and a 293 s file has too few distances
for the spread to be stable. A change-point form is promising; its levers are not yet shown to be
smooth on short inputs or portable across files without an absolute (not relative) magnitude scale.

## 6. Conclusions

- **The premise is sound and is how the field segments audio.** Similar texture along the timeline,
  with a persistence requirement, is a legitimate definition of a scene.
- **The knob the implementation exposes is not a granularity lever.** `clustering_threshold` controls
  how many global texture classes the whole film is sorted into. Scene boundaries are read off the
  label sequence, which at half-second resolution is mostly single-vector churn; the count is then
  fixed by `min_duration` and `snap_window`. Across the exposed slider range the count moves opposite
  to what the CFF2 help text says.
- **The implementation does have a granularity lever — two, in fact — under other names.**
  `min_duration` is the premise's persistence criterion and behaves like one; `snap_window` is an
  equally strong lever that nobody would guess controls scene count. The presets work because they
  change these keys.
- **Structured Ward and the change-point form both turn "how different / for how long" into working
  levers on the same features**, each with a measured limit (B: non-monotone below threshold ~14;
  C: relative *k* is fragile on short files). Whether their boundaries are *better* is untested here
  (C5) and would need the reference-based check the owner deferred.

## 7. Options for the owner (nothing implemented)

1. **Structured Ward.** Pass a chain connectivity matrix to `AgglomerativeClustering` at `:553`. The
   exposed threshold becomes a zoom over the upper part of its range; downstream unchanged. Needs: a
   re-measured slider range (dead zone below ~14), re-examined presets, and — only if the owner wants
   quality evidence — a reference-based check.
2. **Change-point boundary logic.** Replace steps 3–4 with the adjacent-window distance of §5.5, with
   *window* and an **absolute** change threshold (not the per-file *k*) as the two user levers. Slightly
   larger change; avoids the O(n²) clustering. Same caveat on quality evidence.
3. **Keep the algorithm, expose the real levers.** Present `min_duration` (and `snap_window`) as the
   granularity controls and stop presenting `clustering_threshold` as one. Honest and immediate; leaves
   the count non-monotone in the threshold and the raw-boundary churn in place.

Related, also the owner's call: the CFF2 control "Scene Change Threshold" on `dev_v1.9.2` (unreleased)
and its help text describe a direction the measurements contradict over the whole exposed range.
Choices: correct the text, repoint the control at `min_duration`, or withhold the control until
option 1 or 2 lands. I have changed nothing.

## 8. Limits

Two files, both Japanese drama/JAV audio, one feature set, no human scene reference by design. The
counts are what the logics produce, not how right their boundaries are. Arm C's local-maximum test
accepts ties (cosmetic at these counts). The snap-window mechanism in §5.3 is a reading of
`_snap_to_silence`'s duplicate collapse (`:672`), consistent with the numbers but not separately
instrumented.
