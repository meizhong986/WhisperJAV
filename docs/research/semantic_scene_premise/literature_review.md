# The science behind "semantic" audio scene detection: what the literature establishes

Research note, 2026-09-05, companion to `README.md` (the measurements). Owner question: is the premise
"similar audio characteristics = same scene; a sustained change of characteristics = a new scene" sound
science, and if so why does the shipped implementation not behave? Accuracy of any method is out of
scope; this note is about what the field knows, what it assumes, and where the shipped logic departs.

Every quotation was checked against the source as read in this session (papers as extracted text, web
pages as fetched). Causal sentences about the shipped code are labelled **established** (read at file:line
or measured in `README.md`) or **hypothesised**. Sources I could not read are named in §7 and never quoted.
An assessment-adversary pass ran on the first draft; its ten findings were verified against the sources
and are folded in (details in §8).

## 1. Two different things are called "acoustic scene"

**Acoustic scene classification (ASC)** assigns a label to a whole recording that "characterizes the
environment in which it was recorded" — airport, tram, urban park ([DCASE 2020 task
definition](https://dcase.community/challenge2020/task-acoustic-scene-classification)); the 2020
task's ten classes group into indoor / outdoor / transportation ([DCASE 2020 challenge
paper](https://arxiv.org/pdf/2005.14623)). The question is *where was this recorded*, not *where does
one part end and the next begin*. Early ASC systems used MFCC features with GMM/SVM classifiers; the
field has moved to log-mel spectrograms and CNNs ([ASC survey](https://www.sciencedirect.com/science/article/abs/pii/S0957417423024041), abstract).

**Audio scene segmentation** is the temporal problem: partition a stream into contiguous, internally
homogeneous stretches. That is what the WhisperJAV premise describes, and it has its own literature (§3).

The distinction is a preliminary, not the explanation: the two share features, but a method built for
the first question does not by itself answer the second.

## 2. The premise: is "long-term audio consistency = scene" in the literature?

Yes, under a carefully qualified meaning of "scene". The clearest treatment is Sundaram and Chang's
**computable scenes** (Columbia; ACM Multimedia 2000, IEEE Trans. Multimedia 2002, book chapter 2003 —
the chapter is what was read here). They define a video scene by long-term consistency of chromaticity
and lighting, and:

> "an audio scene exhibits a long terms [sic] consistency with respect to ambient sound. We denote them
> to be computable since these properties can be reliably and automatically determined using low-level
> features present in the audio-visual data. **Note that these are not semantic scenes.**"

Five points from that chapter matter here.

- **Persistence is part of the definition.** Footnote 1: "Analysis of experimental data (one hour each,
  from five different films) indicates that for both the audio and the video scene, a minimum of 8
  seconds is required to establish context. These scenes are usually in the same location (e.g. in a
  room, in the marketplace etc.) are and [sic] are typically 40~50 seconds long." The premise's "for a
  period of time" is not decoration; it is what makes the definition computable.
- **Their audio detector compares recent audio against a longer memory; it does not cluster.** A causal
  FIFO memory holds the last *T_m* seconds; the most recent *T_as* of it is the attention span ("typical
  values for the parameters are Tm=32 sec. and Tas=16 sec."). "In order to segment the data into audio
  scenes, we compute correlations amongst the audio features in the attention-span with the data in the
  rest of the memory." Scene changes are read from the extrema of that comparison (the sentence that says
  so also covers the video stream, whose key-frames are compared by colour-histogram coherence). The
  chapter gives typical values for *T_m* and *T_as*; it does not present them as a user granularity
  control, and no sensitivity analysis is given. Their dedicated audio paper ("Audio Scene Segmentation
  Using Multiple Features, Models And Time Scales", ICASSP 2000, their ref. [47]) was not accessible to me.
- **The framework is multimodal and fused.** Computable scenes are audio-visual; the audio and video
  detectors are combined with silence detection ("The silence is detected via a threshold on the average
  energy; we also impose minimum duration constraints on the detector") and with dialogue-structure
  detection (a periodic-analysis transform tested with Student's t-test), because "the computational
  model cannot disambiguate between two cases involving two long and widely differing shots: (a) in a
  dialog sequence and (b) adjoining video scenes." A **minimum-duration constraint is therefore part of
  the science**, not foreign to it.
- **Validation is qualitative.** "We validated the computable scene definition, which appeared out of
  intuitive considerations, with actual film data. The data were from three one-hour segments from three
  English language films. … In most cases, the c-scenes are usually a collection of shots that are
  filmed in the same location and time". The chapter then gives a type census of one film hour; the
  detector evaluation is in their TMM 2002 paper ("Our experiments [51] show that the c-scene change
  detector and the structure detection algorithm work well"), which I did not read.
- **They criticised threshold-driven clustering — of video shots.** On scene-transition graphs (STG,
  Yeung and Yeo: time-constrained clustering of shots with a maximum cluster diameter δ and a time window
  *T*): "An important concern in this work is the setting of the cluster threshold parameter δ and the
  time-window size *T*, both of which critically affect the segmentation result. Unfortunately, neither
  of these parameters can be set with [sic] taking the specific character of the data being analyzed."
  Elsewhere: "cluster thresholds are difficult to set and often have to be manually tuned." The STG
  method itself is described as "time-constrained clustering of video frames along with cluster label
  transition analysis" — boundaries read off changes of cluster label, **with** a time constraint. This
  is the nearest ancestor of the shipped operator in the literature I read.

So: the premise appears in the literature as a *defined and qualitatively validated* construct, with
"scene" meaning an acoustically homogeneous stretch of at least several seconds, explicitly *not* a
narrative scene, and in practice fused with silence and dialogue cues before being called a scene.

## 3. How the temporal problem is solved, and where the granularity lever sits in each family

The music-structure literature names three complementary principles — **repetition**, **homogeneity**,
**novelty** — "principles that also apply to other types of multimedia beyond music"
([AudioLabs FMP, chapter 4](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4.html)). The
WhisperJAV premise is the homogeneity/novelty pair.

**Novelty (Foote 2000).** Build a self-similarity matrix of frame features, correlate a checkerboard
kernel along its diagonal, take peaks of the novelty curve as boundaries
([FMP novelty notebook](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html)).
The [CCRMA report describing the method](https://ccrma.stanford.edu/~unjung/AIR/report4web.pdf) (read as
text; Foote's own paper was not accessible) states the lever: "The width of the kernel W directly affects
the properties of the novelty measure. A small kernel detects novelty on a short time scale, such as beats
or notes. Increasing the kernel size decreases the time resolution, and increases the length of novel
events that can be detected. Larger kernels average over short-time novelty and detect longer
structure". The same report gives Foote's position on features as reported speech: "He mentions that …
the actual parameterization is not crucial as long as 'similar' sounds yield similar parameters", and
that MFCCs "will tend to match similar timbres rather than exact pitches". Kernel width is a
*persistence* lever; the peak threshold is a *magnitude* lever.

**Model-selection change detection (BIC).** Chen and Gopalakrishnan (1998) test, for each candidate
point, whether one Gaussian or two fits the adjacent windows better, penalised by the Bayesian
Information Criterion ([record](https://www.semanticscholar.org/paper/Speaker,-Environment-and-Channel-Change-Detection-Chen-Gopalakrishnan/84f343209a5072509b93d16db408d4e9ad88a8a6),
abstract only). Cettolo, Vescovi and Rizzi ([Computer Speech & Language 19, 2005](https://www.cs.helsinki.fi/group/aprill/asp/papers/CSL19.pdf),
read as text): "Typically, BIC is applied within a sliding variable-size analysis window where single
changes in the nature of the audio are locally searched"; "it is known that short segments are not well
handled by the local algorithm"; a global dynamic-programming variant "outperforms the local one by 2.4%
(relative) F-score in the detection of changes, but is 38 times slower". Evaluation uses a tolerance
around each reference change (0.3–0.5 s in their experiments) and precision/recall. Levers: the penalty
weight (magnitude) and the window (persistence).

**Global partition with a temporal-continuity term.** Some methods do assign every frame of the whole
recording to a recurring class and read boundaries off label changes — but always with a term that makes
the labels temporally coherent. Levy and Sandler cluster frame histograms "with temporal continuity
expressed as constraints modeled by a hidden Markov random field" ([IEEE TASLP 2008](https://ieeexplore.ieee.org/document/4432648),
abstract only); Yeung and Yeo's STG (via Sundaram, §2) is *time-constrained* clustering plus label
transition analysis; spectral clustering as used for music structure operates on a self-similarity matrix
whose construction encodes temporal proximity. Agglomerative clustering restricted to merging temporally
adjacent samples is the simplest member of this family. The continuity term is what turns a class
sequence into contiguous segments; the cut height or class count is then the granularity lever.

**Deep embeddings.** A 2026 evaluation runs Foote kernels, spectral clustering and correlation
block-matching on sequences of pretrained audio embeddings and finds that "modern, generic deep
embeddings generally outperform traditional spectrogram-based baselines, but not systematically"
([arXiv 2603.27218](https://arxiv.org/abs/2603.27218), abstract). The boundary logic is unchanged; only the
features change.

**Common ground.** Every method in this literature carries an explicit temporal term: a kernel width, an
analysis window, a memory and attention span, a time constraint or continuity prior on the clustering.
The granularity lever in every case is that temporal term, a change-magnitude threshold, or both. I found
no published method that partitions a recording into classes with **no** temporal term and then cuts at
every label change.

## 4. Evidence for film-like and JAV-like material

- Sundaram and Chang's film validation (§2) is the closest: qualitative, on three hours of English film,
  for an audio-visual construct fused with silence and dialogue cues.
- Modern movie-scene segmentation defines a scene narratively — "Scene, as the crucial unit of
  storytelling in movies, contains complex activities of actors and their interactions in a physical
  location" ([MovieNet scene segmentation](https://movienet.github.io/projects/cvpr20sceneseg.html)) —
  and is visual-first. A 2026 audit of the MovieNet-SSeg annotations under a stricter narrative definition
  finds only 11.4% of boundaries include the transition type that definition requires (Table 4): "many
  MovieNet-SSeg boundaries align with visually salient transitions even when they do not correspond to
  full narrative scene transitions" ([arXiv 2608.28699](https://arxiv.org/html/2608.28699)). It reports no
  audio-only numbers. Search-engine summaries of other work claim audio alone is insufficient for movie
  scene segmentation — **not verified against a read source**.
- No work found on audio-only scene segmentation of adult video. The material has properties the
  literature's assumptions do not cover: long single-location scenes whose *narrative* changes (new act,
  new partner, new position) are often not *acoustic* changes, and frequent within-scene alternation of
  speech, non-speech vocalisation and music that *is* an acoustic change. Sundaram's caveat applies with
  force: computable scenes are not semantic scenes.

## 5. Reading the shipped implementation against this literature

`whisperjav/vendor/semantic_audio_clustering.py` (read this session). **Established from the code:**
36-dimensional frame features (13 MFCC, 13 delta, RMS, ZCR, 7 spectral-contrast bands, chroma std) are
median-filtered over 15 frames (`:544`), subsampled to one vector per 0.48 s (`:546-548`), standardised
(`:551-552`), and clustered over the **whole file** with
`AgglomerativeClustering(n_clusters=None, distance_threshold=…, linkage='ward')` and no `connectivity`
argument (`:553`). A boundary is placed wherever consecutive vectors carry different labels (`:557-560`);
boundaries are then snapped to silence (`:565` onward) and segments shorter than `min_duration` are
merged (`:674` onward).

**Established (code + §3):** this is the global-partition family of §3 with its temporal-continuity term
removed. The 15-frame median filter smooths each feature over about half a second; it is not a
continuity constraint on the labels. Nothing between clustering and the `min_duration` merge asks how
long a change lasts or how large it is relative to its neighbours. The exposed `clustering_threshold` is
the cut height of a partition that has no temporal term, so it sets how many texture classes the file is
divided into — the δ that Sundaram and Chang said "critically" affects the result and cannot be set from
the data — while the time constraint *T* they paired it with is absent.

**Established (README.md):** the measurements match this reading in one respect: at the two thresholds
where label runs were counted (18 and 90), most runs are a single 0.48 s vector, and the final scene count
is set by `min_duration` and `snap_window`. **Not explained by this reading:** the measured *inversion*
(final scenes rising from 43 to 58 over thresholds 5–30 while clusters fall from 4029 to 52). Global-class
clustering predicts insensitivity to the threshold, not a rise; that remains open.

**Hypothesised, not measured:** whether the 36-dimension z-scored feature vector is itself a good basis
for Ward's variance criterion. No arm of the study varied features (`README.md` §8), and the literature
read here says nothing about this particular vector. The features are neither exonerated nor implicated.

## 6. Answers to the owner's questions

- **C4 — is the premise sound?** It is a recognised construct: Sundaram and Chang's *computable* audio
  scene, an acoustically homogeneous stretch of at least several seconds, defined and qualitatively
  validated on film, explicitly "not semantic scenes", and in their system fused with silence and
  dialogue cues. Every temporal-segmentation family in §3 is built on the same homogeneity/novelty idea.
- **C1/C2 — what is the zoom lever in this science?** A temporal term, a change magnitude, or both:
  kernel width and peak threshold (novelty), window and penalty (BIC), memory and attention span
  (Sundaram), a time constraint or continuity prior plus cut height (global partition). In every case the
  lever acts on *how long* and *how much* nearby audio must differ.
- **C6 — if the science is sound, why does the implementation not work?** Established: the shipped
  operator belongs to the global-partition family but lacks the temporal term that every published
  member of that family carries, so its threshold controls class count rather than boundary
  persistence or magnitude; the only temporal term left, `min_duration`, therefore governs the count.
  The rise in scene count over the slider range is measured but not explained by this reading.
- **What the literature does not settle for us.** Whether any audio-only method places boundaries where a
  JAV viewer would put a scene change. Computable scenes coincide "usually" with same-location film
  scenes; adult video's narrative changes are frequently not acoustic changes. That is an empirical
  question about this material and was deliberately excluded here.

## 7. Sources

Read in this session:
- Sundaram, H.; Chang, S.-F. "Video Analysis and Summarization at Structural and Semantic Levels", book
  chapter (2003), §4 — [PDF](https://sundaram.cs.illinois.edu/pubs/2003/2003sundaram_video_bc.pdf),
  extracted to text.
- Cettolo, M.; Vescovi, M.; Rizzi, R. "Evaluation of BIC-based algorithms for audio segmentation",
  Computer Speech & Language 19 (2005) — [PDF](https://www.cs.helsinki.fi/group/aprill/asp/papers/CSL19.pdf).
- CCRMA student report on Foote's novelty method — [PDF](https://ccrma.stanford.edu/~unjung/AIR/report4web.pdf).
- Heittola, Mesaros, Virtanen, "Acoustic scene classification in DCASE 2020 challenge" — [PDF](https://arxiv.org/pdf/2005.14623);
  [DCASE 2020 task page](https://dcase.community/challenge2020/task-acoustic-scene-classification).
- Web pages: [FMP chapter 4](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4.html),
  [FMP novelty notebook](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html),
  [arXiv 2608.28699](https://arxiv.org/html/2608.28699), [MovieNet scene segmentation](https://movienet.github.io/projects/cvpr20sceneseg.html).

Abstract only: Chen & Gopalakrishnan 1998; Levy & Sandler 2008; arXiv 2603.27218; MovieNet
(arXiv 2007.10937); the ASC survey.

Not accessible, not quoted: Foote 2000 (original); Sundaram & Chang ICASSP 2000 (audio-only paper) and
IEEE TMM 2002 (detector evaluation); Levy & Sandler full text; Yeung & Yeo STG papers.

## 8. Adversary pass

The first draft claimed that no method in the literature derives boundaries from a global partition and
called the shipped step "the ASC operation". The adversary refuted both from the draft's own citations
(Levy & Sandler; spectral clustering) and from Sundaram's description of STG as "cluster label
transition analysis"; the correct differentiator is the **missing temporal-continuity term**, and §3, §5
and §6 were rewritten around it. Also accepted: the features exculpation was unmeasured (removed); the
"predicts the measurements" claim ignored the inversion (now stated as open); "exactly the structure
Sundaram and Chang criticised" overstated a criticism of video shot clustering that had a time
constraint (softened); Foote quotations were attributed to a paper not read (re-attributed to the CCRMA
report); Sundaram's validation was qualitative and multimodal (§2 rewritten); three quotations were
silently repaired (restored with [sic]); one splice hid that the quoted sentence also covered video
(un-spliced); scikit-learn API links read as a fix in citation form (removed); causal sentences lacked
labels (added); the median filter was omitted (added).
