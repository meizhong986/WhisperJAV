# WhisperJAV v1.9.2 — Development Change Log

> Developer-facing record of what changed on `dev_v1.9.2`, why, and how it was
> verified. `docs/release_notes_v1.9.2.md` is the user-facing text derived from
> this file. Keep both current; this one carries the detail (files, decisions,
> tests, follow-ups) that the release notes and user guide are written from.
>
> Conventions: one entry per landed change, newest first. "Decision" lines
> record who decided what, so a later reader can tell policy from mechanism.
>
> **SYNC** — pack r2.6 · 2026-09-06 | tracker rev 51.6 | change log through 2026-09-09 (balanced pipeline: semantic scene default + Silero VAD version selection, **uncommitted**) | `dev_v1.9.2` @ bc2474b + 24 modified / 3 new files, uncommitted |
> GitHub 134 open · 231 closed · 12 PRs · 0 labels applied · 32 owed (8 replies posted 2026-09-06; #413 follow-up at 16:11 UTC). Owner pack: https://claude.ai/code/artifact/73c5c95d-0a92-49d1-b127-fb23c02029c2
> (updated in place; never a second page). Rule: a session that changes the pack, this file or the change
> log brings the other two to the same state before it ends (CLAUDE.md, Assessment discipline, rule A7).

---

## 2026-09-09 — the balanced pipeline picks which Silero VAD it runs, and drops the external speech segmenter

**Owner requirements:** `Requirements_balanced_pipeline_v192.txt` S2, S6, S7, S8, S9, S9.1, S9.2, plus
the clarifications typed the same day. The owner stated the two user-facing problems this addresses:
external VAD makes balanced run at 0.7-1x realtime, and the internal VAD skips major spoken stretches.
S4/S5 (offload and reload the model per scene) he withdrew — the existing 20-minute refresh stands.

**Decision (owner, 2026-09-09):** a NEW `--vad-version` flag rather than overloading
`--speech-segmenter`; the trimming is BALANCED-ONLY, so fidelity, qwen, anime-whisper and non-balanced
ensemble passes keep their segmenters; thresholds are settled at conservative 0.5 / balanced 0.4 /
aggressive 0.3 for every version and are not to be re-measured; all three models ship inside the
wheel; `faster-whisper` and `ctranslate2` are pinned exactly; `--no-vad` goes; one INFO line names the
running version, with no checksums or model probing.

**His premise S6B1/S6B2 is false, and the correction is the whole design.** faster-whisper does not
choose its Silero model from any cache that can be pre-populated. It builds a hard-coded path inside
its own package — `vad.py` → `os.path.join(get_assets_path(), "silero_vad_v6.onnx")` — verified in
1.0.2, PyPI 1.2.1, SYSTRAN git master and the 2.1.1 fork. There is no environment variable, no cache
directory and no hook, and the file cannot simply be swapped because the three Silero generations have
different ONNX input signatures. The reference script the owner cited pre-populates *stable-ts*'s
`cached_model_instances['silero_vad']`, a different library on a code path balanced does not take.
So the choice is made one level up: `whisperjav/modules/silero_vad_adapter.py` rebinds the module-level
`faster_whisper.vad.get_vad_model`. Everything above the model — `get_speech_timestamps`, the
hysteresis, the padding, the chunk assembly — stays faster-whisper's own code, untouched, and it is the
only consumer of that function in the package.

**Three traps decided correctness, and each produces plausible-looking wrong output rather than an
error.** v3.1's ONNX output is two-class and index 1 is speech (index 0 gives a smooth, believable
series that is not speech probability). All three builds are stateful and batch-1, so the LSTM state
carries window to window and must reset per audio. And the 64-sample context concatenation is a v5/v6
input convention that belongs to 6.2 alone. The 6.2 adapter reproduces faster-whisper's own bundled
model to three decimal places on a real clip (mean probability 0.129, 12.1 % of windows above 0.40,
identical), which is the independent check that the context handling is right.

**Area:** `modules/silero_vad_adapter.py` (new); `assets/vad/*.onnx` (new, 4.7 MB, MIT);
`config/components/vad/faster_whisper_vad.py` (`version` field + the owner's thresholds);
`modules/faster_whisper_pro_asr.py` (`_install_vad_version`, the C11 INFO line);
`main.py` (`--vad-version`, `--pass1/2-vad-version`, `validate_balanced_vad_options`, `--no-vad`
removed); `ensemble/pass_worker.py`; `webview_gui/{api.py,assets/app.js,assets/index.html}`;
`pyproject.toml`; `installer/core/registry.py`; tests.

**C11, one INFO line per run.** `VAD: Silero v3.1, threshold 0.40`, printed by the entry point --
`main.py` for a single-pass run, the pass worker for each ensemble pass. Not by the recogniser: it is
rebuilt on every model refresh (a fresh worker process every 20 minutes of scene audio by default), so
a line there repeated about six times on a feature-length film. The recogniser logs what it actually
loaded at DEBUG and **warns** if that is not the requested build, so the one INFO line cannot become a
claim about a version that is not running.

**Where the adapter is installed, and why there.** `FasterWhisperProASR.__init__` installs it when the
backend is the built-in VAD. That constructor runs in whichever process hosts the model — the parent
when `--model-refresh-audio-minutes 0`, the spawned worker otherwise
(`modules/asr_worker_proxy.py:_asr_worker_main`), and its own worker again for each ensemble pass — so
one site covers every case. It is process-global by nature; balanced runs nothing else in that process.

**S2/S9, the external segmenter.** `--speech-segmenter` with `--mode balanced`, and
`--passN-speech-segmenter` with a balanced pass, are now usage errors (exit 2, before any
transcription) instead of silently producing a different pipeline. `pass_worker._apply_gui_overrides`
additionally normalises a balanced pass to the built-in VAD with a warning, so a stale saved preset or
a hand-written `pass_config` that never passed argparse behaves the same way. In the Ensemble tab the
Speech Segmenter dropdown becomes the VAD *version* selector for a balanced pass — Silero 3.1
(selected), 4.0, 6.2 and nothing else — and switches back to the external list for any other pipeline.

**Two defects found on the way, both fixed here.** `get_segmenter_schema("faster-whisper")` returned
"Unknown segmenter backend", so the Customize → Segmenter tab has been showing an error for every
balanced pass; it now returns a schema built from the Pydantic component. And
`_apply_custom_params` routed segmenter parameters by `seg_backend.startswith("silero")`, sending the
built-in VAD's parameters to `params["speech_segmenter"]` where nothing reads them — so every
Customize edit on a balanced pass was silently dropped. Both are pre-existing, neither was reported.

**Verified by running it, not by reading it** (the owner does not accept reported tests otherwise):

| what | evidence |
|---|---|
| the INFO line, in the spawned worker | `VAD: Silero v3.1, threshold 0.40` on a real 293 s run |
| the version actually changes transcription | same clip, same settings: 3.1 → 32 cues, 4.0 → 50, 6.2 → 28 |
| ensemble matches single-pass | `--ensemble --pass1-pipeline balanced --pass1-vad-version 4.0` → 50 cues, identical |
| thresholds | `--dump-params` balanced × 3 sensitivities → 0.5 / 0.4 / 0.3, version 3.1 |
| the rejections | balanced + `--speech-segmenter` → exit 2; balanced pass + `--passN-speech-segmenter` → exit 2; `--vad-version` off balanced → exit 2; a qwen pass keeps `ten` |
| `--no-vad` | absent from `--help`; `--no-vad file.wav` → exit 2, "unrecognized arguments" |
| the GUI dropdown | app.js driven against the real `index.html` option lists: Balanced → `3.1, 4.0, 6.2` with 3.1 selected; Fidelity → the external list back; the choice survives a round trip; the run payload sends `speechSegmenter: null` + `vadVersion` |
| tests | `tests/test_vad_version_v192.py` 32 passed; `test_scene_clustering_threshold_v192.py` 9 passed; `webview_gui/test_api.py` 9 passed |

**Measured, and relevant to the second user problem.** On a 293 s clip at threshold 0.40, the model
faster-whisper bundles finds 21.8 % of the audio to be speech; Silero 3.1 finds 31.2 % and 4.0 finds
44.8 %. That is the mechanism behind "the output skips over major parts of the movie", and it is now a
user-facing choice. **This is one clip and it is not a recommendation** — the default stays 3.1 per S8,
and no claim about which version is better belongs in the release notes until a feature-length A/B exists.

**Throughput.** ONNX only, never the torch-JIT: measured per 60 s of audio, single thread, JIT v3.1 12×
realtime against ONNX v3.1 87×. In the pipeline the adapters run at 77× (3.1), 146× (4.0) and 188×
(6.2) realtime, so a two-hour film costs roughly 90-140 s of CPU for the VAD pass.

**Pins (owner decision, 2026-09-10).** `ctranslate2==4.8.1` exactly, and **faster-whisper pinned to a
COMMIT rather than a release**: SYSTRAN master @ `ed9a06cd89a93e47838f564998a6c09b655d7f43`, which is
exactly three commits past the v1.2.1 tag — `cf42429` (drops a deprecated download argument, #1389),
`2eeafe0` (**replaces the bundled Silero weights with v6.2**, #1390) and `ed9a06c` (adds VAD
parameters, #1386). The owner runs that build, it is what he tests against, and #1390 is the reason:
the 1.2.1 release still carries the older Silero v6 weights.

Established, not inferred: his installed `vad.py` and `transcribe.py` are byte-identical to
`upstream/master`, and the bundled `silero_vad_v6.onnx` (md5 `67e11e5a`) is the file master added in
#1390. It produces the same probabilities as the Silero 6.2 model WhisperJAV ships — measured on a
real clip, mean 0.129 and 12.1 % of windows above 0.40 for both, and the same 31 speech chunks — so
`--vad-version 6.2` and this build's own default are the same weights.

Pinning the commit rather than the branch is what keeps installs reproducible: `master.tar.gz` moves,
a commit does not. `pyproject.toml`, `whisperjav/installer/core/registry.py` (as a git source) and
`uv.lock` all carry it, and the installer validation passes. **This also closes the note that used to
sit here:** the development environment and what users receive are now the same code.

**Both gates ran, and both found real defects.** `call-chain-verifier` returned WIRED on all three
chains but caught two pieces of CLI surface that S2/S9 had made unreachable while still advertising
themselves: `faster-whisper` was still a `--speech-segmenter` choice whose help said "balanced only"
(balanced rejects it), and the firered-vad / ten routing-guard exemption still told users those
backends work on `--mode balanced`. Both removed. Its third finding, that `init()` never calls
`populateSegmenterOptions`, is wrong -- it does, at `app.js:1687`; the agent read the file before that
call landed.

`assessment-adversary` was given **the owner's requirements file**, not a plan written here. It could
not break the adapter: it reproduced all three builds against reference implementations (torch-JIT
v3.1, torch-JIT v4.0, the `silero_vad` pip package) on real audio at correlation 1.00000, and found no
path where a balanced run still receives an external segmenter. What it did find, all now fixed:
`pass_worker` wrote `vad_version` into `params["vad"]` for **any** pipeline, where for a non-balanced
pass that block belongs to the external Silero segmenter -- which has a constructor argument of its own
called `version`; `--passN-vad-version` was accepted and silently ignored on a non-balanced pass while
`--vad-version` exits 2 on a non-balanced mode; **C11 was violated** (two lines, the second once per
model refresh); `--max-group-duration` and `--chunk-threshold` were live no-ops on balanced, which the
prior investigation had named "to trim"; `install()` returned the requested version even after falling
back; four pieces of over-engineering (a prefix-tolerant `normalise_version` no producer can exercise,
a `VAD_VERSION_LABELS` comment claiming two consumers it does not have, an `is_available()` with no
production caller, and a `--vad-version` emitter on a GUI tab that has no VAD control) -- all cut; the
adapter's `__call__` is not reentrant where the code it replaces is; and `uv.lock` still carried
`faster-whisper>=1.1.0` with no ctranslate2 entry, so the exact pins existed in two manifests of three
(`uv lock` regenerated, a 3-line change).

**Owner decisions, typed 2026-09-09 after those asks were put to him.** *"'omitted' mean omitted...
If a CLI user script tries to use those old APIs, that shall be a failure. Ie backward compatibility
shall be broken."* So `--speech-segmenter` on `--mode balanced` stops the run, and so do
`--max-group-duration` and `--chunk-threshold` there — they set how an external speech segmenter
groups what it found, and Balanced has none, so they were changed from a warning to a stop. On the
fallback: *"The fall back shall be to use the original faster whisper. If that too fails then it is a
failure and the process shall stop!"* — a Silero build that will not load now falls back to the model
faster-whisper ships with, loudly, and if that will not load either the run stops with a message
telling the user to reinstall. And the user-facing name is his: **"Internal FW Silero VAD"**, in the
Customize panel and in the dropdown.

Two of the six asks he could not read, because they were written in a metaphor ("escape hatch") and
in an internal constant name. Rewritten in plain language in
`docs/plans/V192_BALANCED_IMPLEMENTATION_PLAN.md` §6 and both answered on 2026-09-10: his 6 September
decision letting FireRedVAD and TEN be chosen on Balanced is superseded — *"removed speech segmnenters
is correct. No exceptions."* — and the loss of the "suspect" result on Balanced he treats as following
from his own requirement rather than a separate decision: *"The question is mute as no external speech
segmente is available for balanced mode."* The rule that came out of the wording failure is recorded in
memory: no metaphors, no internal names, no jargon he has not used himself.

Its most important finding is not a bug: **product decisions were being presented as implementation.**
Failing a run, the exit-code contract and anything a user scripts against are the owner's, not an
implementation detail (CLAUDE.md A4). All six were put to him and all six are answered —
`docs/plans/V192_BALANCED_IMPLEMENTATION_PLAN.md` §6 carries the question, his words and the state.

**Not done / open:** the process-global rebind also reaches kotoba and stable-ts's batched path if one
of them runs in the same process as a balanced recogniser — balanced does not, but scoping it is an
owner decision. The "suspect" result is gone from Balanced for good: the check
needs a second speech detector to compare against and the built-in VAD provides none, and choosing an
external segmenter -- the only way a user could switch it back on -- no longer exists. Owner
2026-09-10: *"The question is mute as no external speech segmente is available for balanced mode."* Four pre-existing test failures in `tests/config/{test_resolver_v3,test_legacy}.py` assert
old *silero* component values and are unrelated to this change.

---

## 2026-09-09 — scene detection: semantic becomes the default, and each backend finally gets its own parameters

**Owner decisions (typed 2026-09-09):** flip the default scene detector to semantic; make it the
default for every Transcription-tab pipeline and the matching CLI; raise the `le=300` bound;
`fast`/`fidelity` keep semantic's own 20 s/420 s while **balanced** gets 28 s/1200 s; a 1200 s ceiling
equal to the 20-minute model-refresh budget is accepted while #394 is open; add a semantic *component*
rather than post-mutating the resolved config. The flip itself was first approved on 2026-04-20 for
v1.8.12 (`memory/project_v1812_default_backends_flip.md`) and never applied to the CLI, while the GUI
Ensemble tab has shipped `semantic selected` since v1.8.11 — the two entry points disagreed until now.

**Area:** `config/components/features/scene_detection.py` (new `SemanticSceneDetectionOptions` +
`SemanticSceneDetection`; `max_duration_s` bound `le=300` → `le=1200` on the auditok **and** silero
option classes); `config/components/features/__init__.py`; `config/legacy.py`
(`SCENE_FEATURE_BY_METHOD`, `_select_scene_feature`, `_normalise_scene_method`, new `scene_method`
parameter on `resolve_legacy_pipeline`, per-backend `scene_overrides` on the balanced entry);
`config/segmenter_presets.py` (`DEFAULT_SCENE_DETECTOR`); `main.py`; `ensemble/pass_worker.py`;
`webview_gui/api.py`; `webview_gui/assets/app.js`;
`config/v4/ecosystems/tools/semantic-scene-detection.yaml`; tests.

**Mechanism it fixes (established, executed not read):** every scene-detecting pipeline declared
`auditok_scene_detection` regardless of `--scene-detection-method`, so a semantic run received
auditok's parameter names. The semantic backend reads `min_duration`/`max_duration` without the `_s`
suffix (`semantic_backend.py:68-73`) and the factory passes kwargs through untranslated
(`factory.py:239`), so **every semantic run since v1.8.11 — including every GUI ensemble pass — ran on
the engine's hard-coded 20/420 defaults**, whatever the configuration said. The feature component is
now selected from the backend that will actually run.

**The trap this closes:** auditok's config builder falls back to the bare names when the `_s` ones are
absent (`auditok_backend.py:152-153`), and its `min_duration` *discards* shorter regions rather than
merging them (`:437` falls through, `:466` drops). Handing auditok a semantic `min_duration` of 28
would therefore delete every region under 28 seconds, with its audio. Verified by `--dump-params`:
`--scene-detection-method auditok --mode balanced` resolves the `_s` names at 1200/1200 and carries no
bare `min_duration` at all.

**Verified by execution:** `--help`; `--dump-params` for balanced/fast/fidelity/faster × 3
sensitivities (balanced 28/1200 on all three, fast/fidelity 30/420·20/420·10/180, `faster` gains no
scene feature); explicit `auditok`/`silero`/`semantic` on balanced; the clustering-threshold warning
now keys off the *effective* method; two end-to-end balanced runs, exit 0.
`tests/test_scene_clustering_threshold_v192.py` 9 passed after updating two tests whose premise was the
old default. `tests/config/test_resolver_v3.py` has 2 failures that are **pre-existing** — identical
with these changes stashed out.

**Not verified, and it is the open question:** what the new default does to output at feature length.
Two single-run A/Bs, same command but for the flag:

| clip | detector | cues | speech span | last cue | characters |
|---|---|---|---|---|---|
| Netflix drama, 293 s (GT: 68 cues, 46.9%) | semantic | 13 | 11.7% | 1:48 | 210 |
| | auditok | 24 | 13.6% | 3:02 | 312 |
| JAV SONE-966, 464 s | semantic | 17 | 7.7% | 2:53 | 369 |
| | auditok | 16 | 7.4% | 4:11 | 200 |

They disagree: auditok is clearly better on the drama clip, semantic yields more text on the JAV clip.
The only pattern common to both is that **semantic's last cue is earlier**, which is an observation,
not a cause. n=2, one run each, both far below ground truth for reasons that pre-date this change.
A feature-length A/B on `EKAI-023` (179 min) is the test that speaks to the reported failure mode
(output stopping after 60-110 minutes) and has **not** been run. The default flip should not be
described to users as an improvement until it has.

**Deliberately not done:** `SceneOverlapResolver` is NOT wired into balanced/fast/fidelity. Semantic
returns padded, overlapping scenes — measured on EKAI-023: 51 seams, 0.100-0.700 s each, median
0.414 s, 18.0 s total, 0.168% of the film — while the resolver's Rule 1a drops any fully nested cue on
geometry alone with no text comparison (`scene_overlap_resolver.py:104-107`). On balanced a stitched
cue can run to 30 s, so a short interjection at a scene head could be deleted. The fix is larger than
the exposure; revisit with a kill switch if duplicate cues are observed.

**Follow-ups:** the feature-length A/B above; `brute_force_chunk_s` stays 29 s (bounded `le=120`), so
that rare auditok fallback still chunks at 29 s regardless of the ceiling; `pass1_max_silence_s` stays
2.5 (measured inert on both films at auditok's energy threshold).

---

## 2026-09-06 — #411: a GPU the PyTorch build has no kernels for stops the run at start-up (owner C5)

**Area:** `whisperjav/utils/device_detector.py` (`cuda_build_supports_device()`, `CUDA_UNUSABLE_REASON`,
`_check_cuda_available`); `whisperjav/utils/preflight_check.py` (the unconditional gate
`enforce_gpu_requirement` stops with the reason; `--check` reports a fatal FAIL); tests
`tests/test_device_detector_arch.py` (new, 38); `tests/test_gui_run_summary.py` (+1: the box reaches the
Ensemble builders).

**Report (#411, GTX 1060 6 GB, sm_61, Windows installer v1.9.0; #333 is the same mechanism from its
reporter's own arch-list output; #326 is a bare "cuda errors on a 10-series" report, probably the same,
unconfirmed):** PyTorch warned that the card's
compute capability is not in the build (sm_75 … sm_120); the run continued, took ~2 min per 28 s scene,
produced 0 cues and reported success. Owner on the thread: "Instead of failing immediately, whisperjav
continues to process the movie. That is a bug." Owner C5: "if the preflight fails, then fail there and
then … not yet another check … the preflight is the placeholder for the correct solution."

**Mechanism (established, code read):** every device choice derives from `get_best_device()`, whose CUDA
branch answered "usable" whenever `torch.cuda.is_available()` was True — which it is for a card the build
cannot run on. The unconditional start-up gate asks that function, so it let the run through. (An earlier
draft of this fix put the failure in the `--check`-only suite, the mis-location the tracker had already
recorded once; the adversary caught the repeat.)

**What changed — one honest answer, one existing gate, no new check:**
- `_check_cuda_available()` now judges the card against `torch.cuda.get_arch_list()` with NVIDIA's
  per-entry cubin/PTX rule (an `sm_XY` cubin runs on the same major version with minor ≥ Y; a
  `compute_XY` PTX on anything ≥ X.Y; arch-specific suffixes `sm_90a`/`sm_100f` stripped as
  `torch.cuda._extract_arch_version` does; a list with no `sm`/`compute` entry is not judged). Recent
  PyTorch's start-up warning (`_warn_unsupported_code`, the one in the reporter's console) applies the same
  idea with family exceptions such as 8.7 and 10.1, which this rule does not model and treats as supported;
  older PyTorch warned only on the list's min/max. Where they disagree this rule errs toward "usable", so it
  never stops a run PyTorch would allow. A card the build cannot run on is reported as no usable GPU, with
  the reason in `CUDA_UNUSABLE_REASON`.
- `enforce_gpu_requirement`, which every run passes through, then **stops and asks** (owner, 2026-09-06:
  "the check has to stop the process and ask for the user input to proceed or to abort"). On a console it
  prints the card, its capability, the build's kernel list and the options, then asks "Continue on the CPU
  anyway? [y/N]" and waits — no timeout, no auto-continue; anything but yes aborts with exit status 1 and
  "Nothing was processed". Where nobody can answer — the GUI's child process (which the GUI now marks with
  `WHISPERJAV_NO_CONSOLE=1`, because on Windows even the null device reports itself as a terminal), or a
  run whose stdin is not a terminal — it aborts and says how to answer: `--accept-cpu-mode`, or the GUI's "Accept CPU-only mode" box, both of which the gate
  already honoured at its top; an explicit `--device cpu` counts as the same answer. The GUI's box now
  reaches the Ensemble tab's builders too — the first version left that tab with no way to answer, caught by
  the adversary. Machines with no GPU at all (no CUDA, no MPS) are asked the same question — owner,
  2026-09-06 second round, item 3 — and the question is printed inside a double-ruled box on both paths so
  it cannot be missed; the 30 s auto-continue, the "press any key" wait and their helper are gone. Where
  nobody can answer, a boxed STOPPED message names `--accept-cpu-mode` and the GUI box instead; an
  end-of-input on a piped stdin (which Windows reports as a terminal) reaches that same message rather
  than counting as a silent "no".
- `--check` reports the same fact as a fatal FAIL, so on such a card it now exits 1 where it exited 0. That
  follows from "fail there and then"; it is an exit-code change and is stated here for the owner.

**Consent is recorded once.** The check runs at import time and again in `main()`, and every spawned
worker (Balanced's recogniser worker, ensemble pass workers) re-runs `whisperjav.main`'s module level; a
"yes", `--accept-cpu-mode` or `--device cpu` sets `WHISPERJAV_CPU_ACCEPTED=1`, which the check honours and
children inherit, so the question is asked once per run (the adversary found it asked twice in the parent
and again in each worker, where an unanswered prompt would have hung the run). `bypass_flags` now also
covers `--check-verbose` and `--dump-params`; the dead `-v` entry is gone. (`main()`'s own call of the check
now also skips `--dump-params`; the adversary found the probe stopped there — see the second-round entry.)

**Owner decisions, 2026-09-06 second round (typed, items 1–5), all landed:** (1) the stop stays for every
pipeline, the CTranslate2 ones included; no exemption, and `--device cuda` is not consent. (2) The GUI keeps
abort-plus-instruction; no dialog. (3) No-GPU machines are asked too, and the question is boxed (above).
(4) A lone 「?」/「!」 entry goes (#413). (5) The #413 follow-up was posted, 16:11 UTC. Consequence of (3),
stated for the notes: a CPU-only machine run from a console is asked on every run until the user answers or
passes `--accept-cpu-mode`; run from the GUI it aborts unless "Accept CPU-only mode" is ticked (the box is
saved with the other GUI settings, so once); a script that used to proceed after 30 s on a CPU-only machine
now needs `--accept-cpu-mode` or `--device cpu`.

**Adversary pass on this round, seven findings adopted.** (a) `--dump-params` was in `bypass_flags` at the
import-time gate but `main()` ran the check unconditionally, so the probe stopped on CPU-only machines —
`main()` now skips the check for `--dump-params` (it never transcribes; verified by a real run with the GPU
hidden). (b) The Windows installer's post-install verification imports `whisperjav.main` in a child process
with no flags, which runs the check; on a CPU-only install that would now have blocked or exited 1 and printed
a false "FAILED" — the template sets `WHISPERJAV_CPU_ACCEPTED=1` for that import check only (generated
installer files are regenerated at release). (c) A piped run on Windows, where the null device reports itself
as a terminal, printed the question and then "no console to ask on" — end-of-input now says "input ended
before you answered", which is also the true message for a real Ctrl+Z / Ctrl+D. (d) The bypass line named
`--accept-cpu-mode` even when the user passed `--device cpu` or ticked the box; reworded. (e) The consent
variable leaked out of the test session and made `test_supported_card_passes_silently` vacuous; an autouse
fixture clears it and the test asserts the GPU branch was taken. (f) The tracker still said "No-GPU machines
unchanged"; corrected in the same sync. (g) The user docs (EN/ZH FAQ, GUI guide, workflows, Windows installation
guide, CLI reference) presented the box as optional; they now say it is required on a CPU-only machine, and the
CLI reference documented a non-existent `--cpu-only`, replaced by `--accept-cpu-mode`. **Stated, not changed
(owner's call):** the Kaggle notebook (`notebook/WhisperJAV_kaggle_parallel_edition.ipynb`) and two developer
suites (`tools/ensemble_failure_rate_suite.py`, `tools/vad_hypothesis_suite/configs.py`) launch
`whisperjav.main` without consent; on a CPU-only session they now abort with the boxed instruction where they
used to continue after 30 s.

**Background to (1), kept for the record — the check judges by a PyTorch-build fact, and CTranslate2 does not depend on it.** The
Balanced, Fast and Faster pipelines transcribe through CTranslate2, which ships its own CUDA kernels, and
`resolver_v3.py:36-91,180` carries a deliberate Pascal path (float32 compute type, added for #123) whose
log line appears in the #411 console. So those pipelines may have run on the GPU on a GTX 10-series card,
and now stop at the check unless the user answers yes and continues on the CPU. Whether the PyTorch mismatch
caused the #411 empty run is therefore a hypothesis, not established: the run used CTranslate2, and the
discriminating measurement (the shipped faster-whisper wheel on an sm_61 card) cannot be made here. The
stop itself is what the owner asked for. Scope decided 2026-09-06 (item 1): keep it for every pipeline.

**The GUI does not ask; it aborts and says which box to tick** (decided 2026-09-06, item 2). The reporter's
environment is the GUI, whose worker has no console. There the check aborts with the boxed instruction rather
than asking; a dialog in the GUI itself was offered and declined.

**Known limitation, stated:** `--accept-cpu-mode` on such a card is a full CPU mode only for the modules
that ask `get_best_device()` (faster-whisper, openai-whisper, stable-ts, kotoba, the resolver). The
ChronosJAV generators (anime-whisper, Qwen3, Cohere), the transformers pipeline, WhisperVAD, NeMo and the
speech-enhancement backends pick CUDA from `torch.cuda.is_available()` on their own and will still try
the card (eleven files: `qwen_asr.py`, `generators/anime_whisper.py`, `generators/cohere.py`,
`transformers_asr.py`, `backends/whisper_vad.py`, `backends/nemo.py`, `speech_enhancement/base.py`,
`pipeline_helper.py`, `bs_roformer.py`, `clearvoice.py`, `zipenhancer.py`). **Owner decision 2026-09-06
(item 6): not fixed in 1.9.2; documented instead** — a "CPU-only users" section in `docs/release_notes_v1.9.2.md`
and `README.md` separates the no-GPU case (every component falls back to the CPU; ChronosJAV, transformers, NeMo,
WhisperVAD, WhisperSeg and the neural enhancers untimed on the CPU) from the unsupported-GPU case, where the
Whisper-family pipelines honour the CPU choice (`get_best_device()` in `faster_whisper_pro_asr.py:49`,
`whisper_pro_asr.py:49-51`, `stable_ts_asr.py:153-155`, `resolver_v3.py:331`; default segmenters faster-whisper
built-in VAD / Silero never leave the CPU) and the rest still try the card. Established for the text, code read
this session: `--device cpu` does NOT reach the ChronosJAV pipelines — their device comes from `--qwen-device`
(`main.py:1391` → `qwen_pipeline.py:493,505,515` `device=cfg["device"]`) and in an ensemble from
`--passN-qwen-params {"device"}` (`pass_worker.py:398` → `:1202`); `pass_config["device"] = args.device`
(`main.py:2828,2857`) is consumed only by `resolve_legacy_pipeline` (`pass_worker.py:1387`); the transformers
pipeline takes `--hf-device` (`main.py:1301`) / `--passN-hf-params` (`pass_worker.py:312,341`). The GUI HAS a
per-pass Device control for these pipelines (schema `api.py:2210`, control `app.js:3619-3624`, collected
`app.js:4545-4560`, sent as `--pass1-qwen-params` `api.py:2944,2971`); the Transcribe tab's `--mode transformers`
sends `hf_device: 'auto'` unconditionally (`app.js:737`). Enhancers: no flag carries a device
(`pass_worker.py:1747-1766` writes only backend/model), auto-detect in `speech_enhancement/base.py:243`
(`resolve_torch_device`); NeMo and ClearVoice leave the device to their libraries (`clearvoice.py:92` stores
`_device` and never uses it). WhisperSeg picks ONNX Runtime's CUDA provider only when `onnxruntime-gpu` is
installed (`whisperseg.py:381-393`; the standard install pins the CPU `onnxruntime`, `pyproject.toml:221`).
WhisperVAD is CTranslate2 (`whisper_vad.py:254,270-273`). Kotoba is not a `--mode` choice (`main.py:205`) and is
not listed. A first draft of the section told users `--device cpu` would fix the ChronosJAV pipelines and that
the GUI had no device selector — both wrong, both caught by the adversary; the first would have turned a run
that stops safely at the start-up check into one that starts and dies in CUDA.

**Two defects found by the adversary while checking the section, for the owner (not fixed, not in the user
text as fixes):** (1) `speech_enhancement/backends/bs_roformer.py:116` calls `BSRoformer(device=device)` and
`:264` `self._separator.separate(...)`, but the installed `bs_roformer_infer 0.1.0` (unpinned in
`pyproject.toml:218`) exports the raw model class with signature `BSRoformer(dim, *, depth, ...)` and no
`separate` method — so in this environment BS-RoFormer fails to initialise on any device and returns the
audio unchanged (error swallowed at `:137-139`). Needs a run on the shipped installer environment to confirm
which package version users get. (2) The GUI's speech-enhancer *Device* dropdown is collected into
`enhancerParams` (`app.js:4528,4543`, and `:4719,4729`) which is never read, so the control has no effect.
The stop at the check is what closes #411. Trade-off also stated: CTranslate2 (Balanced) has its own GPU kernels and may
have used such a card; those users now stop at the gate too, and continue only on the CPU.

**Verification (executed 2026-09-06):** `tests/test_device_detector_arch.py` 38 passed — the rule table
for the cu128 wheel (sm_61 and sm_70 → False, sm_75/86/89/120 → True), PTX forward compatibility,
suffixed names, unparseable lists; the detector with a stand-in torch ((6,1) → no GPU + reason, (8,6)
unchanged); `--check` FAIL/PASS; **the real gate**: on a console the question is asked, "no" or Enter
aborts with exit 1, "y" continues on the CPU; with no console it aborts and names `--accept-cpu-mode` and
the GUI box; the keypress-with-timeout helper no longer exists and the gate has no timeout parameter; the
no-GPU path (stand-in torch with no CUDA and no MPS) asks the same question, "yes" continues and is
remembered, no console aborts and says how, `--accept-cpu-mode` asks nothing; end-of-input reaches the
STOPPED box; the box characters are in the output; `--accept-cpu-mode` passes; (8,6) passes silently.
Real runs on this machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`), stdin from the null device:
the "No GPU found" block, the boxed STOPPED message, exit 1; with `--accept-cpu-mode` no box and the
bypass line. `--help` lists the flag with its new text. `tests/test_preflight_cuda_version.py`
unchanged. Real `--check` on the RTX 3060: PASS, exit 0. Not executed: the GUI on a machine with such a
card — the child exits 1 and the console shows the block; the GUI reports the failed exit as it does for any
failed run.

## 2026-09-06 — #413: subtitle entries that are only punctuation are dropped (owner C1/i1/i2)

**Area:** `whisperjav/modules/subtitle_pipeline/cleaners/nonverbal_line_filter.py` (`_normalizes_to_nothing()`;
`filter_srt_file` counts such entries under `dropped_empty`); tests `tests/test_nonverbal_line_filter.py`
(+16). The shared `NonlinguisticUtteranceFilter` is untouched.

**Report (#413, weifu8435):** in his Qwen3-ASR pass file, 209 of 815 cues are a lone 「。」 (median 0.3 s;
they come in runs at 2–3 s spacing, consistent with moan-only frames). His anime-whisper pass has an
ellipsis in 147 of 157 cues.

**Owner decisions (i1, i2):** anime-whisper's inline ellipses are correct output in the style of anime
text — nothing touches them, the existing post-processing for that pass is right as it is. Entries that
consist of nothing but punctuation are removed whole. C1's "normalise, then check for nonverbal sound"
was already what `NonlinguisticUtteranceFilter` does for lines with kana (「。あ。」 is already dropped);
the case it could not judge is a line with no Japanese character at all, i.e. punctuation only.

**What changed:** one clause in `NonverbalLineFilter`, the Qwen-only Phase 8 filter (the tracker's
recorded placement; a first draft put it in the shared filter, which the legacy pipelines also run — the
adversary caught it, and it was moved so Balanced output cannot change). An entry whose text is only
characters from the shared `PUNCTUATION_CHARS` set (plus line breaks) is dropped and counted as
`dropped_empty`. Inline punctuation is untouched because text remains. Any pass that runs the Qwen
pipeline gets it — the GUI's default pairing (anime-whisper, Qwen3) on both passes; a pass configured as
balanced/fast/faster/fidelity/transformers does not and goes through the legacy sanitizer instead. The
clause sits inside the `--qwen-drop-nonverbal-lines` switch (default on), so `--no-qwen-drop-nonverbal-lines`
also keeps punctuation-only entries; the ensemble worker does not forward that switch (pre-existing), so
ensemble runs always apply it.

**Overridden on purpose, for the owner to confirm:** the anime-whisper pass-1 cleaner deliberately kept a
lone 「?」, 「!」 or 「」」 (its comment: "Preserves (keep): '?' alone, '!' alone, '」' alone"). Under i2 those
are punctuation-only entries and are now dropped later in Phase 8. Half-width katakana sound lines (「ﾊｧ」)
remain outside the nonlinguistic filter's class — a separate follow-up, not part of this change.

**Verification (executed 2026-09-06):** the nonverbal, nonlinguistic, device-detector and preflight suites
plus the #324 symbol-purge suite pass (counts in the commit); the new SRT test drops 「。」, 「、」, 「…」
entries, keeps 「…やらしいことして欲し…」 and 「もっと」, and renumbers. Origin of the 「。」 cues (model output
for non-speech vs. leftover after a removal) is not settled — the rule covers both.

## 2026-09-06 — #415: cached Hugging Face models load without a hub round-trip; `--offline` / "Offline mode"

**Area:** new `whisperjav/utils/offline_mode.py`; `whisperjav/main.py` (raw-argv scan before the
pipeline imports, `--offline`, start-up INFO line, `--dump-params` fields `offline_mode` and
`cli_args.offline`); `whisperjav/cli.py` (same scan before `patch_hf_hub_downloads`);
`whisperjav/utils/model_loader.py` (pass-through under `HF_HUB_OFFLINE`);
`whisperjav/modules/speech_segmentation/backends/whisperseg.py` and
`whisperjav/modules/subtitle_pipeline/generators/anime_whisper.py` (cache-first loads); GUI
`index.html` (checkbox `offlineMode`, Advanced options row 3), `app.js` (both collectors,
`SettingsPersistence.FIELDS`), `api.py` (four live `--offline` emission sites plus the dead `_build_ensemble_args`, `_GUI_SETTINGS_MAP`);
`whisperjav/settings/gui_settings.py` (`offline_mode: False`); tests `tests/test_offline_mode.py` (new,
14) and `tests/test_gui_settings.py` (count 36, drift list, S20).

**Report (weifu8435, #415, v1.9.0, VPN off, pass-2 model `Qwen/Qwen3-ASR-0.6B`):** the reporter says
pass 1 produced subtitles after long hub retries and that pass 2 "was skipped for the same reason"; he
asked for an offline switch. His log (286 lines, read in full) covers pass 1 up to the anime-whisper load
and **ends mid-retry**; it does not contain pass 2, so the pass-2 mechanism below comes from my
reproduction, not from his log. Screenshot read.

**Mechanism, established by code read and by reproduction on the 15 s test clip with the hub endpoint
pointed at an unreachable address (`HF_ENDPOINT=http://127.0.0.1:9`):**
- Every transformers `from_pretrained` performs a per-file freshness check against huggingface.co.
  With the host unreachable, huggingface_hub retries each file five times with 1/2/4/8/8 s backoff
  (`huggingface_hub/utils/_http.py`, 10 s connect timeout each), then serves cached files from the
  local cache. WhisperSeg's `WhisperFeatureExtractor.from_pretrained("openai/whisper-base")` runs
  twice per pass (Phase 4 and framing); in the reproduction anime-whisper's processor and model probed
  nine distinct files. Reporter's log: two WhisperSeg loads of ~4.8 min each (11:40:49→11:45:38,
  11:46:44→11:51:31), then the anime-whisper retries begin and the log ends. Reproduction (run A, before
  the change, no flag, on `1to6arabic_16000_mono_bc_noise.wav`, 18.8 s audio): pass 1 alone 567 s, 75
  retry lines; the run was stopped after pass 2 failed. **Run A is on a different clip from runs B–D**,
  so the before/after below is directional; the mechanism (per-file retries) does not depend on the clip.
- Pass 2 fails outright, not slowly: `qwen_asr`'s `Qwen3ASRModel.from_pretrained` calls
  `AutoProcessor.from_pretrained(..., fix_mistral_regex=True)`; transformers 4.57.6's
  `_patch_mistral_regex` then calls `huggingface_hub.model_info()` — a live API request with no cache
  fallback — unless `is_offline_mode()` is true (`transformers/tokenization_utils_base.py:2409-2437`).
  Reproduction: `requests.exceptions.ConnectionError … /api/models/Qwen/Qwen3-ASR-1.7B`; the run ends
  `suspect` with pass 1's output. Established here, not in the reporter's log; it is the most likely
  reading of his "pass 2 was skipped".
- The `[HF Download] Step 2 FAILED — 'openai/whisper-base' not found in local cache` block in the
  reporter's log is the resilience wrapper (#204) reporting on `processor_config.json`, a file that
  does not exist in that repo (mirror 404); `preprocessor_config.json` was served from the cache. Noise,
  not a missing model.
- `HF_HUB_OFFLINE=1` already removed all of this with no code change (adversary finding, verified:
  cached `preprocessor_config.json` resolved in 0.00 s with the hub unreachable). The switch is standard
  huggingface_hub behaviour read at import time (`huggingface_hub/constants.py:165`); transformers
  honours it (`transformers/utils/hub.py:81,420`).

**What changed**
1. **Cache-first loading (no flag):** `load_cached_first()` calls `from_pretrained` with
   `local_files_only=True` first and falls back to a normal load only on a cache miss (`OSError`; any
   other failure such as corrupt weights propagates from the first attempt, not re-run online). Applied to
   WhisperSeg's feature extractor and anime-whisper's processor and model — the loaders on the reporter's
   pass 1. A downloaded model therefore never waits on huggingface.co again. **Side effect, disclosed in
   the notes and yours to keep or drop:** those two loaders are pinned to the cached copy and will not
   pick up an upstream re-upload of the same model id on their own (harmless for whisper-base's
   preprocessor config; a silent version pin for anime-whisper weights). The owner told the reporter on
   the thread that this change was being made. Under offline mode a miss raises an `OSError` naming the
   model and the cache directory (chained to the hub's own error).
2. **`--offline` (CLI) and "Offline mode (downloaded Hugging Face models only)" (GUI Advanced options,
   both tabs' runs), off by default.** Sets `HF_HUB_OFFLINE=1` from a raw `sys.argv` scan at the top of
   `main.py` and `cli.py`, before any import that pulls huggingface_hub in (the pipeline imports at
   `main.py:~95`; `cli.py`'s `patch_hf_hub_downloads`). Ensemble pass workers and the Balanced
   recogniser worker are spawned with the parent's environment and inherit it. One INFO line announces
   it. This is what makes pass 2 (external `qwen_asr` loader) work offline, and what makes a missing
   model fail at once instead of retrying.
3. Under offline mode the resilience wrapper passes straight through (its cache/mirror steps could only
   add noise); the hub's own exception classes are preserved (transformers tolerates
   `LocalEntryNotFoundError` for optional files — changing the class would break WhisperSeg). "Offline"
   has one definition everywhere: `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` truthy, exactly what
   huggingface_hub reads; no WhisperJAV-private variable. `--dump-params` reports `offline_mode` and
   `hub_constant_offline` (the hub's import-time constant — true only if the scan ran before the import).
4. Scope stated in the flag help, the checkbox tooltip and the release notes: loads that go through
   huggingface_hub. Silero via torch.hub (#263), openai-whisper weights, ModelScope enhancers and NeMo
   configs have their own download paths and are not covered. The GUI's own update check is not affected
   (it runs in the GUI process). The hallucination-list Gist is already cache-first (the reporter's log
   shows "filter list: cache").

**Verification (executed 2026-09-06, hub unreachable via `HF_ENDPOINT=http://127.0.0.1:9`; runs B–D on
`015sec_test-966-00_01_45-00_01_59.wav`, 14 s audio; the reporter's pass configuration except that the
cached `Qwen/Qwen3-ASR-1.7B` stands in for his `0.6B`, which is not cached here; times are the logs'
own "Ensemble summary" figures):**
- Run A (before the change, no flag, different clip — see above): pass 1 567 s, 75 retry lines; pass 2
  failed at `model_info`.
- Run C (`--offline`): **both passes done, 60.4 s, 0 retries**, RUN SUMMARY `done 1`.
- Run D (`--offline`, pass-2 model `Qwen/Qwen3-ASR-0.6B`, never downloaded): pass 1 done, pass 2 failed
  **at once** (0 retries); the hub's `LocalEntryNotFoundError: … outgoing traffic has been disabled` is
  the chained cause, transformers' `OSError: We couldn't connect …` is the top-level message (the
  external loader is not wrapped, so the WhisperJAV-worded message applies to WhisperSeg and
  anime-whisper only); 52.0 s; RUN SUMMARY `suspect 1` with pass 1's output kept.
- Run B (after the change, no flag): pass 1 done in 19.8 s with **0 retries** (run A's pass 1 on its
  clip: 567 s / 75 retries), so cache-first loading alone removes the pass 1 stall; pass 2 still retries
  six files five times each (100 retry lines) and fails at `model_info` as before, because that call is
  inside the external `qwen_asr` loader — offline mode is the fix for that pass; 415.6 s; RUN SUMMARY
  `suspect 1`, pass 1's output kept. **This is the default configuration: without the checkbox, the
  Qwen3 pass still fails when the hub is unreachable.**
- `python -m whisperjav.main --help | grep -- --offline` shows the flag; `--offline --help` exits 0;
  `--offline --dump-params` and `--offl --dump-params` (argparse abbreviation) both report
  `offline_mode: true` and `hub_constant_offline: true` and print the INFO line; without the flag both
  are false.
- `pytest tests/test_offline_mode.py tests/test_gui_settings.py`: 71 passed (14 + 57, re-measured after
  the last edit). `py_compile` on every touched file; `node --check app.js`.
- GUI not rendered here. Call-chain verifier: every hop PASS on both tabs (checkbox → both collectors →
  four live builders → `--offline` → env set before the first hub import, proven with an import-order
  probe → spawn children inherit, proven); it found the argparse-abbreviation gap (`--offl` parsed but
  the exact-match scan missed it), fixed here with a prefix-aware scan and a test. Implementation
  adversary: 11 findings, all folded in (headline, provenance, counts, `except OSError`, one offline
  definition, the dump constant, the version-pin disclosure).

**Owner decisions embedded, flagged for confirmation:** a new flag and checkbox exist (user-visible);
default **off** (the reporter's own suggestion, chosen because a first-run user needs downloads);
the name `--offline` (the word also appears in translation and WhisperSeg-decoder vocabulary);
Hugging Face-only scope. Each is a one-line change if you decide otherwise.

**Prior record:** the resilience wrapper and its mirror advice are #204's decided work, unchanged in
substance. #263 (Silero via torch.hub in China) is the same user problem on a different download
path and is not fixed here.

**Not done:** WhisperSeg still loads its feature extractor twice per pass (Phase 4 and framing) — an
inefficiency independent of networking; cohere generator not switched to cache-first (not on the
reporter's path); no connectivity auto-detection (owner decision, not proposed).

## 2026-09-05 — research: `clustering_threshold` is not a scene-granularity lever (docs only)

**Area:** new `docs/research/semantic_scene_premise/` (README, `semantic_lever_study.py`,
`lever_study_results.json`). No product file changed.

**What was found (measured on `test_media` audio, features extracted once, only the boundary logic
varied):** the semantic detector's `clustering_threshold` sets the number of global texture clusters
(4029 → 1 on the 65-minute file) but the final scene count stays 43–65 across thresholds 5–200 and
*rises* over the exposed slider range 5–30 (43 → 58), then collapses to one scene above ~500. Owner's
EKAI-023 runs: 119 scenes at 50, 121 at 90. Cause: at 0.48 s resolution 6060 of 6735 label runs are a
single vector, so raw boundaries are label churn; the count is set by `min_duration` (5 → 354 scenes,
60 → 12) and `snap_window` (1 s → 14, 12 s → 70). With a temporal connectivity constraint the same
threshold is a monotone zoom above ~14; a change-point form gives window/magnitude levers. Options
listed for the owner; the CFF2 "Scene Change Threshold" control and help text (dev, unreleased) are
contradicted by the measurements and await the owner's choice.

**Verification:** study re-run twice (second run adds fine sweeps, `snap_window`, label-run statistics);
adversary pass reproduced the first run bit-for-bit and supplied the `snap_window` and label-run
attacks, which the second run confirms; arm A is boundary-identical to `SemanticSegmenter.segment`.

**Literature (same day, owner asked for the science, not a slider fix):** `literature_review.md`.
Sundaram & Chang (2000-2003) define computable audio scenes as long-term consistency of ambient sound
(>= 8 s to establish context; typically 40-50 s; explicitly "not semantic scenes") and detect them by
correlating an attention span (16 s) against a memory (32 s); they warn that cluster-threshold methods
"critically" depend on a threshold that cannot be set from the data. Every temporal method in the field
(Foote novelty, BIC, memory-model, global partition with a continuity term, embeddings + the same
detectors) carries an explicit temporal term plus a magnitude lever. The shipped engine is a global
partition WITHOUT any temporal-continuity term (unconstrained Ward, cut at every label change), which is
why its threshold sets class count, not granularity; the measured rise in scene count over the slider
range is not explained by this reading and stays open. No audio-only scene work on adult video was
found. Adversary pass ran on the note; ten findings folded in (note §8).

**Decision:** owner (2026-09-05) asked for the logic to be researched, not accuracy; no change made.
Owner N1/N2 (2026-09-05, after the literature review): the threshold slider STAYS in 1.9.2 unchanged; the
study is marked for 2.x, where the semantic detector's boundary logic and granularity control are to be
revisited (roadmap §9.3).

## 2026-09-05 — tools/scene_inspector.py: scene-detector statistics and per-scene screenshots

**Area:** new `tools/scene_inspector.py` + `tools/scene_inspector.md`. Developer/user utility, no
product code touched.

**What it does:** runs a WhisperJAV scene backend (auditok default; silero, semantic, none) on a
media file the way the pipelines do (AudioExtractor 16 kHz mono, `SceneDetectorFactory.create` with
the backend's own defaults, `--param KEY=VALUE` overrides) and writes to `<media folder>/scenes_info/`
(or `--output-dir`): `<name>.scenes.csv`, `<name>.scenes.json`, `<name>.summary.md` (count, coverage,
duration percentiles + histogram, scenes over the 29 s window and the 180 s aligner limit, gaps,
scenes per minute, per-scene table with hh:mm:ss.mmm) and `screenshots/` with begin/middle/end frames
per scene named `<name>__scene0001__begin__00_12_03.450.jpg` (`--no-screenshots` to skip; skipped with
a recorded reason for audio-only input). `--list-params BACKEND` prints accepted parameters with
their YAML defaults and presets. Statistics use the extracted audio duration; the container header is
reported beside it (the 293 s test clip carries a 30:55 header).

**Verification (executed):** 293 s mkv default backend → 19 scenes, 57 frames, exit 0; semantic with
`clustering_threshold=10` → 6 scenes, effective parameter recorded; audio-only wav → screenshots
skipped with reason; `--list-params` for auditok/semantic; `--help` exit 0; one frame opened and
checked visually.

**Same day, owner accepted all seven P8 suggestions (i2/i3) and asked for the semantic threshold
to be user-settable (i1):** `--backend a,b,...` runs several backends and writes `<name>.compare.md/.json`
(counts, coverage, percentiles, pairwise boundary agreement within 1 s); per-scene RMS/peak dBFS;
`--speech-ratio [SEGMENTER]` (speech seconds and ratio per scene, default firered-vad);
`<name>.<backend>.contact_sheet_pNN.jpg` (begin frames, labelled, 60 per page);
`<name>.<backend>.chapters.ffmeta` (muxed with ffmpeg and read back by ffprobe); `--sensitivity`
(preset keys only, see below); `--srt FILE` (cues per scene, scenes without cues, cues in gaps);
`--scene-threshold FLOAT` (shortcut for `clustering_threshold`, warns when semantic is not selected).
Outputs now carry the backend in the file name; screenshots go to `screenshots/<backend>/`. Coverage is
the union of scene intervals (padded semantic scenes overlap; 1.3 s of overlap on the test clip is
reported separately).

**Finding for the owner (product, unchanged):** the scene tool YAML `spec` blocks are not the backend
constructor defaults — auditok `pass2_max_duration_s` 1800 vs 28 s, `pass2_max_silence_s` 1.8 vs 0.94,
`pass1_max_silence_s` 2.5 vs 1.8 — and the pipelines pass a third set from the Pydantic preset.
Applying the whole spec produced one 222 s scene on the 293 s clip; the tool therefore applies only
the `presets.<name>` keys for `--sensitivity`.

**Decision:** owner request 2026-09-05 (I2 P1–P7; i1–i3 for the seven additions).

## 2026-09-05 — CFF1: the recogniser is unloaded and reloaded after 20 minutes of scene audio

**Area:** new `whisperjav/utils/model_refresh.py` (policy), new `whisperjav/modules/asr_worker_proxy.py`
(child-process recogniser for Balanced); `whisperjav/pipelines/balanced_pipeline.py` (`_ensure_asr`
hook, per-scene accounting, `cleanup` override, metadata); `whisperjav/pipelines/fidelity_pipeline.py`
(in-process `_release_asr` + `_load_fresh_asr`); `whisperjav/utils/asr_telemetry.py` (`model_epoch`); `whisperjav/main.py`
(`--model-refresh-audio-minutes`, validation, sync/async/ensemble plumbing, dump echo); GUI
`index.html` / `app.js` / `api.py` / `settings/gui_settings.py`; tests `tests/test_model_refresh_v192.py`
(new, 16), `tests/fake_asr_for_proxy.py` (new), `tests/test_gui_settings.py` (count 34 → 35).

**What changed**
- `--model-refresh-audio-minutes MINUTES` (default **20**, `0` = never; negative → exit 2 at startup).
  The budget is the sum of the durations of the scenes handed to one recogniser instance (owner D2:
  scene granularity, nothing finer; a scene that failed was still handed over and counts). It is
  counted per instance: a sync Balanced batch shares one instance across its files, while Fidelity
  (per-file ASR) and `--async-processing` (per-file pipeline) start a new instance, and therefore a
  new count, with each file. When it is spent, the instance is replaced *between* two scenes. Same flag for sync, `--async-processing` (via
  `resolved_config`) and `--ensemble` (via `worker_kwargs`; Balanced/Fidelity passes honour it,
  other pipelines ignore it). GUI: one Advanced-options field on the Transcription tab, read by both
  the Transcription and Ensemble runs (like the source-language control), persisted.
- **Balanced (owner D5):** with a non-zero budget the CTranslate2 model lives in a spawned worker
  process behind `RemoteFasterWhisperASR`, which exposes the seven members the scene loop already
  used on `FasterWhisperProASR` plus `record_audio` and `shutdown`. A refresh = ask the worker to
  `os._exit(0)` (no destructor) and start a fresh one; the int8 fallback learned by one generation
  is carried into the next. The worker serves every file of the batch until the budget is spent, so
  the model reuse across files is kept. Telemetry, the #394 streak, the progress bar and metadata stay
  in the parent (one telemetry writer per file). A worker that dies natively fails only the scene it
  was on (its completed scenes keep counting in the filter statistics); the next scene starts a fresh
  worker, which is a new telemetry generation (`model_epoch` +1) but not a refresh. After three
  consecutive deaths or failed restarts the proxy stops restarting for the rest of that file: each
  remaining scene is marked failed and the file is then classified by its output like any other
  (usually `empty`, `suspect` if the segmenter kept detecting speech) — whether repeated recogniser
  death should classify the file as `failed` is an owner decision on the exit-status contract, not
  made here. The counter resets at the start of each file (the second adversary pass caught the
  first version, where it did not, so one bad file could have blanked every later file of a batch).
  With `--model-refresh-audio-minutes 0` the pre-v1.9.2 in-process immortal instance is used unchanged.
- **Fidelity:** in-process, in two steps so the old model is really gone before the new one loads:
  `_release_asr` (fold filter statistics, clean up the external segmenter the ASR owns, `cleanup()`),
  then the loop drops its own reference (`asr = None`) — a reference held anywhere keeps the model
  resident — then `_load_fresh_asr` (`gc.collect`, `empty_cache`, fresh `WhisperProASR`, which also
  rebuilds its segmenter). A reload failure raises and fails the file, as a failed initial load would
  (there is no instance left to continue on). The adversary pass caught the first version of this,
  where `del asr` inside the helper deleted only a local name. Measured after the fix on the RTX 3060
  with large-v2: `torch.cuda.memory_allocated()` around a release/load pair (see verification) — the
  discriminating variable; the whole-run peak device memory (11 704 MiB with three refreshes vs
  11 735 MiB with refresh off) is consistent but, being the caching allocator's reserved peak, not
  by itself proof.
- Records: per-scene `model_epoch` in `scenes_detected` (both pipelines) and in the telemetry JSONL
  (Balanced only — Fidelity has no telemetry); `model_refreshes` (this file's share; the Balanced
  proxy counts across the batch and the pipeline reports the per-file difference) and
  `model_refresh_audio_minutes` in the run summary metadata. `model_epoch` is context for reading a
  trend, not evidence of the cause (the `probe_failed` re-transcribe experiment remains unwired).

**Verification (executed, CPU/GPU auto, `--model tiny`):**
- 293 s clip, Balanced, budget 1 min: 14 scenes, refreshes after scenes 6/9/12 (worker PIDs 26344 →
  2248 → 30416 → 13912), telemetry epochs `[1×6, 2×3, 3×3, 4×2]` in ONE file, 54 cues, exit 0.
- Same clip, Fidelity, budget 1 min: 3 in-process refreshes ("Loading Whisper model" ×4), 38 cues, exit 0.
- `--ensemble` with two Balanced passes, budget 1 min: each pass worker spawned its own recogniser
  worker (grandchild) and refreshed three times; merged 55 cues, exit 0.
- `--async-processing` with two Balanced files (the case that died natively with exit 127 on
  2026-09-04): both files done, RUN SUMMARY, exit 0 — the parent never destroys a CT2 model now.
- Kill test: `taskkill /F` on the worker during scene 7/14 → "Scene 7/14 failed: ASR worker process
  died (exit code 1)", fresh worker for scene 8, file finished with 46 cues, exit 0.
- Budget 0 → "Initializing ASR model (exclusive VRAM block)" (in-process path), exit 0.
- GPU (RTX 3060): Fidelity large-v2, budget 1 min → 3 refreshes, 41 cues, exit 0; whole-run peak
  device VRAM 11 704 MiB vs 11 735 MiB with refresh off (control). Direct measurement of the
  release/load pair with `torch.cuda.memory_allocated()` (large-v2, fp32, same config path):
  0 → 6 018 MiB after load A → 0 after `_release_asr` + `asr = None` + gc + empty_cache → 6 018 MiB
  after load B (a resident second model would read ~12 036). Balanced large-v2, budget 1 min → 3
  refreshes of 13–14 s each, 53 cues, 154 s wall, exit 0.
- Two-file sync batch (293 s + 15 s clips, Balanced tiny, budget 1 min): one worker served both
  files (4 "ASR worker ready" = 1 start + 3 refreshes); per-file metadata `model_refreshes` 3 and
  0; the second file's scene carries `model_epoch` 4 (the instance continued across files).
- `pytest tests/test_model_refresh_v192.py` 16 passed (policy; proxy against a fake recogniser in a
  real spawned worker: handshake, refresh → new PID, statistics across generations and across a
  worker death, learned compute type, ordinary error vs native death, give-up rule, shutdown, bad
  model; CLI parse/validation/echo). The pipeline hook sites (`_ensure_asr` branch, the loop's
  `record_audio`, `cleanup`, the Fidelity release/load pair) are covered by the real runs above, not
  by unit tests.
  `test_gui_settings` + `test_gui_run_summary` + `test_asr_telemetry_default` + `test_run_outcome`
  149 passed. `--help` gate + exit 0. `node --check app.js`.

**Cost measured (RTX 3060, this session):** one interpreter start + model load per Balanced refresh =
13–14 s with large-v2 (three refreshes timed from "Model refresh" to "ASR worker ready": 14, 13, 13 s;
about 8 s with `tiny`). At the 20-minute default a two-hour film refreshes at most about six times
(the budget counts scene audio, which is less than the film length), i.e. at most roughly 80 s — a
few percent of a Balanced run. Fidelity: one `whisper.load_model` plus its segmenter per
refresh. Balanced peak VRAM with the worker: 4 447 MiB total on the device during the large-v2 run.

**Decision:** owner (CFF1, 2026-09-05): mechanism on by default at 20 minutes, user-adjustable, at
scene boundaries; D2 budget in scene-audio minutes; D5 child-process recogniser for Balanced. No
over-engineering (owner): budget counted at scene granularity only; no per-scene timeouts; no
re-transcribe probe. Not a fix for the #394 root cause — containment.

## 2026-09-05 — CFF5: Qwen lone-line filter also drops 「はい。」 and 「うん。」

**Area:** `whisperjav/modules/subtitle_pipeline/cleaners/nonverbal_line_filter.py`,
`whisperjav/main.py` (help text of `--[no-]qwen-drop-nonverbal-lines`); tests
`tests/test_nonverbal_line_filter.py`.

**What changed**
- `NONVERBAL_TOKENS` gains `はい` and `うん`. The predicate is unchanged: a line is dropped only
  when its whole stripped text is exactly one token plus an optional single `。`, so `はいはい。`,
  `うんうん。`, `ううん。`, `あ、うん。` and any sentence stay (tests added for each). Runs in Phase 8
  for all three Qwen backends (qwen3 / anime-whisper / cohere) before the nonlinguistic filter,
  which lists the two words as keep-evidence — no conflict, it only sees surviving lines.
- Docstring rewritten (the v1.9.0 text stated the opposite rule) with the owner's evidence:
  two SRT screenshots of lone `はい。` (0.065–2.720 s) and `うん。` (0.300–2.900 s) cues; "almost
  90% … refer to human moans during intimate scenes".
- Off switch unchanged: `--no-qwen-drop-nonverbal-lines` (single-pass). Ensemble Qwen passes inherit
  the default and have no switch — unchanged by owner decision D8.
- Not covered on purpose: the Balanced pipeline (legacy sanitizer path; `filter_list_v08.json`
  already holds both words as exact-match hallucination entries).

**Verification (executed):** `pytest tests/test_nonverbal_line_filter.py` (positives incl. the two
new tokens with/without `。`/whitespace; 13 negatives; SRT drop/renumber 8 → 2); `--help` shows the
tokens; `--no-qwen-drop-nonverbal-lines --help` exit 0.

**Decision:** owner (CFF5, 2026-09-05) + D8 (token list only; no ensemble off-switch, no GUI
checkbox). Thread owed: #254 (owner's v1.9.0 reply lists the old token set).

## 2026-09-05 — CFF2: semantic scene-change threshold exposed on the CLI and in Customize

**Area:** `whisperjav/main.py` (two flags, features injection, qwen kwargs, decoupled kwargs,
`--dump-params` echo); `whisperjav/pipelines/qwen_pipeline.py`, `whisperjav/pipelines/decoupled_pipeline.py`
(ctor param → `clustering_threshold` kwarg); `whisperjav/ensemble/pass_worker.py` (`prepare_qwen_params`
mapping + lift); `whisperjav/webview_gui/api.py` (Qwen Audio-tab schema); `webview_gui/assets/app.js`
(Qwen "Custom Scene Bounds" slider, `QwenManager.defaults`); the semantic tool YAML (slider label /
description); tests `tests/test_scene_clustering_threshold_v192.py` (new, 8 tests).

**What changed**
- The semantic detector's `clustering_threshold` (Ward distance, default 18, YAML presets 10
  aggressive / 22 conservative) was a constructor kwarg all the way to the vendored engine, but
  nothing above the factory set it. Now: `--scene-clustering-threshold FLOAT` for the legacy modes
  (written into `features["scene_detection"]`, so every legacy pipeline passes it to the factory;
  auditok/silero accept and ignore it, with a WARNING when the effective method is not semantic) and
  `--qwen-scene-clustering-threshold FLOAT` for `--mode qwen` (ctor param, applied independently of
  safe chunking); `--pipeline decoupled` takes the legacy flag. Both echoed in `--dump-params` `cli_args`
  (`scene_clustering_threshold`, `qwen_scene_clustering_threshold`; the Qwen echo was missing in the
  first version — caught by the adversary pass, added, and re-measured: `--mode qwen --dump-params
  --qwen-scene-clustering-threshold 10` → `10.0`).
- Ensemble: legacy passes already accepted `clustering_threshold` in `--passN-params`; Qwen passes
  now accept `scene_clustering_threshold` in `--passN-qwen-params` (mapped to
  `qwen_scene_clustering_threshold`, lifted only when set).
- GUI (Ensemble tab Customize): the legacy modal's Scene tab already rendered the slider from the
  YAML hints (label now "Scene Change Threshold"); the Qwen/anime-whisper/cohere modal gains the same
  slider under Audio → Custom Scene Bounds (5–30, default 18), collected by the shared collector.
  Transformers passes are method-only (`--hf-scene`) and unchanged — noted, not extended.
- Wording: the presets call 10 "more, shorter segments" and 22 "fewer, longer"; on the 293 s clip
  the final count went 7 → 6 scenes when lowering 18 → 10 because the detector's min-duration merge
  runs after clustering, so help text says "tends to", not "=".

**Verification (executed):** `--help` lists both flags; each parses with exit 0. `--dump-params
--mode balanced --scene-detection-method semantic --scene-clustering-threshold 10` →
`features.scene_detection = {method: semantic, clustering_threshold: 10.0}`; without `semantic` the
WARNING names the effective method. Real CPU runs on the 293 s Netflix clip (balanced, tiny,
`--debug`) at 18 and 10: the factory log shows `clustering_threshold: 18.0` / `10.0`, 7 vs 6 scenes,
48 vs 50 cues, both exit 0. `inspect.signature` confirms the Qwen and Decoupled ctor params;
`prepare_qwen_params({"qwen_params": {"scene_clustering_threshold": 12}})` → 12.
`node --check app.js`. New test file 8 passed.

**Decision:** owner (CFF2, 2026-09-05): expose to CLI and the GUI Customize parameters tab.
Customize lives on the Ensemble tab only (D4).

## 2026-09-06 — N3: Balanced default speech segmenter back to faster-whisper's built-in VAD (CFF3 default reversed)

**Area:** `whisperjav/config/segmenter_presets.py` (`BALANCED_DEFAULT_SEGMENTER = "faster-whisper"`;
the fallback chain and `pick_balanced_default_segmenter` removed — no caller left);
`whisperjav/main.py` (default block, `--speech-segmenter` help, guard comment);
`whisperjav/ensemble/pass_worker.py` (balanced pass default); `whisperjav/webview_gui/assets/app.js`
(`applyPipelinePresets` balanced → `faster-whisper`; `pickBalancedDefaultSegmenter`,
`reapplyBalancedDefaultIfUnavailable` and the availability cache removed); `index.html` (FireRedVAD
option titles); `whisperjav/utils/preflight_check.py` (fireredvad message); comments in
`faster_whisper_pro_asr.py`, `api.py`, `factory.py`, `speech_segmentation/__init__.py`,
`backends/__init__.py`, `backends/firered_vad.py`, `installer/core/registry.py` (the `reason=` string
for the fireredvad dependency — the last three found by the adversary pass), the FireRedVAD YAML,
`pyproject.toml`; `tools/ct2_degradation_probe.py` and `tools/scene_inspector.py` (help/comment);
`README.md` segmenter table; tests `tests/test_balanced_defaults_v192.py` rewritten (18 tests).

**Owner decision (typed, 2026-09-05, N3):** "I have changed my mind and revert my earlier
requirements: CFF3. As such the default behaviour for the balanced pipeline would be to use the
internal VAD of faster-whisper." Scope confirmed 2026-09-06: **default only**. Kept from CFF3: the
single-pass path resolves a WhisperJAV segmenter's per-sensitivity YAML preset, and `firered-vad` /
`ten` are exempt from the routing-guard downgrade on `--mode balanced` (an explicit choice honours
`--sensitivity`; Test-D grouping still overlays). CFF6 (FireRedVAD installed, not experimental) stays.

**What changed**
- `--mode balanced` without `--speech-segmenter` runs faster-whisper's built-in VAD again, as in
  v1.9.0/v1.9.1 (one recognizer call per scene). Same at the ensemble pass worker for a balanced pass
  without `--passN-speech-segmenter`, and in the GUI Ensemble-tab preset when a pass is switched to
  balanced. The Transcribe tab sends no segmenter (D4) and so follows the CLI default.
- Consequences reverted with it: Balanced is as fast as v1.9.0/v1.9.1; the #394 corroboration counter
  is inert under the built-in VAD (the recogniser reports segmenter "none"), so a zero-cue Balanced
  file is `empty`, not `suspect`, unless a WhisperJAV segmenter is chosen; D6 and D7 no longer apply.
- `fireredvad` stays a standard dependency; preflight now describes it as the segmenter behind
  `--speech-segmenter firered-vad`, not as the Balanced default.

**Verification (executed 2026-09-06):** `python -m whisperjav.main --help | grep speech-segmenter`
(new default text shown); `--speech-segmenter faster-whisper --help` exit 0; `py_compile` on every
touched Python file; `node --check app.js`; `pytest tests/test_balanced_defaults_v192.py` 18 passed
(`--dump-params`: default backend `faster-whisper` with the native VAD preset for all three
sensitivities and no `_dump_note`; explicit `firered-vad` → preset .5/.4/.3 + Test-D 9.0/0.1, not
downgraded; `ten` exempt; `whisperseg` still downgrades; CLI overrides win; fidelity unchanged);
`pytest tests/test_gui_settings.py tests/test_run_outcome.py tests/test_output_coverage.py` 134
passed. **The ensemble default is verified by code read only** (`pass_worker.py`: backend
`faster-whisper` → not in `SPEECH_SEGMENTER_MAP`, passes through → resolver returns `{}` →
`apply_balanced_vad_defaults` native branch); no executed test covers that arm because the test
module cannot import `pass_worker`. GUI not rendered here: owner to confirm the Ensemble tab shows
Faster-Whisper native when a pass is set to balanced. A user-saved ensemble preset from the
FireRedVAD build still carries `firered-vad` (presets store the segmenter) — that is saved state,
not a failed revert.

## 2026-09-05 — CFF3: Balanced defaults to an external speech segmenter (FireRedVAD), single-pass presets resolved — **DEFAULT REVERSED 2026-09-06 (N3, above); preset resolution and guard exemption remain**

**Area:** new `whisperjav/config/segmenter_presets.py`; `whisperjav/main.py` (default block,
routing guard, preset merge, `--speech-segmenter` help); `whisperjav/ensemble/pass_worker.py`
(re-exports, balanced default); `whisperjav/webview_gui/api.py` (`get_pipeline_defaults`);
`whisperjav/webview_gui/assets/app.js` (`applyPipelinePresets`, `pickBalancedDefaultSegmenter`);
`whisperjav/modules/faster_whisper_pro_asr.py` (stale comment); `tools/ct2_degradation_probe.py`
(help only); `README.md`; tests `tests/test_balanced_defaults_v192.py` (new, 21 tests).

**What changed**
- `--mode balanced` with no `--speech-segmenter` now runs a WhisperJAV external segmenter:
  `firered-vad`, or `ten` if the fireredvad package is missing, or `silero-v3.1` if both are
  (WARNING with the pip command when FireRedVAD is skipped). The same chain is applied by the
  ensemble pass worker for a balanced pass without `--passN-speech-segmenter`, and by the GUI's
  Ensemble-tab preset when the pipeline is switched to balanced (availability-aware).
  `--speech-segmenter faster-whisper` restores the v1.9.0/v1.9.1 native-VAD behaviour.
- The single-pass path now resolves the backend's per-sensitivity YAML preset (the ensemble
  path always did). Order: backend → YAML preset → Test-D grouping overlay → explicit CLI
  overrides. Before this, `--sensitivity` was inert for a non-silero segmenter on
  `--mode balanced` — the guard's stated reason for downgrading such choices.
- Routing guard: `firered-vad` and `ten` are exempt on `--mode balanced` (their presets now
  flow); every other non-silero backend keeps the downgrade to silero-v3.1 (fidelity/fast/faster
  untouched; wider unification is PR #375's scope). Warning text updated.
- `SEGMENTER_PARAMS`, the backend→YAML tool map and `resolve_qwen_sensitivity` moved verbatim
  from `pass_worker.py` (imports every pipeline) to the light module
  `config/segmenter_presets.py`; `pass_worker` re-exports the old names, so existing imports and
  tests are unchanged.
- GUI Customize panel (`get_pipeline_defaults`): for a non-silero external backend the returned
  `vad` block is the segmenter's effective parameters (YAML preset + Test-D) instead of the
  resolver's silero values that the ASR firewall discards at run time.
- Probe tool default left at `faster-whisper` so new runs stay comparable with the reporter
  datasets already collected; its help says how to mirror the v1.9.2 default.

**Consequences stated for users (release notes):** Balanced is slower than in v1.9.0/v1.9.1
(one recognizer call per VAD group instead of per scene); the #394 corroboration counter is
active on balanced by default, so a file with zero cues while speech kept being detected is
`suspect` rather than `empty` (exit code changes only under `--fail-on suspect`); Test-D
grouping (9.0 s / 0.1 s) overrides the FireRedVAD YAML `max_group_duration_s` (7/6/5) and
`chunk_threshold_s` (1.0) — the Segmenter tab in Customize shows the YAML values for those two
keys, not the Test-D ones (pre-existing display gap for every external segmenter on balanced).

**Verification (executed):** `--dump-params --mode balanced` × conservative/balanced/aggressive →
backend firered-vad, threshold .5/.4/.3, max_speech 7/6/5, end_pad 250/150/100, max_group 9.0,
chunk 0.1, `vad` cleared + `_dump_note`; `--speech-segmenter faster-whisper` → native preset kept;
`ten` → exempt, preset + Test-D; `whisperseg` → downgrade warning; `silero-v6.2`, `--mode
fidelity`, `--mode fast` unchanged; `--vad-threshold 0.6 --max-group-duration 6` win. Real CPU
run (`--mode balanced --model tiny`, 15 s clip): "Speech segmenter set to: firered-vad",
Test-D line, "Speech Segmenter initialized: firered-vad", 3 cues, RUN SUMMARY exit 0.
`--help` lists firered-vad; `--speech-segmenter firered-vad --help` exit 0. `get_pipeline_defaults`
called directly for firered-vad / faster-whisper / silero-v3.1. `node --check app.js`.
Suites: new file 21 passed; `test_qwen_sensitivity` + `test_ensemble_params` +
`test_gui_custom_params_simulation` 104 passed / 8 failed — identical to the baseline before
this change (stale silero-v6.2 set); `test_gui_settings` + `test_gui_run_summary` 76 passed.

**Decision:** owner (CFF3, 2026-09-05): default = FireRedVAD, "external" = a WhisperJAV segmenter
as opposed to faster-whisper's native VAD; D3 Test-D stays; D4 no Transcribe-tab dropdown; D6
`suspect` verdicts on balanced accepted after explanation; D7 fallback to another WhisperJAV
segmenter, never native. Guard exemption limited to the two chain members (owner: no
over-engineering). Thread owed: #311.

## 2026-09-05 — CFF6: FireRedVAD becomes a first-class dependency (installed with `[cli]`)

**Area:** `pyproject.toml` (`[cli]` extra), `uv.lock`, `whisperjav/installer/core/registry.py`,
`whisperjav/installer/validation/imports.py`, `installer/templates/requirements.txt.template`,
`whisperjav/utils/preflight_check.py`, labels in `whisperjav/modules/speech_segmentation/`
(`backends/firered_vad.py`, `factory.py`, `__init__.py`), the tool YAML
`firered-vad-speech-segmentation.yaml`, `whisperjav/ensemble/pass_worker.py` (comments),
`whisperjav/main.py` (`--qwen-segmenter` help), `webview_gui/assets/index.html` (both segmenter
dropdowns), `README.md`; tests `tests/test_dependency_cross_match.py`.

**What changed**
- `fireredvad>=0.0.2` (PyPI, Apache-2.0, Python ≥3.10) is a `[cli]` dependency, so `[all]`,
  `[colab]` and `[kaggle]` inherit it and the conda-constructor `requirements_v1.9.2.txt`
  (generated from pyproject by `installer/build_release.py`) carries it. Registry entry at CLI
  order 49; removed from the import scanner's optional list; fallback template updated.
  `uv lock` added exactly four packages (fireredvad, kaldi-native-fbank, kaldiio, textgrid);
  no existing pin moved (uv re-derived some environment markers). `kaldi-native-fbank` ships
  wheels for win/linux/macOS × cp310–cp313, so no platform marker is needed.
- Every "experimental" label on FireRedVAD removed (display name is now `FireRedVAD`; the YAML
  drops the `experimental` tag; `speech_segmentation/backends/__init__.py` was missed at first and
  fixed after the adversary pass). The remaining truthful caveat is kept in words: detection presets
  are upstream-derived, the segment cap was JAV-tuned on 2026-08-14.
- Preflight lists `fireredvad` as an optional dependency with an actionable message (worded for
  the CFF3 default at the time; reworded 2026-09-06 when N3 reversed that default).
- Hygiene found by the sync gate while adding the entry: the registry pinned `numba>=0.60.0`
  while pyproject says `>=0.61.0`, and the fallback template disagreed with the registry on
  `numba` and `transformers`. Aligned the registry and the fallback template to what pyproject
  already ships (pyproject is what pip/uv and the conda-constructor requirements use, so no user
  install changes); without this the sync gate could not confirm the new dependency. **Owner may
  revert** — it is not traceable to CFF1–CFF6.

**Verification:** `python -m whisperjav.installer.validation` → PASSED (was failing on the numba
mismatch before); `pytest tests/test_installer.py tests/test_installation.py
tests/test_dependency_cross_match.py` → 100 passed, 6 failed, all six identical on HEAD before this
change (WJ env has numpy 1.26 / pip-check conflicts / a stale entry-point test); `uv lock` exit 0;
the real generator `build_release.generate_requirements_from_pyproject()` executed against the current
pyproject emits `fireredvad>=0.0.2` (a `--dry-run` alone prints no content, so it proves nothing);
YAML parses;
`tests/test_config_v4.py` 33 passed; `tests/test_speech_segmentation.py` 83 passed, 4 failed = the
known stale silero-v6.2 set.

**Decision:** owner (CFF6, 2026-09-05): FireRedVAD "mandatory" in dependency, setup and
installation; no longer experimental. Fallback when the package is absent at runtime: owner D7
(another WhisperJAV segmenter, never faster-whisper's internal VAD) — implemented with CFF3.
**Superseded 2026-09-06 (N3):** Balanced defaults to the built-in VAD again, so there is no runtime
fallback chain any more; `fireredvad` stays a standard dependency and is used only when selected
explicitly. Thread owed: #311 (requester was told it needs `pip install fireredvad`).

## 2026-09-04 — ASR telemetry on by default and inside ensemble; version 1.9.2; note corrections

**Area:** `whisperjav/utils/asr_telemetry.py` (new `resolve_telemetry_path`),
`whisperjav/pipelines/balanced_pipeline.py`, `whisperjav/ensemble/pass_worker.py`
(new `_configure_telemetry`), `whisperjav/main.py`, `whisperjav/__version__.py`,
`installer/VERSION`, `docs/release_notes_v1.9.2.md`; tests
`tests/test_asr_telemetry_default.py` (new), `tests/test_run_outcome.py`.

**What changed**
- Telemetry is **on by default**. Default location
  `<output_dir>/raw_subs/<name>.asr_telemetry.jsonl` (the folder users already
  attach to bug reports). `--asr-telemetry PATH` moves it (directory → one
  file per media; file path used as given); new `--no-asr-telemetry` disables.
  `resolve_telemetry_path()` is the one place the rule lives.
- Ensemble: `main.py` puts `asr_telemetry` / `asr_telemetry_enabled` into both
  pass configs; the pass worker calls `_configure_telemetry()` per file before
  `pipeline.process`, pointing the pipeline at
  `<file_output_dir>/raw_subs/` with tag `passN`, so each Balanced pass writes
  `<name>.passN.asr_telemetry.jsonl` beside that file's pass outputs. Needed
  because in `source` mode the orchestrator points the pipeline's own
  `output_dir` at the temp directory and moves only the SRT out.
- Async path: the two values travel in `resolved_config` and
  `BalancedPipeline.__init__` reads them (the sync path sets the attributes
  after construction, as before).
- Version: `__version__.py` and `installer/VERSION` set to 1.9.2 (VERSION had
  been left at 1.9.0 while `__version__.py` said 1.9.1).
- Release notes: #395 moved from Known limitations to Fixed (the patch exists,
  v4-flash only, credit @mcdman); persistence entry no longer credits #298
  (closed 2026-04-23) or #381 (two-pass Customize persistence not built);
  telemetry section rewritten.

**Decision:** owner (2026-09-04, CL1–CL3): take items 1 and 2; set the version
in code; make telemetry reach ensemble; telemetry should be on by default.
Location under `raw_subs/` is my choice (stated rationale: must survive the
run and be where reporters already look); owner to veto if unwanted.

**After adversary review (same day), also:**
- `AsrTelemetry` appends each record to disk as its scene finishes (truncate on
  first record, append + flush per scene; `finalize()` only logs, `write` is an
  alias); `BalancedPipeline` finalizes from its error handler too, so a crash,
  hang or interrupt leaves a partial file. The buffered version wrote only on
  the success path, which contradicted the reason for default-on.
- GUI opt-out: Advanced options checkbox "Keep per-scene ASR telemetry"
  (`asrTelemetry`, on by default) → `--no-asr-telemetry` in every builder;
  persisted (settings count 33 → 34).
- `cuda_used_mb` (device-wide, `torch.cuda.mem_get_info`) added: the
  PyTorch allocator counters cannot see CTranslate2's arena.
- **Pre-existing bug fixed:** `--async-processing --output-dir source` wrote
  every file's outputs beside the first input (`main.py` set one output_dir
  from `media_files[0]`; the processor never overrode it per task). Now each
  media carries `output_dir` and `AsyncPipelineProcessor._process_media`
  applies it per task.
- Notes: #395 qualified to the direct DeepSeek provider (OpenRouter route not
  patched); "#96's first question" attribution dropped; file-path
  `--asr-telemetry` with several inputs documented as overwriting.

**Limits:** only `BalancedPipeline` records telemetry, so the GUI's default
ensemble pairing (anime-whisper + Qwen3-ASR) still records nothing (the CLI's
default `--pass1-pipeline` is balanced). `installer/generated/` is still the
v1.9.0 set until `build_release.py` is run (owner-gated).

---

## 2026-09-03 — GUI follows the run-outcome contract

**Area:** `whisperjav/webview_gui/api.py`, `assets/app.js`, `assets/index.html`,
`whisperjav/settings/gui_settings.py`; tests `tests/test_gui_run_summary.py`,
`tests/test_gui_settings.py`.

**What changed**
- On process exit the API reads `whisperjav_run.json` (`_read_run_summary`),
  using the same `default_manifest_path(output_dir, raw_inputs)` call the CLI
  uses, and rejects a manifest whose `started_at` predates the process.
- Closing console line is `[FINISHED] <tally> (exit status N)`,
  `[FINISHED WITH FAILURES] …`, or `[STOPPED] <note> …`. `[SUCCESS]` is gone
  from the API, including the separate translation runner's closing lines.
- `get_process_status` returns `run_summary` (counts, tally, per-file states,
  note, fails_on, manifest_path). The frontend shows `Finished · <tally>` in
  the status label, lists every non-`done` file with state and reason in the
  console, and prints the manifest path (`ProcessManager.logFileStates`).
- Two Advanced-options checkboxes, `failOnEmpty` / `failOnSuspect` →
  `fail_on_empty` / `fail_on_suspect` → `--fail-on empty[,suspect]`, forwarded
  by every builder (`build_args`, transformers, crispasr, ensemble, two-pass).
  Persisted through `_GUI_SETTINGS_MAP` and `DEFAULT_GUI_SETTINGS` (settings
  count 31 → 33). Default off.
- crispasr builder also gained `--skip-existing`, which it had never forwarded.

**Decision:** owner (2026-09-03, CL1–CL3): GUI must be consistent with the CLI
contract; say "Finished" then the tally; make the GUI state-aware.

**Verification:** 14 new tests (builders on every mode-dropdown value,
manifest freshness, folder input, closing lines); owner ran a real ensemble
file through the GUI (2199 cues, 99.8% span → `done`, `[FINISHED]` line,
manifest beside the source). Adversary review found and led to fixes for:
crispasr builder missing the flag; GUI/CLI manifest-path divergence on folder
inputs; mtime-based staleness; per-file states collected but not shown;
`[SUCCESS]` surviving in the translation runner.

**Follow-ups:** none required. `_build_ensemble_args` is unreachable from the
frontend (JS calls `start_ensemble_twopass` only); left in place.

---

## 2026-09-03 — One per-file vocabulary and one exit-status rule for every path

**Area:** new `whisperjav/utils/run_outcome.py`; rewritten
`whisperjav/utils/output_coverage.py`; `whisperjav/main.py`; tests
`tests/test_run_outcome.py` (new), `tests/test_output_coverage.py` (rewritten).

**What changed**
- States: `done`, `empty`, `suspect`, `failed`, `skipped`; coverage `ok` /
  `low` / `not assessed`. `classify_output()` is the only verdict function;
  `exit_status()` the only exit rule: 1 if any file is `failed` or the run did
  not complete, else 0; `--fail-on empty|suspect` adds states.
- `process_files_sync`, `process_files_async` and the `--ensemble` branch all
  append to one `outcomes` list owned by `main()` and end in `_finish_run()`,
  which prints the RUN SUMMARY table, writes `whisperjav_run.json` next to the
  outputs (output dir, or beside the first raw input in `source` mode; inside
  the folder for a folder input), and returns the status. `KeyboardInterrupt`
  and the outer `except Exception` also finish (note + status 1).
- Removed: the ensemble branch's own `sys.exit(1)`, the three per-path summary
  blocks, `_failed_count`, `CoverageReport.is_failure`, `report_coverage`, the
  0.60 "suspicious" tier. `output_coverage.py` now measures only.
- `--fail-on` (repeatable / comma-separated) and `--min-coverage` (0–1) are
  validated at startup; a bad value exits 2 before any model loads.
- `apply_vtt_conversion` returns the VTT path so the manifest names an
  existing file. Duplicate-outcome guard when a late step raises. Ensemble
  outcomes keyed by input path (basename collisions). Degraded ensemble files
  with `--translate` record translation `skipped`. A promised-but-missing SRT
  is `failed`; zero cues with corroboration or after a pass-2 failure is
  `suspect`, not `empty`. stdout/stderr flushed before the ctranslate2
  `os._exit`.
- **Pre-existing defect fixed:** `--async-processing` never waited for its
  tasks (`AsyncPipelineManager.process_files` submits with `wait=False`); the
  CLI summarised queued tasks, then `shutdown()` cancelled them. It now waits
  (`wait_for_task`, then the Future as the authority).

**Decisions**
- Owner (2026-09-03): agreed section 5 of the "Exit Status Precedents"
  research, with C1 no halt control and C2 no pre-flight. Facts set: f1 zero
  subtitles is not a failure; f2 a batch contract is needed; f3 consistency
  over right/wrong.
- Implementation choices inside that scope (flagged to owner, not objected):
  translation error → `failed` in every mode; "not assessed" is a coverage
  column, not a sixth state; manifest has a fixed name, overwritten per run.
- Reverses the pre-release behaviour announced on #394 (2026-08-30) in which
  a zero-cue file failed the run.

**Verification:** 75 tests in the two suites, including `main()` executed
through sync, async and ensemble with stubs and one test through the real
`AsyncPipelineManager`; real CLI runs (faster/tiny, CPU) on the 15 s / 5 s /
piano clips through all three paths, and a piped run through the ctranslate2
fast exit. Three adversary passes; findings applied (interrupt path, missing
file, corroborated/degraded empty, VTT path, duplicate outcome, basename
collision, docstring drift, double summary, grammar).

**Known limits (stated in the release notes):** corroboration exists only in
Balanced with an external segmenter; the ensemble pass worker does not return
the streak, so ensemble `suspect` = span or pass-2 failure only; an ensemble
worker crash still ends the whole batch (orchestrator behaviour);
`--output-dir source` with inputs in several folders writes one manifest
beside the first.

---

## 2026-09-03 — Pre-review audit of the branch (no code change)

Independent audit of `dev_v1.9.2` against `main` before the work above:
found the ensemble exit-1 regression (`_failed_count` unbound at the old
`main.py:2806`, introduced by `bd98edc`), four release-note over-claims
(#395 "cannot be disabled" stale; #298 closed since April and #381 not
addressed by PR #378; zero-cue failure sync-only; `--asr-telemetry` never
reaching ensemble), and two script-style test files that break a one-session
`pytest tests/`. Report: https://claude.ai/code/artifact/73c5c95d-0a92-49d1-b127-fb23c02029c2

---

## 2026-08-30 — Earlier v1.9.2 work (see the release-notes changelog table)

`--asr-telemetry`; Balanced corroboration signal; artifacts summary fix;
DeepSeek v4-flash thinking switch (#395); post-processing stage logging
(#372); PR #378 merge (#328, #96, #298, #381, #309, #382); Linux/Windows
install-guide fixes (#366, #334); seven small defects (#340, #341, #306, #325,
#323, Ollama fallback list); GUI settings suite greened. The coverage gate
from that batch is superseded by the contract above.

---

## Open items carried on this branch

- **Async + Balanced + more than one file dies natively** (exit 127 on
  Windows) when the second task's `FasterWhisperProASR` initialises after the
  first task's pipeline was cleaned up — verified 2026-09-04. **2026-09-05:**
  with the CFF1 default (recogniser in a worker process) the same two-clip run
  completes with both SRTs and a RUN SUMMARY, because the parent never
  destroys a CT2 model. The limitation still applies with
  `--model-refresh-audio-minutes 0` (in-process instance); stated as such in
  the release notes.
- Telemetry outside `BalancedPipeline` (the default ensemble pairing records
  nothing); a Transcribe-tab segmenter control (owner decision); a default cue
  ceiling for the default ensemble (owner decision); #394 containment.
- Test hygiene: `tests/test_gui_refactor.py`, `tests/test_postprocessing_performance.py`
  (rewrap stdout at import) and `tests/test_tab_spacing.py` (opens Tk)
  prevent a single-session `pytest tests/`.
- #314 installer progress streaming never landed.
