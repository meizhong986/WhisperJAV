# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-31 (rev21) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **55 open** on GitHub

---

## Status Legend

| Label | Meaning |
|-------|---------|
| `SHIPPED` | Fix released in a stable version. Waiting for user to test/confirm. |
| `FIX CODED` | Fix committed but not yet released. |
| `AWAITING REPLY` | Response given. Waiting for user reply. |
| `NEEDS RESPONSE` | Issue has no response or needs a follow-up reply. |
| `NEEDS FOLLOW-UP` | User replied after our response — needs another look. |
| `DEFERRED` | Moved to a future release. |

---

## Quick Stats

| Category | Count | Notes |
|----------|------:|-------|
| Total open on GitHub | **55** | Was 54 at rev20. New: #271, #272. Closed: #269. Net +1. |
| **v1.8.10 RELEASED** | — | 2026-03-30. 39 commits. |
| **FIX CODED (on main, not released)** | 2 | Hallucination filter bundle, Colab install |
| **NEW since rev20** | 2 | #271 (Ollama variant error), #272 (XXL model flag ignored) |
| **NEEDS RESPONSE** | 2 | #271, #272 |
| **NEEDS FOLLOW-UP** | 2 | #265, #267 (debug logs arrived) |
| **AWAITING REPLY** | 5 | #234, #251, #258, #260, #261 |
| Feature requests (open) | 29 | See Feature Requests section |

---

## v1.8.10 — RELEASED (2026-03-30)

See `installer/RELEASE_NOTES_v1.8.10.md` for full details.

---

## Fixes on main (committed, not released)

These fixes are committed to main after v1.8.10 was tagged. Colab/pip users get them automatically. .exe installer users will get them in the next release.

| Commit | Item | Issues | Risk |
|--------|------|--------|------|
| `1bfe548` | Hallucination filter: bundled JSON missing from wheel, stale gist URLs, INFO logging | #265 | LOW — packaging + logging |
| `4a600b2` | `--scene-detection-method` ignored in standard pipeline modes | #269 (closed) | LOW — one-line injection |
| `2533438` | Docs: GUI user guide updated for Ollama + enhance-for-VAD | — | LOW — docs only |
| `a6b473d` | Colab install: remove venv, reuse system torch, fix llama-cpp filename | #231 | LOW — Colab-tested |
| `a6b473d` | Colab/Kaggle notebooks: remove venv references | #231 | LOW — config changes |
| `f875303` | Issue tracker rev20 | — | — |

---

## Open Issues by Status

### Bugs / Active Issues (18)

| # | Title | Reporter | Status | Notes |
|---|-------|----------|--------|-------|
| **#272** | XXL ignores --pass2-model large-v2 flag | TinyRick1489 | `NEEDS RESPONSE` | **NEW.** Confirmed bug with debug log. WhisperJAV logs model=large-v2 but XXL subprocess loads large-v3. Flag not forwarded to XXL command line. |
| **#271** | Ollama "Could Not Create Optimized Variant" | justantopair-ai | `NEEDS RESPONSE` | **NEW.** qwen3:14b error. Also reports gist timeout fetching instructions_standard.txt, plot summary cutoff in GUI, translation errors. 3 screenshots. |
| **#267** | Stuck on streaming features (Qwen+Semantic) | OrangeFarmHorse | `NEEDS FOLLOW-UP` | **Debug logs arrived.** All 3 test runs (short clip, long file, balanced pipeline) hang at exact same point: "[1/5] Streaming features (60s chunks)..." inside SemanticAudioClustering v7.0.0. File length irrelevant. RTX 5060 Ti (Blackwell). |
| **#265** | Hallucination in Chinese translation (large-v2) | yangming2027 | `NEEDS FOLLOW-UP` | User provided 3 concrete post-processing suggestions: (1) trailing period inconsistency, (2) leading dash removal, (3) long subtitle line wrapping. Sample SRT files attached. Also requesting merge module (#230). |
| **#263** | GPU not utilized / stuck at VAD | herlong6529424-dot | `AWAITING REPLY` | RTX 3070 8GB. Process hangs during Silero VAD. Asked for debug log + short file test. No response yet. |
| **#261** | Network check: unknown url type https | henry99a | `AWAITING REPLY` | NSIS SSL context issue. User found workaround (run from cmd). |
| **#260** | Uninstall leaves 6GB | hawai-stack | `AWAITING REPLY` | Responded with cache paths. |
| **#259** | Local Translation Issues (hotfix2) | destinyawaits | `RESPONDED` | Responded with v1.8.10 Ollama upgrade instructions. |
| **#258** | 我遇到的问题 (vague) | Uillike | `AWAITING REPLY` | Asked for logs. |
| **#255** | 如何用ollama进行翻译 | cheny7918 | `RESPONDED` | Responded with v1.8.10 Ollama upgrade instructions. 3 users. |
| **#251** | post2 launch failure (fastgit SSL) | zoqapopita93 | `AWAITING REPLY` | Diagnosed: fastgit mirror SSL failure. Provided 3 solutions. |
| **#243** | Install verify fails (RTX 3050) | Trenchcrack | `AWAITING REPLY` | Community member helping. |
| **#240** | GUI access violation Win11 | m739566004-svg | `SHIPPED` post2 | private_mode=True. |
| **#237** | XXL model questions | yangming2027 | `COMMUNITY` | Active discussion. |
| **#234** | CUDA version confusion | techguru0 | `AWAITING REPLY` | Corrected. |
| **#233** | Translation error (local LLM) | WillChengCN | `RESPONDED` | Recommended Ollama + v1.8.10 upgrade. |
| **#231** | Kaggle notebook error | fzfile | `RESPONDED` | llvmlite fix + translate cmd fix on main. |
| **#225** | GUI white screen | github3C | `STALE` | Exhausted hypotheses. |
| **#217** | GUI.exe not found (China network) | loveGEM | `STALE` | Root cause: PyTorch download fails behind GFW. |

### Stale / Low Activity (1)

| # | Title | Last Activity | Recommendation |
|---|-------|---------------|----------------|
| #232 | whisper-ja-anime model | 03-16 | Evaluate model. Keep open. |

### Feature Requests (29)

| # | Title | Priority | Target |
|---|-------|----------|--------|
| **#268** | Thai + Korean translation targets | LOW | v1.9+ (Korean requested by @seminice86 in comments) |
| **#265** | Post-translation hallucination filter (Chinese) | MEDIUM | v1.9 — user contributed 130-rule XML |
| **#264** | Model download location customization | LOW | Responded. HF_HOME workaround. |
| **#262** | Cohere Transcribe model | LOW | Responded. No timestamps — needs integration work. |
| **#254** | Remove non-speech sounds | MEDIUM | v1.9 |
| **#252** | Multi-speaker / diarization | MEDIUM | v1.9+ |
| **#250** | Model folder documentation | LOW | Docs / FAQ |
| **#248** | Diarization | MEDIUM | v1.9+ (dup #252). Responded — waiting for good solution. |
| **#247** | Docker support | LOW | Backlog |
| **#242** | XXL in Pass 1 | MEDIUM | v1.9 |
| **#239** | AMD GPU | MEDIUM | v1.9+ |
| **#230** | Standalone merge module | HIGH | v1.9.0. yangming2027 also requesting (#265). |
| **#224** | Vocal separation | MEDIUM | v1.9 |
| **#215** | Qwen3-ASR quality | LOW | Expected behavior |
| **#213** | Intel GPU (XPU) | LOW | v1.9+ |
| **#206** | Grey out incompatible options | MEDIUM | v1.9 |
| **#205** | VibeVoice ASR | LOW | v1.9+ |
| **#181** | Frameless window | LOW | Backlog |
| **#180** | Multi-language GUI (i18n) | HIGH | v1.9.0 |
| **#175** | Chinese GUI | HIGH | v1.9.0 |
| **#164** | MPEG-TS + Drive | LOW | Backlog |
| **#142** | AMD Radeon ROCm | MEDIUM | v1.9+ |
| **#128** | Gemma 3 models | HIGH | v1.9.0 |
| **#126** | Recursive directory | LOW | Backlog |
| **#114** | DirectML | MEDIUM | v1.9+ |
| **#99** | 4GB VRAM guidance | LOW | Backlog |
| **#96** | Settings persistence | MEDIUM | v1.9.0 |
| **#71** | Google Translate (free) | LOW | Backlog |
| **#59** | Feature plans (meta) | — | Keep open |
| **#51** | Batch translate | LOW | Backlog |
| **#49** | Output to source folder | LOW | Backlog |
| **#44** | GUI drag-drop | LOW | Backlog |
| **#43** | DeepL provider | LOW | v1.9+ |
| **#33** | Linux pyaudio docs | LOW | Backlog |

---

## Immediate Actions

| Action | Issues | Status |
|---|---|---|
| **Respond to #272** (XXL model flag bug — confirmed with debug log) | #272 | **TODO** |
| **Respond to #271** (Ollama variant error — 3 screenshots, needs diagnosis) | #271 | **TODO** |
| **Investigate #267** (debug logs available — semantic hang at streaming features) | #267 | **TODO** — logs ready for code analysis |
| Assess #265 post-processing suggestions (trailing period, leading dash, line wrap) | #265 | TODO — v1.9 scope |
| ~~Respond to #269~~ (scene-detect fix coded) | #269 | **DONE** — issue closed |
| ~~All NEEDS RESPONSE cleared~~ | — | **DONE** at rev20. 2 new NEEDS RESPONSE since. |
| ~~Follow up #263~~ (requested debug log + short file test) | #263 | **DONE** — waiting on user |
| ~~Follow up #267~~ (requested debug log + short file test) | #267 | **DONE** — user responded with 4 log files |
| ~~Push commits to main~~ | — | **DONE** |
| ~~Test Colab install fix on Colab~~ | — | **DONE** |

---

## Release Roadmap

### v1.8.10 Post-Release (on main, no version bump needed)

Colab/pip users get these automatically. .exe installer users get them via `whisperjav-upgrade`.

| Item | Commit | Status |
|---|---|---|
| Hallucination filter bundled in wheel | `1bfe548` | Pushed |
| --scene-detection-method fix (#269) | `4a600b2` | Pushed. #269 closed. |
| Docs: GUI guide for Ollama + enhance-for-VAD | `2533438` | Pushed |
| Issue tracker rev20 | `f875303` | Pushed |
| Colab install: remove venv, reuse system torch | `a6b473d` | Pushed. Colab-tested. |
| Colab/Kaggle notebook: remove venv refs | `a6b473d` | Pushed. Colab-tested. |

### v1.8.10-hotfix — If Needed

**Candidate bugs:**

| Bug | Issue | Severity | Hotfix? |
|---|---|---|---|
| XXL ignores --pass2-model flag | #272 | MEDIUM | **Yes — if fix is small.** Confirmed bug, user on Kaggle CLI. |
| Semantic scene detect hangs on Blackwell GPUs | #267 | HIGH | **No.** Needs investigation first. Workaround exists (auditok). |
| Ollama optimization variant error | #271 | LOW | **No.** Likely Ollama-side issue, not WhisperJAV bug. |

**Recommendation**: Investigate #272 first. If the fix is a one-liner (model flag not forwarded to XXL subprocess args), it's a good hotfix candidate alongside the already-pushed fixes.

### v1.9.0 — Next Major Release

**Theme: Platform Expansion + Translation Overhaul + UX**

| Priority | Item | Issues | Effort | Notes |
|---|---|---|---|---|
| **P0** | Ollama full migration (deprecate llama-cpp-python) | #132, #233, #255, #259 | Large | Remove ~1500 LOC. llama-cpp already broken on Colab. |
| **P0** | Chinese UI (partial i18n) | #175, #180 | Medium | 40%+ issues are Chinese. Biggest support burden reducer. |
| **P0** | Unified CLI override layer | #269 | Small | Fix architectural divergence between standard/ensemble override paths. |
| **P1** | Post-translation hallucination filter (Chinese) | #265 | Medium | User contributed 130-rule XML. Automate as post-process option. |
| **P1** | Post-processing polish (trailing period, leading dash, line wrapping) | #265 | Small | 3 concrete suggestions from yangming2027 with sample SRTs. |
| **P1** | Standalone merge CLI | #230 | Medium | `whisperjav-merge`. Active demand (yangming2027, weifu8435). |
| **P1** | Speaker diarization | #248, #252 | Large | Need quality solution. pyannote-audio or similar. |
| **P1** | XXL in Pass 1 | #242 | Medium | Needs forced alignment (no timestamps). |
| **P1** | AMD ROCm support | #142, #114, #239 | Medium | Document + test. |
| **P1** | Additional translation targets | #268 | Small | Thai, Korean requested. May be PySubtrans config only. |
| **P1** | Full dual-track enhance-for-VAD | — | Medium | ASR module separate VAD/ASR audio paths. |
| **P2** | Semantic scene detection hanging investigation | #267 | Medium | Hangs at "[1/5] Streaming features" on Blackwell GPU. Debug logs available. Needs code-level investigation. |
| **P2** | GUI settings persistence | #96 | Medium | Long-standing request. |
| **P2** | Vocal separation | #224, #254 | Medium | BS-RoFormer / UVR integration. |
| **P2** | Grey out incompatible options | #206 | Small | Prevent invalid GUI combos. |
| **P2** | Uninstall cleanup tool | #260 | Small | Script to remove cached models. |
| **P2** | Model cache paths docs | #250 | Small | FAQ / docs. |
| **P2** | MPS selective policy | #227 | Small | Force CPU for whisper, allow MPS for kotoba. |
| **P3** | Docker support | #247 | Medium | Dockerfile + compose. |
| **P3** | Gemma 3 model configs | #128 | Small | Contributor PR. |
| **P3** | whisper-ja-anime model | #232 | Small | If standard HF format. |

**Estimated timeline**: 6-8 weeks for P0+P1 items.

---

### v2.0 — Strategic Vision

**Theme: Architecture + Scale + Ecosystem**

| Area | Item | Why |
|---|---|---|
| **Architecture** | Plugin system for ASR backends | Hardcoded pipeline classes → community contributions |
| **Architecture** | Separate installer from runtime | Enables Docker, cloud, headless |
| **Architecture** | Web-based UI (replace pywebview) | PyWebView has persistent issues (#225, #240) |
| **Scale** | Batch processing dashboard | Queue, progress, ETA |
| **Scale** | GPU memory management overhaul | Proper memory budget for multi-model workflows |
| **Ecosystem** | Public API / SDK | REST API for remote processing |
| **Quality** | Ground truth test framework | Formalize F7 pattern. CI/CD regression detection |
| **Platform** | Linux native installer | AppImage or deb package |
| **Platform** | China mirror support | #217, #251 — PyTorch download fails behind GFW |

---

## Emerging Patterns

### Semantic Scene Detection Hanging (#267) — DEBUG LOGS AVAILABLE

**Symptom**: Process hangs at "[1/5] Streaming features (60s chunks)..." inside SemanticAudioClustering v7.0.0.

**Evidence from 4 log files (2026-03-31)**:
- 2-minute clip hangs identically to 1.9-hour file → **file length is NOT the cause**
- Both Qwen and Balanced pipelines hang → **pipeline is NOT the cause**
- Auditok works perfectly on same files → **audio content is NOT the cause**
- `SemanticClusteringAdapter initialized` successfully — engine loads, hangs during feature extraction
- RTX 5060 Ti (Blackwell architecture) — possibly GPU/driver-specific
- `--debug` flag produces no additional output from inside the streaming step

**Hypothesis (unverified)**: The feature streaming step may use a GPU operation (torch audio feature extraction) that is incompatible with the Blackwell/RTX 50 series drivers or CUDA version. The hang occurs inside a library call, not in WhisperJAV code.

**Next step**: Read the semantic scene detection source code to identify what "[1/5] Streaming features" does — what library calls, what GPU operations. Determine if there's a known Blackwell compatibility issue.

### Silero VAD Hanging (#263)

**Symptom**: Process hangs during Silero VAD initialization/inference. Log shows "Starting speech segmentation with: silero-v4.0" with no "complete" message. User has RTX 3070 8GB, processing a 2.4h file (314 scenes).
**Status**: Waiting on user debug log + short file test. No response since 2026-03-31 request.

### XXL Model Flag Ignored (#272) — CONFIRMED BUG

**Symptom**: `--pass2-model large-v2` is logged correctly by WhisperJAV but not forwarded to the Faster-Whisper-XXL subprocess. XXL loads large-v3 by default.
**Evidence**: Debug log shows `[Worker 3816] Pass 2: BYOP XXL (exe=..., model=large-v2)` but XXL subprocess output says `'large-v3' model may produce worse results`.
**Root cause**: The `--model` flag is likely missing from the XXL subprocess command construction in `pass_worker.py` or `xxl_runner.py`.

### Hallucination in Chinese Translation (#265)

**Symptom**: After Japanese transcription + DeepSeek translation to Chinese, hallucination phrases appear in Chinese output.
**Assessment**: Two contributing factors: (1) hallucination filter was not bundled in wheel — fix committed; (2) no post-translation sanitization step exists. User provided 3 concrete post-processing suggestions with sample SRTs.

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Action |
|---------|--------|---------|--------|
| **Local LLM / Ollama** | #271, #259, #255, #233, #132 | #132 | v1.8.10 ships Ollama GUI. #271 new (variant error). |
| **Diarization** | #248, #252 | #248 | v1.9+. Both responded. |
| **AMD/Intel GPU** | #239, #142, #114, #213 | #142 | v1.9+. |
| **i18n** | #175, #180 | #180 | v1.9.0 P0. |
| **Speech enhancement** | #254, #224 | #224 | v1.9. |
| **XXL** | #272, #242, #237, #223 | #242 | #272 is a bug (model flag). #242 is feature (Pass 1). |
| **Install/Network** | #261, #251, #243, #240, #225, #217 | — | Individual. |
| **Model management** | #264, #250 | #250 | Docs / FAQ. |
| **Translation targets** | #268 | #268 | Thai + Korean. v1.9+. |
| **Scene detection** | #267 | — | Debug logs available. Needs code investigation. |
| **Hallucination** | #265, #246 | #265 | Post-translation filter for v1.9. |
| **Merge module** | #230, #265 | #230 | v1.9.0 P1. yangming2027 strong advocate. |

---

## Changelog

| Date | Changes |
|------|---------|
| **2026-03-31** | **rev21.** 54→55 open. New: #271 (Ollama variant error), #272 (XXL model flag bug). Closed: #269. #267 debug logs arrived — hangs at "[1/5] Streaming features" regardless of file length/pipeline, Blackwell GPU. #272 confirmed with debug log — model flag not forwarded to XXL subprocess. |
| 2026-03-31 | **rev20.** 55→54 open. New: #265, #267, #268, #269. Closed: #218, #221, #236, #244. All fixes pushed to main (4 commits). Colab tested. All NEEDS RESPONSE cleared. |
| 2026-03-30 | **rev19.** v1.8.10 released. 54 open. #253 closed. |
| 2026-03-29 | **rev18.** 55 open. Session work: aggressive retune, Ollama GUI, bug fixes. |
| 2026-03-28 | **rev17.** Closed 11. Responded 7. |
| 2026-03-28 | **rev16.** Full refresh. |

---
