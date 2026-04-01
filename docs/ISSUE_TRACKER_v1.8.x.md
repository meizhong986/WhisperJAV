# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-04-01 (rev23) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **55 open** on GitHub

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
| Total open on GitHub | **55** | Unchanged from rev22. No new issues. |
| **v1.8.10 RELEASED** | — | 2026-03-30. 39 commits. |
| **FIX CODED (on main, not released)** | 4 | Hallucination filter, Colab, XXL model (#272), translation overwrite (#259) |
| **FIX CODED (local, not pushed)** | 1 | numba cache dir (#267, #263) |
| **NEEDS RESPONSE** | 1 | #267 (user asking whether to wait for code fix or apply temp workaround) |
| **AWAITING REPLY** | 6 | #234, #251, #258, #260, #261, #263 |
| Feature requests (open) | 29 | See Feature Requests section |

---

## v1.8.10 — RELEASED (2026-03-30)

See `installer/RELEASE_NOTES_v1.8.10.md` for full details.

---

## Fixes on main (committed, not released)

These fixes are committed to main after v1.8.10 was tagged. Colab/pip users get them automatically. .exe installer users will get them in the next release.

| Commit | Item | Issues | Risk |
|--------|------|--------|------|
| `e50df15` | XXL model flag not forwarded to subprocess; semantic scene detect diagnostics v7.1.0 | #272, #267 | LOW — XXL arg fix + diagnostic logging |
| `1bfe548` | Hallucination filter: bundled JSON missing from wheel, stale gist URLs, INFO logging | #265 | LOW — packaging + logging |
| `4a600b2` | `--scene-detection-method` ignored in standard pipeline modes | #269 (closed) | LOW — one-line injection |
| `2533438` | Docs: GUI user guide updated for Ollama + enhance-for-VAD | — | LOW — docs only |
| `a6b473d` | Colab install: remove venv, reuse system torch, fix llama-cpp filename | #231 | LOW — Colab-tested |
| `f875303` | Issue tracker rev20 | — | — |

---

## Open Issues by Status

### Bugs / Active Issues (18)

| # | Title | Reporter | Status | Notes |
|---|-------|----------|--------|-------|
| **#272** | XXL ignores --pass2-model large-v2 flag | TinyRick1489 | `FIX CODED` | Fix on main (`e50df15`). Responded with upgrade instructions. Awaiting user confirmation. |
| **#271** | Ollama "Could Not Create Optimized Variant" | justantopair-ai | `RESOLVED` | User confirmed general-purpose models (qwen3:14b, qwen3.5:9b) produce format errors. Guided to use instruct models. gemma3:12b works well. |
| **#267** | Stuck on streaming features (Qwen+Semantic) | OrangeFarmHorse | `NEEDS RESPONSE` | **ROOT CAUSE FOUND**: numba PermissionError — can't write JIT cache to `C:\Tools\whisperjav\lib\site-packages\librosa\__pycache__\`. Fix coded locally (`ensure_numba_cache_dir`). User asking whether to wait or apply temp workaround. |
| **#265** | Hallucination in Chinese translation (large-v2) | yangming2027 | `NEEDS FOLLOW-UP` | 3 post-processing suggestions with sample SRTs: trailing period, leading dash, line wrapping. Also shared Subtitle Edit multi-replace XML. |
| **#263** | GPU not utilized / stuck at VAD | herlong6529424-dot | `AWAITING REPLY` | Likely same root cause as #267 (numba cache permissions). Responded with diagnostic script + upgrade instructions. |
| **#261** | Network check: unknown url type https | henry99a | `AWAITING REPLY` | NSIS SSL context issue. User found workaround (run from cmd). |
| **#260** | Uninstall leaves 6GB | hawai-stack | `AWAITING REPLY` | Responded with cache paths. |
| **#259** | Local Translation Issues (hotfix2) | destinyawaits | `NEEDS FOLLOW-UP` | **Ollama confirmed working.** New bug: temp file overwrites existing English subtitles in same folder — save path not respected until task complete. |
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
| **Respond to #267** (user asking: wait for fix or apply temp workaround?) | #267 | **TODO** — fix coded locally, needs commit+push |
| ~~Respond to #259~~ (temp file overwrite — fix coded) | #259 | **DONE** — fix on main (`cfcbddf`) |
| ~~Respond to #263~~ (diagnostic script + upgrade instructions sent) | #263 | **DONE** — awaiting user response |
| Assess #265 post-processing suggestions (trailing period, leading dash, line wrap) | #265 | TODO — v1.9 scope |
| ~~Respond to #272~~ (fix coded, upgrade instructions given) | #272 | **DONE** — awaiting user confirmation |
| ~~Respond to #271~~ (asked which model, explained variant warning) | #271 | **DONE** — awaiting user reply |
| ~~Respond to #267~~ (added diagnostic logging, asked to upgrade and retest) | #267 | **DONE** — user responded with diagnostic log |
| ~~Follow up #263~~ (requested debug log + short file test) | #263 | **DONE** — waiting on user |

---

## Release Roadmap

### v1.8.10 Post-Release (on main, no version bump needed)

Colab/pip users get these automatically. .exe installer users get them via `whisperjav-upgrade`.

| Item | Commit | Status |
|---|---|---|
| XXL model flag fix (#272) + semantic diagnostics v7.1.0 (#267) | `e50df15` | Pushed |
| Hallucination filter bundled in wheel | `1bfe548` | Pushed |
| --scene-detection-method fix (#269) | `4a600b2` | Pushed. #269 closed. |
| Docs: GUI guide for Ollama + enhance-for-VAD | `2533438` | Pushed |
| Colab install: remove venv, reuse system torch | `a6b473d` | Pushed. Colab-tested. |

### v1.8.10-hotfix — Not Currently Needed

All fixes are on main and accessible via `pip install --no-deps --upgrade`. No blocking bugs requiring a versioned hotfix release. The #272 XXL fix and #267 diagnostics are already available to pip/Colab users.

A hotfix would only be needed if:
- A blocking bug is found that affects .exe installer users specifically
- The #267 semantic hang turns out to be a code bug (currently investigating)

### v1.9.0 — Next Major Release

**Theme: Platform Expansion + Translation Overhaul + UX**

| Priority | Item | Issues | Effort | Notes |
|---|---|---|---|---|
| **P0** | Ollama full migration (deprecate llama-cpp-python) | #132, #233, #255, #259 | Large | Remove ~1500 LOC. llama-cpp already broken on Colab. |
| **P0** | Chinese UI (partial i18n) | #175, #180 | Medium | 40%+ issues are Chinese. Biggest support burden reducer. |
| **P0** | Unified CLI override layer | #269 | Small | Fix architectural divergence between standard/ensemble override paths. |
| **P1** | Translation temp file overwrite fix | #259 | Small | Save path not respected until task complete — overwrites existing SRTs. |
| **P1** | Post-translation hallucination filter (Chinese) | #265 | Medium | User contributed 130-rule XML. Automate as post-process option. |
| **P1** | Post-processing polish (trailing period, leading dash, line wrapping) | #265 | Small | 3 concrete suggestions from yangming2027 with sample SRTs. |
| **P1** | Standalone merge CLI | #230 | Medium | `whisperjav-merge`. Active demand (yangming2027, weifu8435). |
| **P1** | Speaker diarization | #248, #252 | Large | Need quality solution. pyannote-audio or similar. |
| **P1** | XXL in Pass 1 | #242 | Medium | Needs forced alignment (no timestamps). |
| **P1** | AMD ROCm support | #142, #114, #239 | Medium | Document + test. |
| **P1** | Additional translation targets | #268 | Small | Thai, Korean requested. May be PySubtrans config only. |
| **P1** | Full dual-track enhance-for-VAD | — | Medium | ASR module separate VAD/ASR audio paths. |
| **P2** | Semantic scene detection: librosa.feature.mfcc hang | #267 | Medium | Diagnostic log pinpointed hang at first `librosa.feature.mfcc()` call. Versions all compatible. Suspected numba JIT issue — needs further investigation. |
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

### Semantic Scene Detection Hanging (#267) — ROOT CAUSE FOUND

**Root cause**: numba PermissionError. The conda-constructor installs to `C:\Tools\whisperjav\` (system-level, installed as admin). User runs as regular account ("Animal"). numba's JIT cache tries to write to `librosa/__pycache__/` inside site-packages — permission denied. numba hangs instead of failing cleanly (Windows `tempfile._mkstemp_inner` retry/stall on access check).

**Evidence** (Ctrl+C traceback from standalone test):
```
PermissionError: [Errno 13] Permission denied: 'C:\\Tools\\whisperjav\\lib\\site-packages\\librosa\\__pycache__\\tmpktp60s15'
```
Stack: `librosa.feature.mfcc()` → lazy import `librosa.filters` → numba `@jit` decorator → `FunctionCache` → `ensure_cache_path()` → `tempfile.TemporaryFile(dir=__pycache__)` → hang.

**Fix**: `ensure_numba_cache_dir()` in `whisperjav/utils/console.py` — sets `NUMBA_CACHE_DIR` to a user-writable location (`%LOCALAPPDATA%\whisperjav\numba_cache`) before any librosa import. Coded locally, not yet pushed.

### Silero VAD Hanging (#263) — Likely Same Root Cause

**Symptom**: Process hangs during Silero VAD. Log shows "Starting speech segmentation with: silero-v4.0" with no "complete" message. RTX 3070 8GB.
**Hypothesis**: Likely same numba PermissionError as #267. Silero backend uses `librosa.resample` (`silero_backend.py:217`) which triggers the same numba JIT cache chain.
**Status**: Sent diagnostic script + upgrade instructions (2026-04-01). Awaiting user response.

### Translation Temp File Overwrite (#259) — NEW BUG

**Symptom**: During AI SRT Translate, the temp file is created in the same folder as the input SRT and overwrites existing English subtitles generated earlier (e.g., by DeepSeek). The user-specified save path is not used until the task completes.
**Reporter**: destinyawaits (also confirmed Ollama working after v1.8.10 upgrade).
**Status**: Needs investigation — is this PySubtrans behavior or WhisperJAV's translation wrapper?

### Ollama Model Suitability (#271) — RESOLVED

**Symptom**: Translation format errors with Ollama. Model output doesn't match PySubtrans expected format.
**Resolution**: User confirmed general-purpose models (qwen3:14b, qwen3.5:9b) produce format errors. Guided to use instruct models. gemma3:12b confirmed working well for translation.

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Action |
|---------|--------|---------|--------|
| **Local LLM / Ollama** | #271, #259, #255, #233, #132 | #132 | #259 Ollama working, new temp file bug. #271 awaiting model info. |
| **Diarization** | #248, #252 | #248 | v1.9+. Both responded. |
| **AMD/Intel GPU** | #239, #142, #114, #213 | #142 | v1.9+. |
| **i18n** | #175, #180 | #180 | v1.9.0 P0. |
| **Speech enhancement** | #254, #224 | #224 | v1.9. |
| **XXL** | #272, #242, #237, #223 | #242 | #272 fix coded. #242 is feature (Pass 1). |
| **Install/Network** | #261, #251, #243, #240, #225, #217 | — | Individual. |
| **Model management** | #264, #250 | #250 | Docs / FAQ. |
| **Translation targets** | #268 | #268 | Thai + Korean. v1.9+. |
| **Scene detection** | #267 | — | Hang pinpointed at librosa.feature.mfcc(). Next: test NUMBA_DISABLE_JIT=1. |
| **Hallucination** | #265, #246 | #265 | Post-translation filter for v1.9. |
| **Merge module** | #230, #265 | #230 | v1.9.0 P1. yangming2027 strong advocate. |
| **Translation UX** | #259 | #259 | Temp file overwrite bug. New. |

---

## Changelog

| Date | Changes |
|------|---------|
| **2026-04-01** | **rev23.** 55 open. #267 ROOT CAUSE FOUND: numba PermissionError on conda-constructor installs (can't write JIT cache to site-packages). Fix coded: `ensure_numba_cache_dir()` in `console.py`. #259 translation overwrite fix coded (`cfcbddf`). #271 resolved — user guided to instruct models, gemma3:12b works. #263 responded with diagnostic script (likely same root cause as #267). |
| 2026-04-01 | **rev22.** 55 open. `e50df15`: XXL model flag fix (#272) + semantic diagnostics v7.1.0 (#267). `cfcbddf`: translation overwrite fix (#259) + diagnostic script (#267). |
| 2026-03-31 | **rev21.** 54→55 open. New: #271, #272. Closed: #269. #267 debug logs arrived. #272 confirmed. |
| 2026-03-31 | **rev20.** 55→54 open. New: #265, #267, #268, #269. Closed: #218, #221, #236, #244. All NEEDS RESPONSE cleared. |
| 2026-03-30 | **rev19.** v1.8.10 released. 54 open. #253 closed. |
| 2026-03-29 | **rev18.** 55 open. Session work: aggressive retune, Ollama GUI, bug fixes. |
| 2026-03-28 | **rev17.** Closed 11. Responded 7. |
| 2026-03-28 | **rev16.** Full refresh. |

---
