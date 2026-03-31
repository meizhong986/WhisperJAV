# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-31 (rev20) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **55 open** on GitHub

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
| Total open on GitHub | **55** | Was 54 at rev19. New: #265, #267, #268, #269. Closed: #218, #221, #236. |
| **v1.8.10 RELEASED** | — | 2026-03-30. 39 commits. |
| **FIX CODED (on main, not released)** | 3 | #269 (scene-detect flag), hallucination filter bundle, Colab install |
| **NEW since rev19** | 4 | #265, #267, #268, #269 |
| **NEEDS RESPONSE** | 5 | #246, #247, #250, #259, #269 |
| **NEEDS FOLLOW-UP** | 3 | #263, #265, #267 |
| **AWAITING REPLY** | 5 | #234, #244, #251, #258, #261 |
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
| `4a600b2` | `--scene-detection-method` ignored in standard pipeline modes | #269 | LOW — one-line injection |
| `2533438` | Docs: GUI user guide updated for Ollama + enhance-for-VAD | — | LOW — docs only |
| (pending) | Colab install: remove venv, reuse system torch, fix llama-cpp filename | — | LOW — needs Colab testing |
| (pending) | Colab/Kaggle notebooks: remove venv references | #231 | LOW — config changes |
| (pending) | Issue tracker rev20 | — | — |

---

## Open Issues by Status

### Bugs / Active Issues (17)

| # | Title | Reporter | Status | Notes |
|---|-------|----------|--------|-------|
| **#269** | --scene-detection-method ignored | TinyRick1489 | `FIX CODED` | Fix on main (commit `4a600b2`). Respond to user. |
| **#267** | Stuck on streaming features (Qwen+Semantic) | OrangeFarmHorse | `NEEDS FOLLOW-UP` | User confirmed: process hangs during semantic scene detection. Auditok works as workaround. RTX 5060 Ti. Cause not yet diagnosed — needs investigation. |
| **#265** | Hallucination in Chinese translation (large-v2) | yangming2027 | `COMMUNITY` | Not a bug. User shared 130-rule post-processing XML. Agreed to share with community. Also requesting standalone merge module (#230). Hallucination filter bundling fix may help. |
| **#263** | GPU not utilized / stuck at VAD | herlong6529424-dot | `NEEDS FOLLOW-UP` | RTX 3070 8GB. User confirmed: log stops after "Starting speech segmentation with: silero-v4.0" — no "complete" message. Process hangs during Silero VAD. Cause not yet diagnosed. |
| **#261** | Network check: unknown url type https | henry99a | `AWAITING REPLY` | NSIS SSL context issue. User found workaround (run from cmd). Asked if install completed. |
| **#260** | Uninstall leaves 6GB | hawai-stack | `AWAITING REPLY` | Responded with cache paths. |
| **#259** | Local Translation Issues (hotfix2) | destinyawaits | `NEEDS RESPONSE` | v1.8.10 ships Ollama GUI. Respond with upgrade path. |
| **#258** | 我遇到的问题 (vague) | Uillike | `AWAITING REPLY` | Asked for logs. |
| **#255** | 如何用ollama进行翻译 | cheny7918 | `RESPONDED` | Responded with v1.8.10 Ollama upgrade instructions. 3 users. |
| **#251** | post2 launch failure (fastgit SSL) | zoqapopita93 | `RESPONDED` | Diagnosed: fastgit mirror SSL failure. Provided 3 solutions. |
| **#243** | Install verify fails (RTX 3050) | Trenchcrack | `AWAITING REPLY` | Community member helping. |
| **#240** | GUI access violation Win11 | m739566004-svg | `SHIPPED` post2 | private_mode=True. |
| **#237** | XXL model questions | yangming2027 | `COMMUNITY` | Active discussion. |
| **#234** | CUDA version confusion | techguru0 | `AWAITING REPLY` | Corrected. |
| **#233** | Translation error (local LLM) | WillChengCN | `NEEDS RESPONSE` | Recommend Ollama + v1.8.10 upgrade. |
| **#231** | Kaggle notebook error | fzfile | `RESPONDED` | llvmlite fix + translate cmd fix on main. |
| **#225** | GUI white screen | github3C | `STALE` | Exhausted hypotheses. |
| **#217** | GUI.exe not found (China network) | loveGEM | `STALE` | Root cause: PyTorch download fails behind GFW. |

### Stale / Low Activity (2)

| # | Title | Last Activity | Recommendation |
|---|-------|---------------|----------------|
| #244 | search for XXL | 03-19 | Close for inactivity. |
| #232 | whisper-ja-anime model | 03-16 | Evaluate model. Keep open. |

### Feature Requests (29)

| # | Title | Priority | Target |
|---|-------|----------|--------|
| **#268** | Thai translation target | LOW | v1.9+ (also Korean requested in comments) |
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
| **Respond to #269** (scene-detect fix coded) | #269 | TODO |
| **Respond to #259** (local translation → upgrade to v1.8.10 Ollama) | #259 | TODO |
| **Follow up #263** (Silero VAD hanging — user confirmed) | #263 | TODO — investigate |
| **Follow up #267** (Semantic scene hanging on long files — user confirmed) | #267 | TODO — investigate |
| **Push 3 commits to main** | — | TODO |
| **Test Colab install fix on Colab** | — | TODO |
| **Close #244** (stale) | #244 | TODO |

---

## Release Roadmap

### v1.8.10 Post-Release (on main, no version bump needed)

Colab/pip users get these automatically. .exe installer users get them via `whisperjav-upgrade`.

| Item | Commit | Status |
|---|---|---|
| Hallucination filter bundled in wheel | `1bfe548` | Committed |
| --scene-detection-method fix (#269) | `4a600b2` | Committed |
| Docs: GUI guide for Ollama + enhance-for-VAD | `2533438` | Committed |
| Colab install: remove venv, reuse system torch | — | Coded, needs Colab test |
| Colab/Kaggle notebook: remove venv refs | — | Coded, needs Colab test |

No version bump needed — these are incremental fixes on main. Colab always installs from main.

### v1.8.10.post1 — If Needed

Only release a post1 if:
- A blocking bug is found that affects .exe installer users
- The Silero VAD hanging issue (#263) turns out to be a code bug (not hardware-specific)

Currently not planned.

---

### v1.9.0 — Next Major Release

**Theme: Platform Expansion + Translation Overhaul + UX**

| Priority | Item | Issues | Effort | Notes |
|---|---|---|---|---|
| **P0** | Ollama full migration (deprecate llama-cpp-python) | #132, #233, #255, #259 | Large | Remove ~1500 LOC. llama-cpp already broken on Colab. |
| **P0** | Chinese UI (partial i18n) | #175, #180 | Medium | 40%+ issues are Chinese. Biggest support burden reducer. |
| **P0** | Unified CLI override layer | #269 | Small | Fix architectural divergence between standard/ensemble override paths. |
| **P1** | Post-translation hallucination filter (Chinese) | #265 | Medium | User contributed 130-rule XML. Automate as post-process option. |
| **P1** | Standalone merge CLI | #230 | Medium | `whisperjav-merge`. Active demand (yangming2027, weifu8435). |
| **P1** | Speaker diarization | #248, #252 | Large | Need quality solution. pyannote-audio or similar. |
| **P1** | XXL in Pass 1 | #242 | Medium | Needs forced alignment (no timestamps). |
| **P1** | AMD ROCm support | #142, #114, #239 | Medium | Document + test. |
| **P1** | Additional translation targets | #268 | Small | Thai, Korean requested. May be PySubtrans config only. |
| **P1** | Full dual-track enhance-for-VAD | — | Medium | ASR module separate VAD/ASR audio paths. |
| **P2** | GUI settings persistence | #96 | Medium | Long-standing request. |
| **P2** | Vocal separation | #224, #254 | Medium | BS-RoFormer / UVR integration. |
| **P2** | Grey out incompatible options | #206 | Small | Prevent invalid GUI combos. |
| **P2** | Uninstall cleanup tool | #260 | Small | Script to remove cached models. |
| **P2** | Model cache paths docs | #250 | Small | FAQ / docs. |
| **P2** | MPS selective policy | #227 | Small | Force CPU for whisper, allow MPS for kotoba. |
| **P2** | Semantic scene detect hanging | #267 | Medium | Process hangs during semantic scene detection. Cause unknown — needs investigation before scoping. |
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

### Silero VAD Hanging (#263)
**Symptom**: Process hangs during Silero VAD initialization/inference. Log shows "Starting speech segmentation with: silero-v4.0" with no "complete" message. User has RTX 3070 8GB, processing a 2.4h file (314 scenes).
**Status**: Cause unknown. Not yet investigated. Need to determine whether it's system-specific, audio-specific, file-length-related, or a code bug. Next step: ask user to test a short file to help isolate.

### Semantic Scene Detection Hanging (#267)
**Symptom**: Process hangs during semantic scene detection at "[1/5] Streaming features (60s chunks)...". User has RTX 5060 Ti. Auditok works as workaround.
**Status**: Cause unknown. Not yet investigated. User confirmed the hang occurs with both Qwen and Balanced pipelines when semantic is selected. Next step: inspect the semantic engine code, attempt local reproduction.

### Hallucination in Chinese Translation (#265)
**Symptom**: After Japanese transcription + DeepSeek translation to Chinese, hallucination phrases appear in the Chinese output. User contributed 130-rule Subtitle Edit XML for post-processing cleanup.
**Assessment**: Two contributing factors identified: (1) hallucination filter was not bundled in wheel — fix committed; (2) no post-translation sanitization step exists. User agreed to share rules for integration. Potential v1.9.0 feature.

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Action |
|---------|--------|---------|--------|
| **Local LLM / Ollama** | #259, #255, #233, #132 | #132 | v1.8.10 ships Ollama GUI. Respond with upgrade. |
| **Diarization** | #248, #252 | #248 | v1.9+. Both responded. |
| **AMD/Intel GPU** | #239, #142, #114, #213 | #142 | v1.9+. |
| **i18n** | #175, #180 | #180 | v1.9.0 P0. |
| **Speech enhancement** | #254, #224 | #224 | v1.9. |
| **XXL** | #242, #237, #223, #244 | #242 | v1.9. |
| **Install/Network** | #261, #251, #243, #240, #225, #217 | — | Individual. |
| **Model management** | #264, #250 | #250 | Docs / FAQ. |
| **Translation targets** | #268 | #268 | Thai + Korean. v1.9+. |
| **Scene detection** | #267, #269 | — | #269 fix coded. #267 needs investigation. |
| **Hallucination** | #265, #246 | #265 | Post-translation filter for v1.9. |
| **Merge module** | #230, #265 | #230 | v1.9.0 P1. yangming2027 strong advocate. |

---

## Changelog

| Date | Changes |
|------|---------|
| **2026-03-31** | **rev20.** 55 open. New: #265, #267, #268, #269. Closed: #218, #221, #236. Fixes coded on main: hallucination filter bundle, scene-detect flag (#269), docs. #263 user confirmed process hangs during Silero VAD (cause unknown). #267 user confirmed process hangs during semantic scene detection (cause unknown). #265 user agreed to share hallucination rules. Added emerging patterns section. Added Rule 0.5 to CLAUDE.md (symptom vs cause). |
| 2026-03-30 | **rev19.** v1.8.10 released. 54 open. #253 closed. Responded to #255, #264, #263, #231. Closed #221, #218, #236. |
| 2026-03-29 | **rev18.** 55 open. Session work: aggressive retune, Ollama GUI, bug fixes. |
| 2026-03-28 | **rev17.** Closed 11. Responded 7. |
| 2026-03-28 | **rev16.** Full refresh. |

---
