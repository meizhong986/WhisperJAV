# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-29 (rev18) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **55 open** on GitHub

---

## Status Legend

| Label | Meaning |
|-------|---------|
| `SHIPPED` | Fix released in a stable version. Waiting for user to test/confirm. |
| `FIX VERIFIED` | User confirmed the fix resolves their issue. Safe to close the GitHub issue. |
| `AWAITING CONFIRMATION` | Response given. Waiting for user reply. |
| `NEEDS RESPONSE` | Issue has no response or needs a follow-up reply. |
| `DEFERRED` | Moved to a future release. |

---

## Quick Stats

| Category | Count | Notes |
|----------|------:|-------|
| Total open on GitHub | **55** | Was 53 at rev16. Closed 11 in rev17, but recount found 2 older uncounted. |
| Closed in this session | 11 | #228, #229, #235, #222, #220, #200, #204, #207, #209, #210, #212 |
| Responded in this session | 8 | #263, #261, #260, #251, #258, #234, #239, #253 |
| **FIX CODED (for v1.8.10)** | 3 | #253 (Silero trust_repo), XXL exe persistence, XXL model in extra args |
| **NEEDS RESPONSE (no reply)** | 7 | #246, #247, #248, #250, #254, #259, #262 |
| **AWAITING USER REPLY** | 7 | #251, #258, #261, #263, #234, #243, #244 |
| **SHIPPED (awaiting test)** | 3 | #132, #236, #240 |
| Feature requests (open) | 28 | See Cluster J |

---

## v1.8.10 — IN DEVELOPMENT (dev_v1.8.10, 37 commits ahead of main)

### What's committed

| Category | Item | Issues | Commits |
|----------|------|--------|---------|
| **Quality** | Aggressive sensitivity retune (F7 ground truth validated) | — | `e6ffd52`, `abc84c6`, `1ccdd9d`, `a6a9ad0` |
| **Quality** | compression_ratio=2.6, condition_on_previous_text=True (tuner-validated) | — | `a6a9ad0` |
| **Tooling** | Whisper param tuner utility (`scripts/whisper_param_tuner.py`) | — | `57e9f0a` |
| **Tooling** | Diagnostic JSON per scene (full transcribe() results) | — | `c0dbf60` |
| **Bug fix** | Config contamination firewall, dead code removal | — | `ae2f0f3` |
| **Bug fix** | GUI ensemble presets not applied on pipeline switch | — | `6eaa07d` |
| **Bug fix** | XXL exe path not restored on app restart | — | `fd1648c` |
| **Bug fix** | XXL --model moved to user-editable Extra Args | — | `ff692bb` |
| **Bug fix** | Silero VAD crashes in Colab/Kaggle (trust_repo) | #253 | `9329840` |
| **Feature** | Enhance-for-VAD checkbox in ensemble UI, all pipelines | — | `06f0c2d` |

### What still needs doing for v1.8.10

| Item | Priority | Status |
|---|---|---|
| Respond to #263 (new screenshot from user) | HIGH | TODO |
| Respond to remaining 7 NEEDS RESPONSE issues | MEDIUM | TODO |
| Write Ollama translation FAQ | MEDIUM | TODO (#255, #259, #233) |
| Write model cache paths doc | LOW | TODO (#250) |
| Test enhance-for-VAD checkbox in GUI | HIGH | TODO |

---

## Open Issues by Status

### Bugs / Active Issues (14)

| # | Title | Reporter | Status | Notes |
|---|-------|----------|--------|-------|
| **#263** | GPU not utilized | herlong6529424-dot | `AWAITING REPLY` | Responded. User replied with screenshot — needs follow-up. |
| **#261** | Network check: unknown url type https | henry99a | `AWAITING REPLY` | Responded with SSL fix. |
| **#260** | Uninstall leaves 6GB | hawai-stack | `AWAITING REPLY` | Responded with cache paths. |
| **#259** | Local Translation Issues (hotfix2) | destinyawaits | `NEEDS RESPONSE` | Same user as closed #212. |
| **#258** | 我遇到的问题 (vague) | Uillike | `AWAITING REPLY` | Asked for logs. |
| **#253** | Colab silero trust_repo | KinhoLeung | `FIX CODED` v1.8.10 | trust_repo=True added. Responded with workaround. |
| **#251** | post2 launch failure | zoqapopita93 | `AWAITING REPLY` | Asked for details. |
| **#246** | anime-whisper hallucination bug | dadlaugh | `NEEDS RESPONSE` | Valuable technical report. |
| **#243** | Install verify fails (RTX 3050) | Trenchcrack | `AWAITING REPLY` | Asked for confirmation. |
| **#240** | GUI access violation Win11 | m739566004-svg | `SHIPPED` post2 | private_mode=True fix shipped. |
| **#237** | XXL model questions | yangming2027 | `NEEDS FOLLOW-UP` | liugngg question about post-processing. |
| **#234** | CUDA version confusion | techguru0 | `AWAITING REPLY` | Corrected. |
| **#233** | translation error (local LLM) | WillChengCN | `NEEDS RESPONSE` | Recommend Ollama. |
| **#225** | GUI white screen | github3C | `STALE` | WebView2 confirmed OK. Exhausted hypotheses. |

### Stale / Low Activity (6)

| # | Title | Last Activity | Recommendation |
|---|-------|---------------|----------------|
| #236 | WebUI cache stale | 03-19 | Close (fixed in post1) |
| #231 | Kaggle llvmlite | 03-27 | Respond with fix: `pip install -U llvmlite numba` |
| #227 | M1 MAX hang | 03-17 | Keep open as known issue |
| #221 | cublas64_12.dll missing | 03-14 | Close for inactivity |
| #218 | cu118 wheel mismatch | 03-14 | Close (shipped in post1) |
| #217 | GUI.exe not found | 03-17 | Close for inactivity |

### Feature Requests (28)

| # | Title | Priority | Target |
|---|-------|----------|--------|
| **#262** | Cohere Transcribe model | LOW | Evaluate |
| **#254** | Remove non-speech sounds | MEDIUM | v1.9 |
| **#252** | Multi-speaker / diarization | MEDIUM | v1.9+ |
| **#250** | Model folder documentation | LOW | v1.8.10 (docs) |
| **#248** | Diarization | MEDIUM | v1.9+ (dup #252) |
| **#247** | Docker support | LOW | Backlog |
| **#242** | XXL in Pass 1 | MEDIUM | v1.9 |
| **#239** | AMD GPU | MEDIUM | v1.9+ |
| **#232** | whisper-ja-anime model | LOW | Evaluate |
| **#230** | Standalone merge module | HIGH | v1.9.0 |
| **#224** | Vocal separation | MEDIUM | v1.9 |
| **#223** | XXL comparison/integration | — | `SHIPPED` v1.8.9 |
| **#215** | Qwen3-ASR quality | LOW | Expected behavior |
| **#213** | Intel GPU (XPU) | LOW | v1.9+ |
| **#206** | Grey out incompatible options | MEDIUM | v1.9 |
| **#205** | VibeVoice ASR | LOW | v1.9+ |
| **#181** | Frameless window | LOW | Backlog |
| **#180** | Multi-language GUI (i18n) | HIGH | v1.9.0 |
| **#175** | Chinese GUI | HIGH | v1.9.0 |
| **#164** | MPEG-TS + Drive | LOW | Backlog |
| **#142** | AMD Radeon ROCm | MEDIUM | v1.9+ |
| **#132** | Local LLM Kaggle | — | `SHIPPED` post1 |
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
| ~~Close 6 stale/verified issues~~ | #228, #229, #235, #222, #220 | **DONE** (rev17) |
| ~~Batch respond to 7 simple issues~~ | #263, #261, #260, #251, #258, #234, #239 | **DONE** (rev17) |
| ~~Fix Silero trust_repo for Colab~~ | #253 | **DONE** (rev18) |
| ~~Fix XXL exe path persistence~~ | — | **DONE** (rev18) |
| ~~Fix XXL model in Extra Args~~ | — | **DONE** (rev18) |
| ~~Restore enhance-for-VAD checkbox~~ | — | **DONE** (rev18) |
| **Follow up #263** (user replied with screenshot) | #263 | TODO |
| **Close 3 more stale issues** | #221, #218, #217 | TODO |
| **Write Ollama translation FAQ** | #255, #259, #233 | TODO |
| **Write model cache paths doc** | #250 | TODO |
| **Respond to #246** (anime-whisper hallucination) | #246 | TODO |

---

## F3: Release Roadmap & Recommendations

### v1.8.10 — Release Candidate

**Theme: Quality + Stability + Developer Tools**

37 commits ahead of main. Ready to stabilize and release.

| Category | Items | Risk |
|---|---|---|
| **Aggressive sensitivity retune** | 6 commits, F7 ground truth validated with param tuner | LOW — tuner-verified |
| **Diagnostic JSON per scene** | Full whisper results saved alongside SRTs | LOW — additive only |
| **Param tuner utility** | `scripts/whisper_param_tuner.py` | LOW — standalone script |
| **Bug fixes** | XXL persistence, config contamination, GUI presets, Colab trust_repo | LOW — targeted fixes |
| **Enhance-for-VAD UI** | Checkbox restored in ensemble panel | LOW — additive UI |

**Recommendation**: Release as v1.8.10 after closing remaining stale issues and writing the Ollama FAQ. No blockers.

---

### v1.8.x — Potential post-release patches

| Item | Trigger | Scope |
|---|---|---|
| #263 follow-up | If GPU not utilized is a real bug | Patch |
| Ollama FAQ doc | Written as docs, not code | Docs only |
| Additional Colab/Kaggle fixes | If more reports come in | Patch |

---

### v1.9.0 — Proposed Scope

**Theme: Platform Expansion + Translation Overhaul + UX**

| Priority | Item | Issues | Effort | Notes |
|---|---|---|---|---|
| **P0** | Ollama full migration (deprecate llama-cpp-python) | #132, #233, #255, #259 | Large | Remove ~1500 LOC fragile code. GUI wiring for Ollama. |
| **P0** | Chinese UI (partial i18n) | #175, #180 | Medium | Biggest support burden reducer. 40%+ of issues are in Chinese. |
| **P1** | Speaker diarization | #248, #252 | Large | New capability. pyannote-audio or similar. |
| **P1** | Standalone merge CLI | #230 | Medium | `whisperjav-merge` command |
| **P1** | XXL in Pass 1 | #242 | Medium | Needs forced alignment since XXL has no timestamps. |
| **P1** | AMD ROCm support | #142, #114, #239 | Medium | Document + test. FishYu-OWO proved it works. |
| **P1** | Full dual-track enhance-for-VAD | — | Medium | ASR module needs separate VAD/ASR audio paths for balanced/fidelity. |
| **P2** | GUI settings persistence | #96 | Medium | Long-standing request. |
| **P2** | MPS selective policy | #227 | Small | Force CPU for whisper, allow MPS for kotoba. |
| **P2** | Vocal separation investigation | #224, #254 | Medium | BS-RoFormer or UVR integration. |
| **P2** | Grey out incompatible options | #206 | Small | Prevent invalid GUI combinations. |
| **P2** | Uninstall cleanup tool | #260 | Small | Script to find and remove cached models. |
| **P3** | Docker support | #247 | Medium | Dockerfile + compose. |
| **P3** | Gemma 3 model configs | #128 | Small | Contributor PR from hyiip. |
| **P3** | whisper-ja-anime model | #232 | Small | If standard HF format. |

**Estimated timeline**: 6-8 weeks for P0+P1 items.

---

### v2.0 — Strategic Vision

**Theme: Architecture + Scale + Ecosystem**

| Area | Item | Why |
|---|---|---|
| **Architecture** | Plugin system for ASR backends | Currently hardcoded pipeline classes. Plugin architecture enables community contributions without core changes. |
| **Architecture** | Separate installer from runtime | Currently entangled. Clean separation enables Docker, cloud, and headless deployments. |
| **Architecture** | Web-based UI (replace pywebview) | PyWebView has persistent issues (#225, #240, WebView2 dependency). A proper web UI (Flask/FastAPI) eliminates platform-specific GUI bugs. |
| **Scale** | Batch processing dashboard | Current batch is sequential. Dashboard with queue, progress, ETA for large libraries. |
| **Scale** | GPU memory management overhaul | Current JIT load/unload is fragile. Proper memory budget system for multi-model workflows. |
| **Ecosystem** | Public API / SDK | Enable third-party integrations. REST API for remote processing. |
| **Ecosystem** | Community model registry | Centralized config sharing for community-tuned models/presets. |
| **Quality** | Ground truth test framework | Formalize the F7 acceptance test pattern. CI/CD with regression detection on quality metrics. |
| **Platform** | Linux native installer | Currently source-only on Linux. AppImage or deb package. |
| **Platform** | Cloud deployment guide | AWS/GCP/Lambda with GPU. Serverless transcription service pattern. |

**v2.0 is a major version bump** — breaking changes acceptable. The key insight from v1.8.x: the biggest technical debts are the PyWebView GUI, the llama-cpp-python translation stack, and the tightly-coupled pipeline architecture.

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Action |
|---------|--------|---------|--------|
| **Local LLM** | #259, #255, #233, #132 | #132 | Ollama FAQ resolves most |
| **Diarization** | #248, #252 | #248 | Merge. v1.9 roadmap. |
| **AMD/Intel GPU** | #239, #142, #114, #213 | #142 | v1.9+. Link all. |
| **i18n** | #175, #180 | #180 | v1.9.0 P0. |
| **Speech enhancement** | #254, #224 | #224 | v1.9. |
| **XXL** | #242, #237, #223, #244 | #242 | v1.9. |
| **Install/Network** | #261, #253, #251, #243, #240, #225 | — | Individual fixes. |

---

## Changelog

| Date | Changes |
|------|---------|
| **2026-03-29** | **rev18.** Refreshed from GitHub. 55 open. Session work: fixed XXL exe persistence, XXL model in extra args, Silero trust_repo for Colab (#253), restored enhance-for-VAD checkbox. Added F3 roadmap for v1.8.10, v1.9.0, v2.0. |
| 2026-03-28 | **rev17.** Closed 11. Responded 7. |
| 2026-03-28 | **rev16.** Full refresh. |
| 2026-03-19 | **rev15.** 3 fixes coded for post2. |

---
