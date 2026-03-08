# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-08 (all items complete) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **19 open** on GitHub

---

## Quick Stats

| Category | Count | Notes |
|----------|------:|-------|
| Total open on GitHub | **19** | Was 40 — closed 22 issues this session |
| Closed this session | 22 | #69, #143, #146, #150, #158, #159, #161, #162, #174, #176, #177, #178, #179, #184, #185, #186, #187, #189, #191, #193, #194, #195 |
| Previously closed (since last tracker) | 5 | #196, #197, #198, #200, #201 |
| New issues (not in prior tracker) | 2 | #203, #204 |
| **Active bugs (need code work)** | 1 | #132 (Local LLM on Colab) |
| **Active bugs (fixed, awaiting confirmation)** | 2 | #204 (SSL fallback), #203 (serial mode reply) |
| Fixed in v1.8.7 (all committed on main) | 8 | #196, #198, #195, #204, #159, setuptools pin, zipenhancer GUI, installer diagnostics |
| Feature requests (open) | 11 | |
| Deferred to v1.9+ | 8 | |

---

## Cluster Analysis

Issues are grouped by root cause / theme rather than by number. This reveals the real problem areas.

### Cluster 1: Network / SSL / Model Download Failures (5 issues)

The most impactful cluster for Chinese users. HuggingFace model downloads fail through proxies.

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#204** | VPN users (v2rayN) SSL failures | yangming2027 | **OPEN** | faster-whisper/HF hub tries online model validation even when cache exists. SSL errors through proxies kill the run. Works offline. | **FIXED** commit `ffde153` — SSL fallback to local cache. Awaiting user confirmation. |
| **#201** | Install SSL cert error on fresh Win10 | jl6564 | CLOSED | Missing root CA certs on fresh Windows. Post-install network check fails. | Closed — env issue |
| **#200** | NVML "Driver Not Loaded" on Optimus laptop | Ywocp | CLOSED | Dual-GPU Optimus: NVML can't load in non-GPU context. Installer falls back to CPU. | Closed — documented |
| **#191** | Pass2 missing — SSL error | yangming2027 | OPEN | Same SSL/proxy issue as #204. Pass2 model download fails. | **Close as dup of #204** |
| **#193** | How to update packages | Faraway-D | OPEN | Support question — urllib3 warning. | **Close — answered** |

**Analysis:** #204 is the real issue here. The HF hub / faster-whisper `download_model()` attempts network validation even when the model is already cached locally. Chinese users behind proxies (v2rayN) hit SSL failures. The user discovered that **disabling network entirely** makes it work — proving the local cache is complete. The fix should be: prefer local cache, only download on cache miss, handle SSL errors gracefully with fallback to cache.

**Impact:** HIGH for Chinese user base (significant portion of users).

---

### Cluster 2: Local LLM Translation (5 issues)

Recurring theme across versions. Token limits, server timeouts, parsing failures.

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| ~~#196~~ | Token limit + "No matches" | destinyawaits | CLOSED | Batch too large for 8K context. Fixed: max_tokens cap, streaming, auto-batch. | **Fixed v1.8.7b0** |
| **#132** | Local LLM on Colab | TinyRick1489 | OPEN | Colab networking + timeout issues. User says "few lines getting translated" — partial success. | **Active — needs investigation** |
| ~~#146~~ | Local server error | leefowan | OPEN | Fixed v1.8.3. Stale 5+ weeks. | **Close — stale** |
| ~~#162~~ | Local model ggml.dll | aaxz886 | OPEN | Fixed v1.8.4. Stale. | **Close — stale** |
| ~~#158~~ | Linux LLM build | parheliamm | OPEN | Fixed hotfix3. | **Close — fixed** |

**Analysis:** #196 was the flagship bug here — now fixed in v1.8.7b0 with three root causes addressed (max_tokens, streaming, CustomClient). #132 on Colab is a separate environment issue (port binding, timeout in notebook). The other 3 are stale — fixes shipped long ago.

---

### Cluster 3: GUI Settings Persistence (4 issues)

Users repeatedly report settings lost on restart.

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#96** | Full settings persistence | sky9639 | OPEN | Translation + ensemble presets done. Full pipeline tab remaining. | **v1.9 — partial** |
| ~~#176~~ | Translation settings lost | Ywocp | OPEN | Fixed hotfix2 (file-based persistence). | **Close — fixed** |
| ~~#174~~ | Settings reset every time | wazzur1 | OPEN | Duplicate of #96. | **Close as dup** |
| ~~#184~~ | Save configurations request | aikotanaka6699 | OPEN | Duplicate of #96. | **Close as dup** |

**Analysis:** Core issue is #96 — partial implementation done (translation, ensemble). Full pipeline settings persistence is a v1.9 effort. The other 3 are duplicates or already fixed.

---

### Cluster 4: Encoding / Unicode Crashes (4 issues)

All stem from non-UTF-8 data in subprocess I/O.

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| ~~#195~~ | UnicodeDecodeError in audio extraction | v2lmmj04 | CLOSED | M4A/M4B with Japanese metadata. Already fixed in commit `55df512`. | **CLOSED — already fixed** |
| ~~#190~~ | GBK codec crash | MatteoHugo | CLOSED | Fixed v1.8.6 — process-wide UTF-8 mode. | Done |
| ~~#186~~ | UnicodeEncodeError subprocess | teijiIshida | OPEN | Fixed hotfix1. | **Close — fixed** |
| ~~#177~~ | cp950 codec translation | stonecfc | OPEN | Fixed hotfix1 + v1.8.6. | **Close — fixed** |

**Analysis:** #195 is the only remaining bug — a simple `errors='replace'` fix in `audio_extraction.py`. The rest are all shipped.

---

### Cluster 5: Ensemble Mode Issues (3 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#203** | Serial mode not working / logic complaint | yangming2027 | **OPEN** | User wants per-file serial processing. Feature exists (`--ensemble-serial` / "Finish each file"). | **REPLIED** — confirmed feature exists in v1.8.6. Awaiting user confirmation. |
| ~~#189~~ | Smart Merge clears one pass | Ywocp | OPEN | Fixed hotfix2. | **Close — fixed** |
| ~~#179~~ | Pass ordering request | yangming2027 | OPEN | Fixed v1.8.6 `--ensemble-serial`. Same user as #203. | **Close — fixed** |

**Analysis:** #203 is from the same user as #179. The feature they requested (`--ensemble-serial`) was built in v1.8.6. But the user doesn't seem to know it exists — they asked "is this already developed?" after meizhong986 pointed them to "Finish each file". This is a **GUI discoverability / documentation** issue, not a code bug. The user needs a clear response confirming it works.

---

### Cluster 6: Platform Support — Apple Silicon / AMD (3 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| ~~#198~~ | No MPS on M1 Mac | francetoastVN | CLOSED | Fixed v1.8.7b0 — MPS detection in TransformersASR. | Done |
| **#142** | AMD Radeon 9600XT not detected | MatthaisUK | OPEN | DirectML/ROCm not supported. | **Defer v1.9+** |
| **#114** | DirectML for AMD/Intel | SingingDalong | OPEN | Major platform enablement. | **Defer v1.9+** |

**Analysis:** MPS is done. AMD/DirectML is a v1.9+ undertaking.

---

### Cluster 7: Translation Provider Requests (4 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| ~~#178~~ | Custom endpoint ignored | Ywocp | OPEN | Fixed hotfix3. | **Close — fixed** |
| **#69** | Grok translation | lingyunlxh | OPEN | Already covered by custom provider. | **Close with note** |
| **#71** | Google Translate (free, no key) | x8086 | OPEN | Fragile unofficial API. | **Defer v1.9+** |
| **#43** | DeepL provider | teijiIshida | OPEN | Non-LLM adapter. | **Defer v1.9+** |

---

### Cluster 8: Feature Requests (Unclustered)

| # | Title | Reporter | State | Summary | Target |
|---|-------|----------|-------|---------|--------|
| **#194** | M4B file support | v2lmmj04 | OPEN | Done v1.8.6 commit `5769688`. | **Close — done** |
| **#181** | Frameless window | QQ804218 | OPEN | Cosmetic. PyWebView `frameless=True`. | v1.9+ |
| **#180** | Multi-language GUI (i18n) | QQ804218 | OPEN | Full i18n framework. High effort. | v1.9+ |
| **#175** | Chinese language GUI | yangming2027 | OPEN | Subset of #180. | v1.9+ |
| **#164** | MPEG-TS + Google Drive | hosmallming | OPEN | MPEG-TS remux + Kaggle Drive. | v1.8.7 |
| **#159** | CLI --vad-threshold | SingingDalong | OPEN | **Fixed** `c092db9`. | **Close — done** |
| **#150** | Xiaomi Mimo API | rr79510 | OPEN | Covered by custom provider. | **Close with note** |
| **#143** | Custom local models + VTT | ymilv | OPEN | VTT done v1.8.6. Custom model path done. | **Close — done** |
| **#126** | Recursive directory | jl6564 | OPEN | Walk subdirectories. | v1.9+ |
| **#99** | 4GB VRAM guidance | hosmallming | OPEN | Log GPU VRAM before model load. | v1.8.7 |
| **#59** | Feature plans for 1.x | meizhong986 | OPEN | Meta-issue (roadmap). | Keep open |
| **#51** | Batch translate wildcard | lingyunlxh | OPEN | Glob/directory in translate CLI. | v1.8.7 |
| **#49** | Output SRT to source folder | meizhong986 | OPEN | `--output-dir source` exists. Docs gap. | v1.8.7 |
| **#44** | GUI drag-drop filename only | lingyunlxh | OPEN | Drag-drop sends filename not full path. | v1.8.7+ |
| **#33** | Linux pyaudio docs | org0ne | OPEN | Documentation gap. | v1.8.7 |

---

### Remaining Unclustered

| # | Title | Reporter | State | Category | Status |
|---|-------|----------|-------|----------|--------|
| **#187** | v1.8.5 can't generate | weifu8435 | OPEN | Stale — fixed in hotfix2/v1.8.6. | **Close — stale** |
| **#185** | v1.8.4 regression | weifu8435 | OPEN | Stale — fixed hotfix2. Same user as #187. | **Close — stale** |
| **#161** | Colab translate error | kokor594-ai | OPEN | Stale since Feb. No response to info request. | **Close — stale** |
| **#132** | Local LLM on Colab | TinyRick1489 | OPEN | Active — user posted new debug log Mar 7. | Needs investigation |

---

## Tally

### By Resolution Status

| Status | Count | Issues |
|--------|------:|--------|
| **Closeable now** (fix shipped, answered, stale, or dup) | 20 | #194, #193, #191, #189, #187, #185, #186, #184, #179, #178, #177, #176, #174, #162, #161, #159, #158, #150, #146, #143 |
| **Active bugs needing code work** | 3 | #204, #195, #132 |
| **Active bugs needing response only** | 1 | #203 (user needs confirmation that serial mode exists) |
| **Feature requests (open, not done)** | 11 | #181, #180, #175, #164, #126, #114, #99, #71, #69, #51, #49, #44, #33 |
| **Meta / roadmap** | 1 | #59 |
| **Deferred to v1.9+** | 8 | #96, #180, #175, #181, #142, #114, #126, #43 |
| **Recently closed** | 5 | #196, #197, #198, #200, #201 |

### By Category

| Category | Open | Closeable | Active | Deferred |
|----------|-----:|----------:|-------:|--------:|
| Bugs | 16 | 12 | 4 | 0 |
| Feature Requests | 15 | 4 | 0 | 8 |
| Enhancements | 4 | 1 | 2 | 1 |
| Questions / Support | 4 | 3 | 0 | 0 |
| Meta | 1 | 0 | 0 | 0 |
| **Total** | **40** | **20** | **6** | **9** |

### By Priority (Active Work Only)

| Priority | # | Issue | Effort | Why |
|----------|---|-------|--------|-----|
| **HIGH** | #204 | HF model download SSL fallback | Medium | Blocks significant Chinese user base |
| **MEDIUM** | #195 | UnicodeDecodeError audio extraction | Tiny (2 lines) | Simple fix, clear root cause |
| **MEDIUM** | #203 | Serial mode discoverability | Tiny (reply) | User just needs confirmation |
| **LOW** | #132 | Local LLM on Colab | Unknown | Needs investigation of new debug log |

---

## Recommended GitHub Actions (Immediate)

### Close These 20 Issues

| # | Close Reason | Comment Template |
|---|-------------|-----------------|
| #194 | Fixed v1.8.6 | "M4B support shipped in v1.8.6. Closing." |
| #193 | Answered | "The urllib3 warning is harmless. Use `whisperjav-upgrade` to update packages. Closing." |
| #191 | Dup of #204 | "This was an SSL/network error, same root cause as #204. Closing as duplicate." |
| #189 | Fixed hotfix2 | "Smart Merge fix shipped in v1.8.5-hotfix2. Closing." |
| #187 | Stale/fixed | "This was fixed in v1.8.6. Please upgrade. Closing." |
| #185 | Stale/fixed | "Fixed in v1.8.5-hotfix2. Same root cause as #173. Closing." |
| #186 | Fixed hotfix1 | "UTF-8 encoding fix shipped in v1.8.5-hotfix1. Closing." |
| #184 | Dup of #96 | "Duplicate of #96 (settings persistence). Closing." |
| #179 | Fixed v1.8.6 | "Ensemble serial mode (`--ensemble-serial` / 'Finish each file') shipped in v1.8.6. Closing." |
| #178 | Fixed hotfix3 | "Custom endpoint fix shipped in v1.8.5-hotfix3. Closing." |
| #177 | Fixed hotfix1+v1.8.6 | "Encoding fix shipped. Closing." |
| #176 | Fixed hotfix2 | "File-based settings persistence shipped in hotfix2. Closing." |
| #174 | Dup of #96 | "Duplicate of #96. Closing." |
| #162 | Stale/fixed | "Fixed in v1.8.4. Closing." |
| #161 | Stale | "No response to info request. Closing — reopen if still an issue." |
| #159 | Fixed v1.8.7 | "`--vad-threshold` and `--speech-pad-ms` shipped. Closing." |
| #158 | Fixed hotfix3 | "Linux LLM build fix shipped. Closing." |
| #150 | Answered | "Custom provider endpoint covers this. Closing." |
| #146 | Stale/fixed | "Fixed in v1.8.3. Closing." |
| #143 | Fixed v1.8.6 | "VTT output and custom model path shipped in v1.8.6. Closing." |

After closing 22 issues, open count is now **19**.

---

## v1.8.7 — Summary of Fixes (all committed on main)

| Commit | Issue | Summary |
|--------|-------|---------|
| `10fbf30` | #198 | MPS device detection for Apple Silicon |
| `55df512` | #195 | `errors='replace'` in audio extraction subprocess |
| `c092db9` | #159 | `--vad-threshold` and `--speech-pad-ms` CLI flags |
| `7407ab6` `e5328c8` `090fd43` | #196 | Local LLM: max_tokens cap, streaming, CustomClient |
| `f81d278` | — | zipenhancer GUI option fix |
| `2abb3b3` | — | setuptools>=61.0,<82 pin (pkg_resources fix) |
| `ffde153` | #204 | SSL/network error fallback to local model cache |

### v1.8.7 Remaining Work (Enhancements — not blockers)

| Priority | # | Description | Effort |
|----------|---|-------------|--------|
| **MEDIUM** | #99 | Log GPU VRAM at INFO before model load | Tiny |
| **MEDIUM** | #164 | MPEG-TS auto-remux | Small |
| **LOW** | #51 | Batch translate wildcard/directory | Medium |
| **LOW** | #49 | Document `--output-dir source` | Tiny (docs) |
| **LOW** | #33 | Linux pyaudio install docs | Tiny (docs) |
| **LOW** | #44 | GUI drag-drop full path | Small |

### Responded / Closed This Session

| # | Action | Done |
|---|--------|------|
| #203 | Replied: "Finish each file" serial mode exists in v1.8.6. | Yes |
| #69 | Closed: custom provider endpoint covers Grok. | Yes |
| #195 | Closed: already fixed in `55df512`. | Yes |
| 21 others | Closed: stale, fixed, dup, or answered. | Yes |

---

## v1.9+ Backlog

| # | Issue | Category |
|---|-------|----------|
| #96 | Full GUI settings persistence (all pipeline tabs) | Enhancement |
| #180/#175 | Multi-language GUI (i18n) — Chinese first | Enhancement |
| #114/#142 | DirectML / ROCm for AMD/Intel GPUs | Platform |
| #126 | Recursive directory + mirror output structure | Feature |
| #181 | Frameless window | Cosmetic |
| #43 | DeepL translation provider | Feature |
| #71 | Google Translate (no API key) | Feature |

---

## Recently Closed Issues

| # | Title | Closed | Resolution |
|---|-------|--------|------------|
| #201 | Install SSL cert error on fresh Win10 | 2026-03-07 | Environment issue — missing root CAs |
| #200 | NVML dual-GPU Optimus detection | 2026-03-07 | Documented — Optimus limitation |
| #198 | MPS not used on M1 Mac | 2026-03-07 | Fixed v1.8.7b0 — MPS detection added |
| #197 | Installation problem v1.8.6 | 2026-03-08 | Closed by user after v1.8.7b0 comment |
| #196 | Local LLM token limit | 2026-03-07 | Fixed v1.8.7b0 — 3 root causes addressed |

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Status |
|---------|--------|---------|--------|
| **Network/SSL/HF download** | #204, #201, #200, #191, #193 | **#204** | Active |
| **Local LLM translation** | ~~#196~~, #132, ~~#162~~, ~~#146~~, ~~#158~~ | ~~#196~~ fixed, #132 active | Mostly resolved |
| **GUI settings** | #96, ~~#176~~, ~~#174~~, ~~#184~~ | **#96** | Partial (v1.9) |
| **Encoding/Unicode** | #195, ~~#190~~, ~~#186~~, ~~#177~~ | **#195** | 1 remaining |
| **Ensemble mode** | #203, ~~#189~~, ~~#179~~ | **#203** | Needs reply only |
| **Apple Silicon / MPS** | ~~#198~~ | Done | Resolved |
| **VAD parameter crashes** | ~~#173~~, ~~#185~~, ~~#187~~ | Done | Resolved |
| **AMD/non-NVIDIA** | #142, #114 | Deferred | v1.9+ |
| **Translation providers** | ~~#178~~, #69, #71, #43 | Deferred | v1.9+ |
| **i18n** | #180, #175 | Deferred | v1.9+ |

---

*This tracker is a point-in-time analysis. For live status, see [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues).*
