# WhisperJAV v1.8.4 Release Notes

**Release Date:** February 2026
**Type:** Stabilization + Hardening
**Base:** v1.8.3

---

## Overview

v1.8.3 introduced the Qwen3-ASR pipeline as a preview. v1.8.4 stabilizes it. This release is almost entirely about making the Qwen pipeline more reliable, fixing timestamp drift, and improving the architecture so that future work doesn't involve patching the same problems in three places.

If you've been using v1.8.3 and ran into timing issues, hallucination loops, or inconsistent results across Qwen input modes -- this release addresses those directly.

The existing Whisper pipelines (`faster`, `fast`, `balanced`, `fidelity`) are unchanged. If you're not using Qwen, this release is still worth taking for the scene detection improvements, the new SRT translation GUI tab, and the bug fixes.

---

## What Changed

### Qwen Pipeline: Decoupled Subtitle Pipeline (ADR-006)

The biggest change isn't user-visible, but it's the foundation everything else sits on. The Qwen assembly mode has been rebuilt around a proper architectural separation:

| Component | What It Does |
|-----------|-------------|
| **Temporal Framers** | Decide *when* to split audio (full-scene, VAD-grouped, SRT-guided, manual) |
| **Text Generators** | Produce transcript text from audio |
| **Text Cleaners** | Sanitize generated text before alignment |
| **Text Aligners** | Produce word-level timestamps |
| **Hardening** | Resolve null timestamps, clamp boundaries, sort chronologically |

Each is a protocol with a factory. You can swap any component without touching the others. The orchestrator ties them together with proper logging, per-scene diagnostics, and error isolation.

**Why this matters to you:** Before this, assembly mode was a monolithic function that mixed generation, alignment, cleaning, and timestamp fixup in a single 400-line method. Bugs in one step were hard to diagnose because they cascaded. Now each step is independent, logged separately, and produces its own diagnostics.

### Tighter Scene and Segment Bounds

Timestamp drift was the top complaint with v1.8.3's Qwen pipeline. The root cause: scenes were too long and segments too wide. The forced aligner accumulates error over time, so smaller chunks = less drift.

| Parameter | v1.8.3 | v1.8.4 | What Changed |
|-----------|--------|--------|-------------|
| Scene min duration | 30s (assembly) / 12s (coupled) | 12s (all modes) | Unified minimum |
| Scene max duration | 120s (assembly) / 90s (coupled) | 48s (all modes) | Halved for assembly, tightened for coupled |
| VAD group max | 29s | 6s | Much tighter grouping |
| Step-down Tier 1 | 30s | 6s | Matches new group size |

The pipeline now owns these defaults as single source of truth. The CLI and ensemble GUI no longer impose their own defaults -- they only forward values if you explicitly set them.

### Unified Hardening Across All Qwen Modes

In v1.8.3, the three Qwen input modes (assembly, context-aware, VAD slicing) each had their own timestamp resolution code -- duplicated static methods that had drifted out of sync. Assembly mode had boundary clamping and chronological sorting. The coupled modes didn't.

This meant context-aware mode could produce timestamps beyond scene duration, and VAD slicing mode could produce out-of-order segments after sentinel recovery.

Now all three modes route through the same `harden_scene_result()` function:
- Timestamp resolution (interpolation, VAD fallback, or aligner-only -- your choice)
- Boundary clamping to `[0, scene_duration]`
- Chronological sort
- Diagnostics tracking

~270 lines of duplicated code removed.

### Sentinel Recovery Fix

The alignment sentinel detects when the forced aligner fails (all words get timestamp 0.0) and redistributes them proportionally. In v1.8.3, sentinel-recovered words in the coupled modes were being reconstructed with `suppress_silence=True`, which let stable-ts adjust the carefully redistributed timestamps. Assembly mode correctly used `suppress_silence=False`.

Fixed: all modes now use `suppress_silence=False` for sentinel-recovered words.

### Scene Detection Refactoring

The scene detection system has been rebuilt around a Protocol + Factory pattern with four backends:

| Backend | How It Works | Default For |
|---------|-------------|-------------|
| **auditok** | Energy-based silence detection. Fast, reliable. | All Whisper pipelines |
| **silero** | Neural VAD model. Better at ignoring music/effects. | Available via `--scene-detection-method silero` |
| **semantic** | Sentence-boundary-aware splitting using ASR. | Qwen pipeline |
| **none** | No splitting. Single scene per file. | When you want to skip scene detection |

All seven pipelines + the ensemble system now use `SceneDetectorFactory`. The old `DynamicSceneDetector` still works (it's a thin wrapper that emits a deprecation warning) but all production code has moved to the new system.

Added `SafeSceneDetector` -- a wrapper that catches backend crashes and falls back to single-scene mode rather than killing the entire pipeline.

### GUI: SRT Translation Tab

A new tab in the GUI for translating existing SRT files using AI providers (DeepSeek, Gemini, Claude, GPT, OpenRouter). Previously this was CLI-only via `whisperjav-translate`.

- Tab-aware file filtering (shows SRT files when the translation tab is active)
- Real-time progress tracking
- All CLI translation arguments available through the GUI

### SRT Output to Source Folder

SRT files now default to being saved next to the source video, instead of the current working directory. This matches what most users expect.

Batch translate (`whisperjav-translate`) now supports glob patterns: `whisperjav-translate -i "movies/*.srt"`.

### Local LLM Server Reliability

Fixes for the local LLM translation server (#149, #148, #157):
- **AVX2 detection:** Checks whether your CPU supports AVX2 before trying to load llama.cpp models. If it doesn't, you get a clear error instead of a crash.
- **Viability gating:** After the server starts, measures steady-state inference speed. Reports whether performance is GOOD (>10 tps), MARGINAL (2-10 tps), or UNUSABLE (<2 tps) with actionable guidance.
- **Server diagnostics:** `start_local_server()` now returns a `ServerDiagnostics` object alongside the API base URL.

### macOS GUI Support

Improvements from community testing (#155):
- WebKit backend detection and configuration for macOS
- Updated install script (`installer/install_mac.sh`)

---

## Bug Fixes

| Issue | What Happened | Fixed |
|-------|---------------|-------|
| #125 | Cleanup crash on pipeline cancellation | Defensive cleanup handler |
| #163 | TEN speech segmenter timing drift | Config correction |
| #162 | ggml.dll not found in subprocess workers | DLL path forwarding |
| #149, #148 | Local LLM server startup failures, unreliable performance | AVX2 detection + viability gating |
| #157 | Crash on CPUs without AVX2 when loading llama.cpp | Pre-check with clear error message |
| F-01 | Orchestrator alignment batching sent wrong audio paths | Fixed batch construction |
| F-02 | Hardening speech regions array was stale after scene slicing | Fresh region extraction |
| Critical | Aligner adapter reading wrong attribute from `ForcedAlignResult` | Fixed attribute name |
| C1 | OOM from stale closure holding scene audio in memory | Closure cleanup |
| C2 | Wrong post-processor selected for Qwen output | Corrected processor routing |

---

## Logging and Diagnostics Improvements

Assembly mode now produces structured per-step logging:
- Step banners with scene index and duration
- Per-scene progress tracking across all orchestrator methods
- Enriched diagnostics (scene duration, input mode, word count, sentinel stats, timing sources, VAD regions)
- Phase 2 scene duration statistics
- Phase 5 assembly summary
- Batch summary logging from the text cleaner

These show up in `whisperjav.log` and in verbose console output. Useful for diagnosing why a particular scene produced bad subtitles.

---

## Breaking Changes

None for end users. If you have custom scripts that:
- Import `VAD_PARAMS` from `pass_worker.py` -- this was removed in a prior refactor
- Expect `--qwen-japanese-postprocess` to default to `True` -- now defaults to `False` (Qwen3 uses its own text cleaner instead)
- Expect `--qwen-max-group-duration` to default to 29 -- now defaults to 6 (pipeline-owned default)

---

## Installation

### Upgrading from v1.8.3

This is a safe upgrade -- same dependency set, no new packages. Use:

```bash
whisperjav-upgrade
```

Or for wheel-only (code changes only, no dependency reinstall):

```bash
whisperjav-upgrade --wheel-only
```

### Fresh Install

See [v1.8.3 release notes](RELEASE_NOTES_v1.8.3.md) for full platform-specific installation instructions. The process is identical for v1.8.4.

---

## Downloads

| File | Description |
|------|-------------|
| [WhisperJAV-1.8.4-Windows-x86_64.exe](https://github.com/meizhong986/WhisperJAV/releases/download/v1.8.4/WhisperJAV-1.8.4-Windows-x86_64.exe) | Windows Standalone Installer |
| [whisperjav-1.8.4-py3-none-any.whl](https://github.com/meizhong986/WhisperJAV/releases/download/v1.8.4/whisperjav-1.8.4-py3-none-any.whl) | Python Wheel (for upgrades) |

---

## Full Changelog

**26 commits since v1.8.3.** [v1.8.3...v1.8.4](https://github.com/meizhong986/WhisperJAV/compare/v1.8.3...v1.8.4)

---

*If you run into issues, please open a [GitHub issue](https://github.com/meizhong986/WhisperJAV/issues) with your platform, GPU, and the error output. That helps us fix things faster than a vague "it doesn't work".*
