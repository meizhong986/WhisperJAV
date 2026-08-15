# CrispASR Ecosystem Synthesis — Phase 0 (Part 1: Real-World Consumer Patterns)

This document is Part 1 of the cross-cutting synthesis for the CrispASR Phase 0 research dossier. Part 1 is descriptive only: it inventories the real-world third-party consumers of CrispASR that primary-source research could locate as of 2026-05-13, documents each one's integration pathway with file:line citations, and surfaces the verbatim findings from SubtitleEdit issue #10775 ("CrispASR aligners"). It makes no comparisons to WhisperJAV and contains no recommendations. Part 2 — cross-ecosystem relationships (CrispASR ↔ CrisperWeaver ↔ Susurrus version pinning, whisper.cpp posture, C-ABI surface alignment) — will be appended below the marker at the end of this file by a follow-on task.

---

## A. SubtitleEdit's CrispASR integration

### A.1 Snapshot

- **Repo URL**: <https://github.com/SubtitleEdit/subtitleedit>
- **Project type**: Cross-platform desktop subtitle editor written in C# / .NET (Avalonia UI). Mature OSS project (`niksedk` is the maintainer; reply on issue #10775 confirms membership).
- **Stated CrispASR capability** (per `change-log.txt`, fetched 2026-05-13):
  - First introduced no later than the v5.0.0-beta17 entry (21 April 2026): "Update Crisp ASR to the latest version + add more models". This wording implies it was already shipping before beta17.
  - **v5.0.0-beta18 (24 Apr 2026)**: "Fix Crisp ASR Qwen3 + Granite" + "Add Crisp ASR Omni" + "Pick Crisp ASR cpu/vulkan/cuda engine for Windows" — multi-variant Windows picker shipped.
  - **v5.0.0-beta20 (29 Apr 2026)**: "Update CrispASR download links".
  - **v5.0.0-beta21 (3 May 2026)**: "Update CrispASR to v0.5.5 + add Granite 4.1 models (base/plus/nar) + add Kyutai STT backend".
  - **v5.0.0-beta22 (5 May 2026)**: "Update CrispASR to v0.5.7".
  - **v5.0.0-beta23 (9 May 2026)**: "Add forced aligner picker for CrispASR" + "Update CrispASR to v0.6.0". (Same beta in which issue #10775 reporter `subof` is asked to retest — see §A.6.)
  - **v5.0.0-beta25 (13 May 2026)**: "Update CrispASR to v0.6.2".
- **Supported CrispASR backends** (from `src/libse/AudioToText/WhisperChoice.cs` and the engine class inventory in `src/ui/Features/Video/SpeechToText/Engines/`):
  - `CrispAsrParakeet` → `BackendName = "parakeet"`
  - `CrispAsrCanary` → `BackendName = "canary"`
  - `CrispAsrCohere` → `BackendName = "cohere"`
  - `CrispAsrQwen3` → `BackendName = "qwen3"`
  - `CrispAsrFireRed` → `BackendName = "firered-asr"`
  - `CrispAsrGlm` → `BackendName = "glm-asr"`
  - `CrispAsrGranite` → `BackendName = "granite-4.1"`
  - `CrispAsrKyutai` → `BackendName = "kyutai-stt"`
  - `CrispAsrOmni` → `BackendName = "omniasr"`
- **Version range pinning** (from `src/ui/Logic/Download/CrispAsrDownloadService.cs`): URLs reference `releases/download/v0.6.6/crispasr-windows-x86_64-*.zip` for Windows, with `.tar.gz` equivalents for macOS and Linux x86_64.
  - Note: at the time of search there was a brief observation of a fork (`Ironship/subtitleedit-plus`) pinning to `v0.6.2`; SubtitleEdit upstream is currently at `v0.6.6` in code, while the changelog text mentions a planned `v0.6.2` bump in beta25 — this is a slight drift between code and changelog text that the maintainer may reconcile.

### A.2 Integration pathway

Integration is **subprocess-based**, not in-process / not FFI.

- **Executable**: per-backend override in each `CrispAsr*` engine class returns `"crispasr.exe"` on Windows (e.g. `CrispAsrCanary.cs`, `CrispAsrParakeet.cs`, `CrispAsrCohere.cs`, etc.); on Linux/macOS the equivalent (no `.exe`) is selected.
- **Install root**: `$"{baseFolder}/CrispASR"` (the `baseFolder` is `Se.SpeechToTextFolder`, i.e. `[Data Folder]/SpeechToText`). Documented in `docs/third-party-components.md`:
  > `| **Crisp ASR** | crispasr.exe, models/ folder | [Data Folder]/SpeechToText/CrispASR |`
- **Process construction**: invoked from `src/ui/Features/Video/SpeechToText/SpeechToTextViewModel.cs`. The same view-model logs the invocation:
  > `$"Calling speech-to-text ({settings.WhisperChoice}) with : {_whisperProcess.StartInfo.FileName} {_whisperProcess.StartInfo.Arguments}{Environment.NewLine}"`
  WebFetch was unable to return the exact lines of the CrispASR branch of `GetWhisperProcess` due to the file being large enough to be truncated mid-method by the WebFetch summarizer. The structure that is visible:
  - A facade `CrispAsrEngine` (`src/ui/Features/Video/SpeechToText/Engines/CrispAsrEngine.cs`) holds `List<CrispAsrEngineBase> _backends` containing instances of all nine `CrispAsr*` engines. `SelectedBackend` tracks the active one and delegates property reads (`BackendName`, `IsEngineInstalled()`, etc.). This is the standard delegated-facade pattern.
  - `CrispAsrEngineBase` is abstract and defines `Name`, `Choice`, `Url`, `BackendName`, `DefaultLanguage`, plus `GetHelpText()` which loads `avares://SubtitleEdit/Assets/SpeechToText/CrispASRCommon.txt` and a per-backend help asset (e.g. `CrispASRCanary.txt`).
- **Per-backend CLI argument defaults** (`src/ui/Logic/Config/SeAudioToText.cs`):
  > ```
  > public string CommandLineParameterCrispAsrCanary { get; set; } = "--max-len 50 --split-on-punct";
  > public string CommandLineParameterCrispAsrCohere { get; set; } = "--max-len 50 --split-on-punct";
  > public string CommandLineParameterCrispAsrKyutai { get; set; } = "--max-len 50 --split-on-punct";
  > public string CrispAsrForcedAligner { get; set; } = "built-in";
  > ```
  Per-backend `CommandLineParameter` is exposed in the **Advanced** window so users can override flags (`SpeechToTextAdvancedWindow.cs` shows a `StandardCrispAsrCommand` button bound visible by `IsCrispAsrVisible`, alongside an `EnableVadCrispAsrCommand` "Enable VAD" button).
- **Binary acquisition**: auto-download via `ICrispAsrDownloadService` / `CrispAsrDownloadService`. The user is prompted with a download dialog. From `SpeechToTextViewModel.cs`:
  > `$"{Environment.NewLine}\"{engine.Name}\" requires downloading the CrispASR engine.{Environment.NewLine}{Environment.NewLine}Select a version to download:"`
- **Per-platform variant selection** (`CrispAsrDownloadService.cs`):
  - Windows defaults to **Vulkan**: `"if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) { return WindowsVulkanUrl; }"`.
  - Four Windows variants exposed as separate methods: `DownloadEngineWindowsCuda`, `DownloadEngineWindowsVulkan`, `DownloadEngineWindowsCpu`, `DownloadEngineWindowsCpuLegacy`. Caller picks which one.
  - macOS / Linux x86_64: single archive (`.tar.gz`).
- **User-facing variant picker** (`DownloadSpeechToTextEngineViewModel.cs`):
  > ```csharp
  > public string CrispAsrWindowsVariant { get; set; } = "vulkan";
  > // …
  > _downloadTask = CrispAsrWindowsVariant switch
  > {
  >     "cuda"       => _crispAsrDownloadService.DownloadEngineWindowsCuda(...),
  >     "cpu"        => _crispAsrDownloadService.DownloadEngineWindowsCpu(...),
  >     "cpu-legacy" => _crispAsrDownloadService.DownloadEngineWindowsCpuLegacy(...),
  >     _            => _crispAsrDownloadService.DownloadEngineWindowsVulkan(...),
  > };
  > ```
  The same view-model handles **stale-binary cleanup** before unpacking — it removes `.exe`, `.dll`, `.so`, `.dylib` files in the target folder so switching variants doesn't leave a contaminated runtime. The unpack step uses platform-aware archive-internal folder names (`crispasr-linux-x86_64`, `crispasr-macos`, `crispasr-windows-x86_64-{variant}`).

### A.3 Capability-flag handling

- **CrispASR side** (issue #10775 maintainer comment, see §A.6 for verbatim): CrispASR exposes backend capabilities as a bitset including `CAP_WORD_TIMESTAMPS` (backend produces native word-level timing) and `CAP_TIMESTAMPS_CTC` (backend can have CTC-aligner output overlaid).
- **CrispASR-side dispatch logic** (verbatim from the maintainer's comment, citing `examples/cli/crispasr_run.cpp`):
  ```cpp
  const bool want_align =
      !params.aligner_model.empty() &&
      ((backend.capabilities() & CAP_TIMESTAMPS_CTC) || params.force_aligner);
  // …
  if (!seg.words.empty() && !params.force_aligner)
      continue;  // skip CTC pass — keep the backend's native words
  ```
- **SubtitleEdit-side handling**: SubtitleEdit does not query `CAP_WORD_TIMESTAMPS` directly; rather, it exposes a UI-level "Forced aligner" combo-box that maps to the CrispASR `--aligner-model` flag, plus (added in beta23) lets the user select a built-in or external GGUF aligner. The aligner path is resolved by `GetForcedAlignerPath(ICrispAsrEngine crispEngine, ForcedAlignerOption aligner)`:
  ```csharp
  private static string GetForcedAlignerPath(
      ICrispAsrEngine crispEngine, ForcedAlignerOption aligner)
  {
      if (aligner.IsBuiltIn || string.IsNullOrEmpty(aligner.FileName)
          || crispEngine is not CrispAsrEngineBase baseEngine)
      {
          return string.Empty;
      }
      return baseEngine.GetModelForCmdLine(aligner.FileName);
  }
  ```
- **`--force-aligner` flag emission**: as of the version available at research time, SubtitleEdit's per-backend default `CommandLineParameter*CrispAsr*` strings (Canary, Cohere, Kyutai — `"--max-len 50 --split-on-punct"`) **do not** include `--force-aligner` (cf. `SeAudioToText.cs`). The maintainer of CrispASR (in issue #10775) suggested SubtitleEdit append `--force-aligner` whenever a non-built-in aligner is selected. Whether SubtitleEdit's `GetWhisperProcess` for CrispASR conditionally appends `--force-aligner` from the new UI picker was not visible to the WebFetch summarizer (file truncated mid-method). Issue #10775 comment thread shows the user reporter could not get `--force-aligner` to work in beta23 — see §A.6.
- **Per-backend UI conditionals**: `SpeechToTextAdvancedWindow.cs` shows `WithBindIsVisible(nameof(vm.IsCrispAsrVisible))` gating the "Standard CrispAsr command" and "Enable VAD" buttons; the "forced aligner picker" combo is bound via `comboCrispAsrBackend = UiUtil.MakeComboBox(vm.CrispAsrBackends, vm, nameof(vm.SelectedCrispAsrBackend))` in `SpeechToTextWindow.cs`.

### A.4 Model management

- **Model registry**: each `CrispAsr*` engine class hard-codes a list of downloadable GGUF models. Parakeet example (`CrispAsrParakeet.cs`):
  - `parakeet-tdt-0.6b-v3-q4_k.gguf` (489 MB), `q5_0.gguf` (541 MB), `q8_0.gguf` (745 MB), full-precision `.gguf` (1.26 GB)
  - Japanese variant `parakeet-tdt-0.6b-ja.gguf` (1.24 GB) from `huggingface.co/cstr/parakeet-tdt-0.6b-ja-GGUF`
  - All from `huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/main/...`
- **Canary models** (`CrispAsrCanary.cs`): four GGUF models 705 MB – 1.97 GB (q4_k, q5_0, q8_0, full) from HuggingFace `cstr/...-GGUF` repos. Supported languages: 25 European languages including BG, HR, CS, DA, NL, EN, ET, FI, FR, DE, EL, HU, IT, LV, LT, MT, PL, PT, RO, SK, SL, ES, SV, RU, UK.
- **Storage**: under `Se.SpeechToTextFolder/CrispASR/models/`. `IsModelInstalled()` validates file existence + a minimum 10 MB size threshold to avoid loading a half-downloaded file.
- **Download UI**: `SpeechToTextViewModel.cs` shows the model-selection prompt and triggers `engine.GetAndCreateWhisperModelFolder(model.Model)` to ensure the destination exists. `GetModelForCmdLine(model.Name)` returns the path string that goes into the `-m` argument.
- **Aligner model**: separately downloadable (e.g. `canary-ctc-aligner-q8_0.gguf` 704 MB, `canary-ctc-aligner.gguf` 1.25 GB) — exposed in the same combo-box as the "Forced aligner" picker, alongside the "built-in" option (which is the value `Se.Settings.Tools.AudioToText.CrispAsrForcedAligner = "built-in"` default).
- **No hash verification on download path**: WebFetch confirmed that `CrispAsrDownloadService.cs` "Hash Verification: Not implemented" — the download stream itself is not validated by hash inside the service. Instead, **post-extraction** validation is done via `DownloadHashManager`.

### A.5 Error handling + UX

- **DownloadHashManager** (`src/ui/Logic/Download/DownloadHashManager.cs`, approx. lines 29-467) is the integrity-check layer:
  - **Archive-level keys** (lines 29-40): `CrispAsr.Windows.Cuda`, `CrispAsr.Windows.Vulkan`, `CrispAsr.Windows.Cpu`, `CrispAsr.Windows.CpuLegacy`, `CrispAsr.MacOs`, `CrispAsr.Linux`.
  - **Executable-level keys** (lines 43-49): `WindowsCudaExecutable`, `WindowsVulkanExecutable`, etc. — used as a fallback when the archive sidecar is missing.
  - **`KnownHashes` dictionary** (lines 56-188): SHA-256 values ordered newest-first (index 0 is the latest known hash).
  - **`ResolveCrispAsrKey(string? windowsVariant)`** (lines 347-365): platform/variant → hash-key.
  - **`DetectCrispAsrWindowsVariant(string installFolder)`** (lines 409-453): inspects backend-specific DLLs in the install folder (e.g. CUDA / Vulkan runtimes) and falls back to hashing `crispasr.exe` against the known sets.
  - **`TryReadInstalledKey(string installFolder)`** (lines 455-467): reads a `.installed.sha256` sidecar file written at install time to identify the variant on subsequent launches.
- **Progress reporting**: The `change-log.txt` entry observed under "Allow speech-to-text ETA to increase - thx Ironship" indicates SubtitleEdit has an ETA-monotonic-growth fix. WebFetch could not return the exact `Parameters: …` ETA-parsing block from `SpeechToTextViewModel.cs` (truncated), but the logging signature is visible:
  > `"Parameters: " + _waveExtractProcess.StartInfo.Arguments + Environment.NewLine + …`
- **Error surface for non-zero exit codes**: standard pattern across all engines; CrispASR error text is visible in the SubtitleEdit log window. (The specific `OutputDataReceived` / `ErrorDataReceived` handlers for the CrispASR branch were not returnable in this research pass — see §E.)
- **"CrispASR update required" dialog** (`TextToSpeechViewModel.cs`): the integrated Chatterbox-TTS feature reuses the CrispASR install. If the installed CrispASR is older than v0.6.0 it shows:
  > `"\"Chatterbox TTS\" needs CrispASR v0.6.0 or newer. Re-download now?"`
  This is a version-gate UX surface that reads the installed binary's hash via `DownloadHashManager` and compares against the registry.

### A.6 Issue #10775 — verbatim findings

**Issue URL**: <https://github.com/SubtitleEdit/subtitleedit/issues/10775>
**Title**: "CrispASR aligners"
**Reporter**: `subof` (NONE — non-member)
**Opened**: 2026-05-04T19:15:37Z
**State at research time**: `open`, 12 comments, last updated 2026-05-10T22:29:53Z. Not closed.

**Opening report verbatim**:
> Hi, Unlike Crisp-ASR with Qwen, Canary doesn't offer to download the aligner. I manually downloaded the model, placed it in the model folder, and specified the path, but I received inaccurate subtitles. It seems that the alignment doesn't work for Canary.
> https://github.com/CrispStrobe/CrispASR#word-level-timestamps-via-ctc-alignment
>
> `--max-len 50 --split-on-punct true --aligner-model "C:\Users\User\AppData\Roaming\Subtitle Edit\SpeechToText\CrispASR\models\canary-ctc-aligner-q8_0.gguf"`

**Comment 1 — CrispASR maintainer (`CrispStrobe`, 2026-05-05T08:59:13Z)** identifies the root cause as a per-segment dispatch bypass in CrispASR's CLI:
> "The aligner *is* being loaded for canary; it's the per-segment dispatch logic that bypasses it."

The maintainer quotes the dispatch loop in `examples/cli/crispasr_run.cpp`:
```cpp
const bool want_align =
    !params.aligner_model.empty() &&
    ((backend.capabilities() & CAP_TIMESTAMPS_CTC) || params.force_aligner);
// ...
if (!seg.words.empty() && !params.force_aligner)
    continue;  // skip CTC pass — keep the backend's native words
```
and explains:
> "Canary advertises both `CAP_TIMESTAMPS_CTC` *and* `CAP_WORD_TIMESTAMPS`, so the aligner GGUF gets loaded (good) but each segment arrives with `seg.words` already populated by canary's native timing — the CTC pass is then skipped per-segment (the footgun). The subtitles you're seeing are canary's native word timing, not the aligner's."

The proposed fix is the `--force-aligner` (short `-falign`) flag. The maintainer also acknowledges a **UX defect** on the CrispASR side and ships a fix in the same comment (commit `93f9050f`):
> "when `--aligner-model` is explicitly set but `--force-aligner` isn't, and the backend has `CAP_WORD_TIMESTAMPS`, CrispASR now prints: `crispasr: warning: --aligner-model is set, but backend 'canary' already produces native word timestamps. The aligner will be loaded but skipped per-segment …`"

The auto-aligner registry path is documented in the same comment:
> "`-am auto` (or `--aligner-model auto`) does work for canary too: the registry lookup pulls `cstr/canary-ctc-aligner-GGUF` (Q4_K, ~442 MB) and caches it under `~/.cache/crispasr/`."

Maintainer's suggestion to SubtitleEdit:
> "the simplest thing is to always append `--force-aligner` to the command CrispASR invocation when the user has selected the CTC aligner option — same shape as a checkbox that turns the aligner from 'fallback only' into 'always on.'"

**Comment 2 — `subof` (2026-05-05T19:02:23Z)**: reports the user-side blocker — at the time of the reply, the installed CrispASR build does not accept `--force-aligner`:
> "error: unknown argument: --force-aligner. I've tried both versions of the command: detailed and concise."

(This is a version-skew problem: the user's installed CrispASR predates v0.6.0 where `--force-aligner` was added; see comment 6 below.)

**Comment 3 — `subof` (2026-05-06T10:40:34Z)** posts an AI-generated comparison framing CrispASR as accurate but timing-deficient relative to WhisperX, which is the recurring theme:
> "Until Subtitle Edit integrates a full-fledged Forced Aligner for NVIDIA models (as it does for WhisperX), WhisperX will remain the leader for subtitles. Canary is currently a powerful 'engine' that lacks a good 'gearbox' for time synchronization in the GUI software."

**Comment 4 — SubtitleEdit maintainer (`niksedk`, 2026-05-06T10:48:43Z)** acknowledges the timing gap on the CrispASR + VAD path and announces the upcoming UI:
> "Purfview Faster Whisper with VAD and CTranslate2 with VAD give very good timestamps (CPP with VAD is not really working). Next beta has choice of forced aligner in the UI as a combo-box."

**Comment 5 — `subof` (2026-05-06T10:52:00Z)** repeats the AI explanation about cross-attention DTW vs. CTC forced alignment for encoder-decoder architectures (Canary).

**Comment 6 — CrispASR maintainer (`CrispStrobe`, 2026-05-06T11:55:40Z)**:
> "Yes, cross-attention DTW is much less accurate than CTC forced alignment — that's measured. That's why we ship canary-ctc-aligner-GGUF as the recommended path; once you upgrade to ≥ v0.6.0, `-am auto --force-aligner` gives you the specialized alignment the AI said you need. … But i see the usage might be counter-intuitive, with a much worse path as default, so next CrispASR release will have it so that `--backend canary` auto-uses canary-ctc-aligner for any subtitle-style output unless `--no-auto-aligner` is passed."

This commits to an **auto-aligner-on-by-default for Canary** in a future CrispASR release.

**Comment 7 — `subof` (2026-05-06T15:47:38Z)** asks for a quality vs. size choice (full-precision vs. q8_0):
> "I hope we will have the opportunity to choose from the options? I'm not willing to sacrifice quality for speed. canary-ctc-aligner.gguf | 1.25 GB | F16, full precision / canary-ctc-aligner-q8_0.gguf | 704 MB | Q8_0, near-lossless"

**Comment 8 — `niksedk` (2026-05-09T07:28:45Z)**:
> "@subof: how is beta 23: https://github.com/SubtitleEdit/subtitleedit/releases"

(Beta 23 is the one with the forced-aligner combo-box added per change-log.)

**Comment 9 — `subof` (2026-05-09T10:28:08Z)**: tested beta 23, reports it still doesn't work for him:
> "I couldn't get it to work. There's a button for VAD, but I think I need another one for the aligner. I tried using the command `--force-aligner -am` auto or `--aligner-model canary-ctc-aligner-q8_0.gguf`, but it didn't work. … After completing the process, it always returns to the built-in model in the menu, as if it's not using the external one. The subtitles are inaccurate and overlap each other. I think we don't have a replacement for WhisperX for precise positioning yet."

This is a **state-persistence defect** in the UI: the combo-box reverts to "built-in" after the run completes.

**Comment 10 — `rikimtasu` (2026-05-09T11:55:00Z)** reports an unrelated layout bug for Parakeet:
> "not canary but when using Parakeet build in aligner the windows wont show everything on screen"
> (with screenshot showing the dialog truncated at 1074×617).

**Comment 11 — `subof` (2026-05-10T22:29:53Z)**, in beta 24 pre-release, reports a CUDA crash:
> "D:\a\CrispASR\CrispASR\ggml\src\ggml-cuda\ggml-cuda.cu:97: CUDA error. I don't know what I'm doing wrong, but I can't seem to use CrispASR in any way now. v5.0.0-beta24 Pre-release"

**Status summary as of 2026-05-13**:
- **Acknowledged defects**:
  1. CrispASR's `--aligner-model` is silently bypassed per-segment when the backend has `CAP_WORD_TIMESTAMPS`. CrispASR side: warning emitted in v0.6.0 commit `93f9050f`; default-on auto-aligner promised for next CrispASR release. SubtitleEdit side: forced-aligner combo-box added in beta23.
  2. SubtitleEdit beta23 forced-aligner combo-box does not appear to persist or actually apply (reporter still observed "built-in" after the run).
  3. Parakeet aligner dialog layout truncation (beta23).
  4. CUDA error path (`ggml-cuda.cu:97`) on the reporter's system in beta24 pre-release — root cause unstated.
- **Still open**: alignment lag vs WhisperX is the recurring user perception; both maintainers acknowledge that cross-attention DTW (used by encoder-decoder models like Canary) is less accurate than CTC forced alignment.

---

## B. Meetily's CrispASR integration

### B.1 Snapshot

- **Repo URL**: <https://github.com/Zackriya-Solutions/meetily>
- **Stated CrispASR usage**: **None found.** Meetily's `README.md`, `CLAUDE.md`, repository top-level tree, and `Cargo.toml` were examined; there are no mentions of "CrispASR", "crispasr", "Crisp ASR", "CrispStrobe", or "crispasr-sys".
- **Actual transcription stack** (per Meetily's `CLAUDE.md` content returned by WebFetch):
  > "Audio Processing: Rust (cpal, whisper-rs, professional audio mixing)"
  > "Transcription: Whisper.cpp (local, GPU-accelerated)"
  > "frontend/src-tauri/src/whisper_engine/whisper_engine.rs - Whisper model management and transcription"
  Repository structure confirms direct use of `whisper.cpp` (vendored as a git submodule at top-level commit `d682e150908e10caa4c15883c633d7902d385237`) plus the `whisper-rs` Rust crate.
- **Language / framework**: Rust + Tauri + Next.js. Top-level workspace `Cargo.toml` declares only `anyhow`, `serde`, `serde_json`, `tokio` — actual transcription deps live in workspace member crates not surfaced in this search.
- **License**: Per the search result, Meetily Community Edition is open source. License file `LICENSE.md` is 1074 bytes — typical MIT — but not opened for verbatim quote in this pass.

### B.2 Integration pathway

**Not applicable** — Meetily does not consume CrispASR. The user-supplied claim that "Meetily uses CrispASR via Rust FFI" is **not corroborated by primary sources** as of 2026-05-13. Meetily uses `whisper-rs` (a different, established Rust binding to `whisper.cpp`) and a vendored `whisper.cpp` submodule.

### B.3 Backend coverage

Not applicable. (Meetily mentions Parakeet and Whisper as the user-visible "live transcription" engines, but the implementation route is whisper.cpp / whisper-rs, not via CrispASR's multi-backend hub.)

### B.4 Memory safety + state teardown

Not applicable for CrispASR-context lifecycle in Meetily. (Whisper context teardown patterns in `whisper-rs` are well-documented in that crate but are not CrispASR's C-ABI and therefore out of scope here.)

---

## C. Other consumers (discovered via search)

This section lists every additional consumer that `gh search code "CrispASR"` and related searches surfaced as of 2026-05-13. Each entry is descriptive; no comparative judgments.

### C.1 parakit (`pszemraj/parakit`)

- **Repo URL**: <https://github.com/pszemraj/parakit>
- **Self-description** (from its `Cargo.toml`): "Push-to-talk desktop dictation daemon backed by NVIDIA Parakeet through CrispASR."
- **Language / framework**: Rust (MSRV 1.87), edition 2021. MIT-licensed.
- **Integration pathway**: **FFI via `crispasr-sys` plus the higher-level safe `crispasr` Rust crate.** Confirmed by:
  - `Cargo.lock` contains `crispasr-sys`.
  - `parakit/Cargo.toml` declares "Inference runtime. CrispASR's Rust crates are supplied by the pinned [vendored submodule]".
  - `parakit/docs/build.md` (verbatim): "parakit is a Rust 1.87+ binary that links to the vendored CrispASR submodule. The default build is CPU-only and local-machine optimized."
  - `parakit/build.rs` (comment quoted in code search): "Emits a `rustc-link-search` so `crispasr-sys`'s `link-lib=crispasr` resolves to the in-tree build". The same `build.rs` carries the comment "own cargo features, not by `crispasr-sys` (which is a pure shim and [does not enable backends])".
- **Source-of-binary**: vendored as a git submodule (`vendor/CrispASR/`) and built **from source via the `cmake` Rust crate** during `cargo build`. The `build.rs` invokes CMake with:
  ```
  cfg.profile("Release")
     .define("BUILD_SHARED_LIBS", "ON")
     .define("GGML_NATIVE", "ON")
     .define("GGML_OPENMP", "ON")
     .define("GGML_CPU_REPACK", "ON")
     .define("WHISPER_BUILD_TESTS", "OFF")
     .define("WHISPER_BUILD_EXAMPLES", "ON")
     .define("GGML_BUILD_TESTS", "OFF")
     .define("GGML_BUILD_EXAMPLES", "OFF")
  ```
- **Feature flags** (parakit `Cargo.toml`):
  - `default = ["bundled", "daemon"]`
  - `bundled` → builds `libcrispasr` via CMake in-tree
  - GPU backends: `cuda`, `metal`, `vulkan` — each **implies `bundled`**. Quoted comment: "GPU backends. Each implies `bundled` because crispasr-sys does NOT" [enable backends on its own].
- **BLAS handling**: `PARAKIT_BLAS` env var with values `auto | off | mkl | openblas | accelerate | generic`; auto-detects on Apple platforms (Accelerate) and Linux (MKL/OpenBLAS) via `pkg-config`.
- **Backend coverage**: Parakeet TDT (per project name). The model is loaded via the safe `Engine::open_with_threads(path, threads)` API exposed by the higher-level `crispasr` crate. From `parakit/src/main.rs` (approx. lines 370-385):
  ```rust
  fn open_engine(path: &Path, threads: usize, verbose: bool) -> Result<Engine> {
      if verbose {
          return Engine::open_with_threads(path, threads);
      }
      with_stderr_suppressed(|| Engine::open_with_threads(path, threads))
  }
  ```
- **Memory safety + state teardown**: Engine context lives exclusively inside a worker thread. Cleanup is by Rust-idiomatic ownership transfer:
  ```rust
  drop(tx);          // close worker channel
  worker.join()...?; // join thread → Drop runs on Engine → C-ABI teardown
  ```
  No explicit `crispasr_*_free` call is exposed in parakit; the safe `crispasr` crate's `Drop` impl is the teardown surface.
- **Platform notes from `build.md`**:
  - **Linux/BSD**: transitive RPATH (`$ORIGIN`) baked into `libwhisper.so` so siblings resolve automatically.
  - **Windows**: no rpath — `build.md` warns users to "copy generated DLLs next to the binary or put the generated `out\bin` directory on `PATH`."
  - **macOS**: requires Xcode command-line tools. Uses `@loader_path` rpath.
  - Warning: "treat GitHub auto-generated source archives as unsupported because they do not include the CrispASR submodule" — i.e. the consumer must clone with submodules.

### C.2 SubtitleEdit-plus (`Ironship/subtitleedit-plus`)

- **Repo URL**: <https://github.com/Ironship/subtitleedit-plus>
- **Description**: Fork of SubtitleEdit. Its `CrispAsrDownloadService.cs` pins to **CrispASR v0.6.2** (vs. upstream's v0.6.6 at research time):
  > `private const string WindowsCudaUrl = "https://github.com/CrispStrobe/CrispASR/releases/download/v0.6.2/crispasr-windows-x86_64-cuda.zip";`
- **Integration pathway**: identical to upstream SubtitleEdit (subprocess + `CrispAsrDownloadService` family); fork is a version-pin and feature delta only.
- **Notable**: the changelog credit "Allow speech-to-text ETA to increase - thx Ironship" suggests this fork is also the source of one of the ETA-monotonicity fixes upstream merged.

### C.3 WyomingCrispAsrServer (`AlexanderMaleckij/HomeAssistantWyomingServices`)

- **Repo URL**: <https://github.com/AlexanderMaleckij/HomeAssistantWyomingServices>
- **Description**: ASR server speaking the **Wyoming protocol** (used by Home Assistant) and forwarding to CrispASR.
- **Language / framework**: C# / .NET. Self-described as **Native AOT-compatible**.
- **Integration pathway**: **subprocess + bidirectional stream**. The class `CrispAsrStreamingSession` (`WyomingCrispAsrServer/Services/CrispAsrStreamingSession.cs`) spawns CrispASR with `--stream` and writes PCM audio to stdin while reading transcript lines from stdout. Verbatim invocation (lines 102-127):
  ```csharp
  var processStartInfo = new ProcessStartInfo
  {
      FileName = crispAsr.FullName,
      Arguments = $"--stream -m \"{modelPath}\" {(language is not null ? $"-l {language}" : "")} -np true --stream-step {stepMs}",
      RedirectStandardInput = true,
      RedirectStandardOutput = true,
      RedirectStandardError = true,
      UseShellExecute = false,
      CreateNoWindow = true,
      StandardOutputEncoding = Encoding.UTF8,
      StandardErrorEncoding = Encoding.UTF8,
  };
  processStartInfo.Environment["PATH"] = $"{crispAsr.DirectoryName};";
  ```
- **PCM streaming** (lines 129-132):
  ```csharp
  public ValueTask AddPcmDataAsync(byte[] data, CancellationToken cancellationToken)
  {
      return _process.StandardInput.BaseStream.WriteAsync(data, cancellationToken);
  }
  ```
- **Lifecycle** (lines 134-150): explicit close of stdin → `WaitForExitAsync` → check `ExitCode` → flush ANSI-escape reader → return text. Non-zero exit throws `InvalidOperationException` with stderr buffer attached.
- **ANSI/terminal-escape handling**: a dedicated `AnsiLineReader` (lines 11-81) strips `ESC[K`, `ESC[2K`, and `\r` overwrite sequences because CrispASR's interactive output uses them for progress redraws. The reader commits a line only on `\n`.
- **Default step interval** (line 90): `stepMs = 3_600_000` (one hour), effectively "no chunking" — Home Assistant pushes whole utterances at once.
- **Language validation**: maximum 3-character language code accepted (`if (language is not null && language.Length > 3)` → `ArgumentException`).
- **Configuration** (`appsettings.json`): Host/Port (default 0.0.0.0:10300), `CrispAsrPath`, `FallbackLanguage`, and a `Models[]` array each with `name`, attribution, languages list, model file path, auto-detection support. The README's sample model is NVIDIA `parakeet-tdt-0.6b-v3` Q4_K.

### C.4 whatdeysay (`darkspadez/whatdeysay`)

- **Repo URL**: <https://github.com/darkspadez/whatdeysay>
- **Self-description** (from its `CLAUDE.md`): "Whatdeysay is a self-hosted subtitle manager. Bazarr-style provider search with **CrispASR** (a C++ ASR binary) as AI fallback, plus TS-native sync. Bun + Hono + tRPC v11 + Drizzle + SQLite + BullMQ on Dragonfly. React 19 + shadcn/ui + TailGrids on the frontend."
- **Language / framework**: TypeScript on Bun + Hono + tRPC. Frontend React 19.
- **Integration pathway**: **subprocess** via a project-specific `spawnLogged` wrapper that adds streaming logs and cancellation. Verbatim from CLAUDE.md §4.2:
  ```ts
  const result = await spawnLogged({
    command: "crispasr",
    args: ["-m", modelPath, "-f", wavPath, "-osrt", "-of", outBase, "--vad"],
    signal: job.cancelSignal,
    onStdout: (line) => job.log("info", line),
    onStderr: (line) => job.log("warn", line),
    timeoutMs: 30 * 60 * 1000,
  })
  if (result.exitCode !== 0) throw new ProcError("crispasr failed", result)
  ```
- **Argument shape**: `-m model -f input.wav -osrt -of <outBase> --vad`. Output is SRT directly (CrispASR has built-in SRT writer).
- **Timeout**: 30 minutes hard-cap per call. `signal: job.cancelSignal` wires the job's `AbortSignal` through to subprocess cancellation.
- **Operational guidance** (CLAUDE.md §7): "Don't run CrispASR on the full file when probing. For language detection or sync probing, extract a 60-second sample, not the whole audio track." This implies the project uses CrispASR for cheap language-detection passes as well as full-transcript passes — a usage pattern documented but not implemented at the layer surfaced here.

### C.5 WhisperInc (`praxeo/whisperinc`)

- **Repo URL**: <https://github.com/praxeo/whisperinc>
- **Files of interest**:
  - `CrispAsrServerTranscriber.cs` — adapter that invokes CrispASR's `--server` HTTP mode.
  - `scripts/build-crispasr.ps1` — PowerShell build script that compiles CrispASR from source on Windows with explicit DLL deployment.
- **Integration pathway**: **HTTP / OpenAI-compatible API** against a locally-spawned CrispASR server. The class spawns `crispasr --server --host 127.0.0.1 --port <port> -m <model> -t <threads> -np ...` (with optional `--backend <name>` and GPU control via `-ng` or `--gpu-backend {cuda|vulkan|metal|auto}`).
- **Health-check protocol**: polls `GET /health` every 200 ms with a 500 ms per-call timeout, up to a 45 s total budget. `SemaphoreSlim` ensures single startup attempt across concurrent callers.
- **Transcription request**: POSTs multipart form data to `/v1/audio/transcriptions` with:
  - `file` part (audio/wav)
  - optional `language`
  - optional `prompt`
  - hard-coded `response_format = "json"`
  Static `HttpClient` with 120-second per-request timeout.
- **Response parsing**: JSON deserialization → extract `"text"` property → trim whitespace.
- **Lifecycle**: lazy init on first `TranscribeAsync`. `KillServer()` terminates the process tree. `Dispose()` calls `KillServer()`. Exception path resets `_serverReady` to false.

### C.6 Kurnevsky's Nix module (`kurnevsky/nixfiles`)

- **Repo URL**: <https://github.com/kurnevsky/nixfiles>
- **File**: `modules/crispasr.nix`
- **Purpose**: NixOS / Nix package definition that builds CrispASR from source via CMake. Lets Nix consumers install CrispASR declaratively.
- **Version pin**: `rev = "2dc5f28b0c9ceb986f68465bbe4225c50384c110"` (not a release tag — a raw commit), `hash = "sha256-GeS/ULpysdY+C63l+8bFBGc9T2YejhdXzx+cOUjq38Q="`. Package `version = "0"` (placeholder).
- **Build inputs** (conditional on `vulkanSupport`): `shaderc`, `vulkan-headers`, `vulkan-loader`.
- **Native build inputs**: `cmake`, `git`.
- **CMake flags**: `CRISPASR_BUILD_TESTS=false`, `GGML_VULKAN={vulkanSupport}`.
- **Metadata**: `meta.mainProgram = "crispasr"` → `nix run` works.
- **Integration pathway**: this is a packaging consumer, not a runtime consumer — it builds the binary; downstream tools invoke it however they like.

### C.7 fairphone_voxtral_offline build helper (`ananta888/ananta`)

- **Repo URL**: <https://github.com/ananta888/ananta>
- **File**: `tools/fairphone_voxtral_offline/build-crispasr.sh`
- **Purpose**: shell script that clones CrispASR (`git clone https://github.com/CrispStrobe/CrispASR "$CRISPASR_DIR"`) into `${CRISPASR_DIR:-$HOME/src/CrispASR}` and builds the binary for an off-device Fairphone workflow targeting the Voxtral backend.
- **Integration pathway**: build tooling only. The same repo's `crash.txt` mentions "pressure during CrispASR/Voxtral model execution" and "Runner selection now prefers names containing 'voxtral' or starting with 'crispasr'", indicating it is also a runtime subprocess consumer at the runner-selection layer.

### C.8 hbd-qwen3-tts.cpp (`dbrain/hbd-qwen3-tts.cpp`)

- **Repo URL**: <https://github.com/dbrain/hbd-qwen3-tts.cpp>
- **File**: `src/qwen3_fa/core/attention.h`
- **Reference type**: **shared-environment-variable** consumer. The code reads `CRISPASR_KV_QUANT` as the KV-cache quantization dtype, with a CrispASR-style env-var contract:
  > `return kv_dtype_parse(std::getenv("CRISPASR_KV_QUANT"), backend_tag, "CRISPASR_KV_QUANT", GGML_TYPE_F16);`
  Plus a comment: `// PLAN #60e: KV cache dtype selection from CRISPASR_KV_QUANT.`
- **Implication**: not a CrispASR runtime caller, but evidence of CrispASR's env-var contract being adopted by a sibling project (likely shared author/team).

### C.9 KoeNote (`TommyKammy/KoeNote`) — phase-0 runbook reference

- **Repo URL**: <https://github.com/TommyKammy/KoeNote>
- **File**: `docs/archive/phases/phase0/RUNBOOK.md`
- **Quotation**: lists `tools/crispasr.exe` as a planned dependency and notes "Pending final `crispasr.exe` command-line confirmation."
- **Integration pathway**: planned / in-progress; no live invocation code visible in the search results.

### C.10 CrispEmbed (`CrispStrobe/CrispEmbed`)

- **Repo URL**: <https://github.com/CrispStrobe/CrispEmbed>
- **Relationship**: sibling project by the same author. CrispEmbed's `CMakeLists.txt` declares: "crisp_audio — shared audio-encoder library, lives in CrispASR. The path is the team uses on dev machines (~/code/CrispEmbed alongside ~/code/CrispASR)." Its `PLAN.md` states: "Same philosophy as CrispASR: pure C/C++, GGUF models, quantisation, … Copy ggml as submodule (same version as CrispASR)."
- **Integration pathway**: source-level dependency (shared `crisp_audio` library); not a runtime consumer of `crispasr.exe`. Listed here only because it appears in code search and shows the same author's design conventions.

### C.11 Forks / monitor / archive surfaces (informational only)

The remaining search hits are operational rather than functional integrations:
- `gmh5225/CrispASR` — a fork (recorded in `CrackerCat/feed:archive/2026-05-03.md`).
- `arun-gupta/repo-pulse`, `dukanov/research-monitor`, `w00tzenheimer/feed` — automated repo-watchers / trend trackers.
- `WiiPlayer2/nixos-common` — a NixOS home-manager snippet referencing CrispASR by comment ("# crispasr") in `whisper-stream.nix`.
- `latin-ocr/...` — Latin-OCR text files where the literal string `crispasr` appears in old botanical Latin text and is unrelated (a homophone of "crispa" + a noise byte).

---

## D. Common patterns across consumers

This section observes patterns that emerge across the consumers documented above. Pattern frequency is reported as observed; no normative weight is implied.

### D.1 Invocation modality

| Consumer | Modality | Note |
|---|---|---|
| SubtitleEdit | subprocess (CLI) | one process per file; user-facing |
| SubtitleEdit-plus | subprocess (CLI) | fork of above |
| parakit | FFI via `crispasr-sys` + safe `crispasr` crate | in-process; daemon retains engine |
| WyomingCrispAsrServer | subprocess (CLI, `--stream`) | bidirectional stdin/stdout PCM streaming |
| whatdeysay | subprocess (CLI, batch) | per-file SRT generation |
| WhisperInc | subprocess (CLI, `--server`) → HTTP | OpenAI-compatible API; lazy server |
| Nix module | source build only | packaging |
| fairphone_voxtral_offline | source build + subprocess | hybrid |
| hbd-qwen3-tts.cpp | env-var contract only | not a caller |
| KoeNote | planned subprocess | not yet implemented |

**Observation**: subprocess is the dominant integration pathway (5 / 6 runtime consumers). The single in-process FFI consumer (parakit) is a Rust daemon that benefits from holding engine state warm across many push-to-talk activations.

### D.2 Server-vs-batch invocation

Three distinct subprocess-invocation modes appear:
1. **Per-file batch** — `crispasr -m <model> -f <wav> -osrt -of <out>`. Used by SubtitleEdit and whatdeysay.
2. **Streaming stdin** — `crispasr --stream -m <model> -np true --stream-step <ms>`. Used by WyomingCrispAsrServer; audio flows on stdin, transcript flows on stdout.
3. **HTTP server** — `crispasr --server --host 127.0.0.1 --port <p> -m <model> [--gpu-backend …]`. Used by WhisperInc; clients POST to `/v1/audio/transcriptions`. Exposes `/health` endpoint.

### D.3 Capability-introspection patterns

- **No consumer was observed introspecting `CAP_WORD_TIMESTAMPS` or `CAP_TIMESTAMPS_CTC` programmatically.** CrispASR's capability bitset is internal to its own CLI dispatch (`examples/cli/crispasr_run.cpp`, per issue #10775).
- **SubtitleEdit** abstracts the capability differently: it exposes a `HasNativeTimestamps` boolean per engine class (`CrispAsrCanary.HasNativeTimestamps = true`, `CrispAsrParakeet.HasNativeTimestamps = true`). This is a SubtitleEdit-side mirror of the CrispASR capability, not a query against the binary.
- **The forced-aligner UX gap is therefore foreseeable**: consumers that mirror capabilities statically (rather than querying) must keep their mirror in sync with CrispASR. SubtitleEdit's `beta23` forced-aligner combo-box is the response to issue #10775; the user comment of 2026-05-09 reports the response did not fully land.

### D.4 Model-management patterns

| Consumer | Model storage | Download mechanism |
|---|---|---|
| SubtitleEdit | `[Data Folder]/SpeechToText/CrispASR/models/` | in-app download from `huggingface.co/cstr/*-GGUF`; size-floor validation (10 MB) |
| parakit | user-managed (passed as `path` arg) | none — user supplies the GGUF |
| WyomingCrispAsrServer | configured via `appsettings.json`'s `Models[]` array | none — config-driven |
| whatdeysay | configured via app state | none visible in this pass |
| WhisperInc | passed as `-m <path>` | none |

**Observation**: SubtitleEdit is the only consumer that ships a full model registry + downloader. All others delegate model acquisition to the user or to a sister tool.

### D.5 Binary acquisition + versioning patterns

| Consumer | Acquisition | Version pin |
|---|---|---|
| SubtitleEdit | downloads pre-built ZIPs from CrispASR GitHub releases | hard-coded URL pin (`v0.6.6` at research time) + SHA-256 registry |
| SubtitleEdit-plus | same | hard-coded `v0.6.2` |
| parakit | builds from vendored git submodule | submodule commit SHA |
| Nix module | builds from CrispASR GitHub at a pinned rev | commit SHA + content hash |
| WyomingCrispAsrServer | user supplies path (`CrispAsrPath`) | none enforced |
| whatdeysay | user supplies binary on PATH | none enforced |
| WhisperInc | builds via `scripts/build-crispasr.ps1` | not surfaced in this pass |
| fairphone_voxtral_offline | clones + builds | not surfaced |

**Observation**: two distinct version-pin strategies — hash-pinned (Nix, SubtitleEdit) versus submodule-pinned (parakit). Self-supplied paths (Wyoming, whatdeysay) skip pinning entirely and rely on the user's environment.

### D.6 Error-handling patterns

- **Exit-code check** is universal: subprocess consumers all check non-zero exit and surface stderr (SubtitleEdit, WyomingCrispAsrServer line 142-145, whatdeysay `ProcError`).
- **stderr buffer accumulation**: WyomingCrispAsrServer captures all stderr into a `StringBuilder` and includes it in the exception message — pattern visible in the verbatim quote in §C.3.
- **ANSI / progress-redraw stripping**: only WyomingCrispAsrServer explicitly parses CrispASR's progress output (which uses `ESC[K`, `\r`); other consumers either ignore or log raw lines.
- **Health-check polling**: only WhisperInc (server mode) implements a readiness probe before sending requests.
- **Timeout**: explicit per-call timeouts vary widely — whatdeysay caps each transcription at 30 min, WhisperInc's HttpClient at 120 s, WyomingCrispAsrServer relies on the upstream Wyoming protocol to cancel.

### D.7 Lifecycle / teardown patterns

- **One-shot CLI consumers** (SubtitleEdit, whatdeysay): no teardown — the binary exits.
- **Streaming-stdin consumer** (WyomingCrispAsrServer): explicit `StandardInput.BaseStream.Close()` → `WaitForExitAsync` → `_process.Dispose()`; event handlers detached before disposal.
- **HTTP-server consumer** (WhisperInc): `KillServer()` terminates the process tree; lazy re-init on next call.
- **FFI consumer** (parakit): Rust's `Drop` impl on the safe `Engine` type runs the C-ABI teardown when the worker thread joins. No explicit `crispasr_*_free` in parakit's own code — it relies entirely on the `crispasr` crate's safe Drop.

### D.8 Version skew + UX defects observed

- The user-side blocker in issue #10775 ("error: unknown argument: --force-aligner") is a direct **consumer / runtime version-skew** failure: the consumer (SubtitleEdit) tried to emit a flag that the installed CrispASR (an older one) did not understand. Resolution requires either (a) consumer-side capability gating before emitting the flag, or (b) re-download of the upgraded binary.
- The `DownloadHashManager.DetectCrispAsrWindowsVariant` logic in SubtitleEdit (lines 409-453) is an existing infrastructural answer to part of this problem: it can identify which CrispASR build is installed by inspecting DLLs and hashing the executable, but it currently maps to {Cuda, Vulkan, Cpu, CpuLegacy} variants, not to CrispASR-version numbers.

---

## E. Could not verify

The following claims, hypotheses, or details could not be confirmed against primary sources in this research pass. They are flagged here so a later pass can target them.

1. **Meetily uses CrispASR via Rust FFI**. The user-supplied input asserted this. Primary-source verification fails: Meetily's CLAUDE.md and README do not mention CrispASR/crispasr/CrispStrobe anywhere; the project uses `whisper-rs` + a vendored `whisper.cpp` submodule. **The input claim is not corroborated.** It is possible this was a planning-stage intent or a misattribution; if Meetily has internal CrispASR work in a non-public branch, that is out of reach for primary-source research.
2. **The exact CrispASR-branch `ProcessStartInfo` construction inside `SpeechToTextViewModel.cs`** could not be returned by WebFetch — the file is large enough that the summarizer truncated mid-method and never reached the CrispASR construction block. Consequently:
   - The exact line numbers of CrispASR argument assembly in SubtitleEdit.
   - Whether `--force-aligner` is appended conditionally from the new combo-box value as of beta23/beta25.
   - The full ETA-parsing regex / state-machine that the "ETA monotonic" change touched.
   These would require either (a) a smaller-scope fetch of a specific line-range, (b) a local clone, or (c) reading the file via the GitHub raw URL with explicit byte ranges, none of which were available in this pass.
3. **CrispASR's verbatim `docs/cli.md` section on `--force-aligner` / `-falign` / `--no-auto-aligner`**. The maintainer's comment on issue #10775 references "`docs/cli.md` under '`--force-aligner` / `-falign` — override native timestamps (issue #62)'", but the docs/cli.md content itself was not fetched in this pass.
4. **Hash values inside `DownloadHashManager.KnownHashes`**. WebFetch summarized this dictionary at line range 56-188 but did not return verbatim hashes. The fact that the dictionary is "ordered newest-first (index 0 = latest)" came from the summarizer's description — a verbatim sample of one or two `KnownHashes` entries would confirm format.
5. **Whether SubtitleEdit's combo-box state-persistence defect** (the reporter's beta23 complaint that the picker "always returns to the built-in model in the menu") is a UI binding bug or a settings-persistence bug. Reading `SpeechToTextWindow.cs` more carefully would resolve this.
6. **Whether `Ironship/subtitleedit-plus` has merged any of its CrispASR-related deltas upstream**, beyond the changelog credit "Allow speech-to-text ETA to increase - thx Ironship".
7. **CrisperWeaver's CrispASR integration (Dart-FFI / Flutter)** — out of scope for Part 1; covered in dossier `02_crisperweaver_dossier.md`. Listed here only so readers don't expect it.
8. **Susurrus's CrispASR integration (Python multi-backend)** — out of scope for Part 1; covered in dossier `03_susurrus_dossier.md`.
9. **Reddit / HackerNews / blog-post mentions of CrispASR**. WebSearch returned no relevant results — CrispASR appears to have no significant discussion presence on those forums as of 2026-05-13. (Search hits for "Cosmic Crisps" and "Crisp chat" were unrelated noise.)

---

<!-- CROSS-ECOSYSTEM SECTIONS APPENDED BELOW (Part 2) -->

---

# Part 2: Cross-Ecosystem Relationships

Cross-cutting synthesis across the three CrispStrobe-owned repos (CrispASR, CrisperWeaver, Susurrus) plus their relationship to upstream `ggerganov/whisper.cpp` and to the third-party consumer set documented in Part 1. Descriptive only. All facts sourced from the per-repo dossiers (`01_crispasr_dossier.md`, `02_crisperweaver_dossier.md`, `03_susurrus_dossier.md`) and the consumer-patterns sections above.

---

## F. Repo relationships and version pinning

### F.1 Shared ownership

All three ecosystem repos are owned by the same GitHub identity `CrispStrobe` (user id `154636388`, per `03_susurrus_dossier.md` §0). The README of each repo explicitly references the others in a "Crisp ecosystem" table. CrispASR's README references `CrispEmbed` as a fourth sibling (out of scope for this dossier).

### F.2 Repo metadata snapshot

| Repo | Created | HEAD commit (research time) | License | Stars | Forks | Open issues | Releases |
|------|---------|-----------------------------|---------|-------|-------|-------------|----------|
| CrispASR | 2026-03-29 | `bac5f8f` (2026-05-13) | MIT | 176 | 19 | 9 | 40 (v0.1.0 → v0.6.6) |
| CrisperWeaver | 2025-09-11 | `9b93f86` (2026-05-12) | AGPL-3.0 | 12 | 3 | 0 (closed: 1) | 10+ visible (v0.1.6 → v0.5.0) |
| Susurrus | (not captured in `gh api`) | `7073a77` (2026-04-19) | MIT | 16 | 2 | 0 | 0 (no tags, no releases) |

Sources: `01_crispasr_dossier.md` §0; `02_crisperweaver_dossier.md` §0; `03_susurrus_dossier.md` §0.

### F.3 Version-pinning between repos

| Consumer | Producer | Pinning mechanism | Strength |
|----------|----------|-------------------|----------|
| CrisperWeaver | CrispASR (Dart FFI package) | `pubspec.yaml` local `path:` dependency `../CrispASR/flutter/crispasr` | **None.** No version constraint; resolves against whatever is checked out in the sibling directory. |
| CrisperWeaver CI | CrispASR | Env vars `CRISPASR_REPO` and `CRISPASR_REF` (visible in `ci.yml` and `release.yml`) | Optional — defaults unpinned. |
| Susurrus | CrispASR (binary download) | URL pattern `https://github.com/CrispStrobe/CrispASR/releases/latest/download/{asset}` | **None.** Always pulls `latest`, no checksum, no signature, no per-version manifest. (`03_susurrus_dossier.md` §11) |
| SubtitleEdit | CrispASR (binary download) | `CrispAsrDownloadService.cs` hardcodes release tag `v0.6.6` in URL, with SHA-256 via `DownloadHashManager` | **Strong.** Tag-pinned and hash-verified. (`04_ecosystem_synthesis.md` §A.1) |
| parakit | CrispASR | Git submodule + `crispasr-sys` Cargo crate | **Commit-level.** Submodule pin = sha. (`04_ecosystem_synthesis.md` consumer section) |

Conclusion: the sibling ecosystem repos themselves do NOT version-pin CrispASR; pinning is only seen in external production consumers (SubtitleEdit hash+tag, parakit submodule sha). For an integrator following the SubtitleEdit pattern, the model "pin to a known CrispASR release tag and verify SHA-256" is concretely documented.

### F.4 Release-cadence comparison

| Repo | Commits in last 4 weeks (research time) | Release cadence | Activity tier |
|------|------------------------------------------|-----------------|---------------|
| CrispASR | 1330 | 40 tagged releases in ~6.5 weeks (~6.5/week) | Hyperactive |
| CrisperWeaver | 137 attributed commits total; release pattern ~10 releases in 24 days | High | Highly active |
| Susurrus | 7 total commits, latest 2026-04-19 (no activity for ~24 days at research time) | None (zero tags, zero releases) | Effectively dormant |

The CrispASR/CrisperWeaver cadence vs the Susurrus stall is sharp: Susurrus appears to have been a one-shot demonstration of the CrispASR backend integration (4 commits authored 2026-04-19 by a placeholder identity, see §J below) rather than an ongoing project. CrispASR/CrisperWeaver are the two repos under active development.

### F.5 Sibling-engine consumption pattern

Both production sibling consumers (CrisperWeaver via Dart FFI, Susurrus via subprocess) hold CrispASR at arm's length:
- CrisperWeaver does NOT touch `DynamicLibrary` directly; it delegates to the upstream `package:crispasr` Dart package (in `CrispASR/flutter/crispasr`). (`02_crisperweaver_dossier.md` §5)
- Susurrus does NOT use Python bindings at all; it calls the `crispasr` binary as a subprocess and parses stdout. (`03_susurrus_dossier.md` §12)

In other words: even the maintainer's own Python consumer chose subprocess over the in-process Python bindings. The Python bindings (`python/crispasr/_binding.py` 83.3 KB pure-ctypes per `01_crispasr_dossier.md` §15) exist but are not exercised by the sibling Python project.

---

## G. License posture across the three repos

### G.1 The three license states

- **CrispASR — MIT**, with copyright line preserved from upstream: *"Copyright (c) 2023-2026 The ggml authors"* (`01_crispasr_dossier.md` §2). Permits both subprocess and static-linking use, with the standard attribution requirement.
- **CrisperWeaver — AGPL-3.0-or-later** (full GNU AGPL v3 text in `LICENSE`, 34,523 bytes; `02_crisperweaver_dossier.md` §0).
- **Susurrus — MIT**, copyright "2026 CrispStrobe" (`03_susurrus_dossier.md` §0).

### G.2 The AGPL boundary observation

CrisperWeaver's AGPL-3.0 license applies to CrisperWeaver's own source code. It does NOT propagate to CrispASR (separate repo, MIT-licensed). A Python or other host that consumes CrispASR's MIT-licensed runtime (whether via subprocess, Python ctypes bindings, Rust `crispasr-sys`, or direct C-ABI linkage) is bound by CrispASR's MIT terms — not by CrisperWeaver's AGPL.

The reference pattern in CrisperWeaver's Dart FFI bridge could still be studied as a design reference under AGPL viewing without copying code into a non-AGPL project. Direct code transcription from CrisperWeaver into a non-AGPL host would require either an AGPL host or independent reimplementation.

### G.3 Per-model GGUF license dimension

Distinct from the engine license, individual GGUF model weights carry their own per-model licenses tracked in `01_crispasr_dossier.md` §18 (e.g. whisper MIT, kyutai-stt MIT; other backends not enumerated in this Phase 0 pass). The CrispASR README itself documents *"Per-model weights covered by respective HuggingFace licenses"*. Any bundling/redistribution decision needs to be evaluated per-model, not at the runtime layer.

---

## H. Relationship to upstream whisper.cpp

### H.1 Fork posture (verified)

CrispASR is a **logical fork** of `ggerganov/whisper.cpp`, not a GitHub-API fork (per `01_crispasr_dossier.md` §0: GitHub metadata `fork: false`, but the repo description and README both state *"Fork of whisper.cpp"*). The `LICENSE` file preserves the original "The ggml authors" copyright line.

The `AUTHORS` file (21.2 KB, auto-generated 2025-02-04 per its own header) carries the full pre-fork whisper.cpp / ggml / llama.cpp contributor roster alphabetically — confirming the fork heritage at the code-history level.

### H.2 Whisper kept as a regression gate

Per `01_crispasr_dossier.md` §1: *"Whisper is intentionally not migrated"* — whisper-specific code remains in the ggml subtree rather than in CrispASR's `src/`. The README and dossier flag this as a deliberate regression-gate: the whisper backend stays byte-identical to upstream so any divergence in shared primitives is detectable against the whisper.cpp baseline.

### H.3 ggml as a vendored subtree

ggml lives in `CrispASR/ggml/` as a git subtree with 5 fork-local patches enumerated in `UPSTREAM.md` (9.1 KB). One of those five patches was merged upstream as `ggml/pull/1477`. The other four remain fork-local.

### H.4 Implication for whisper.cpp consumers

A project (such as WhisperJAV via `faster-whisper` or `openai-whisper`) that already consumes whisper-family models indirectly is consuming whisper.cpp's heritage at one degree of separation. CrispASR would represent a second consumption path of the same lineage via a different runtime (ggml C++ vs CTranslate2 / PyTorch). The two paths are not exclusive and not (license-wise) coupled.

---

## I. Integration-pathway prevalence across all consumers found

Aggregating the consumers documented in Part 1 above (SubtitleEdit, Meetily (not corroborated), parakit, WyomingCrispAsrServer, whatdeysay, WhisperInc, plus packaging consumers) and adding the two sibling ecosystem repos:

| Consumer | Language / Runtime | Integration pathway | Streaming? |
|----------|--------------------|---------------------|------------|
| SubtitleEdit | C# / .NET (Avalonia) | Subprocess (stdout parse) | No (batch) |
| CrisperWeaver | Dart / Flutter | In-process FFI via Dart C-ABI | Yes (microphone streaming via `NativeCallable.listener`) |
| Susurrus | Python | Subprocess (stdout regex parse) | No (buffers all output, yields after wait) |
| parakit | Rust | In-process FFI via `crispasr-sys` + safe `crispasr` crate | Worker-thread |
| WyomingCrispAsrServer | C# / AOT | Subprocess (stdin/stdout PCM streaming) | Yes |
| whatdeysay | TypeScript / Bun | Subprocess (batch, `-osrt` flag) | No |
| WhisperInc | C# | HTTP server mode (`/v1/audio/transcriptions`) | No |
| SubtitleEdit forks (e.g., Ironship/subtitleedit-plus) | C# | Subprocess (inherits from SubtitleEdit) | No |

### I.1 Pathway breakdown

- **Subprocess (stdout parse)**: 5 consumers (SubtitleEdit, Susurrus, WyomingCrispAsrServer, whatdeysay, forks)
- **In-process FFI**: 2 consumers (CrisperWeaver via Dart, parakit via Rust)
- **HTTP server mode**: 1 consumer (WhisperInc)

Subprocess is the dominant pattern by a 5:2 margin across the observed consumer set. The two FFI consumers are also the two most language-runtime-aligned with CrispASR's own C-ABI surface (Dart/native, Rust/native).

### I.2 Common subprocess-pattern characteristics

- All subprocess consumers parse the standard stdout text format `[hh:mm:ss.fff --> hh:mm:ss.fff] text` inherited from whisper.cpp (Susurrus regex documented at `03_susurrus_dossier.md` §12; WyomingCrispAsrServer has a custom `AnsiLineReader` to strip ESC[K progress-redraw codes interleaved with output).
- Multiple consumers handle the C++ destructor-crash class explicitly. WhisperJAV's own BYOP path documents the same (`05_whisperjav_integration_surfaces.md` §1.1).
- Capability-flag introspection (the `CAP_*` bits enumerated in `01_crispasr_dossier.md` §10) is NOT done by any of the surveyed subprocess consumers. SubtitleEdit exposes a `--force-aligner` switch (per `04_ecosystem_synthesis.md` §A.6 and the verbatim issue #10775 finding) but does not query CrispASR for its capability bitmask first; it surfaces a user toggle instead.

### I.3 Capability introspection gap

There is currently NO production consumer of CrispASR observed to call its capability-introspection API. Subprocess consumers parse stdout opaquely; FFI consumers use the type-safe Rust/Dart wrappers but the surveyed consumer code does not show capability bitmask probing. This is descriptive (not a value judgment): it means future integrators following established patterns can defer the capability-introspection design until needed, and there is no battle-tested reference for it.

---

## J. Authorship and AI-coauthor signals

A pattern visible across all three ecosystem repos and worth flagging for risk-assessment purposes:

### J.1 Placeholder commit-author identity

- Susurrus's three CrispASR-integration commits (`1048d6d`, `0af3e7e`, `7073a77`) are authored by the placeholder identity `crispasr integration <crispasr-dev@localhost>` (`03_susurrus_dossier.md` §0).
- CrispASR's tip-of-main commit `bac5f8f` is also authored by `crispasr integration <crispasr-dev@localhost>` (`01_crispasr_dossier.md` §0).

This identity is consistent across repos but is NOT a real GitHub user — it's an automated/scripted commit identity, likely from a CI pipeline or templated workflow.

### J.2 Claude co-authorship signal

- CrisperWeaver's commits are co-authored `Claude Opus 4.7 (1M context) <noreply@anthropic.com>` (`02_crisperweaver_dossier.md` §0).
- Susurrus's CrispASR-integration commits are co-authored `Claude Opus 4.6 (1M context) <noreply@anthropic.com>` (`03_susurrus_dossier.md` §0).

This is the standard Claude Code co-author trailer and indicates AI-assisted development throughout. It is neither a positive nor negative signal in isolation; it is captured here because it materially shapes the contributor-count picture.

### J.3 Contributor count realism

GitHub's contributor counts (CrispStrobe 754 / vkrmch 3 / DBMePls 2 for CrispASR; CrispStrobe 137 sole for CrisperWeaver; CrispStrobe + 1 typo-fix PR for Susurrus) materially overstate the effective bus-factor if the Claude co-author trailer means the work is human-supervised AI-generated. Conversely, AI-co-authored code can be reviewed and accepted by a single human maintainer at high throughput. **Descriptive observation, not a recommendation:** the effective sustainable maintenance picture of all three repos depends on a single human (`CrispStrobe`) plus that maintainer's tooling.

---

## K. Cross-ecosystem could-not-verify

Items where the cross-ecosystem picture is incomplete:

1. **Whether Susurrus's CrispASR integration was abandoned, parked, or is being actively re-spun in a different repo.** The 24-day-and-counting silence on `main` after the 4-commit integration burst could indicate any of those states. No issue thread or roadmap commits to a position.
2. **Whether CrispASR's hyperactive release cadence (40 releases in 6.5 weeks) is sustainable.** No documented release-cadence commitment in the repo; the AGENTS.md / PLAN.md / TODO.md docs noted in `01_crispasr_dossier.md` §25 were not read.
3. **Whether the C-ABI surface is committed to be stable across point releases**, or whether v0.x cadence implies pre-1.0 breakage. CrispASR's README does not state an ABI-stability policy that this research surfaced.
4. **Whether the placeholder author identity `crispasr integration <crispasr-dev@localhost>` is owned and used solely by CrispStrobe.** Could be a script, a contributor, or shared between collaborators — primary-source evidence does not disambiguate.
5. **Cross-ecosystem dependency graph fragility.** CrisperWeaver depends on CrispASR via local path; if CrispASR's Dart FFI package layout changes, CrisperWeaver breaks. The `CRISPASR_REF` env-var escape hatch exists in CI but is documented as optional.
6. **Whether `CrispEmbed` (the fourth ecosystem repo) shares maintainer cadence and patterns — not researched in this Phase 0 pass.**
7. **Meetily's stated Rust-FFI CrispASR usage** is not corroborated by primary sources (`04_ecosystem_synthesis.md` §B). The Meetily repo appears to use `whisper-rs` / `whisper.cpp` directly, not CrispASR. If a future Meetily integration emerges it would be an additional FFI-consumer data point but at present should not be counted as one.

---

_End of cross-ecosystem synthesis (Part 2). Next dossiers: `06_ideation_inputs_verification.md` (ideation-inputs claim verification) and `07_open_questions.md` (aggregated could-not-verify across all dossiers)._
