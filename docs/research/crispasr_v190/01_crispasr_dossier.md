# CrispASR — Phase 0 Dossier

Primary-source reference dossier on the CrispStrobe/CrispASR repository as a candidate runtime hub for ASR. This document describes what exists in the repo at research time (commit `bac5f8f`, 2026-05-13). It is **descriptive only**: no recommendations, no integration design, no comparisons to alternatives. Where evidence is insufficient, the section "25. Could not verify" makes that explicit.

Research period: 2026-05-13. Repository state captured: tag `v0.6.6` (2026-05-12), tip-of-main commit `bac5f8ffabb948f691b0f84f2c9ff936bb1a9d6d` (2026-05-13). Primary sources include the GitHub REST API (`gh api`), raw files on `raw.githubusercontent.com`, individual issue/PR pages, and one downstream consumer issue (SubtitleEdit#10775).

---

## 0. Snapshot

| Field | Value | Source |
| --- | --- | --- |
| Repo URL | `https://github.com/CrispStrobe/CrispASR` | `gh api repos/CrispStrobe/CrispASR` |
| Default branch | `main` | same |
| Repo created | 2026-03-29T21:24:48Z | same |
| Last push | 2026-05-13T10:11:28Z | same |
| Tip-of-main commit (research time) | `bac5f8ffabb948f691b0f84f2c9ff936bb1a9d6d` ("docs: fix HF truecaser-de table (CRF 8.5 MB, stat 1.7 MB, add missing columns)") authored 2026-05-13T10:11:26Z by `crispasr integration <crispasr-dev@localhost>` | `gh api repos/CrispStrobe/CrispASR/commits?per_page=1` |
| License (SPDX) | MIT | `gh api repos/CrispStrobe/CrispASR` → `license.spdx_id` = `MIT` |
| License file copyright line | `Copyright (c) 2023-2026 The ggml authors` | `raw.githubusercontent.com/CrispStrobe/CrispASR/main/LICENSE` (1078 bytes) |
| Stars | 176 | `gh api repos/CrispStrobe/CrispASR` |
| Forks | 19 | same |
| Watchers / subscribers | 176 / 7 | same |
| Open issues + PRs | 9 (combined GitHub `open_issues_count`; 9 open issues, 0 open PRs at research time) | same; cross-check via `gh api .../pulls?state=open` |
| Total commits in trailing 52 weeks | 1687 | `gh api repos/CrispStrobe/CrispASR/stats/participation` (`recent_13w` and `total_commits_52w` both report 1687 — repo is younger than 52 weeks, so they coincide) |
| Total commits in last 4 weeks | 1330 | same (`last_4w` field) |
| Distinct contributors per GitHub `contributors` endpoint | 3 (`CrispStrobe` 754, `vkrmch` 3, `DBMePls` 2) | `gh api repos/CrispStrobe/CrispASR/contributors?per_page=100` |
| `AUTHORS` file length | 21,736 bytes; alphabetized | `gh api .../contents/AUTHORS` |
| Topics | `cohere-transcribe`, `cohere-transcribe-03-2026`, `ggml`, `parakeet`, `speech-recognition`, `speech-to-text`, `stt`, `transcription`, `voxtral`, `whisper-cpp` | `gh api repos/CrispStrobe/CrispASR` |
| Latest release tag | `v0.6.6` published 2026-05-12T22:37:09Z (not pre-release, not draft) | `gh api .../releases/tags/v0.6.6` |
| Earliest release tag | `v0.1.0` published 2026-04-12T05:34:20Z | `gh api .../releases?per_page=100` |
| Total tagged releases | 40 (v0.1.0 → v0.6.6) | same |

**Stated relationship to upstream `ggerganov/whisper.cpp`** — the repo description on GitHub states verbatim: *"C++ ggml runtime hub for multilingual ASR models: Cohere Transcribe, Parakeet TDT, Voxtral, Canary 1B v2, etc, plus universal forced alignment via NeMo Forced Aligner-style CTC, and others. **Fork of whisper.cpp**."* The `README.md` repeats this in the License section: *"MIT — same as upstream whisper.cpp."* The repo metadata field `fork` is `false`, meaning the GitHub UI does not show a fork badge — the relationship is a logical/historical fork, not a GitHub fork relationship. The `LICENSE` retains the original ggml-authors copyright. The `AUTHORS` file (auto-generated 2025-02-04 per its header) preserves the full original whisper.cpp / ggml contributor roster (alphabetized; sampled first 60 names show whisper.cpp / llama.cpp / ggml-org contributors). The repo's own commit history begins 2026-03-29; the AUTHORS file was generated against the imported upstream history *before* that fork date.

**Note on commit-stats coincidence.** `stats/participation` is a rolling 52-week histogram. Because the repo's first commit is 2026-03-29 and the research date is 2026-05-13 (~6.5 weeks of history), the 52-week and 13-week totals coincide at 1687. The 4-week figure (1330) and the 13-week figure (1687) imply roughly 357 commits in weeks 4-13 of the repo's life and 1330 in weeks 1-4 — i.e. a large initial-port burst followed by sustained ~80-100 commits/week.

---

## 1. Repository structure

Top-level directory listing (`gh api .../contents/`):

```
.clang-format            (1.5 KB)        .clang-tidy             (4.9 KB)
.devops/                                 .dockerignore           (139 B)
.env.example             (301 B)         .github/
.gitignore               (1.6 KB)        ARCHITECTURE.md         (10.7 KB)
AUTHORS                  (21.2 KB)       CMakeLists.txt          (12.3 KB)
CMakePresets.json        (5.1 KB)        COMPARISON.md           (3.1 KB)
HANDOVER_INDEXTTS.md     (4.4 KB)        HISTORY.md              (189.9 KB)
LEARNINGS.md             (225.4 KB)      LICENSE                 (1.0 KB)
Makefile                 (39.7 KB; CMake-generated, not hand-written)
PERFORMANCE.md           (54.9 KB)       PLAN.md                 (146.8 KB)
README.md                (51.0 KB)       README_sycl.md          (6.4 KB)
TODO.md                  (77.7 KB)       Testing/
UPSTREAM.md              (9.1 KB)        VERSION                 (6 B; "0.6.6")
bindings/                ci/             cmake/
crisp_audio/             crispasr-sys/   crispasr/
docker-compose.cuda.yml  docker-compose.yml
docs/                    examples/       flutter/
ggml/                    grammars/       hf-space/
hf_readmes/              include/        models/
python/                  ref/            samples/
scripts/                 src/            tests/
tools/                   build-android.sh / build-ios.sh / build-vulkan.bat /
                         build-windows.bat / build-xcframework.sh
```

### Key directories

- **`src/`** — Per-backend C++ runtimes plus shared core; ~120 files including `crispasr.cpp` (334.9 KB), `crispasr_c_api.cpp` (175.5 KB), and per-backend implementations (`whisper.cpp` is omitted from `src/` per `docs/architecture.md` — "Whisper is intentionally not migrated"; instead whisper-related code stays in the ggml subtree).
- **`src/core/`** — 27 shared primitives (header-only + a few `.cpp`): `mel.{h,cpp}`, `gguf_loader.{h,cpp}`, `attention.h` (35.9 KB), `fastconformer.h`, `conformer_ibm.h`, `beam_decode.h`, `greedy_decode.h`, `bpe.h`, `ctc.h`, `ffn.h`, `fft.h`, `conv.h`, `lstm.h`, `qformer.h`, `granite_llm.h`, `rvq.{h,cpp}`, `kaldi_fbank.{h,cpp}`, `activation.h`, `align.h`, `audio_chunking.h`, `audio_resample.{h,cpp}`, `cpu_ops.h`.
- **`include/`** — Two public-headers: `crispasr.h` (35.3 KB), `crispasr_chat.h` (9.3 KB).
- **`examples/cli/`** — CLI dispatcher + per-backend adapters: `cli.cpp` (97.7 KB), `crispasr_run.cpp` (111.7 KB), `crispasr_server.cpp` (53.2 KB), `crispasr_backend.{h,cpp}` (factory base), `crispasr_backend_<name>.cpp` adapters for 28 backends (canary, chatterbox, cohere, crispasr, fastconformer_ctc, firered_asr, gemma4_e2b, glm_asr, granite, granite_nle, indextts, kokoro, kyutai_stt, m2m100, mimo_asr, moonshine, moonshine_streaming, omniasr, orpheus, parakeet, qwen3, qwen3_tts, t5, vibevoice, voxtral, voxtral4b, wav2vec2).
- **`examples/server/`** — `server.cpp` (54.7 KB), `ws_stream.{cpp,h}` (15.7 KB), embedded `httplib.h` (354.7 KB).
- **`examples/`** (other) — `cli/`, `server/`, `bench/`, `command/`, `crispasr-quantize/`, `crispasr.android/`, `crispasr.objc/`, `crispasr.swiftui/`, `crispasr.wasm/`, `crispasr.nvim/`, `crispasr.android.java/`, `demo/`, `stream/`, `talk-llama/`, `wchess/`, `lsp/`, `nfa-align/`, `cohere-align/`, `qwen3-asr-test-*`, `voxtral-test-*`, `vibevoice-test-stages/`, `quantize/`, `addon.node/`.
- **`bindings/`** — `go/`, `java/`, `javascript/`, `ruby/`. (No `bindings/dart/` or `bindings/python/` directory; Dart lives under `flutter/crispasr/`, Python under `python/crispasr/`.)
- **`crispasr-sys/`** — Rust low-level FFI sys crate (`Cargo.toml` 1.5 KB, `build.rs` 10.1 KB).
- **`crispasr/`** — Rust high-level safe wrapper crate (`Cargo.toml` 1.2 KB).
- **`python/crispasr/`** — Pure-Python ctypes wrapper. Files: `__init__.py` (1.0 KB), `_binding.py` (83.3 KB), `_helpers.c` (740 B).
- **`flutter/crispasr/`** — Dart/Flutter package (`pubspec.yaml` 690 B, `lib/`, `test/`).
- **`crisp_audio/`** — Separate audio-encoder library, own CMakeLists. `include/crisp_audio.h` (4.4 KB), `src/audio_tower.cpp` (29.5 KB), `src/crisp_audio.h` (4.4 KB).
- **`ggml/`** — Vendored ggml as a git subtree. Carries 5 fork-local patches; see §3 / §24.
- **`tools/`** — 35+ Python utilities including `test-all-backends.py` (74.4 KB), `benchmark_asr_engines.py` (34.2 KB), `gen-feature-matrix.py`, `audit-backend-capabilities.py`, `audit-hf-licenses.py`, `kaggle-benchmark-all-backends.py`, `macbook-benchmark-all-backends.py`, dump_*_reference.py scripts, `upstream-prs/` (routing notes for upstream contributions).

### Entry points

- **Main CLI binary**: `crispasr` (built from `examples/cli/`; entry in `cli.cpp` dispatches to `crispasr_run.cpp` which contains the unified pipeline `crispasr_run_backend()`). `README.md`: *"Produces: `build/bin/crispasr` (main CLI), `build/bin/crispasr-quantize` (model re-quantization), `build/bin/crispasr-diff` (regression testing)."*
- **Auxiliary CLIs**: `crispasr-quantize`, `crispasr-diff` (binary uses `crispasr_diff_main.cpp` 117.4 KB), `crispasr-lid` (`crispasr_lid_main.cpp`), `crispasr-server` (`server.cpp`), `crispasr-chat` (`crispasr_chat_main.cpp`).
- **Mode-flags inside the main binary**: `--server` flips to HTTP server mode; `--mic`, `--stream`, `--live` switch the input source; `--list-backends` and `--list-backends-json` enumerate the compiled-in backend set.

Source: `README.md` § *Build Instructions* and § *HTTP Server Mode*; file inventory via `gh api .../contents/examples/cli`.

---

## 2. License + redistribution posture

### License file contents

`LICENSE` (1078 bytes, `raw.githubusercontent.com/CrispStrobe/CrispASR/main/LICENSE`) is the standard MIT license, copyright line **"Copyright (c) 2023-2026 The ggml authors"** — unchanged from upstream whisper.cpp / ggml. The clauses observed:

- Permission grant: *"to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software"*
- Attribution requirement: *"the above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software"*
- Standard disclaimer: *"THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND"*
- Standard limitation-of-liability clause

### Implications for static vs subprocess linking (descriptive)

The MIT license permits both static linking into a closed/different-license host program and subprocess invocation, subject to the attribution requirement. There are no copyleft or share-alike clauses in MIT.

### Per-component license notes visible in the repo

`README.md` makes per-model license calls out explicitly in the *License* section: *"Per-model weights covered by respective HuggingFace licenses (see Supported backends section). The `crispasr` binary links mostly permissively licensed runtimes (MIT / Apache-2.0 / CC-BY-4.0 for weights)."*

Specific per-model license flags visible in `README.md`:

| Component | License flag in README |
| --- | --- |
| whisper | MIT |
| kyutai-stt | MIT |
| moonshine-de / moonshine-tiny-de | CC-BY-NC-SA-4.0 |
| data2vec-audio | Apache-2.0 |
| hubert | Apache-2.0 |
| firered post-processor (punctuation) | Apache-2.0 |
| fullstop post-processor | MIT (XLM-R-large) |
| punctuate-all post-processor | MIT (XLM-R-base) |
| PCS post-processor | Apache-2.0 |
| CLD3 LID | Apache-2.0 |
| GlotLID-V3 LID | Apache-2.0 |
| LID-176 LID | CC-BY-SA-3.0 (README explicitly warns: *"viral — redistributors inherit ShareAlike"*) |
| Granite, Voxtral, Qwen3, OmniASR-LLM | "Apache-licensed speech-LLM" per the README's *Backend Selection Guide* |
| Translation: m2m100 / WMT21 / MADLAD | README does not state explicit license flags inline (per-model HuggingFace cards govern) |

### Embedded third-party components in the source tree

- `examples/json.hpp` (886.6 KB) — nlohmann/json
- `src/miniaudio.h` (4.0 MB) and `examples/miniaudio.h` (4.0 MB) — David Reid's miniaudio (single-file audio I/O library)
- `examples/stb_vorbis.c` (188.3 KB) — Sean Barrett's stb_vorbis
- `examples/server/httplib.h` (354.7 KB) — yhirose/cpp-httplib
- `ggml/` — vendored ggml as a git subtree, with five fork-local patches inside marked `// CrispASR patch` (see `UPSTREAM.md` § *ggml — fork-local patches we already carry*)

The repo does not present a consolidated NOTICE / THIRD_PARTY_LICENSES file at the root. The `AUTHORS` file is an alphabetized contributors list, not a licensing notice.

Source: `LICENSE`, `README.md` § *License*, `README.md` § *Supported backends*, `UPSTREAM.md`, `gh api .../contents/`.

---

## 3. Build system

### Hand-written `CMakeLists.txt` (root)

`CMakeLists.txt` (12.6 KB) — the actual build root. Key observations from the verbatim content fetched:

- `cmake_minimum_required(VERSION 3.10...3.29)`.
- `project(crispasr VERSION ${PROJECT_VERSION_CONTENT} LANGUAGES C CXX)` — version read from `VERSION` file.
- `if(APPLE) enable_language(OBJC) enable_language(OBJCXX) endif()` — Objective-C/-C++ for Apple platforms.
- `BUILD_SHARED_LIBS_DEFAULT` — `OFF` on Emscripten and MinGW, `ON` elsewhere.

#### Hand-defined `option()` declarations (CMakeLists.txt root):

```
CRISPASR_ALL_WARNINGS                ON
CRISPASR_ALL_WARNINGS_3RD_PARTY      OFF
CRISPASR_FATAL_WARNINGS              OFF
CRISPASR_USE_SYSTEM_GGML             OFF
CRISPASR_SANITIZE_THREAD             OFF
CRISPASR_SANITIZE_ADDRESS            OFF
CRISPASR_SANITIZE_UNDEFINED          OFF
CRISPASR_BUILD_TESTS                 ${CRISPASR_STANDALONE}
CRISPASR_BUILD_EXAMPLES              ${CRISPASR_STANDALONE}
CRISPASR_BUILD_SERVER                ${CRISPASR_STANDALONE}
CRISPASR_CURL                        OFF        # libcurl for HF download
CRISPASR_SDL2                        OFF
CRISPASR_FFMPEG                      OFF        # Linux only
CRISPASR_COREML                      OFF
CRISPASR_COREML_ALLOW_FALLBACK       OFF
CRISPASR_OPENVINO                    OFF
COHERE_MKL                           OFF        # Intel MKL GEMM for Cohere
CRISPASR_WASM_SINGLE_FILE            ON         # Emscripten only
```

#### Deprecated option aliases (auto-forwarded to `GGML_*`):

```
CRISPASR_CUBLAS              → GGML_CUDA              (FATAL_ERROR)
CRISPASR_CUDA                → GGML_CUDA              (WARNING)
CRISPASR_KOMPUTE             → GGML_KOMPUTE           (WARNING)
CRISPASR_METAL               → GGML_METAL             (WARNING)
CRISPASR_METAL_EMBED_LIBRARY → GGML_METAL_EMBED_LIBRARY (WARNING)
CRISPASR_NATIVE              → GGML_NATIVE            (WARNING)
CRISPASR_OPENMP              → GGML_OPENMP            (WARNING)
CRISPASR_RPC                 → GGML_RPC               (WARNING)
CRISPASR_SYCL                → GGML_SYCL              (WARNING)
CRISPASR_SYCL_F16            → GGML_SYCL_F16          (WARNING)
CRISPASR_CCACHE              → GGML_CCACHE            (WARNING)
```

This list enumerates the GPU/CPU backends that the build system understands. The active flag names live in `ggml/CMakeLists.txt` (per `add_subdirectory(ggml)` call); the `CRISPASR_*` versions are kept only for backward compatibility.

#### Conditional behaviour observed in the root CMakeLists.txt:

- **MKL**: if `COHERE_MKL=ON`, sets `GGML_BLAS=ON` and `GGML_BLAS_VENDOR=Intel10_64lp`.
- **CUDA**: if `GGML_CUDA AND NOT MSVC`, adds `-Wno-deprecated-gpu-targets`.
- **System GGML**: if `CRISPASR_USE_SYSTEM_GGML`, uses `find_package(ggml REQUIRED)` instead of the vendored subtree.
- **Emscripten**: forces `-pthread`, `TOTAL_STACK=5242880`, suppresses deprecation warnings.
- **MSVC**: suppresses specific warning codes via per-target `target_compile_options` on `crispasr`, `common`, `common-sdl`, `lsp`, `wchess-core`, `crispasr-command`, `crispasr-cli`, `crispasr-server`, `crispasr-stream`, `crispasr-talk-llama`, `crispasr-bench`, `quantize`, `vad-speech-segments`.
- **Public header install**: `set_target_properties(crispasr PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/crispasr.h)`.
- **CMake package config**: generates `crispasr-config.cmake`, `crispasr-version.cmake`, and `crispasr.pc` for downstream find_package + pkg-config use.

#### Subdirectories driven from root CMakeLists.txt:

```
add_subdirectory(ggml)         # if not using system ggml
add_subdirectory(src)
add_subdirectory(crisp_audio)
if (CRISPASR_BUILD_TESTS)  add_subdirectory(tests)
if (CRISPASR_BUILD_EXAMPLES) add_subdirectory(examples)
```

### `Makefile` (root, 40.7 KB)

The 39.7 KB file at the root named `Makefile` was confirmed via `WebFetch` to be a **CMake-generated** Makefile (its header reads `# CMAKE generated file: DO NOT EDIT! / # Generated by "Unix Makefiles" Generator, CMake Version 3.28`), not a hand-maintained build script. No `ifeq`/`ifdef`-style GPU-backend conditionals live in this file. All GPU dispatch lives in `CMakeLists.txt` + `ggml/`.

### `CMakePresets.json` (5.2 KB)

Defines configure presets: `default`, `debug`, `linux`, `asan`, `ubsan`, `sanitize`, `tsan`, `coverage`, `tidy` — sanitizer- and code-quality-focused. No GPU-specific presets (no `cuda` / `metal` / `vulkan` presets in this file). GPU enablement is via `-DGGML_CUDA=ON` etc. on the cmake command line.

### Platform helper scripts (root)

| Script | Purpose (inferred from name + README) |
| --- | --- |
| `build-android.sh` (1.5 KB) | Android NDK build |
| `build-ios.sh` (1.9 KB) | iOS build |
| `build-vulkan.bat` (1.9 KB) | Windows Vulkan build |
| `build-windows.bat` (1.4 KB) | Windows CPU build |
| `build-xcframework.sh` (24.7 KB) | macOS / iOS / visionOS / tvOS XCFramework build |

### Conditional features (GPU + CPU + accelerator backends)

Per `README.md` § *Build Instructions* and § *GPU Backend Selection*, the build system supports the following backends (flags passed to cmake at configure time):

- **CUDA** (`-DGGML_CUDA=ON`) — NVIDIA
- **Metal** (`-DGGML_METAL=ON`) — Apple Silicon
- **Vulkan** (`-DGGML_VULKAN=ON`) — cross-vendor GPU
- **MUSA** — Moore Threads (`README.md` mentions, flag is `GGML_MUSA` per ggml convention)
- **SYCL** (`-DGGML_SYCL=ON`) — Intel
- **CoreML** (`-DCRISPASR_COREML=ON` + optional `-DCRISPASR_COREML_ALLOW_FALLBACK=ON`)
- **OpenVINO** (`-DCRISPASR_OPENVINO=ON`)
- **OpenMP** (`-DGGML_OPENMP=ON`) — CPU parallelism
- **BLAS** (`-DGGML_BLAS=ON`, optional `-DCOHERE_MKL=ON` shortcut for Intel MKL)
- **RPC** (`-DGGML_RPC=ON`) — distributed inference (inherited from ggml)
- **Kompute** (`-DGGML_KOMPUTE=ON`) — Vulkan-compute via Kompute (inherited from ggml)
- **FFmpeg** (`-DCRISPASR_FFMPEG=ON`, Linux only per CMakeLists) — in-process audio decoding via libav*

Per the README: *"Multiple backends can be compiled simultaneously, with `ggml will pick the highest-priority compiled backend at runtime.`"* Runtime selection is controlled by `--gpu-backend {auto,cuda,metal,vulkan,cpu}` and `-dev N` (per-device pin) flags, as documented in the README's *GPU Backend Selection* section.

### Target platforms (per prebuilt release artifacts on v0.6.6)

Confirmed via `gh api .../releases/tags/v0.6.6` asset list:

| Asset | Bytes |
| --- | --- |
| `crispasr-android-arm64-v8a.tar.gz` | 21.7 MB |
| `crispasr-linux-arm64.tar.gz` | 4.9 MB |
| `crispasr-linux-x86_64.tar.gz` | 5.4 MB |
| `crispasr-linux-x86_64-avx512.tar.gz` | 5.3 MB |
| `crispasr-macos.tar.gz` | 3.7 MB |
| `crispasr-python-linux-arm64.tar.gz` | 3.5 MB |
| `crispasr-python-linux-x86_64.tar.gz` | 3.7 MB |
| `crispasr-windows-x86_64-cpu.zip` | 3.1 MB |
| `crispasr-windows-x86_64-cpu-legacy.zip` | 3.1 MB |
| `crispasr-windows-x86_64-cuda.zip` | **683.7 MB** |
| `crispasr-windows-x86_64-vulkan.zip` | 24.7 MB |
| `libcrispasr-linux-arm64.tar.gz` | 3.5 MB |
| `libcrispasr-linux-x86_64.tar.gz` | 3.8 MB |
| `libcrispasr-linux-x86_64-avx512.tar.gz` | 3.8 MB |
| `libcrispasr-linux-x86_64-cuda.tar.gz` | 100.6 MB |
| `libcrispasr-macos-arm64.tar.gz` | 3.0 MB |
| `libcrispasr-windows-x86_64.tar.gz` | 11.8 MB |
| `libcrispasr-windows-x86_64-cpu-legacy.tar.gz` | 11.7 MB |
| `libcrispasr-windows-x86_64-cuda.tar.gz` | **789.2 MB** |
| `libcrispasr-windows-x86_64-vulkan.tar.gz` | 48.0 MB |

Total: 20 prebuilt artifacts per release. The "v0.6.0" release additionally shipped `crispasr-v0.6.0-xcframework.zip` (139.9 MB) and split CUDA artifacts for CUDA 11.8 / CUDA 12.4. The most recent v0.6.6 release does not include an iOS / xcframework asset, suggesting the xcframework is built less frequently.

Source: `CMakeLists.txt` (verbatim, 12.6 KB); `CMakePresets.json` (5.2 KB); `gh api .../releases/tags/v0.6.6`; `README.md`.

---

## 4. Layered architecture

`docs/architecture.md` (27.0 KB) and `ARCHITECTURE.md` (10.7 KB at root) both describe the architecture; the docs version is more comprehensive.

### Three-layer stack (as stated in `docs/architecture.md`)

> *"The split between `src/` (library) and `examples/cli/` (presentation) is deliberate: **every algorithm** — VAD, diarization, LID, CTC alignment, HF download/cache, model registry — lives in `src/` behind a stable C-ABI."*

- **Layer 1 — CLI (`examples/cli/`)**: dispatch based on `--backend` flag or GGUF architecture auto-detect; orchestration of model-load → audio-load → VAD → ASR → align → punctuation → output. `cli.cpp` (97.7 KB) is the entry point; `crispasr_run.cpp` (111.7 KB) contains the unified pipeline (`crispasr_run_backend()`).
- **Layer 2 — CLI Adapters (`examples/cli/crispasr_backend_*.cpp`)**: thin (~120 LOC each per the contributing doc) wrappers that handle policy: auto-download prompts, TTY prompts, subprocess fallbacks (e.g. sherpa-ONNX for diarization). They delegate algorithmic work down to layer 3. 28 backend adapter files observed.
- **Layer 3 — Library (`src/`)**: per-model runtimes (`canary.cpp`, `cohere.cpp`, `qwen3_asr.cpp`, etc.) + shared C-ABI (`crispasr_c_api.cpp` 175.5 KB) + shared primitives (`src/core/`).

### `src/core/` primitives

The `src/core/` directory exposes a `crispasr-core` static library. Files observed (`gh api .../contents/src/core`):

| File | Size | Stated role (docs/architecture.md or filename) |
| --- | --- | --- |
| `mel.{h,cpp}` | 10.6 + 15.8 KB | Unified log-mel spectrogram extraction; supports both NeMo-family (z-score + per-mel normalization) and HuggingFace/Whisper-family (log10 + global clip) |
| `gguf_loader.{h,cpp}` | 8.5 + 37.2 KB | Two-pass GGUF load with mmap + pread/fseek fallback; `core_gguf::WeightLoad` owns ggml_context + backend buffer + tensor map (move-only) |
| `attention.h` | 35.1 KB | Llama-style multi-head attention + flash-attention + GQA + RoPE (header-only) |
| `ffn.h` | 5.9 KB | SwiGLU / SiLU feed-forward (header-only) |
| `fastconformer.h` | 13.3 KB | NeMo FastConformer block (conv subsampling + MHA with relative position encoding) |
| `conformer_ibm.h` | 13.5 KB | IBM Macaron Conformer variant (intentionally separate from fastconformer.h per the docs) |
| `beam_decode.h` | 18.1 KB | Beam search decoder primitive |
| `greedy_decode.h` | 12.4 KB | Autoregressive greedy decode with EOS handling |
| `bpe.h` | 10.6 KB | GPT-2 byte-pair tokenizer |
| `ctc.h` | 4.8 KB | CTC decoding utilities |
| `fft.h` | 3.4 KB | Header-only FFT |
| `conv.h` | 6.6 KB | Convolution helpers |
| `cpu_ops.h` | 3.9 KB | CPU intrinsic helpers |
| `lstm.h` | 6.3 KB | LSTM cell primitive |
| `qformer.h` | 10.0 KB | Q-Former (BLIP-style cross-attention sampler) primitive |
| `granite_llm.h` | 6.6 KB | Granite LLM forward (audio-LLM head) |
| `rvq.{h,cpp}` | 1.9 + 1.9 KB | Residual Vector Quantization (for kyutai-stt-style codec ASR) |
| `kaldi_fbank.{h,cpp}` | 2.2 + 7.8 KB | Kaldi-compatible filterbank (for legacy NeMo models) |
| `activation.h` | 2.9 KB | Activation functions (GELU, SiLU, etc.) |
| `align.h` | 2.1 KB | Forced-alignment primitive (CTC alignment) |
| `audio_chunking.h` | 4.0 KB | Long-audio chunking helpers |
| `audio_resample.{h,cpp}` | 1.6 + 4.8 KB | PCM resampling |

The README states this is auditable bit-identically: *"Every `src/core/` migration commit includes md5-level validation against `samples/jfk.wav`."* (`docs/architecture.md`).

### `src/` per-backend runtimes

Per-backend C++ files in `src/` (counts based on `gh api .../contents/src`). Each ASR backend has one or two main `.cpp` files plus a header:

ASR backends found: `canary.cpp` + `canary_ctc.cpp` + `canary.h` + `canary_ctc.h`; `cohere.cpp` + `cohere-arch.h` + `cohere.h`; `crispasr.cpp` (the whisper runtime entry); `firered_asr.cpp` + `firered_vad.cpp` + `firered_lid.cpp` + `fireredpunc.cpp`; `gemma4_e2b.cpp`; `glm_asr.cpp`; `granite_nle.cpp` + `granite_speech.cpp`; `kyutai_stt.cpp`; `mimo_asr.cpp` + `mimo_tokenizer.cpp`; `moonshine.cpp` + `moonshine_streaming.cpp` + `moonshine-tokenizer.cpp`; `omniasr.cpp`; `parakeet.cpp`; `qwen3_asr.cpp`; `voxtral.cpp` + `voxtral4b.cpp`; `wav2vec2-ggml.cpp` + `wav2vec2-ggml-debug.cpp`.

TTS backends: `chatterbox.cpp` + `chatterbox_campplus.cpp` + `chatterbox_s3gen.cpp` + `chatterbox_s3tok.cpp` + `chatterbox_ve.cpp`; `indextts.cpp` + `indextts_voc.cpp`; `kokoro.cpp`; `orpheus.cpp` + `orpheus_snac.cpp`; `qwen3_tts.cpp`; `vibevoice.cpp`.

LID providers: `crispasr_lid.cpp`, `ecapa_lid.cpp`, `firered_lid.cpp`, `lid_cld3.cpp`, `lid_fasttext.cpp`, `silero_lid.cpp`, `text_lid_dispatch.cpp`.

VAD providers: `crispasr_vad.cpp` (Silero default), `crispasr_vad_encdec.cpp` (Whisper-VAD-EncDec), `firered_vad.cpp`, `marblenet_vad.cpp`.

Post-processing: `fireredpunc.cpp` (punctuation), `pcs.cpp` (punctuation + casing + sentence boundary detection), `truecaser.cpp` + `truecaser_crf.cpp` + `truecaser_lstm.cpp` (German truecasing), `pyannote_seg.cpp` (diarization segmentation), `titanet.cpp` (speaker embedding for diarization), `speaker_db.cpp`.

Translation: `m2m100.cpp`, `t5_translate.cpp` (MADLAD-400).

C-ABI: `crispasr_c_api.cpp` (175.5 KB) is the single export surface consumed by all language bindings.

Other infrastructure: `crispasr_cache.cpp` (HF model cache with zombie-file detection), `crispasr_model_registry.cpp` (36.2 KB; backend→GGUF URL mapping with fuzzy filename hints), `crispasr_aligner.cpp` (forced alignment), `crispasr_diarize.cpp` (4 diarization methods), `crispasr_mic.cpp` (mic via miniaudio).

### `bindings/` structure

| Path | Contents (`gh api .../contents/bindings/<name>`) |
| --- | --- |
| `bindings/javascript/` | `whisper.js` (937.8 KB pre-built), `emscripten.cpp`, `libwhisper.worker.js`, `package.json` |
| `bindings/go/` | `crispasr_session.go` (32.9 KB), `whisper.go` (17.8 KB), `params.go` (5.2 KB), `whisper_test.go`, `doc.go`, `pkg/`, `samples/`, `examples/` |
| `bindings/java/` | Android Gradle module (`build.gradle`, `gradlew`, `src/`) |
| `bindings/ruby/` | `Rakefile`, `whispercpp.gemspec`, `ext/`, `lib/`, `sig/`, `test/` |
| (Python) `python/crispasr/` | `__init__.py`, `_binding.py` (81.4 KB ctypes wrapper), `_helpers.c` |
| (Dart) `flutter/crispasr/` | `pubspec.yaml`, `lib/`, `test/`, `CHANGELOG.md` |
| (Rust low-level) `crispasr-sys/` | `Cargo.toml`, `build.rs` (10 KB), `src/` |
| (Rust high-level) `crispasr/` | `Cargo.toml`, `src/`, `tests/` |

### `tools/` highlights

Per `gh api .../contents/tools`, 35+ Python utilities (numbers approximate):

- `test-all-backends.py` (74.4 KB) — regression gate (see §22 / `docs/regression-matrix.md`)
- `benchmark_asr_engines.py` (34.2 KB) + `benchmark_asr_engines.README.md` + `benchmark_asr_engines.results.md`
- `bench_streaming_latency.py` (7.8 KB)
- `gen-feature-matrix.py` (12.9 KB) — generates `docs/feature-matrix.html` from `crispasr --list-backends-json`
- `audit-backend-capabilities.py` (8.9 KB)
- `audit-hf-licenses.py` (13.9 KB)
- `kaggle-benchmark-all-backends.py` (29.1 KB)
- `macbook-benchmark-all-backends.py` (20.8 KB)
- `_audio_diff.py` (8.0 KB) — TTS audio cosine-similarity validator
- `dump_*_reference.py` family — per-backend PyTorch reference dumpers used by `crispasr-diff`
- `upstream-prs/` — routing notes for upstream contributions to ggml-org/ggml vs llama.cpp
- `format.sh` (3.2 KB) — clang-format-v18 enforcement wrapper (`docs/contributing.md`)

Source: `docs/architecture.md`, `ARCHITECTURE.md`, `gh api .../contents/src`, `gh api .../contents/src/core`, `gh api .../contents/examples/cli`, `gh api .../contents/bindings`, `gh api .../contents/tools`.

---

## 5. The "5-file recipe" extension pattern

Documented in `docs/contributing.md`. Adding a new ASR backend requires modifying exactly five files (verbatim list):

1. **`src/yourmodel.{h,cpp}`** — C API implementation
2. **`examples/cli/crispasr_backend_yourmodel.cpp`** — backend adapter class (~120 LOC)
3. **`examples/cli/crispasr_backend.cpp`** — factory registration (constructor in conditional chain) + listing in `crispasr_list_backends()`
4. **`examples/cli/CMakeLists.txt`** — build configuration
5. **`src/crispasr_model_registry.cpp`** — optional `-m auto` entry (backend → default HF repo + filename → URL mapping)

### C API signature pattern

```c
struct yourmodel_context * yourmodel_init_from_file(const char * path, ...)
void yourmodel_free(struct yourmodel_context *)
char * yourmodel_transcribe(struct yourmodel_context *, const float * samples, ...)
```

### Backend adapter class

The adapter extends `CrispasrBackend` (declared in `examples/cli/crispasr_backend.h`) and implements six virtual methods:

- `name()` — human-readable identifier
- `capabilities()` — bitmask of `CAP_*` flags (see §10)
- `init(const whisper_params&)` — loads model; returns success boolean
- `transcribe()` — processes mono 16 kHz PCM; returns vector of segments with absolute timestamps
- `transcribe_stereo()` — optional stereo variant; default falls back to mono
- `shutdown()` — resource cleanup

Optional TTS / translation methods on the same class: `synthesize()`, `translate_text()`.

### Shared primitives reused (per `docs/architecture.md`)

A new ASR backend implementation is expected to call into `src/core/` for: mel extraction, GGUF loading, attention blocks, FFN, BPE tokenization, conv stems, FFT, beam/greedy decode. The architecture doc lists *"Three categories remain intentionally duplicated: Per-model Cooley-Tukey FFT implementations (9 variants), GGUF tensor naming schemes (genuinely model-specific), Forward graph topologies and KV cache threading patterns."* — so the FFT and forward-graph code are explicitly not shared.

### Regression validation step

For ASR: bit-identical transcript output before/after, via `diff before.txt after.txt`.
For stochastic TTS: pinned Gaussian noise + audio-cosine similarity vs official model output (`tools/_audio_diff.py`); ≥0.999 indicates equivalence modulo F16 quantization.

### Style enforcement

`docs/contributing.md` states: *"**‼️ `clang-format` MUST be v18 — never use v22**"*. v22 silently rewraps lines and produces ~80+ CI lint failures. The repo enforces v18 via `./tools/format.sh` (hardcoded version check). Install paths: Homebrew `llvm@18`, apt `clang-format-18`, pip `clang-format==18.1.8`.

Source: `docs/contributing.md`, `docs/architecture.md`, `examples/cli/crispasr_backend.h`.

---

## 6. Memory model

### ggml context lifecycle

Documented in `docs/architecture.md` as `core_gguf::WeightLoad`: *"owns the `ggml_context`, backend buffer, and tensor map in a single struct supporting move semantics."* This struct is the per-backend handle owning all GPU/CPU memory for one loaded model. Move-only semantics ensure each backend instance has unique ownership of its tensors.

The whisper-style C API in `include/crispasr.h` exposes the classical whisper.cpp lifecycle (preserved unchanged for the whisper path):

```c
struct whisper_context * whisper_init_from_file_with_params(const char*, struct whisper_context_params);
struct whisper_state   * whisper_init_state(struct whisper_context*);
void whisper_free(struct whisper_context*);
void whisper_free_state(struct whisper_state*);
void whisper_free_params(struct whisper_full_params*);
void whisper_free_context_params(struct whisper_context_params*);
```

### mmap GGUF loading

Per `docs/architecture.md`: *"Two-pass GGUF loading with mmap support and fallback to pread/fseek for incompatible filesystems."* The README documents an environment variable `CRISPASR_GGUF_MMAP=1` to opt into mmap weights for "most backends."

### VRAM teardown / context-switch semantics

The architecture document does not state explicit timing/ordering guarantees for VRAM release. What is documented is that `crispasr_server.cpp` supports hot-swap via `POST /load` (server mode replaces in-memory model — implies previous model is freed before the new one allocates). Outside server mode, the unified pipeline `crispasr_run_backend()` runs one model per process invocation; teardown happens on `shutdown()` + process exit.

Behaviour around opening a second model in the *same* process while the first is still loaded — and behaviour around environment variable `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (which the README documents as a way to swap to RAM when VRAM is exhausted) — is documented only at the env-var level. Detailed VRAM-release semantics (synchronous vs deferred, lazy CUDA stream sync) are **not** discussed in the public docs.

The `Session` Python class in `python/crispasr/_binding.py` implements `close()` + context-manager `__enter__`/`__exit__` — and `Mic` similarly. Both wrap ctypes `_free` calls. There is no documented per-tensor or per-layer eviction API.

Source: `docs/architecture.md`, `include/crispasr.h`, `python/crispasr/_binding.py`, `README.md` § *GPU Backend Selection*.

---

## 7. Hardware acceleration matrix

### Compile-time backend selection (per `README.md` § *Build Instructions* + `CMakeLists.txt`)

| Backend | CMake flag | Notes |
| --- | --- | --- |
| CUDA | `-DGGML_CUDA=ON` | NVIDIA; Windows artifact size ~683 MB suggests bundled cuBLAS/cuDNN |
| Metal | `-DGGML_METAL=ON` + optional `-DGGML_METAL_EMBED_LIBRARY=ON` | Apple Silicon; macOS artifacts ship by default |
| Vulkan | `-DGGML_VULKAN=ON` | Cross-vendor (AMD / Intel Arc / NVIDIA fallback) |
| MUSA | `-DGGML_MUSA=ON` (ggml convention) | Moore Threads GPUs — README mentions, no prebuilt artifact at v0.6.6 |
| SYCL | `-DGGML_SYCL=ON` (optional `GGML_SYCL_F16=ON`) | Intel — `README_sycl.md` (6.4 KB) exists as a dedicated build guide |
| CoreML | `-DCRISPASR_COREML=ON` (+ optional `-DCRISPASR_COREML_ALLOW_FALLBACK=ON`) | Apple Neural Engine for Whisper encoder |
| OpenVINO | `-DCRISPASR_OPENVINO=ON` | Intel hardware acceleration |
| CPU + BLAS | `-DGGML_BLAS=ON` + vendor (OpenBLAS / Apple Accelerate / Intel MKL via `-DCOHERE_MKL=ON`) | |
| OpenMP | `-DGGML_OPENMP=ON` | CPU parallelism |
| RPC | `-DGGML_RPC=ON` | Distributed inference (inherited from ggml) |
| Kompute | `-DGGML_KOMPUTE=ON` | Vulkan-compute via Kompute (inherited from ggml) |

### D3D12

**No D3D12 backend was found** in primary sources. The ggml family includes D3D12 in some forks (DirectML-style) but `CMakeLists.txt` does not declare a `GGML_D3D12` option, and the deprecated-option forwarding list in CMakeLists.txt does not include a D3D12 alias. See §25.

### CPU SIMD paths

The release-asset list distinguishes `crispasr-linux-x86_64-avx512.tar.gz` from `crispasr-linux-x86_64.tar.gz`, and `crispasr-windows-x86_64-cpu.zip` from `crispasr-windows-x86_64-cpu-legacy.zip`. This implies:

- AVX2 baseline ("default") and AVX-512 (separate artifact, ~3% smaller binary, 5.3 vs 5.4 MB on Linux).
- "legacy" Windows artifact targets older CPUs (likely AVX-only, perhaps SSE4 fallback).
- ARM NEON is implied for the `linux-arm64` and `android-arm64-v8a` artifacts (4.9 MB and 21.7 MB respectively).

The exact SIMD selection logic lives in `ggml/src/ggml-cpu/` (the vendored ggml subtree). `UPSTREAM.md` mentions a planned (but not-yet-implemented) AVX-VNNI / AVX512-VNNI Q8_0 dot-product path: *"ggml's `vec_dot_q8_0_q8_0` uses AVX2 `pmaddubsw` / `pmaddwd`, which is ~2× slower than the AVX-VNNI `vpdpbusd` instruction."* Status: design plan only, not implemented.

### Runtime selection

Per `README.md` § *GPU Backend Selection*:

```bash
crispasr -m model.gguf -f audio.wav                              # Auto-select: CUDA > Metal > Vulkan > CPU
crispasr --gpu-backend vulkan -m model.gguf -f audio.wav         # Force Vulkan
crispasr --gpu-backend vulkan -dev 1 -m model.gguf -f audio.wav  # Pin specific GPU
crispasr -ng -m model.gguf -f audio.wav                          # Force CPU (`--no-gpu`)
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 crispasr -m model.gguf -f audio.wav   # CUDA unified memory
```

Selection rule (quoted): *"Auto-select best (CUDA > Metal > Vulkan > CPU)"*. Multiple backends can be compiled simultaneously; ggml picks the highest-priority compiled backend at runtime.

Source: `CMakeLists.txt`, `README.md` § *GPU Backend Selection* + § *Build Instructions*, `UPSTREAM.md`, `gh api .../releases/tags/v0.6.6`.

---

## 8. Audio I/O subsystem (`crisp_audio`)

There are **two distinct things** named "crisp_audio" in this repo — careful disambiguation required:

### 8.1 The `crisp_audio/` directory (the audio-encoder library)

Despite the directory name, this is **not** an audio-I/O / format-decoding library. It is an **audio-feature encoder** (i.e., the audio tower portion of multimodal models). Per the file header of `crisp_audio/include/crisp_audio.h` (verbatim quote):

> *"shared C++ audio-encoder library. Models multimodal audio embedding paths used in many speech models — BidirLM-Omni (CrispEmbed), Qwen3-ASR + Voxtral + Whisper-derivatives (CrispASR), and most other Conv-stem + Transformer-encoder topologies in current use."*

It contains exactly one supported "dialect" (`CRISP_AUDIO_DIALECT_QWEN_OMNI = 1`) with an `AUTO` placeholder for future expansion. Public API (verbatim from the header):

```c
struct crisp_audio_context;
enum crisp_audio_dialect { CRISP_AUDIO_DIALECT_AUTO = 0, CRISP_AUDIO_DIALECT_QWEN_OMNI = 1 };
struct crisp_audio_params { int n_threads; int verbosity; bool use_gpu;
                            const char* tensor_prefix; const char* meta_prefix;
                            enum crisp_audio_dialect dialect; };

struct crisp_audio_params  crisp_audio_params_default(void);
struct crisp_audio_context * crisp_audio_init_from_file(const char* gguf_path,
                                                        const struct crisp_audio_params* params);
void                       crisp_audio_free(struct crisp_audio_context*);
float *                    crisp_audio_compute_mel(struct crisp_audio_context*, const float* samples,
                                                   int n_samples, int* out_n_mels, int* out_T_mel);
float *                    crisp_audio_encode(struct crisp_audio_context*, const float* mel,
                                              int n_mels, int T_mel, int* out_n_frames, int* out_dim);
int                        crisp_audio_d_model(struct crisp_audio_context*);
int                        crisp_audio_output_dim(struct crisp_audio_context*);
int                        crisp_audio_n_layers(struct crisp_audio_context*);
int                        crisp_audio_n_window(struct crisp_audio_context*);
enum crisp_audio_dialect   crisp_audio_dialect_of(struct crisp_audio_context*);
```

**Input contract**: raw 16 kHz mono float32 PCM → `(n_frames, output_dim)` features. **Threading**: *"All functions are thread-unsafe per context — wrap with a mutex in multi-threaded callers."* (header comment, verbatim).

This library is `add_subdirectory(crisp_audio)`'d from the root CMakeLists.txt.

### 8.2 In-process audio file decoding (the actual format-ingestion path)

In-process audio loading uses two embedded single-file libraries — both at `src/miniaudio.h` (4.0 MB) and `examples/miniaudio.h` (4.0 MB), and `examples/stb_vorbis.c` (188.3 KB). `crispasr_mic.cpp` and the mic callback in `_binding.py` reference miniaudio's audio thread.

### 8.3 Supported audio formats (file ingestion)

`README.md` (§ *Audio formats*) and `UPSTREAM.md` together indicate:

- **Always supported via embedded decoders**: WAV (PCM s16le standard), Ogg Vorbis (via stb_vorbis), MP3 (via miniaudio), FLAC (via miniaudio).
- **Opt-in via `-DCRISPASR_FFMPEG=ON` (Linux only)**: bare-codec files like `.opus`, `.aac`. The build flag is gated `if (CMAKE_SYSTEM_NAME MATCHES "Linux")` in CMakeLists.txt.
- **Known broken even with `CRISPASR_FFMPEG=ON`** (per `UPSTREAM.md`):
  - `.m4a` (AAC in mp4): "crashes with `munmap_chunk(): invalid pointer` on the first audio chunk read"
  - `.webm` (Opus in WebM): "hangs indefinitely after the libavformat headers are parsed"
- **Workaround documented in `UPSTREAM.md`**: users pre-convert mp4-family containers with `ffmpeg -i in.X -ar 16000 -ac 1 -c:a pcm_s16le out.wav`.

### 8.4 Codec independence claim vs ffmpeg

The repo does **not** claim full codec independence from ffmpeg. It claims native (no ffmpeg) decoding for WAV / MP3 / FLAC / Ogg Vorbis via miniaudio + stb_vorbis. For other formats (Opus / AAC / WebM / m4a / mov), ffmpeg is required and the in-process libav* path has known mp4-container bugs documented in `UPSTREAM.md`.

### 8.5 Resampling behavior

`src/core/audio_resample.{h,cpp}` (1.6 + 4.8 KB) handles PCM resampling. The Python `Session.transcribe_vad()` docstring (per `_binding.py` summary) states: *"VAD + streaming assume 16 kHz mono PCM; resampling done in-place on Python side before ctypes calls."* Whether resampling is also performed transparently inside the C library is documented at the API surface level (the C functions take a `sample_rate` argument), but the algorithmic details (linear / sinc / polyphase) are not stated in the docs.

### 8.6 Stream vs file ingestion

Three streaming paths are documented in `docs/streaming.md`:

- **Pipe mode** (`--stream`): stdin piping (e.g. from ffmpeg) — sliding-window with `--stream-step` 3000 ms / `--stream-length` 10000 ms / `--stream-keep` 200 ms.
- **Microphone** (`--mic`): auto-detects `arecord` / `sox` / `ffmpeg` from `$PATH`; alternative is the library-level `Mic` class in the Python binding (miniaudio backend, runs on miniaudio's audio thread).
- **Continuous live** (`--live`): runs indefinitely, one transcript line per chunk; `--monitor` adds visual indicators.

Source: `crisp_audio/include/crisp_audio.h` (verbatim header text), `README.md` § *Audio formats*, `UPSTREAM.md` § *crispasr — `examples/ffmpeg-transcode.cpp` mp4-container handling*, `docs/streaming.md`, `python/crispasr/_binding.py` (summarized).

---

## 9. Configuration model

### GGUF metadata schema fields

The repo does not publish a single consolidated schema spec. Architecture-specific metadata fields are read by per-backend code in `src/<backend>.cpp` plus the shared GGUF loader in `src/core/gguf_loader.cpp`. Auto-detection from GGUF metadata is documented:

> *"Auto-detection support demands registering the architecture string in `crispasr_detect_backend_from_gguf()`."* (`docs/contributing.md`)

The factory function `crispasr_detect_backend_from_gguf(path)` reads an architecture string from the GGUF header to select which backend handles a given file. Specific GGUF metadata keys per architecture are not documented in `docs/`; they would need to be inferred from each backend's source.

### Runtime parameters (per `whisper_full_params` and per-backend equivalents)

From `include/crispasr.h`:

```c
enum whisper_sampling_strategy { CRISPASR_SAMPLING_GREEDY, CRISPASR_SAMPLING_BEAM_SEARCH };

struct whisper_full_params {
    enum whisper_sampling_strategy strategy;
    int n_threads;
    bool translate;
    /* ...80+ fields per WebFetch summary; specific field list not enumerated... */
};
```

The full field list of `whisper_full_params` is not enumerated by the WebFetch summary; the header is 35.3 KB and contains preserved whisper.cpp typedefs. The CLI flag set (from `README.md` § *CLI Reference - Common Flags*) maps these conceptually:

| Flag | Maps to |
| --- | --- |
| `-tp F` | sampling temperature |
| `-bs N` | beam search width |
| `--temperature-inc` | (not in CLI table but standard whisper.cpp param) |
| `--max-len N` | max segment length |
| `-l <lang>` / `-l auto` | language code or LID pre-step |
| `-am <gguf>` | CTC aligner GGUF path |
| `-tl <lang>` | target language for translation-capable backends |
| `-sl <lang>` | source language (Canary explicit-control mode) |
| `-ck N` | fallback chunk size (default 30 s) |
| `-ml N` | output granularity (1=word, 2=segment) |
| `--vad`, `--vad-model`, `--vad-threshold`, `--vad-min-speech-duration-ms`, `--vad-min-silence-duration-ms`, `--vad-speech-pad-ms` | Silero/FireRed/MarbleNet VAD parameters |
| `--lid-backend <name>` | LID provider (whisper / silero / ecapa / firered) |
| `--punc-model <name>` | Post-processing punctuation model |
| `--truecase-model <name>` | German truecasing |
| `--diarize` | Speaker diarization toggle |
| `--gpu-backend <name>` | GPU selection (cuda / metal / vulkan / cpu) |
| `-dev N` | Multi-GPU device pin |
| `--flash-attn` | Flash attention toggle |
| `--flush-after N` | Flush output after N segments |
| `--split-on-punct` | Split subtitles at punctuation |

### Initial-prompt biasing

The README mentions `--prompt` indirectly via the OpenAI-compatible HTTP API: `prompt` is one of the form fields on `POST /v1/audio/transcriptions` (per `docs/server.md` § *OpenAI-Compatible Audio*). Not explicitly listed in the `--list-backends` CLI flag table; whether all backends honor it is not stated.

### `-m auto` probe behavior

From `docs/cli.md` (verbatim): *"When you pass `-m auto` (or `-m default`), CrispASR downloads the default quantized model for the selected backend into `~/.cache/crispasr/` on first use."* The mapping is in `src/crispasr_model_registry.cpp` (36.2 KB). Example sizes per the doc: parakeet ~467 MB; voxtral4b ~3.3 GB.

Cache directory: `~/.cache/crispasr/` (or `cache_dir()` in the Python binding). Zombie-file detection per `docs/architecture.md`: *"`crispasr_cache.{h,cpp}` — Model downloads to `~/.cache/crispasr/` with zombie-file detection."*

### `-l auto` LID pre-step

From `docs/cli.md` (verbatim): *"Auto-detect the input language. Backends without native lang-detect (cohere, canary, granite, voxtral, voxtral4b) get it via the LID pre-step."* Default LID provider is `whisper` (whisper-tiny GGUF, 75 MB, 99 languages); selectable via `--lid-backend {whisper,silero,ecapa,firered}`.

JSON output (with `-oj -l auto`) includes `language_source` indicating which LID backend chose the language. Per the README:

```json
{
  "crispasr": {
    "backend": "cohere",
    "language": "en",
    "language_detected": "en",
    "language_confidence": 0.977,
    "language_source": "ecapa"
  },
  "transcription": [...]
}
```

Source: `include/crispasr.h` (summarized), `docs/cli.md`, `README.md` § *CLI Reference*, § *Language Identification (Pre-Transcription)*.

---

## 10. Per-backend capability flags

### CAP_* bitmask (from `examples/cli/crispasr_backend.h`, verbatim WebFetch)

19 capability bits enumerated:

| Flag | Comment |
| --- | --- |
| `CAP_TIMESTAMPS_NATIVE` | model produces segment timestamps natively |
| `CAP_TIMESTAMPS_CTC` | can use CTC aligner for timestamps |
| `CAP_WORD_TIMESTAMPS` | word-level timestamps available |
| `CAP_TOKEN_CONFIDENCE` | per-token probability |
| `CAP_LANGUAGE_DETECT` | auto language detection |
| `CAP_TRANSLATE` | speech translation |
| `CAP_DIARIZE` | speaker diarization |
| `CAP_GRAMMAR` | GBNF grammar constraints |
| `CAP_TEMPERATURE` | temperature/sampling control |
| `CAP_BEAM_SEARCH` | beam search |
| `CAP_FLASH_ATTN` | flash attention toggle |
| `CAP_PUNCTUATION_TOGGLE` | can enable/disable punctuation |
| `CAP_SRC_TGT_LANGUAGE` | separate source/target language (canary) |
| `CAP_AUTO_DOWNLOAD` | supports -m auto via HF hub |
| `CAP_PARALLEL_PROCESSORS` | whisper-style n_processors |
| `CAP_VAD_INTERNAL` | backend handles VAD internally (whisper) |
| `CAP_TTS` | text-to-speech synthesis |
| `CAP_VOICE_CLONING` | TTS: synthesise with --voice <reference.wav> |
| `CAP_PUNCTUATION_NATIVE` | backend already emits punctuation by default |

Capabilities are reported by each backend's `capabilities()` virtual method (returns `uint32_t` bitmask) on the `CrispasrBackend` base class.

### How consumers query capabilities

```c
crispasr_list_backends();             // enumerate compiled backends
crispasr_print_backend_matrix();      // human-readable
crispasr_print_backend_matrix_json(); // machine-readable
```

CLI: `crispasr --list-backends` and `crispasr --list-backends-json`. The latter feeds `tools/gen-feature-matrix.py` to generate `docs/feature-matrix.html` (24.7 KB interactive). `docs/feature-matrix.md` (4.6 KB) summarizes: *"All 42 backends compiled into the `crispasr` binary, with their declared capability bits."* (Note: 42 includes ASR + TTS + translation + auxiliary backends.)

### Known dispatch issue (SubtitleEdit#10775 referenced by the integration prompt)

SubtitleEdit issue #10775 ("CrispASR aligners", opened by @subof on 2026-05-04) reports that **the CTC aligner does not appear to be applied for the Canary backend** in the SubtitleEdit GUI flow, despite the user supplying `--aligner-model "...\canary-ctc-aligner-q8_0.gguf"`. Verbatim quote from the issue body:

> *"Unlike Crisp-ASR with Qwen, Canary doesn't offer to download the aligner. I manually downloaded the model, placed it in the model folder, and specified the path, but I received inaccurate subtitles. It seems that the alignment doesn't work for Canary."*

The issue references the CrispASR README section *"word-level timestamps via ctc alignment"*. The issue has **no comments** at research time. SubtitleEdit is a downstream consumer (third-party GUI), so the report is consistent with a possible dispatch bug in `crispasr_run.cpp` where Canary's native-timestamp path is taken even when `-am` is supplied — but this is not confirmed by any commit, comment, or test in the CrispASR repo itself, and the CrispASR contributing doc says `docs/cli.md` is the canonical source for the `-am`/aligner flag's semantics. SubtitleEdit may also be invoking the CLI with flag names that differ from CrispASR's (the SubtitleEdit issue uses `--aligner-model` while CrispASR docs use `-am`). See §25.

The CLI doc `docs/cli.md` describes the aligner: *"CTC aligner GGUF for word-level timestamps... particularly useful for LLM-based backends... Auto-download the aligner — Q4_K (~442 MB) lives in the registry."* — and elsewhere: *"The LLM-based backends (`qwen3`, `voxtral`, `voxtral4b`, `granite`) don't emit timestamps natively."* This phrasing suggests the aligner is *primarily* a remedy for backends that lack `CAP_TIMESTAMPS_NATIVE`; whether `-am` *forces* a re-alignment when a backend already has native timestamps is **not explicitly documented**. Issue #62 (closed 2026-05-04) was titled *"[FEATURE REQUEST] Option to use forced aligner even for backends that natively support timestamps"* — implying the previous behaviour was that `-am` was ignored when native timestamps were available, and an opt-in flag was added in v0.6.x.

Source: `examples/cli/crispasr_backend.h` (verbatim CAP_* list via WebFetch), `docs/feature-matrix.md`, `docs/feature-matrix.html`, `gh issue view 10775 --repo SubtitleEdit/subtitleedit`, `gh issue view 62 --repo CrispStrobe/CrispASR` (title only, surfaced in §22 issue sample).

---

## 11. Forced alignment subsystem

### NeMo-style CTC path

The repo description states: *"plus universal forced alignment via NeMo Forced Aligner-style CTC."* The library implementation is in `src/crispasr_aligner.cpp` (7.5 KB) + `src/crispasr_aligner.h` (2.2 KB), with the underlying CTC alignment primitive in `src/core/align.h` (2.1 KB) and `src/core/ctc.h` (4.8 KB). The CLI surface is `-am <aligner.gguf>` (sometimes referenced as `--aligner-model`).

Auxiliary CTC alignment models bundled per-backend:

- **Canary**: `canary-ctc-aligner-q8_0.gguf` (extracted from NVIDIA's `canary-1b-v2.nemo` tarball; `UPSTREAM.md` documents the extraction path as wishlist for upstream — NVIDIA does not ship the aux CTC model standalone). The extractor script is `models/convert-canary-ctc-to-gguf.py`. The HF readme is `hf_readmes/canary-ctc-aligner-GGUF.md`.
- **Qwen3-ForcedAligner**: GGUF aux model in the Qwen3 family. README acknowledges Qwen team's Qwen3 aligner: *"Qwen team (Alibaba) — Qwen3 ASR + TTS + aligner."*

### Dispatch logic (where the aligner is invoked)

The unified pipeline lives in `examples/cli/crispasr_run.cpp` (111.7 KB) — function `crispasr_run_backend()`. Per `docs/cli.md` and the README, the conceptual dispatch is:

1. Run primary backend → get segments (with or without native timestamps).
2. If `-am` is supplied **and** the backend lacks `CAP_WORD_TIMESTAMPS`/`CAP_TIMESTAMPS_NATIVE` granularity desired by the user → run CTC aligner against the backend's text output + audio → emit word-level timestamps.

The repo issue #62 (*"Option to use forced aligner even for backends that natively support timestamps"*, closed 2026-05-04T18:38:48Z) indicates the dispatch condition was extended in v0.6.x to honour `--force-aligner` even when the primary backend already has native timestamps. The v0.6.0 release notes confirm: *"Aligner `--force-aligner` flag and end-to-end smoke (#62)."*

### Available aligner GGUF models

Two are documented:

- `canary-ctc-aligner-q8_0.gguf` (per SubtitleEdit#10775 user's file path, and `hf_readmes/canary-ctc-aligner-GGUF.md`)
- Qwen3-ForcedAligner GGUF (per the credits list and `docs/cli.md`)

### Known limitations

`UPSTREAM.md` flags one: NVIDIA does not publish a standalone HF release of the Canary CTC alignment model; CrispASR extracts it from the `.nemo` tarball. If NeMo's internal layout changes, the converter must be updated.

Source: `src/crispasr_aligner.cpp/h`, `src/core/align.h`, `src/core/ctc.h`, `docs/cli.md` § *Aligner Model Flag*, `README.md` § *Credits*, `UPSTREAM.md` § *NeMo Forced Aligner — official ONNX export of the auxiliary CTC model*, GitHub issue #62.

---

## 12. LID subsystem

### Providers + sizes (per `README.md` § *Language Identification (Pre-Transcription)*)

| Provider | Mechanism | Size | Languages | License |
| --- | --- | --- | --- | --- |
| `whisper` (default) | whisper-tiny GGUF | 75 MB | 99 | MIT |
| `silero` | native GGUF | 16 MB | 95 | (per Silero repo; not stated inline) |
| `ecapa` (README notes "recommended") | ECAPA-TDNN | 43-40 MB | 107 / 45 | (per HuggingFace card) |
| `firered` | Conformer+Transformer | 544 MB | 120 | (per HuggingFace card) |

CLI: `-l auto` + `--lid-backend <provider>`. The output JSON exposes `language_source` (e.g. `"ecapa"`), `language_detected`, and `language_confidence`.

### Text LID providers (separate from audio LID)

Three text-classification LID models are available via `crispasr-lid` CLI:

| Model | Labels | Size (F16) | License |
| --- | --- | --- | --- |
| CLD3 | 109 (ISO 639-1) | 440 KB | Apache-2.0 |
| GlotLID-V3 | 2102 (ISO 639-3 + script) | 250 MB | Apache-2.0 |
| LID-176 | 176 (ISO 639-1) | 63 MB | **CC-BY-SA-3.0** (viral; README warns redistributors inherit ShareAlike) |

Implementations: `src/lid_cld3.cpp`, `src/lid_fasttext.cpp`, `src/silero_lid.cpp`, `src/ecapa_lid.cpp`, `src/firered_lid.cpp`. Dispatch: `src/text_lid_dispatch.cpp`. Post-ASR text routing via `--lid-on-transcript auto`.

### Confidence reporting

Audio LID: numeric probability returned in the JSON `language_confidence` field. Text LID: depends on model — CLD3 reports per-label probability, GlotLID-V3 supports `-k N` top-k labels.

### Supported languages — Japanese specifically

The `whisper` LID provider covers Japanese (`ja`) as one of its 99 languages (per upstream Whisper documentation referenced via the README's whisper credit). The `ecapa` LID model covers 107 languages in its larger variant; the README does not enumerate which 107. `firered` LID covers 120 languages; again no enumeration in the README. Japanese coverage for `silero` and `ecapa` and `firered` is not explicitly enumerated in the primary sources.

Source: `README.md` § *Language Identification (Pre-Transcription)*, § *Text Language Identification (Standalone)*, `src/` per-file inventory.

---

## 13. Output formats

### File output formats (CLI flags)

| Flag | Format |
| --- | --- |
| `-osrt` | SRT |
| `-ovtt` | WebVTT |
| `-otxt` | Plain text |
| `-oj` | JSON |
| `-ojf` | JSON-full (likely includes per-token data; not explicitly defined in README excerpt) |
| `-ocsv` | CSV |
| `-olrc` | LRC (timed lyrics) |

### Timestamps per backend

- **Native segment-level**: whisper, parakeet, canary, cohere, kyutai-stt
- **CTC-aligned segment-level**: most other backends, when `-am` is provided
- **Native word-level**: parakeet (per README *Backend Selection Guide*: "Multilingual + word timestamps: parakeet")
- **Word-level via `-am`**: most backends

Per `docs/cli.md`: *"The LLM-based backends (`qwen3`, `voxtral`, `voxtral4b`, `granite`) don't emit timestamps natively."* These require `-am` for any timestamp output.

### JSON output shape (verbatim from README)

```json
{
  "crispasr": {
    "backend": "cohere",
    "language": "en",
    "language_detected": "en",
    "language_confidence": 0.977,
    "language_source": "ecapa"
  },
  "transcription": [...]
}
```

The exact shape of the `transcription` array elements (which timestamp keys, confidence keys, word arrays) is not enumerated in the public README. `-oj` vs `-ojf` distinction not detailed.

### Streaming event surface (`--stream --stream-json`)

Per `docs/streaming.md`, structured JSON-Lines events with three types:

| Event type | Trigger | Fields |
| --- | --- | --- |
| `partial` | New text in open utterance | `utterance_id`, `text`, `t0`, `t1` |
| `final` | Silence ≥ `--stream-final-on-silence-ms` (default 800 ms) | `utterance_id`, `text`, `t0`, `t1` |
| `silence` | No speech detected in step | `t` |

Guarantees: *"Once an `utterance_id` finalizes, its audio is bookmarked and never re-opens a later `utterance_id`."* Finalization mode choice (`--stream-final-mode`): `redecode` (re-runs backend on buffered utterance PCM) or `prefix` (longest-common-prefix accumulator).

### `--flush-after N`

Per the README CLI table: *"Flush output after N segments"*. Used for progressive subtitle output (e.g. PotPlayer, real-time SRT readers). Example: `crispasr --backend parakeet -m parakeet.gguf --vad --flush-after 1 -osrt -f long_audio.wav` produces SRT entries that appear progressively as each VAD segment finishes.

Source: `README.md` § *CLI Reference - Common Flags*, § *JSON Output with Language Detection*, § *Progressive Subtitle Output*; `docs/streaming.md`, `docs/cli.md`.

---

## 14. HTTP server mode

### Endpoints (from `docs/server.md`)

**Native API:**

| Endpoint | Verb | Notes |
| --- | --- | --- |
| `/inference` | `POST` | Multipart audio upload; returns `{"text": "...", "segments": [...], "backend": "...", "duration": ...}` |
| `/load` | `POST` | Multipart `model=path/to/model.gguf` — hot-swap |
| `/health` | `GET` | Public, no auth: `{"status": "ok", "backend": "..."}` |
| `/backends` | `GET` | Lists available + active backend |

**OpenAI-compatible (auth required when keys enabled):**

| Endpoint | Verb | Notes |
| --- | --- | --- |
| `/v1/audio/transcriptions` | `POST` | Form fields: `file` (required), `model` (ignored), `language`, `prompt`, `response_format`, `temperature`; response formats: `json` (default), `verbose_json`, `text`, `srt`, `vtt` |
| `/v1/models` | `GET` | Returns loaded model in OpenAI schema |
| `/v1/audio/speech` | `POST` | JSON body: `input`, `voice`, `instructions`, `speed` (0.25–4.0), `response_format`, `model` (ignored); output formats: `wav` (default 24 kHz mono int16), `pcm` (raw int16 LE), `f32` (raw float32) |
| `/v1/voices` | `GET` | Lists voices from `--voice-dir` |

### Lifecycle

> *"The server loads the model once at startup and keeps it in memory. Subsequent `/inference` requests reuse the loaded model with no reload overhead."* (`docs/server.md`)

- One model loaded persistently.
- Mutex-serialized request handling (single-model, single-stream).
- Hot-swap via `POST /load` replaces in-memory model.
- TTS long-form input auto-chunks on sentence boundaries (`. ! ? 。 ।`) with 200 ms silence padding.
- TTS speed applied server-side via linear-interpolation resampling post-synthesis.

### Authentication

> *"Set `CRISPASR_API_KEYS=key-one,key-two` (env var, not CLI args). Protected endpoints accept `Authorization: Bearer <key>` or `X-API-Key: <key>`. `/health` remains public."*

### Network exposure / CORS

- Default bind: localhost.
- `--host 0.0.0.0` for remote.
- `--cors-origin '*'` (dev) or `--cors-origin 'https://app.example.com'` (prod).
- Docker Compose templates provided: `docker-compose.yml` (1.1 KB) + `docker-compose.cuda.yml` (171 B). Env-var overrides documented: `CRISPASR_MODEL`, `CRISPASR_BACKEND`, `CRISPASR_LANGUAGE`, `CRISPASR_API_KEYS`.

### Implementation

`examples/server/server.cpp` (54.7 KB) + `ws_stream.cpp` (13.7 KB) — yhirose/cpp-httplib embedded as `examples/server/httplib.h` (354.7 KB). The `ws_stream.*` files suggest a websocket streaming path (not enumerated in `docs/server.md`).

Breaking change in v0.6.0: TTS `response_format=pcm` was renamed to `response_format=f32` (the old `pcm` returned 24-bit float anyway and was misleading; servers now reject `pcm` with HTTP 400 + helpful message).

Source: `docs/server.md`, `examples/server/server.cpp`, `gh api .../releases/tags/v0.6.0` body.

---

## 15. Python binding API (the ACTUAL surface)

### Module + install path

- Module name: `crispasr`
- PyPI package name: `crispasr` (per `python/pyproject.toml`, version `0.5.7` at research time; the `__init__.py` declares `__version__ = "0.4.9"` — **the two values are out of sync** — see §25).
- Install: `pip install crispasr` (pure-Python wheel). **The wheel does not bundle native libs.** Users must separately install `libcrispasr`.
- Native lib discovery: `_binding._find_lib()` probes env var `CRISPASR_LIB_PATH` → Homebrew / `/usr/local` / `/usr` → repo-relative `build/` → bare filename for `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` resolution. Accepts both `libcrispasr.{so,dylib,dll}` and legacy `libwhisper.{so,dylib,dll}` filenames.

### Public surface (from `python/crispasr/__init__.py`, verbatim)

```python
from ._binding import (
    AlignedWord, CrispASR, DiarizeMethod, DiarizeSegment, KokoroResolved,
    LidMethod, LidResult, Mic, RegistryEntry, Segment, Session,
    SessionSegment, SessionWord,
    align_words, cache_dir, cache_ensure_file, detect_language_pcm,
    diarize_segments, kokoro_resolve_for_lang, list_known_models,
    mic_default_device_name, registry_lookup, registry_lookup_by_filename,
)
__version__ = "0.4.9"
```

### Classes (signatures from `_binding.py`, summarised)

```python
class CrispASR:
    """Speech-to-text model using ggml inference (Whisper-compatible)."""
    def __init__(self, model_path: str, lib_path: Optional[str] = None,
                 helpers_lib_path: Optional[str] = None): ...
    def transcribe(self, audio_path: str, language: str = "auto",
                   strategy: int = CRISPASR_SAMPLING_GREEDY) -> List[Segment]: ...
    def transcribe_pcm(self, pcm: np.ndarray, sample_rate: int = 16000,
                       language: str = "auto", strategy: int = ...,
                       vad: bool = False, vad_model_path: Optional[str] = None,
                       vad_threshold: float = 0.5, ...) -> List[Segment]: ...
    @property
    def detected_language(self) -> str: ...
    def close(self): ...
    def __enter__(self) / def __exit__(self, *args): ...

class Session:
    """Backend-agnostic transcription session over any CrispASR GGUF."""
    def __init__(self, model_path: str, lib_path: Optional[str] = None,
                 n_threads: int = 4, backend: Optional[str] = None): ...
    @staticmethod
    def available_backends(lib_path: Optional[str] = None) -> List[str]: ...
    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000,
                   *, language: Optional[str] = None) -> List[SessionSegment]: ...
    def transcribe_vad(self, pcm: np.ndarray, vad_model_path: str,
                       *, sample_rate: int = 16000, threshold: float = 0.5,
                       min_speech_duration_ms: int = 250, ...) -> List[SessionSegment]: ...
    def stream_open(self, *, step_ms: int = 3000, length_ms: int = 10000,
                    keep_ms: int = 200, language: str = "",
                    translate: bool = False, live: bool = False) -> "_Stream": ...
    # TTS
    def set_codec_path(self, path: str) -> None
    def set_voice(self, path: str, ref_text: Optional[str] = None) -> None
    def set_speaker_name(self, name: str) -> None
    def set_instruct(self, instruct: str) -> None
    def synthesize(self, text: str) -> np.ndarray
    # Session state
    def set_source_language(self, lang: str) -> None
    def set_target_language(self, lang: str) -> None
    def set_translate(self, enable: bool) -> None
    def set_temperature(self, temperature: float, seed: int = 0) -> None
    def close(self) / __enter__ / __exit__

class Mic:
    """Library-level microphone capture (miniaudio backend).
    Callback runs on audio thread — keep short and non-blocking."""
    _CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float),
                                ctypes.c_int, ctypes.c_void_p)
    def __init__(self, *, sample_rate: int = 16000, channels: int = 1,
                 callback=None, lib_path: Optional[str] = None): ...
    def start(self) -> None / stop() / close() / __enter__ / __exit__
```

### Dataclasses

`Segment` (legacy whisper format), `SessionSegment`, `SessionWord`, `AlignedWord`, `DiarizeSegment`, `RegistryEntry`, `LidResult`, `KokoroResolved`.

### Enums

`DiarizeMethod`, `LidMethod`.

### Standalone functions

```python
align_words(...)                 # forced alignment
cache_dir() -> Path              # returns ~/.cache/crispasr/
cache_ensure_file(...)           # download + cache a HF file
detect_language_pcm(pcm, ...)    # audio LID
diarize_segments(...)            # speaker diarization
kokoro_resolve_for_lang(lang)    # Kokoro TTS voice selection
list_known_models() -> List      # registry enumeration
mic_default_device_name()        # default capture device
registry_lookup(backend_name)    # → RegistryEntry
registry_lookup_by_filename(name)
```

### Sync vs async, callback support

- **Sync only** — `def transcribe(...)`, `def synthesize(...)`. No `async def`, no `asyncio` integration in the public surface.
- **Callbacks**: `Mic` accepts a `callback=` parameter of type `ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p)`. **The callback runs on miniaudio's audio thread**, per the docstring: *"keep short and non-blocking."*
- **Generators / yield**: `_binding.py` does not expose a streaming generator in the high-level surface per the WebFetch summary. A `_Stream` private class is returned from `Session.stream_open()` but its iteration interface is not described in the surface summary.

### Maturity tier

`pyproject.toml`: `"Development Status :: 4 - Beta"`. PyPI version `0.5.7` (per pyproject), in-tree `__version__` `0.4.9` (per __init__.py) — version skew between the two files. Python compatibility: `requires-python = ">=3.8"`, classifiers list Python 3.8–3.13.

### Internal FFI mechanism

**ctypes** — the binding is a pure Python ctypes wrapper. Imports observed: `ctypes`, `os`, `platform`, `wave`, `dataclasses`, `pathlib`, `typing`, `numpy`. No `pybind11`, `cffi`, or `cython` involvement. A separate compiled helper (`libcrispasr_helpers.so`, source `python/crispasr/_helpers.c` — 740 bytes) is used as a thin shim to work around by-value struct passing.

### Backward compatibility

Per the WebFetch summary: *"Dynamic symbol lookup: Many newer features (0.4.2+) guarded by `hasattr(lib, "symbol")` to ensure backward compatibility with older dylib builds."* This means Python wheels can be used against older `libcrispasr.so` versions, with feature degradation for newer APIs.

### Error reporting

Not enumerated in the WebFetch summary. The C API returns `int` status codes (per whisper.cpp convention preserved in `include/crispasr.h`); the Python wrapper would translate these to exceptions, but the exception class hierarchy is not enumerated in the summary.

Source: `python/crispasr/__init__.py` (verbatim), `python/crispasr/_binding.py` (signature summary), `python/pyproject.toml` (verbatim), `python/README.md` (summary).

---

## 16. Threading and GIL behavior at the FFI boundary

### Stated thread-safety contracts

- **`crisp_audio` library** (`crisp_audio/include/crisp_audio.h`, verbatim): *"All functions are thread-unsafe per context — wrap with a mutex in multi-threaded callers."* Per-context, not per-library.
- **Whisper-style C API** (`include/crispasr.h`): *"thread-safe as long as the sample whisper_context is not used by multiple threads concurrently."* (WebFetch summary quoting a header comment.) Same model: per-context mutex required.
- **Server mode**: mutex-serialized request handling per `docs/server.md` — single in-memory model, requests run in series.

### GIL release in the Python binding

Per the WebFetch inspection of `_binding.py`: *"No explicit GIL release: Code does not show `Py_BEGIN_ALLOW_THREADS` or `release_gil()` decorators."*

`ctypes` calls **do release the GIL automatically** for function calls into native code by default (this is standard CPython behaviour for `ctypes.CDLL` calls). Whether explicit annotations would have been necessary depends on whether the wrapper holds Python objects across the call. The summary indicates the wrapper does not have explicit GIL-release annotations.

### Multi-threaded inference support

- **`n_threads` parameter**: `Session(..., n_threads: int = 4)`. The underlying ggml backend uses OpenMP (if compiled with `-DGGML_OPENMP=ON`) for intra-op parallelism.
- **Whisper's `whisper_full_parallel(...)`**: declared in `include/crispasr.h` — runs N parallel processors over chunked audio (legacy whisper.cpp feature). Capability flag: `CAP_PARALLEL_PROCESSORS`.

### Streaming vs synchronous

- File-based `transcribe()` is synchronous; returns after full audio is processed.
- `Session.stream_open()` returns a `_Stream` object — internal iteration interface not enumerated in the summary, but the CLI `--stream` mode implies chunked block processing.
- Mic callback runs on a separate audio thread (miniaudio).

### Callback boundary

The `Mic._CB_TYPE` is `ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p)`. The wrapper "copies buffer to numpy array before callback dispatch to avoid use-after-free."

Source: `crisp_audio/include/crisp_audio.h` (verbatim), `include/crispasr.h` (WebFetch summary), `python/crispasr/_binding.py` (WebFetch summary), `docs/server.md`.

---

## 17. Build composition + distribution artifacts

### Dynamic vs static linking

`CMakeLists.txt`: `option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})` where `BUILD_SHARED_LIBS_DEFAULT` is `OFF` on Emscripten and MinGW, `ON` elsewhere. The `crispasr` library target is shared-by-default on Linux / macOS / non-MinGW Windows, static on WASM and MinGW.

### Per-platform packaging (`gh api .../releases/tags/v0.6.6/assets`)

For the latest release (v0.6.6):

| Platform | CLI artifact | Library artifact |
| --- | --- | --- |
| Linux x86_64 (AVX2) | `crispasr-linux-x86_64.tar.gz` (5.4 MB) | `libcrispasr-linux-x86_64.tar.gz` (3.8 MB) |
| Linux x86_64 (AVX-512) | `crispasr-linux-x86_64-avx512.tar.gz` (5.3 MB) | `libcrispasr-linux-x86_64-avx512.tar.gz` (3.8 MB) |
| Linux x86_64 CUDA | — | `libcrispasr-linux-x86_64-cuda.tar.gz` (100.6 MB) |
| Linux ARM64 | `crispasr-linux-arm64.tar.gz` (4.9 MB) | `libcrispasr-linux-arm64.tar.gz` (3.5 MB) |
| macOS (universal) | `crispasr-macos.tar.gz` (3.7 MB) | `libcrispasr-macos-arm64.tar.gz` (3.0 MB) |
| Windows x86_64 CPU | `crispasr-windows-x86_64-cpu.zip` (3.1 MB) | `libcrispasr-windows-x86_64.tar.gz` (11.8 MB) |
| Windows x86_64 CPU legacy | `crispasr-windows-x86_64-cpu-legacy.zip` (3.1 MB) | `libcrispasr-windows-x86_64-cpu-legacy.tar.gz` (11.7 MB) |
| Windows x86_64 CUDA | `crispasr-windows-x86_64-cuda.zip` (683.7 MB) | `libcrispasr-windows-x86_64-cuda.tar.gz` (789.2 MB) |
| Windows x86_64 Vulkan | `crispasr-windows-x86_64-vulkan.zip` (24.7 MB) | `libcrispasr-windows-x86_64-vulkan.tar.gz` (48.0 MB) |
| Android ARM64 | `crispasr-android-arm64-v8a.tar.gz` (21.7 MB) | — |
| Linux x86_64 Python | `crispasr-python-linux-x86_64.tar.gz` (3.7 MB) | — |
| Linux ARM64 Python | `crispasr-python-linux-arm64.tar.gz` (3.5 MB) | — |

The `crispasr-python-*.tar.gz` artifacts package Python wheels alongside the native lib for Linux. No Windows or macOS Python wheel artifact is present at v0.6.6.

### iOS / xcframework

v0.6.0 (2026-05-05) shipped `crispasr-v0.6.0-xcframework.zip` (139.9 MB). v0.6.6 does not include an xcframework asset — the xcframework appears to be built less frequently. `build-xcframework.sh` (24.7 KB) exists in the repo root.

### Code signing posture (macOS specifically)

Not stated in the docs I fetched. The macOS / iOS artifacts ship as `.tar.gz` and `.zip`, which do not enforce signing. `docs/install.md` (per WebFetch summary): *"Prebuilt binaries from some model cards require glibc 2.38, causing failures on systems with older C libraries like Ubuntu 22.04's glibc 2.35."* — points to Linux ABI mismatches but does not address macOS signing. See §25.

### Asset download counts (v0.6.6 at research time)

Most-downloaded:
- `crispasr-windows-x86_64-cpu.zip` — 9 downloads
- `crispasr-windows-x86_64-vulkan.zip` — 8
- `crispasr-linux-x86_64.tar.gz` — 4
- `crispasr-windows-x86_64-cuda.zip` — 4
- `crispasr-python-linux-x86_64.tar.gz` — 3
- `crispasr-macos.tar.gz` — 3
- `libcrispasr-windows-x86_64-cuda.tar.gz` — 3

(Counts are low because v0.6.6 was published 2026-05-12, ~24 hours before research time.)

Source: `gh api .../releases/tags/v0.6.6`, `gh api .../releases/tags/v0.6.0`, `CMakeLists.txt`, `docs/install.md`.

---

## 18. Backend catalog

Two enumerations exist in primary sources:

- The `README.md` § *Supported ASR backends (24 Total)* table (24 ASR backends).
- The `docs/feature-matrix.md` (4.6 KB): *"All 42 backends compiled into the `crispasr` binary, with their declared capability bits."* — 42 includes ASR + TTS + translation + auxiliary (LID, VAD, punctuation, truecasing) backends.

The following catalogue follows the README's ASR enumeration and adds verbatim README detail where available.

### 18.1 whisper

- **CLI identifier**: `whisper` (default if `-m <whisper-ggml.bin>`)
- **Representative GGUF**: `ggml-base.en.bin`, etc., downloaded via `./models/download-ggml-model.sh` (upstream whisper.cpp script)
- **Architecture**: Encoder-decoder transformer (preserved unchanged from upstream whisper.cpp — README: *"Whisper itself intentionally remains outside `src/core/` migration, kept byte-identical to upstream whisper.cpp as a regression test gate."*)
- **Languages**: 99
- **Native word timestamps**: Yes (via DTW / token-time alignment)
- **JAV/Japanese relevance**: Japanese (`ja`) is one of the 99 languages
- **License**: MIT
- **Capability flag highlights**: `CAP_TIMESTAMPS_NATIVE`, `CAP_WORD_TIMESTAMPS`, `CAP_LANGUAGE_DETECT`, `CAP_TRANSLATE`, `CAP_GRAMMAR`, `CAP_BEAM_SEARCH`, `CAP_VAD_INTERNAL`, `CAP_PARALLEL_PROCESSORS`

### 18.2 parakeet (NVIDIA NeMo Parakeet TDT)

- **CLI**: `--backend parakeet` (or auto-detect from GGUF)
- **Representative GGUF**: `parakeet-tdt-0.6b-v3-q4_k.gguf` (~467 MB per `docs/cli.md`) at `https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF`
- **Architecture**: FastConformer encoder + Transducer Decoder Time-Synchronous (TDT)
- **Languages**: 25 European languages (per README *Supported ASR backends* table); auto-detect
- **Native word timestamps**: Yes (per README *Backend Selection Guide*: "Multilingual + word timestamps: parakeet")
- **JAV/Japanese relevance**: README says 25 EU languages → Japanese not covered

### 18.3 canary (NVIDIA Canary 1B v2)

- **CLI**: `--backend canary`
- **Representative GGUF**: `canary-1b-v2-q5_0.gguf` (size not given inline); aligner aux: `canary-ctc-aligner-q8_0.gguf` (Q4_K ~442 MB per `docs/cli.md`)
- **Architecture**: FastConformer encoder + Transformer decoder
- **Languages**: README emphasizes "Explicit language control" — supports `-sl` (source) and `-tl` (target) explicitly. Capability: `CAP_SRC_TGT_LANGUAGE`
- **Native word timestamps**: Native segment-level; word-level via the bundled CTC aligner
- **JAV/Japanese relevance**: Canary-1B-v2 base language set is English / German / French / Spanish / Italian / Portuguese (per NVIDIA Canary card); Japanese coverage not asserted in primary sources

### 18.4 cohere (Cohere Transcribe 03-2026)

- **CLI**: `--backend cohere`
- **Representative GGUF**: `cohere-transcribe-q5_0.gguf` (size not given inline)
- **Architecture**: Conformer encoder + Transformer decoder
- **Native word timestamps**: Native (segment)
- **JAV/Japanese relevance**: README *Backend Selection Guide* notes *"Lowest English WER"* — primary positioning is English. Issue #67 (closed) reported *"Cohere is consistently dropping Japanese characters"*; the fix in v0.6.0 release notes mentions *"cohere SentencePiece byte-fallback decode"* — meaning Japanese characters were being lost in tokenizer decode prior to v0.6.0. Japanese support is partial; reporter exn251 raised the issue.

### 18.5 granite / granite_speech / granite_nle (IBM Granite Speech)

- **CLI**: `--backend granite`
- **Architecture**: Conformer encoder + Q-Former + LLM (Granite series, IBM)
- **Variants**: granite-speech-3.2-8b → granite-4.0-1b (per README credits); `granite_nle.cpp` is a separate "NLE" variant (Non-Linear Embedding?)
- **Native timestamps**: No (LLM-based; needs `-am` for word-level)
- **License**: Apache (per README *Backend Selection Guide*)

### 18.6 voxtral (Mistral Voxtral Mini 3B)

- **CLI**: `--backend voxtral`
- **Architecture**: Whisper encoder + Mistral LLM
- **Languages**: 8 (per README)
- **Native timestamps**: No (LLM-based)
- **JAV/Japanese relevance**: README *"8 languages, speech-LLM hybrid"* — Japanese coverage not specifically called out; per Mistral Voxtral docs the 8 are English, French, Spanish, German, Italian, Portuguese, Dutch, Hindi (this is from the Voxtral model card, not the CrispASR README).

### 18.7 voxtral4b (Voxtral 4B Realtime)

- **CLI**: `--backend voxtral4b`
- **Architecture**: Causal encoder + 3.4B LLM
- **Languages**: 13
- **Realtime streaming**: Native streaming (per *Backend Selection Guide*: "Realtime streaming: voxtral4b")
- **Native timestamps**: No

### 18.8 qwen3 (Alibaba Qwen3-ASR)

- **CLI**: `--backend qwen3`
- **Representative GGUF**: `-m auto` resolves to a default Qwen3-ASR quantization (size not given inline)
- **Architecture**: Whisper encoder + Qwen3 LLM
- **Languages**: 30 + 22 Chinese dialects (per README)
- **Native timestamps**: No (LLM-based; needs `-am` via Qwen3-ForcedAligner)
- **JAV/Japanese relevance**: Per Qwen team's own documentation (Qwen3-ASR repo card found via WebSearch), Qwen3-ASR-1.7B and Qwen3-ASR-0.6B finely support 30 languages including Japanese (`ja`). The CrispASR README repeats the 30-language claim. This makes Qwen3 the most explicitly Japanese-capable backend in the catalog.

### 18.9 wav2vec2 (Meta Wav2Vec2)

- **CLI**: `--backend wav2vec2` (or auto)
- **Representative GGUF**: `wav2vec2-xlsr-en-q4_k.gguf` (`https://huggingface.co/cstr/wav2vec2-large-xlsr-53-english-GGUF`); also a German variant convertable via `models/convert-wav2vec2-to-gguf.py`
- **Architecture**: CNN + transformer + CTC
- **Native timestamps**: CTC-aligned (segment / word via aligner)
- **Languages**: Any HuggingFace `Wav2Vec2ForCTC` model — depends on which checkpoint is converted
- **JAV/Japanese relevance**: A Japanese-finetuned wav2vec2 checkpoint (e.g. `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`) could be converted via the conversion script, per the README's German example. Not bundled as a default. Native lacking punctuation (`CAP_PUNCTUATION_NATIVE` not set) — needs `--punc-model`.

### 18.10 moonshine (UsefulSensors Moonshine)

- **CLI**: `--backend moonshine`
- **Variants**: tiny, base; German fine-tunes `moonshine-tiny-de-fidoriel-GGUF`, `moonshine-tiny-de-dattazigzag-GGUF`, `moonshine-base-de-fidoriel-GGUF` (found via WebFetch of `huggingface.co/cstr`, sizes 27.1 MB / 27.1 MB / 61.5 MB).
- **Architecture**: Conv stem + 6L encoder + 6L decoder
- **Native timestamps**: Native (segment); word-level via aligner
- **JAV/Japanese relevance**: README says "Smallest/fastest option"; German fine-tunes are CC-BY-NC-SA-4.0 (viral non-commercial). Japanese fine-tunes not bundled in `cstr/` org per the partial HF profile inspection.

### 18.11 omniasr / omniasr-llm (Meta MMS / OmniASR)

- **CLI**: `--backend omniasr`
- **Architecture**: wav2vec2 CNN + transformer + CTC (omniasr); omniasr-llm uses an LLM head
- **Languages**: **1600+** (per README — among the highest in the catalog)
- **Native timestamps**: CTC (segment)
- **JAV/Japanese relevance**: 1600+ languages implies Japanese is covered; the underlying Meta MMS model card confirms Japanese support.

### 18.12 vibevoice (vibevoice-asr / vibevoice-tts)

- **CLI**: `--backend vibevoice-tts` / vibevoice ASR via `vibevoice-asr-GGUF` (cstr/vibevoice-asr-GGUF, 8B per HF profile)
- **Architecture**: σ-VAE ConvNeXt + Qwen2.5-7B
- **Languages**: 50+
- **Dual ASR/TTS**: Yes (`CAP_TTS` + ASR)
- **Native timestamps**: No (LLM-based)
- **Issue context**: #74 (open) reports "1.5B WAV-clone path produces generic voice"; #78 (merged) "Fix vibevoice 1.5B tts voice cloning, CFG, and WAV parsing (Addresses #74)" — partial fix landed.

### 18.13 glm-asr / glm-asr-nano

- **CLI**: `--backend glm-asr`
- **Architecture**: Whisper encoder + Llama LLM
- **Languages**: 17 including Mandarin (per README); README *Backend Selection Guide* lists glm-asr under "Mandarin + dialects"
- **JAV/Japanese relevance**: README doesn't specifically call out Japanese; 17 languages including Mandarin doesn't guarantee Japanese coverage.

### 18.14 kyutai-stt

- **CLI**: `--backend kyutai-stt`
- **Architecture**: Mimi codec + causal LM
- **Languages**: English / French
- **License**: MIT (per README *Backend Selection Guide*)
- **Native timestamps**: Yes (native)
- **JAV/Japanese relevance**: English/French only — not Japanese-capable.

### 18.15 firered-asr (FireRedASR2-AED)

- **CLI**: `--backend firered-asr`
- **Architecture**: Conformer encoder + CTC + beam search
- **Languages**: Mandarin / English + 20+ Chinese dialects
- **Native timestamps**: CTC
- **JAV/Japanese relevance**: Mandarin/English/dialects — no Japanese support.
- Companion models: `fireredpunc` (Apache-2.0 punctuation, Chinese+English), `firered-vad` (DFSMN VAD), `firered-lid` (Conformer+Transformer LID, 120 languages).

### 18.16 gemma4-e2b (Google Gemma 4 Audio Encoder 2B)

- **CLI**: `--backend gemma4-e2b`
- **Architecture**: USM Conformer encoder + Gemma4 LLM
- **Languages**: 140+ (per README — broadly multilingual)
- **JAV/Japanese relevance**: 140+ languages — Japanese coverage likely; not explicitly enumerated by CrispASR doc.
- **Auth**: `HF_TOKEN=hf_xxx` required for gated Gemma4 model download (per README *Debugging & Profiling*).

### 18.17 mimo-asr (Xiaomi MiMo)

- **CLI**: `--backend mimo-asr`
- **Architecture**: 6L transformer + 36L Qwen2 + RVQ
- **Languages**: Mandarin dialects + English
- **JAV/Japanese relevance**: Mandarin/English only.

### 18.18 fastconformer-ctc / fc-ctc

- **CLI**: `--backend fastconformer-ctc`
- **Architecture**: 24L / 42L FastConformer + CTC
- **Languages**: 80-mel feature; English-focused
- **JAV/Japanese relevance**: English-focused.

### 18.19 data2vec-audio

- **CLI**: presumably `--backend data2vec` (not enumerated in README CLI examples)
- **Size**: 79 MB Q4_K
- **License**: Apache-2.0
- **Languages**: English
- **JAV/Japanese relevance**: English only.

### 18.20 hubert

- **CLI**: presumably `--backend hubert`
- **Size**: 212 MB Q4_K
- **License**: Apache-2.0
- **Languages**: English

### 18.21 glotlid

- README lists this as a backend (apparently "GLM-ASR-Nano variant"), 17 languages. Possibly internal LID/labelling rather than full ASR.

### 18.22 moonshine-streaming

- **Architecture**: Sliding-window encoder + AR
- Native streaming.

### 18.23 distil-whisper

- **Architecture**: 32L encoder + 2L decoder
- **Speed**: 6.3× faster
- **Languages**: English only

### Summary of Japanese-relevant ASR backends

Based on primary-source enumeration:

| Backend | Japanese coverage (per primary source) | Notes |
| --- | --- | --- |
| whisper | Yes (1 of 99 languages, MIT, well-documented) | Confirmed via README + upstream whisper.cpp |
| qwen3 | **Yes — explicitly stated** (Qwen team docs, 30 langs, ja) | LLM-based, no native timestamps |
| omniasr | Likely (1600+ langs via MMS) | CTC backbone |
| gemma4-e2b | Likely (140+ langs) | Gated model — HF_TOKEN required |
| voxtral | Maybe (8 langs, Japanese not in canonical Voxtral 8) | |
| wav2vec2 | Possible via custom HF Japanese-finetuned checkpoint conversion | Not bundled |
| vibevoice | Possibly (50+ langs) | Not explicitly enumerated |

Source: `README.md` § *Supported ASR backends*, § *Backend Selection Guide*, § *Quick Start Examples*, § *Credits*; `huggingface.co/cstr` partial inventory; WebSearch result on Qwen3-ASR Japanese support; GitHub issues #67, #74, #78.

---

## 19. TTS catalog

### Supported TTS engines (5, per `README.md` § *Text-to-Speech Backends*)

| Backend | CLI identifier | Architecture | Languages | Notable |
| --- | --- | --- | --- | --- |
| vibevoice-tts | `--backend vibevoice-tts` | σ-VAE ConvNeXt + Qwen2.5-7B | 50+ | Voice cloning (via reference WAV); DPM-Solver++; English+Chinese in README, broader per HF card |
| qwen3-tts | `--backend qwen3-tts` / `qwen3-tts-customvoice` | 3 model sizes 0.6B / 1.7B (per README) | Multilingual | Voice design; auto-download via `-m auto` |
| kokoro | `--backend kokoro` | 82M, per-voice GGUF | 9 languages including German | Smallest |
| orpheus | `--backend orpheus` | Llama-3.2-3B + SNAC codec | English, German | Multi-speaker via `--speaker speaker_N` |
| chatterbox | `--backend chatterbox` (+ siblings) | T3 AR (Llama / GPT-2) + S3Gen flow-matching | English, German, Arabic | Two-GGUF runtime; ~150–200 KB per-voice GGUF |
| indextts | `--backend indextts` | GPT-2 AR + voice cloning | Chinese, English | `HANDOVER_INDEXTTS.md` (4.4 KB) documents the implementation handover |

Counts: README header says "5 TTS engines" but the table enumerates 6 (kokoro, qwen3-tts, vibevoice-tts, orpheus, chatterbox, indextts). The discrepancy is likely because chatterbox + siblings are counted as one engine family.

### TTS HTTP endpoint behaviour

(See §14.) `/v1/audio/speech` produces WAV / raw int16 LE PCM / raw float32. Speed 0.25–4.0. Voice selection from `--voice-dir`. Voice-cloning capability surfaced via `CAP_VOICE_CLONING`.

Source: `README.md` § *Text-to-Speech Backends*, § *Text-to-Speech Examples*, `docs/server.md`, `docs/tts.md` (25.2 KB, not fetched verbatim).

---

## 20. Versioning policy + breaking-change history

### Tag/release cadence

40 tagged releases from `v0.1.0` (2026-04-12) through `v0.6.6` (2026-05-12). That's 40 releases in 30 days — **~1.3 releases per day on average**, varying by day. Multiple releases on the same calendar day: v0.4.5/v0.4.6/v0.4.7/v0.4.8 all on 2026-04-19; v0.6.4/v0.6.5/v0.6.6 all on 2026-05-12.

### Stability commitments

None documented. `pyproject.toml` marks `"Development Status :: 4 - Beta"`. There is no SEMVER policy statement in the README or in `docs/`. The version skew between `python/pyproject.toml` (`0.5.7`) and `python/crispasr/__init__.py` (`__version__ = "0.4.9"`) and the engine `VERSION` file (`0.6.6`) suggests Python-binding versions and engine versions are independently tracked (and that the in-tree `__version__` is occasionally stale).

### Documented breaking changes

**v0.6.0 release notes** (verbatim from `gh api .../releases/tags/v0.6.0`):

> *"## Breaking changes — TTS `/v1/audio/speech` `response_format`: `pcm` → `f32`. The old name returned 24-bit float anyway and was misleading; servers now reject `pcm` with a 400 + helpful message."*

That is the only explicit "Breaking changes" section observed in the recent release-note bodies. (Earlier release bodies fetched do not include a "Breaking changes" header.)

### Release-note style

Release-note bodies that exist (e.g. v0.6.0) are technically rich, organised by: New backends → New features → Performance → Breaking changes → Bug fixes → Internals. v0.6.6's body is `null` (no release notes — only the auto-generated asset list). v0.6.5, v0.6.4, v0.6.3, v0.6.2, v0.6.1 also have no body (per the WebFetch summary not showing them). Many minor releases ship without prose release notes.

Source: `gh api .../releases?per_page=100`, `gh api .../releases/tags/v0.6.0`, `python/pyproject.toml`, `python/crispasr/__init__.py`, `VERSION`.

---

## 21. Maintainer activity

### Commit cadence (last 6 months → repo only 6.5 weeks old)

Repo first commit: 2026-03-29. Research date: 2026-05-13 → 6.5 weeks of history.

- Total commits in trailing 52 weeks: **1687** (`gh api stats/participation`).
- Trailing 4 weeks: **1330**.
- Implied weeks 5-6 (the first 2.5 weeks of repo life): 1687 − 1330 = 357 (about 143 commits/week average for the very early period; numbers approximate because GitHub's `stats/participation` aggregates by ISO week).

That equates to ~80-100 commits/week average over the most recent 4 weeks. The latest-commit author email is `crispasr-dev@localhost` (no GitHub-account attribution, hence the missing entry in `gh api contributors`).

### Contributor count + breadth

Per `gh api repos/CrispStrobe/CrispASR/contributors?per_page=100`: **3 contributors with attributed merged commits**:

- `CrispStrobe` — 754 contributions
- `vkrmch` — 3 contributions (e.g. PRs #57, #63, #73)
- `DBMePls` — 2 contributions (e.g. PR #78)

The `gh api .../stats/contributors` endpoint returned empty for our query (GitHub may not have computed the stats yet for a young repo). The `AUTHORS` file contains thousands of historical names — these are inherited from the upstream whisper.cpp / ggml history captured before the fork. They do not represent CrispASR-specific contributors.

**Bus-factor signal**: 754 / (754+3+2) = **99.3% of attributed merged commits are from `CrispStrobe`** at research time. Two external contributors have landed multiple PRs (vkrmch x3, DBMePls x1 merged + x1 closed). The repo is currently single-maintainer-dominant.

### Recent issue response times (sample of 10, ordered by closure recency)

Computed from `gh api .../issues?state=closed&per_page=30&sort=updated`:

| # | Title | Created | Closed | Comments | Response time |
| --- | --- | --- | --- | --- | --- |
| #86 | FR: serving a TTS model | 2026-05-11 16:07 | 2026-05-11 16:49 | 1 | ~42 min |
| #84 | CLI streaming `--stream-length` does not keep rolling context | 2026-05-10 15:39 | 2026-05-12 12:50 | 11 | ~46 h |
| #82 | Silero LID does not work | 2026-05-09 08:31 | 2026-05-09 11:45 | 1 | ~3 h |
| #80 | granite-speech-4.1-2b-nar outputs double punctuation | 2026-05-08 17:19 | 2026-05-08 18:48 | 1 | ~1.5 h |
| #79 | VibeVoice ASR reuses too-small KV cache across repeated calls | 2026-05-08 13:11 | 2026-05-09 08:04 | 2 | ~19 h |
| #77 | Language ID not carried out in server mode | 2026-05-07 17:10 | 2026-05-09 08:23 | 2 | ~39 h |
| #70 | Streaming stdin path lacks VAD + punctuation parity | 2026-05-06 16:09 | 2026-05-09 15:15 | 9 | ~71 h |
| #71 | Chatterbox TTS triggers ggml tensor OOB | 2026-05-06 19:31 | 2026-05-08 07:00 | 7 | ~36 h |
| #65 | ggml-cuda/cpy.cu assert on large codec graphs | 2026-05-04 22:53 | 2026-05-06 03:11 | 3 | ~28 h |
| #67 | Cohere consistently dropping Japanese characters | 2026-05-05 01:33 | 2026-05-05 04:50 | 4 | ~3 h |

Median close time on the sample: ~28 h. Range: ~1.5 h – ~71 h. Most issues are closed within 1-3 days; some (e.g. #70 the streaming VAD parity) take longer when they require coordinated multi-file changes.

### PR turnaround (sample of 5 merged PRs)

| # | Title | Created | Merged | Lag |
| --- | --- | --- | --- | --- |
| #57 | perf(qwen3-tts): QWEN3_TTS_CODEC_GPU env var | 2026-05-04 02:19 | 2026-05-04 05:39 | **3 h** |
| #63 | feat(server,qwen3-tts): /v1/audio/speech + /v1/voices | 2026-05-04 22:46 | 2026-05-05 05:15 | **6.5 h** |
| #73 | feat(vibevoice-tts): voice-dir bare-name resolution | 2026-05-06 23:03 | 2026-05-07 03:36 | **4.5 h** |
| #78 | Fix vibevoice 1.5B tts voice cloning, CFG, WAV parsing | 2026-05-07 20:07 | 2026-05-08 10:52 | **15 h** |
| #4 | feat(server): OpenAI-compatible /v1/audio/transcriptions | 2026-04-15 15:48 | (closed 2026-04-16 03:36, not merged) | — |

Merged PRs land in 3-15 hours typically. One PR (#4) was closed without merge — title suggests the same feature was reimplemented internally by the maintainer.

### Issue-to-PR ratio in sample

Of the 20 most-recently-touched issues, 12 closed and 8 open at research time. Of the closed: 9 closed-with-fix-shipped (commit reference in body) and 3 closed-as-answered-only (e.g. #16 *"How to reduce segments/slices?"* — questions).

Source: `gh api .../contributors`, `gh api .../stats/participation`, `gh api .../issues?state=closed&per_page=30&sort=updated`, `gh api .../pulls?state=closed&per_page=20&sort=updated`.

---

## 22. Recent issues (last ~6 weeks)

The repo is only 6.5 weeks old, so "last 6 months" = entire repo history. Total open issues at research time: **9** (per `gh api .../issues?state=open`).

### Currently open issues (research time)

| # | Title | Reporter | Opened | Comments | Theme |
| --- | --- | --- | --- | --- | --- |
| #88 | parakeet TDT greedy decode: blank + duration-0 forces frame advance, diverging from NeMo/MLX reference | pszemraj | 2026-05-12 | 1 | Algorithmic divergence vs reference impl |
| #87 | Regression: voxtral 4b stopped working | thiswillbeyourgithub | 2026-05-12 | 4 | Recent-version regression |
| #85 | "Problems" | Oplay66 | 2026-05-10 | 1 | (Vague title — likely a user-question issue) |
| #83 | Whisper-VAD-EncDec not working | exn251 | 2026-05-10 | 4 | Backend-specific bug |
| #81 | onnx-asr comparison | grikdotnet | 2026-05-08 | 21 | Performance/comparison discussion (largest comment thread) |
| #76 | [bug] chatterbox-turbo produces distorted and very quiet audio | racheandre | 2026-05-07 | 5 | TTS quality bug |
| #75 | add indextts | ipp9 | 2026-05-07 | 8 | Feature request (indextts already added — perhaps follow-up) |
| #74 | [bug] vibevoice-tts 1.5B WAV-clone path produces generic voice | vkrmch | 2026-05-07 | 3 | TTS voice-cloning bug (partial fix in #78) |
| #46 | Open MOSS model | Oplay66 | 2026-05-01 | 1 | Feature request for OpenMOSS support |

### Recurring themes

1. **TTS reliability** — multiple TTS quality reports (#71, #74, #76, #80, vibevoice CFG/WAV parsing).
2. **Streaming-mode parity** — #70 and #84 both reported gaps between streaming and file modes (VAD parity, structured partial/final events, rolling context window docs mismatch).
3. **Language identification consistency** — #77 (LID not run in server mode), #82 (Silero LID broken), #67 (Cohere dropping Japanese chars due to SentencePiece byte-fallback).
4. **Backend regressions** — #87 voxtral 4b regression, #88 parakeet TDT greedy-decode divergence, #65 ggml-cuda assert on large codec graphs (resolved with fork patch #3).
5. **Aligner / timestamps UX** — #62 (resolved: `--force-aligner` flag added), SubtitleEdit#10775 (downstream, unresolved at research time).

### Maintainer-acknowledged defects

- `UPSTREAM.md` documents 5 fork-local ggml patches + 1 ffmpeg-transcode mp4-container bug + 1 NeMo missing standalone CTC export — all acknowledged in writing.
- Issues #65, #67, #69, #70 are linked to commits / release-note entries (acknowledged + shipped fixes).
- #74 partially fixed by #78 (community PR); voice-cloning fully working not confirmed.

### Work-in-progress signals

- `TODO.md` is 77.7 KB (not fetched verbatim) — likely a long roadmap doc.
- `PLAN.md` is 146.8 KB — likely planning doc.
- `LEARNINGS.md` is 230.8 KB — session log / bug-class lessons doc.
- `HISTORY.md` is 194.4 KB — likely changelog.
- `HANDOVER_INDEXTTS.md` (4.4 KB) is a per-task handover document.

These suggest highly active solo-maintainer workflow with extensive process documentation.

Source: `gh api .../issues?state=open&per_page=30&sort=created`, `gh api .../issues?state=closed&per_page=30&sort=updated`, `gh api .../contents/`, `UPSTREAM.md`.

---

## 23. Recent PRs

External PR activity is low. Per `gh api .../pulls?state=closed&per_page=30&sort=updated`, 5 merged PRs in repo history at research time:

| # | Title | Author | Merged | Direction |
| --- | --- | --- | --- | --- |
| #57 | perf(qwen3-tts): QWEN3_TTS_CODEC_GPU env var for clean GPU codec path | vkrmch | 2026-05-04 | Performance |
| #63 | feat(server,qwen3-tts): /v1/audio/speech + /v1/voices, per-request voice switch | vkrmch | 2026-05-05 | Feature: HTTP API |
| #73 | feat(vibevoice-tts): voice-dir bare-name resolution + per-request voice switch | vkrmch | 2026-05-07 | Feature: HTTP API |
| #78 | Fix vibevoice 1.5B tts voice cloning, CFG, and WAV parsing (Addresses #74) | DBMePls | 2026-05-08 | Bug fix |
| #4 | feat(server): add OpenAI-compatible /v1/audio/transcriptions endpoint | ubaldus | 2026-04-16 (closed, not merged) | Feature: HTTP API |

### Open PRs

**None** at research time (`open_issues_count: 9` is all issues, `gh api .../pulls?state=open` returned an empty array).

### In-flight architectural changes (per release-note bodies and `UPSTREAM.md`)

- **ggml subtree bump** — v0.6.0 bumped ggml from v0.10.0 to master `05adcae`; 5 fork-local patches re-applied. Continuing churn.
- **KV-cache quantization migration** (per v0.6.0 notes): *"full-quant K/V via cast-on-read, KV-on-CPU split, flash-attention migration on canary + cohere"* — ongoing across backends.
- **Layer offload to GPU** (v0.6.0): *"ported to 10+ backends — vibevoice, voxtral/4b, glm_asr, orpheus, omniasr-llm, gemma4_e2b, mimo_asr, qwen3_asr, granite_speech, chatterbox"* — significant cross-backend rework.
- **Upstream PRs in flight**: ggml-org/ggml#1477 (Metal conv_transpose_1d, merged 2026-05-10), ggml-org/llama.cpp#22944 (CUDA im2col OW>65535, filed 2026-05-11).

### Community contributors

Three external contributors observed (vkrmch, DBMePls, ubaldus). The first two have landed multiple changes; vkrmch's PRs cluster around HTTP-server TTS features.

Source: `gh api .../pulls?state=all&per_page=20&sort=updated`, `UPSTREAM.md`, `gh api .../releases/tags/v0.6.0`.

---

## 24. Relationship to upstream whisper.cpp

### Stated fork posture

`README.md` § *License*: *"MIT — same as upstream whisper.cpp."* The repo description (visible in `gh api repos/CrispStrobe/CrispASR`) ends with: *"...**Fork of whisper.cpp**."* The GitHub-level `"fork": false` flag means it is not a GitHub-tracked fork but a logical/historical fork that imported the whisper.cpp + ggml history (full `AUTHORS` list preserved, copyright header unchanged).

`docs/architecture.md`: *"Whisper itself intentionally remains outside `src/core/` migration, kept byte-identical to upstream whisper.cpp as a regression test gate."* — whisper-the-backend stays unchanged in this fork.

### Merge cadence with whisper.cpp

The `UPSTREAM.md` file does **not** discuss synchronization with whisper.cpp. The document explicitly says (verbatim per WebFetch): *"The document is titled 'Upstream issues / patches we depend on' and tracks fixes, features, and local modifications that CrispASR maintains relative to its dependencies (crispasr, ggml, NeMo). The document contains no information about: CrispASR's relationship to whisper.cpp, Whether CrispASR was forked from whisper.cpp, Any merge cadence or synchronization strategy with whisper.cpp, Divergence from whisper.cpp, Commitments about upstream sync with whisper.cpp."*

The repo's upstream tracking is focused on **ggml** (via the vendored `ggml/` subtree), **NeMo** (NVIDIA's CTC aligner extraction), and the historical whisper.cpp examples directory (`examples/ffmpeg-transcode.cpp` mp4 bug). Per the upstream-PR routing rule in `UPSTREAM.md`: *"`ggml-cuda/**` and `ggml-vulkan/**` future PRs file to llama.cpp; `ggml-cpu/**`, `ggml.c`, standalone-ggml stays at ggml-org/ggml; Metal is mixed (both work)."*

### Divergence rate

Visible signals:

- **5 fork-local ggml patches**, all enumerated in `UPSTREAM.md` (Q F16 mul_mat saturation; CUDA im2col OW>65535 for both 2D + 3D; CUDA cpy_scalar_transpose grid_y; Metal kernel_conv_transpose_1d input-loop tightening — this last merged upstream as ggml#1477; F32 cast in ggml.c conv builders to companion patch #1).
- **24 new ASR backends + 6 TTS backends + multiple LID/VAD/punctuation/translation engines** added beyond what upstream whisper.cpp ships.
- **`src/core/` library** is a new architectural layer not present in upstream whisper.cpp.
- **`crisp_audio/` separate audio-encoder library** is a new sublibrary.
- **Whisper code itself is kept byte-identical** as a regression test gate.

The divergence is **architectural extension** (whisper untouched + new backends added in parallel) rather than **divergent whisper** (whisper rewritten). The fork's `crispasr_run.cpp` dispatch layer is new; the whisper transcription path delegates to the legacy code.

Source: `README.md`, `UPSTREAM.md` (verbatim), `docs/architecture.md`, `gh api repos/CrispStrobe/CrispASR`.

---

## 25. Could not verify

The following claims/areas have insufficient primary-source evidence in this dossier. Future research should target these gaps before drawing operational conclusions.

1. **D3D12 support** — the integration prompt asked about D3D12. No `GGML_D3D12` flag, no deprecated-D3D12 alias in `CMakeLists.txt`, no D3D12 mention in README or `docs/install.md`. Likely not supported. Need to check the vendored `ggml/` subtree directly to confirm.

2. **Full `whisper_full_params` field list** — header is 35.3 KB; the WebFetch summary indicated 80+ fields exist in `whisper_full_params` but did not enumerate them. Need direct line-level read of `include/crispasr.h` to enumerate all runtime parameters.

3. **Exact GGUF metadata schema fields** read by each backend's `init_from_file` — would need to read each `src/<backend>.cpp` file directly. Per-backend metadata keys (e.g. expected GGUF architecture name strings, expected tensor names) are not documented in `docs/`.

4. **`crispasr_run.cpp` aligner dispatch logic** — the file is 111.7 KB. Not fetched verbatim due to size. The SubtitleEdit#10775 report and the v0.6.0 release-note entry about `--force-aligner` together strongly suggest that when a backend has `CAP_TIMESTAMPS_NATIVE`, the `-am` aligner was historically ignored — but the exact condition in code (and the v0.6.x behaviour for Canary specifically) was not directly traced. Need to read `crispasr_run.cpp` and `crispasr_aligner.cpp` to confirm.

5. **SubtitleEdit#10775 specifically as a CrispASR dispatch bug** — the prompt referenced this as *"the CTC aligner bypass for Canary"*. The issue reporter (@subof, SubtitleEdit) describes the *symptom* (aligner doesn't seem to apply, transcript inaccurate), but the issue is filed against SubtitleEdit (the GUI), has no comments, and includes a flag spelled `--aligner-model` whereas the CrispASR CLI uses `-am`. Whether the bug is in CrispASR's dispatch, in SubtitleEdit's flag-translation layer, or in the user's environment/model file is not established by primary sources. The issue is genuinely open and reproducible per the user's narrative, but the *cause* is not pinned to CrispASR by any commit, comment, or test.

6. **Streaming `_Stream` Python API** — `Session.stream_open()` returns a `_Stream` object. Its iteration / consumption API was not enumerated in the WebFetch summary of `_binding.py`. Whether it is a generator, an explicit polling object, or an async iterator is not stated.

7. **GIL behaviour in the ctypes path** — the WebFetch summary says no explicit `Py_BEGIN_ALLOW_THREADS` annotations. ctypes does release the GIL by default for native function calls — but whether the wrapper *re-acquires* the GIL during long-running calls, or holds Python objects across the call, was not directly verified.

8. **macOS code-signing posture** — not stated in `docs/install.md` per the WebFetch summary. The macOS prebuilt `.tar.gz` artifacts presumably ship unsigned. Whether signed builds are available on request, whether notarisation is performed, and Gatekeeper behaviour on first run were not documented.

9. **Per-backend Japanese language coverage** — only `whisper`, `qwen3`, and `omniasr` were verified Japanese-capable from primary sources. `gemma4-e2b` (140+ langs), `vibevoice` (50+ langs), `voxtral` (8 langs but list not enumerated in CrispASR README) — coverage depends on each model's HF card. Need per-backend HF-card lookups to confirm.

10. **Per-backend native sampling rates** — the audio I/O subsystem ingest is described as "16 kHz mono PCM" in multiple places, but the per-model encoders may expect different sample rates (e.g. 8 kHz for some legacy NeMo, 24 kHz for some TTS outputs). The unified resampling path in `src/core/audio_resample.cpp` should handle this, but per-backend SR contracts were not enumerated.

11. **`bindings/` Python directory mismatch with `docs/bindings.md`** — the docs reference a Python binding but the binding code lives at `python/crispasr/` (not `bindings/python/`). This is consistent and the `python/README.md` confirms install steps, but a future reader of `docs/bindings.md` might expect `bindings/python/`. Likely a docs-vs-layout style choice rather than a bug.

12. **Version skew between `python/pyproject.toml` (0.5.7), `python/crispasr/__init__.py` (0.4.9), and root `VERSION` (0.6.6)** — these three version sources do not agree. Whether this is intentional (Python binding is on its own cadence vs the engine) or an oversight is not stated. The PyPI publish history would tell, but PyPI was not queried in this dossier.

13. **`docs/cli.md` full flag inventory** — the doc is 34.7 KB. Only a few sections (aligner, `-m auto`, `-l auto`, backend list) were fetched. The full enumerated flag list (which would tell us precisely which `whisper_full_params` fields are CLI-exposed) was not fetched verbatim.

14. **`docs/streaming.md` "Pipe Mode" `--alt`/`--alt-n` documentation gap** — the doc says *"`--alt` / `--alt-n` don't output token alternatives in streaming modes"* but doesn't enumerate which non-streaming modes do. Per-mode token-confidence detail not fully traced.

15. **`docs/regression-matrix.md` capability-tier full list** — three tiers (`ignore` / `smoke` / `full`) and per-capability flag list mentioned but not enumerated per-backend in primary sources.

16. **`HISTORY.md`, `PLAN.md`, `TODO.md`, `LEARNINGS.md` content** — combined ~648 KB of documentation not read in this dossier. Likely contains additional roadmap, recent-changes-log, and per-bug postmortems that would change the picture in §22-23.

17. **`build.rs` for Rust `crispasr-sys` crate** — 10.1 KB; describes how the Rust sys crate links against `libcrispasr`. Not fetched. Would tell us which version of `libcrispasr` ABI the Rust crate targets.

18. **Number of distinct backends actually compiled** — `docs/feature-matrix.md` says 42 backends compiled. `examples/cli/` has 28 `crispasr_backend_*.cpp` files. The 14-backend gap consists of LID/VAD/punctuation/truecasing/diarization/translation auxiliary backends. Exact 42-backend enumeration is in `feature-matrix.html` (not fetched as raw markdown).

19. **`AUTHORS` content beyond first 60 lines** — full file is 21.7 KB. The first 60 lines confirm alphabetised whisper.cpp / ggml contributor list inherited from upstream. Whether CrispASR-specific contributors are at the bottom of the file or are missing from it entirely was not verified.

20. **GH Pages / `has_pages: false`** — `gh api repos/CrispStrobe/CrispASR` returns `"has_pages": false`. There is no project website beyond the README + `docs/feature-matrix.html` (served from raw GitHub).

---

## Appendix A: Primary-source URLs

All fetched at research time 2026-05-13 unless noted. URLs preserved for re-verification.

### Repository metadata
- `https://api.github.com/repos/CrispStrobe/CrispASR`
- `https://api.github.com/repos/CrispStrobe/CrispASR/contributors`
- `https://api.github.com/repos/CrispStrobe/CrispASR/stats/participation`
- `https://api.github.com/repos/CrispStrobe/CrispASR/releases?per_page=100`

### Raw files (raw.githubusercontent.com/CrispStrobe/CrispASR/main/...)
- `README.md` (51.0 KB)
- `LICENSE` (1.0 KB)
- `VERSION` (6 B)
- `ARCHITECTURE.md` (10.7 KB)
- `UPSTREAM.md` (9.1 KB)
- `COMPARISON.md` (3.1 KB)
- `CMakeLists.txt` (12.3 KB)
- `CMakePresets.json` (5.0 KB)
- `AUTHORS` (first 60 lines)
- `HANDOVER_INDEXTTS.md` (first 60 lines)
- `docs/architecture.md` (26.4 KB)
- `docs/contributing.md` (9.1 KB)
- `docs/server.md` (13.6 KB)
- `docs/bindings.md` (3.8 KB)
- `docs/cli.md` (selected sections only)
- `docs/streaming.md` (9.2 KB)
- `docs/feature-matrix.md` (4.5 KB)
- `docs/install.md` (6.7 KB)
- `docs/regression-matrix.md` (2.6 KB)
- `examples/cli/crispasr_backend.h` (CAP_* + class summary)
- `include/crispasr.h` (API surface summary)
- `crisp_audio/include/crisp_audio.h` (full header, 4.4 KB)
- `python/crispasr/__init__.py` (full, 1.0 KB)
- `python/pyproject.toml` (full)
- `python/README.md` (summary)
- `python/crispasr/_binding.py` (signature summary)

### Directory listings (`gh api .../contents/...`)
- `/`, `/src`, `/src/core`, `/include`, `/bindings`, `/examples`, `/examples/cli`, `/examples/server`, `/python`, `/python/crispasr`, `/crisp_audio`, `/crispasr-sys`, `/crispasr`, `/docs`, `/tools`, `/flutter/crispasr`, `/bindings/javascript`, `/bindings/go`, `/bindings/ruby`, `/bindings/java`, `/crisp_audio/src`, `/crisp_audio/include`

### Issues + PRs
- `gh api repos/CrispStrobe/CrispASR/issues?state=all&per_page=20&sort=updated`
- `gh api repos/CrispStrobe/CrispASR/issues?state=open&per_page=30&sort=created`
- `gh api repos/CrispStrobe/CrispASR/issues?state=closed&per_page=30&sort=updated`
- `gh api repos/CrispStrobe/CrispASR/pulls?state=all&per_page=20&sort=updated`
- `gh api repos/CrispStrobe/CrispASR/pulls?state=closed&per_page=30&sort=updated`
- `gh api repos/CrispStrobe/CrispASR/releases/tags/v0.6.6`
- `gh api repos/CrispStrobe/CrispASR/releases/tags/v0.6.0`

### Downstream consumer
- `https://github.com/SubtitleEdit/subtitleedit/issues/10775` (CrispASR aligners — Canary)

### External
- `https://huggingface.co/cstr` (partial inventory of GGUF model repos)
- WebSearch: "CrispASR Japanese ASR support qwen3-asr Japanese language"
- WebSearch: "SubtitleEdit issue 10775 CrispASR CTC aligner Canary bypass"

---

*End of dossier. 2026-05-13.*
