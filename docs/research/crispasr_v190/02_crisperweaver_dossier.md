# CrisperWeaver — Phase 0 Dossier

**Scope statement.** This dossier reports primary-source observations about the
[CrispStrobe/CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver)
Flutter application, the reference consumer of CrispASR via Dart FFI. The
dossier is descriptive only. It catalogues the FFI bridge mechanics, the
library-discovery fallback chain, per-platform packaging scripts, model-download
plumbing, UI surface, and maintainer activity exactly as they exist in the
repository on 2026-05-13. No recommendations, comparisons, or integration
proposals are offered; gaps where primary-source evidence was unavailable are
flagged in section 17.

---

## 0. Snapshot

- **Repo URL:** `https://github.com/CrispStrobe/CrisperWeaver`
  ([`gh api repos/CrispStrobe/CrisperWeaver`](https://api.github.com/repos/CrispStrobe/CrisperWeaver))
- **Default branch:** `main`
- **Repo created:** 2025-09-11T06:36:05Z
- **Last push (HEAD activity):** 2026-05-12T06:08:21Z
- **Latest commit on `main` (HEAD as of 2026-05-13):**
  `9b93f863d5cf70fcafd504ec3fa5711417c3071e` — `chore: bump version to 0.5.0`
  (authored 2026-05-12T06:08:12Z).
- **License:** `agpl-3.0` (SPDX `AGPL-3.0`) per the GitHub API license object.
  The repo `LICENSE` file is 34,523 bytes — consistent with the full GNU
  AGPL v3 text. The README states *"GNU AGPL-3.0-or-later"*.
- **Stars:** 12. **Watchers:** 12. **Forks:** 3. **Subscribers:** 0.
- **Open issues:** 0. **Total issues opened (open+closed):** 1
  (issue #1, `MissingPluginException` for `just_audio`, opened 2026-04-24,
  closed by `BFG-BFG`). No pull requests have been opened against the repo
  (the `pulls?state=all` query returns an empty list).
- **Contributors:** 1 (`CrispStrobe`, 137 contributions). All commits in the
  recent log are co-authored `Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
  alongside the human author.
- **Latest release:** `v0.5.0`, published 2026-05-12T06:13:04Z, not flagged as
  pre-release. Release cadence (10 releases visible in the API page-1 slice):
  v0.1.6 (2026-04-18) → v0.1.7 → v0.1.8 → v0.1.9 (2026-04-20) → v0.2.0
  (2026-05-02) → v0.2.1 → v0.3.0 → v0.4.0 (2026-05-03) → v0.4.1 (2026-05-10)
  → v0.5.0 (2026-05-12). Roughly 10 releases in 24 days, all published by
  `github-actions[bot]`.
- **Flutter version targeted:** Flutter 3.38.x (stable channel). The
  `pubspec.yaml` pins `environment.flutter: ">=3.10.0"` but the README and
  shields.io badge both state Flutter 3.38.
- **Stated relationship to CrispASR:** *Sibling repos, not a mono-repo.* The
  README's *Building → Clone the two repos side-by-side* section instructs:
  `git clone https://github.com/CrispStrobe/CrispASR.git` + `git clone
  https://github.com/CrispStrobe/CrisperWeaver.git`. `pubspec.yaml` consumes
  the Dart FFI package as a `path:` dependency: `crispasr: path:
  ../CrispASR/flutter/crispasr`. The README also exposes `CRISPASR_REPO` /
  `CRISPASR_REF` env vars in both `ci.yml` and `release.yml` so a fork or
  version pin can be substituted.

The CrispASR sibling repo itself, for context only: stars 176, forks 19,
license `MIT` (not AGPL), language `C++`, last pushed 2026-05-13T10:11:28Z
(`gh api repos/CrispStrobe/CrispASR`). The Crisp ecosystem also includes
`CrispEmbed` and `Susurrus` per the README ecosystem table.

---

## 1. Repository structure

Root listing from `gh api repos/CrispStrobe/CrisperWeaver/contents/`:

```
.github/                      (CI workflows)
.gitignore
.metadata
AppLogo.png                   (1.8 MB)
CHANGELOG.md                  (30 KB)
HISTORY.md                    (51 KB)
LEARNINGS.md                  (28 KB)
LICENSE                       (34.5 KB — AGPL v3 text)
PLAN.md                       (25 KB roadmap)
README.md                     (26 KB)
analysis_options.yaml         (strict lints, see §3)
android/                      (Gradle module)
assets/                       (models/, images/, licenses/, vad/)
build_all.sh                  (top-level dispatcher)
build_and_run.sh
build_android.sh              (thin wrapper → gradlew assembleDebug)
build_ios.sh                  (thin wrapper → pod install + flutter build ios)
build_windows.bat             (forwards to scripts/build_windows.ps1)
dart_test.yaml                (tag config — `slow` opt-in)
docs/                         (e.g., ios-share-extension-setup.md)
ios/                          (Runner + Frameworks/ + ShareExtension/)
l10n.yaml                     (ARB → Dart codegen config)
lib/                          (Flutter app source — see below)
linux/                        (CMake-based Linux runner)
macos/                        (Xcode Runner project + OpenWithReceiver.swift)
pubspec.lock                  (40 KB resolved deps)
pubspec.yaml                  (see §3 verbatim)
scripts/                      (per-platform build + bundle scripts — see §3)
test/                         (flutter unit/widget tests; 349 passing at v0.5.0)
windows/                      (CMake-based Windows runner)
```

`lib/` subdirectories
(`gh api repos/CrispStrobe/CrisperWeaver/contents/lib`):

```
lib/
├── main.dart                 (24 KB — entry point)
├── constants/
├── engines/                  (FFI bridge surface — see §5)
│   ├── crispasr_engine.dart  (45 KB)
│   ├── engine_factory.dart   (7 KB)
│   ├── mock_engine.dart      (10 KB)
│   └── transcription_engine.dart (7 KB — interface)
├── l10n/                     (ARB sources for en, de)
├── screens/                  (15 screens — see §4)
├── services/                 (36 services — see §4)
├── theme/
├── utils/
└── widgets/
```

`lib/screens/` files (full list with sizes):
`about_screen.dart` (7 KB), `cloud_llm_settings_screen.dart` (3 KB),
`edit_audio_screen.dart` (28 KB), `history_screen.dart` (14 KB),
`hotkey_settings_screen.dart` (2.5 KB),
`local_llm_settings_screen.dart` (3 KB), `logs_screen.dart` (7 KB),
`model_management_screen.dart` (19 KB), `settings_screen.dart` (44 KB),
`storage_screen.dart` (7 KB), `synthesize_screen.dart` (27 KB),
`transcription_screen.dart` (94 KB), `translate_screen.dart` (11 KB),
`voice_bake_screen.dart` (9 KB), `voice_clone_wizard_screen.dart` (18 KB).

`lib/services/` files (selected):
`audio_service.dart` (16 KB), `batch_persistence_service.dart` (14 KB),
`batch_queue_service.dart` (17 KB), `desktop_open_with_bridge.dart` (4 KB),
`diarization_service.dart` (5 KB), `history_service.dart` (7 KB),
`hotkey_service.dart` (12 KB), `lid_service.dart` (8 KB),
`local_llm_backend.dart` (7 KB), `log_service.dart` (11 KB),
`memory_estimator.dart` (10 KB), `model_service.dart` (95 KB — largest
service file), `preset_service.dart` (13 KB), `punc_service.dart` (7 KB),
`server_service.dart` (18 KB — OpenAI-compatible HTTP server),
`settings_service.dart` (17 KB),
`system_audio_capture_service.dart` (16 KB),
`text_translation_service.dart` (7 KB),
`transcript_cleanup_service.dart` (9 KB),
`transcript_summarize_service.dart` (13 KB),
`transcription_service.dart` (23 KB), `transcription_worker.dart` (11 KB),
`transcription_worker_pool.dart` (13 KB), `tts_service.dart` (16 KB),
`vad_service.dart` (5 KB), `voice_baking_service.dart` (6 KB).

`scripts/` (top-level entry points and bundlers):

```
scripts/build_ios_xcframework.sh   (13 KB)
scripts/build_linux.sh             (5 KB)
scripts/build_macos.sh             (6 KB)
scripts/build_windows.ps1          (5.6 KB)
scripts/bundle_linux_libs.sh       (4.6 KB)
scripts/bundle_macos_dylibs.sh     (5.7 KB)
scripts/bundle_windows_dlls.ps1    (4 KB)
scripts/check.sh                   (2 KB)
scripts/wire_ios_xcframework.rb    (3 KB)
```

**Entry points:**
- Flutter Dart entry: `lib/main.dart` — calls `runApp(ProviderScope(... CrisperWeaverApp()))`.
- Desktop runners: `macos/Runner` (Xcode), `linux/runner` (CMake),
  `windows/runner` (CMake).
- Android entry: `android/app` Gradle module.
- iOS entry: `ios/Runner` + `ios/ShareExtension/` (template files for the
  share-extension target, not yet wired in pbxproj — per the 2026-05-11
  commit `8a6e5df`).
- macOS Swift bridges in `macos/Runner/`: `OpenWithReceiver.swift`,
  `SystemAudioCapture.swift`, plus `AppDelegate.swift` overrides for
  `application(_:open:)` / `openFile:` / `openFiles:` (commit `7e5d2e2`,
  2026-05-11).

---

## 2. License + redistribution implications

**CrisperWeaver license:** the GitHub API license field returns
`{"spdx_id": "AGPL-3.0", "key": "agpl-3.0", "name": "GNU Affero General
Public License v3.0"}`. The README's *License & author* section confirms
*"CrisperWeaver is GNU AGPL-3.0-or-later"* and points to `LICENSE` for the
full text. The in-app *About* screen is described as auto-aggregating the
third-party license list via Flutter's `showLicensePage`
(`lib/services/native_licenses.dart` is 1.7 KB, dedicated to this purpose).

**Sibling CrispASR engine license:** the engine repo
(`CrispStrobe/CrispASR`) reports SPDX `MIT` via `gh api
repos/CrispStrobe/CrispASR` (license object: `MIT`). The Dart FFI binding
package at `flutter/crispasr/LICENSE` (1,078 bytes) is consistent with an
MIT or BSD-style header (size and the `README.md` statement "MIT licensed"
match). Thus the runtime engine + FFI-binding package are MIT, while the
Flutter consumer app on top is AGPL-3.0-or-later.

**AGPL-3.0 high-level terms (descriptive, from the standard license text):**
- Source-availability requirement: distributing the program (or modified
  versions) in binary form requires offering corresponding source under the
  same license to all recipients.
- Network-use clause (§13, the "Affero" distinction from GPLv3): if a
  modified version is *interacted with by users remotely through a computer
  network*, those users must be offered the corresponding source of the
  modified version.
- Combined / linked works: anything that combines AGPL-3.0 code with other
  code in a single program is, by default, governed by the AGPL terms for
  redistribution purposes (the standard GPL family "derivative work"
  interpretation; the FSF's own FAQ is the canonical reference for edge
  cases).
- Compatibility: AGPL-3.0 is one-way-compatible with GPL-3.0 (combinable
  *into* AGPL but not vice-versa).
- Bundling: the application bundles a third-party MIT-licensed engine
  (`libcrispasr` / `libwhisper`) and MIT-licensed Dart FFI package
  (`package:crispasr`). MIT is permissive and AGPL-compatible for inclusion;
  the resulting redistributable bundle is governed by AGPL terms.
- The README's "third-party license list" wording acknowledges that the
  shipped binaries aggregate notices from upstream dependencies.

**Non-AGPL components observed in the repo:** the `pubspec.yaml` direct
dependency list (see §3) names packages including `flutter_riverpod`,
`go_router`, `just_audio`, `record`, `dio`, `ffi`, `crypto`, `archive`,
`receive_sharing_intent`, `desktop_drop`, `hotkey_manager`,
`share_plus`, `package_info_plus`, `device_info_plus`,
`flutter_localizations`, `permission_handler`, `path_provider`,
`url_launcher`, `shelf`, `shelf_router`, `media_kit_libs_windows_audio`,
`media_kit_libs_linux`. Their individual licenses are not catalogued in
the repo metadata but are surfaced via `showLicensePage` at runtime per
the README. The CrispASR engine and its `flutter/crispasr` package are
MIT.

**iOS sideload posture (license-adjacent fact, not legal advice):** the
README explicitly states the iOS IPA is *unsigned* and is distributed via
SideStore / AltStore / Feather sideload paths. The build script
`build_ios_xcframework.sh` sets `CODE_SIGNING_REQUIRED=NO`,
`CODE_SIGN_IDENTITY=""`, `CODE_SIGNING_ALLOWED=NO`. The README notes
"We don't pay for the Apple Developer Program yet."

---

## 3. Build system + packaging

### 3.1 `pubspec.yaml` structure (verbatim, current at `main` HEAD)

```yaml
name: crisper_weaver
description: A cross-platform Flutter app for fully-offline audio
  transcription + speech synthesis. 24+ ASR + TTS families via CrispASR.
publish_to: 'none'
version: 0.5.0+1
environment:
  sdk: '>=3.0.0 <4.0.0'
  flutter: ">=3.10.0"

dependencies:
  flutter: { sdk: flutter }
  flutter_localizations: { sdk: flutter }

  # UI and state management
  cupertino_icons: ^1.0.6
  material_color_utilities: ">=0.8.0 <0.12.0"
  flutter_riverpod: ^2.4.9
  go_router: ^17.2.1

  # Audio handling
  just_audio: ^0.10.5
  audio_session: ^0.2.3
  just_audio_media_kit: ^2.1.0      # Windows/Linux backend
  media_kit_libs_windows_audio: any
  media_kit_libs_linux: any
  record: ^6.2.0
  path_provider: ^2.1.1
  permission_handler: ^12.0.1

  # File handling
  file_picker: ^11.0.2
  path: ^1.8.3

  # Network and downloads
  http: ^1.1.2
  dio: ^5.4.0
  url_launcher: ^6.2.2

  # OpenAI-compatible HTTP server (Settings → "Server mode")
  shelf: ^1.4.1
  shelf_router: ^1.1.4

  # Native integration
  ffi: ^2.1.0

  # On-device speech recognition via ggml
  crispasr:
    path: ../CrispASR/flutter/crispasr

  # Utility
  uuid: ^4.3.3
  shared_preferences: ^2.2.2
  intl: ">=0.19.0 <0.21.0"

  # Additional required dependencies
  share_plus: ^12.0.2
  package_info_plus: ^9.0.1
  device_info_plus: ">=11.0.0 <12.4.0"   # 12.4.0 calls iOS 26.1 selector

  # Model downloads + verification
  crypto: ^3.0.3
  archive: ^4.0.9

  # Progress indicators and animations
  percent_indicator: ^4.2.3

  # Better HTTP handling
  retry: ^3.1.2

  # Inbound share: receive audio from OS share sheet
  receive_sharing_intent: ^1.8.0

  # Desktop drag-and-drop
  desktop_drop: ^0.7.0

  # Desktop global hotkey (§5.1.11)
  hotkey_manager: ^0.2.3

dev_dependencies:
  flutter_test: { sdk: flutter }
  flutter_lints: ^6.0.0
  build_runner: ^2.4.7
  json_annotation: ^4.8.1
  json_serializable: ^6.7.1
  flutter_launcher_icons: ^0.14.4
  ffigen: ^20.1.1

dependency_overrides:
  record_linux: ^1.3.0   # platform-interface mismatch in transitive 0.7.2

flutter:
  uses-material-design: true
  generate: true   # ARB → Dart localizations codegen via l10n.yaml
  assets:
    - assets/models/
    - assets/images/
    - assets/licenses/
    - assets/vad/
```

### 3.2 Per-platform packaging scripts

**macOS — `scripts/bundle_macos_dylibs.sh`** (5,667 bytes). Verified
behaviour:

- Copies a single CrispASR shared library
  (`libcrispasr.<version>.dylib` *or* `libwhisper.<version>.dylib`) from
  CrispASR's CMake output into the `.app` bundle's `Contents/Frameworks/`,
  storing it as `libwhisper.dylib`.
- Creates symlinks for backward-compatibility aliases:
  `libcrispasr.dylib → libwhisper.dylib`, `libcrispasr.1.dylib → libwhisper.dylib`.
- Walks `otool -L` output and pulls in any Homebrew / system dylibs
  referenced from absolute paths (`/opt/homebrew/...`, `/usr/local/...`);
  espeak-ng is named as a typical case for Kokoro phoneme support.
- Copies all `libggml*.dylib` files from CrispASR's `ggml/src/` build
  output (multiple ggml subcomponents — see §3.3 Windows for the
  enumerated list).
- Rewrites absolute external dependency paths to `@rpath/` references via
  `install_name_tool -change`, so the bundled copies in
  `Contents/Frameworks/` are dyld-resolved first.
- Applies ad-hoc code signing to the full `.app` bundle:
  `codesign --force --deep --sign -`. The `LEARNINGS.md` notes ad-hoc
  signing is *mandatory* and that skipping it manifests as opaque
  "library not found" errors.

**Linux — `scripts/bundle_linux_libs.sh`** (4,586 bytes). Verified
behaviour:

- Destination: `$BUNDLE/lib/` (default
  `build/linux/x64/{release|debug|profile}/bundle/lib`).
- Copies `libwhisper.so` from `$CRISPASR_DIR/$CRISPASR_BUILD_SUBDIR/src/`.
- Creates SONAME-resolution symlinks
  `libcrispasr.so → libwhisper.so` and `libcrispasr.so.1 → libwhisper.so`.
- Copies all `libggml*.so*` (with version aliases) from
  `$CRISPASR_DIR/$CRISPASR_BUILD_SUBDIR/ggml/src/`.
- An in-script comment states explicitly:
  *"Per-backend .so files are NOT copied: every CrispASR backend is built
  as a STATIC archive and pulled into libwhisper.so at link time.
  Bundling libwhisper.so alone is sufficient."*
- Preserves any `flutter_*`, `app.*`, or `*_plugin*` libraries already in
  the bundle; cleans up previous CrispASR artifacts before re-bundling.

**Windows — `scripts/bundle_windows_dlls.ps1`** (3,995 bytes). Verified
behaviour:

- Destination: `build\windows\x64\runner\Release` (overridable via
  `$RUNNER_DIR`).
- Copies `whisper.dll`, then creates a second copy named `crispasr.dll`
  (the Windows aliasing strategy is "copy" not symlink — per
  `LEARNINGS.md`: *"`libcrispasr.dylib` is a symlink (Unix) or copy
  (Windows) of `libwhisper.dylib` — they're identical, not independently
  versionable."*).
- Attempts to bundle ~30 per-backend DLLs *if* a given backend was built
  as a dynamic library (most are static-linked into `whisper.dll` and
  are skipped with a "static archive" notification):
  `parakeet`, `canary`, `canary_ctc`, `qwen3_asr`, `cohere`,
  `granite_speech`, `granite_nle`, `voxtral`, `voxtral4b`,
  `wav2vec2-ggml`, `glm-asr`, `kyutai-stt`, `firered-asr`,
  `firered-vad`, `marblenet-vad`, `firered-lid`, `omniasr`, `vibevoice`,
  `ecapa-lid`, `moonshine`, `moonshine_streaming`, `gemma4_e2b`,
  `mimo_tokenizer`, `mimo_asr`, `qwen3_tts`, `orpheus`, `kokoro`,
  `pyannote-seg`, `silero-lid`, `fireredpunc`.
- Copies four ggml runtime DLLs: `ggml`, `ggml-cpu`, `ggml-base`,
  `ggml-blas`.
- Prints final runner directory contents with file sizes.

**iOS — `scripts/build_ios_xcframework.sh`** (13,324 bytes). Verified
behaviour:

- Output: `$REPO_ROOT/ios/Frameworks/crispasr.xcframework` (also mirrored
  from `${CRISPASR_DIR}/build-apple/crispasr.xcframework`).
- Builds two slices: arm64 iOS-device (`iphoneos` SDK) and arm64 iOS
  simulator (`iphonesimulator` SDK), both honouring the deployment
  target (default 13.0; 14.0+ required for CoreML).
- Includes ~30 per-backend static libs (libparakeet, libvoxtral,
  libkokoro, etc.) linked into the framework.
- espeak-ng phonemisation is *disabled* in the iOS build path due to
  link incompatibility; iOS Kokoro relies on the shellout fallback which
  is itself unavailable on iOS.
- Signing posture: `CODE_SIGNING_REQUIRED=NO`, `CODE_SIGN_IDENTITY=""`,
  `CODE_SIGNING_ALLOWED=NO`. The framework expects a manual "Embed &
  Sign" step inside Xcode for any device install. The `wire_ios_xcframework.rb`
  helper (3,267 bytes) drives that wiring at build time.

**macOS desktop build orchestrator — `scripts/build_macos.sh`** (6,282
bytes):

- Configures CrispASR via CMake with `DGGML_METAL=ON` and
  `DCRISPASR_COREML=ON`. Uses a dedicated `build-flutter-bundle/`
  subdirectory to keep build outputs separate.
- Builds ~25 backend targets as static archives.
- Links them into `libwhisper.dylib`.
- Runs `flutter build macos` (debug or release based on argv).
- Invokes `bundle_macos_dylibs.sh` to assemble the final `.app`.
- Supports `--rebuild-cmake` for forced reconfigure;
  `$CRISPASR_DIR` and `$JOBS` env vars for CrispASR location and
  parallel job count.

**Top-level wrappers (thin):**
- `build_android.sh` — `cd android && ./gradlew assembleDebug` only.
  No CrispASR compilation here (the README + PLAN.md note real-ASR APKs
  are cross-built only inside the GitHub Actions `release.yml`).
- `build_ios.sh` — `cd ios && pod install && cd .. && flutter build ios
  --debug --no-codesign`.

**`release.yml` GitHub Actions workflow** (summarised from
`.github/workflows/release.yml`):

- Triggers: tag matching `v*` *or* manual `workflow_dispatch`.
- Build matrix (5 jobs):
  - macOS-latest → `crisper_weaver-macos.zip` (ad-hoc-signed, Metal+CoreML).
  - Ubuntu-x64 → `crisper_weaver-linux-x64.tar.gz` (GTK-3 desktop bundle).
  - Android-arm64 → `crisper_weaver-android-arm64.apk` (cross-compiled
    `libwhisper.so` dropped into `jniLibs/arm64-v8a/`).
  - iOS-unsigned → `crisper_weaver-ios-unsigned.ipa`.
  - Windows-x64 → `crisper_weaver-windows-x64.zip` (currently
    `continue-on-error` until real-machine verification per PLAN.md).
- Every job emits a sibling `.sha256` file using platform tools
  (`shasum` / `sha256sum` / `Get-FileHash`).
- CrispASR repo is checked out from `${CRISPASR_REPO:-CrispStrobe/CrispASR}`
  at `${CRISPASR_REF:-main}` in every job — both env vars are documented
  overrides at the top of `release.yml`.
- v0.5.0 release-page asset sizes (downloaded counts in parentheses, all
  from the API as of 2026-05-13):
  `crisper_weaver-android-arm64.apk` 36.3 MB (4 downloads),
  `crisper_weaver-ios-unsigned.ipa` 16.9 MB (0),
  `crisper_weaver-linux-x64.tar.gz` 22.9 MB (0),
  `crisper_weaver-macos.zip` 29.2 MB (2),
  `crisper_weaver-windows-x64.zip` 28.3 MB (4),
  plus the five `.sha256` files.

### 3.3 Code-signing posture summary

- macOS: ad-hoc `codesign --force --deep --sign -` only. No Developer ID,
  no notarisation. `LEARNINGS.md`: *"Ad-hoc codesigning is mandatory;
  skipping it causes 'library not found' errors that obscure permission
  failures."*
- iOS: deliberately unsigned, intended for SideStore/AltStore/Feather.
- Android: APK is built but unsigned per PLAN.md "Unresolved technical
  debt" section.
- Windows: no signing step; no MSI/EXE installer is produced — the
  release artifact is a `.zip` of `build\windows\x64\runner\Release`.
- Linux: no signing.

---

## 4. Flutter app architecture

### 4.1 Top-level widget tree (from `lib/main.dart`)

```dart
runApp(
  ProviderScope(
    overrides: [
      settingsServiceProvider.overrideWithValue(settingsService),
      presetServiceProvider.overrideWithValue(presetService),
      hotkeyServiceProvider.overrideWithValue(hotkeyService),
    ],
    child: const CrisperWeaverApp(),
  ),
);
```

`CrisperWeaverApp` returns `MaterialApp.router`:

```dart
return MaterialApp.router(
  title: 'CrisperWeaver',
  debugShowCheckedModeBanner: false,
  theme: AppTheme.lightTheme,
  darkTheme: AppTheme.darkTheme,
  themeMode: ThemeMode.system,
  routerConfig: _router,
  locale: locale,
  localizationsDelegates: AppLocalizations.localizationsDelegates,
  supportedLocales: AppLocalizations.supportedLocales,
);
```

### 4.2 State management

**Riverpod 2.x** (`flutter_riverpod: ^2.4.9`). The `main()` function
synchronously instantiates three eager singletons (settings, presets,
hotkeys) and injects them via `overrideWithValue` so they are immediately
available to the widget tree without async `FutureProvider` boilerplate.
Other services follow the more typical `Provider` / `StateNotifierProvider`
pattern. `engine_factory.dart` exposes `engineManagerProvider` as a
`StateNotifierProvider` with `EngineManagerNotifier` (a `StateNotifier`
extending state transitions during engine switching).

### 4.3 Routing

**go_router 17.x.** Routes named in `main.dart`:
- Primary: transcription, settings, models, history, logs, about.
- Settings sub-routes: cloud-llm, local-llm, hotkey (dialog on desktop,
  full screen on mobile per the 2026-05-12 v0.5.0 commit message which
  references *"Settings sub-screens on mobile"*).
- Specialised: synthesize, voice-clone, translate, voice-bake,
  edit-audio. Synthesize and edit-audio accept query parameters and
  `extra` data for state handoff.

### 4.4 Threading / isolate usage

Two distinct surfaces:

**(a) Per-transcription event-loop yielding (no real isolate).**
`crispasr_engine.dart` inserts
`await Future<void>.delayed(Duration.zero);` immediately before the
blocking FFI call in `_runTranscription` (~line 716) and
`_runSessionTranscription` (~line 765). This is a single microtask yield;
the FFI call still blocks the platform thread for its full duration.

**(b) Worker-pool isolates for parallel-file transcription.**
`lib/services/transcription_worker_pool.dart` (12,972 bytes) spawns N
persistent worker isolates with `Isolate.spawn()`. Each worker holds its
own FFI engine instance (FFI handles are *not* shared across isolates).
Across the boundary it passes the audio as a `Float32List` via
`sendPort.send()`; segments come back the other way. `count >= 1` is
required; the pool size is parameter-driven (`maxConcurrentTranscriptions`
and `maxConcurrentSessions` in `settings_service.dart`). When one worker
dies, its dispatch future surfaces the exception; the remaining workers
keep absorbing queued jobs.

**(c) Isolated SHA-1 verification.** `model_service.dart` runs SHA-1
verification inside `Isolate.run(() async { ... sha1.convert(bytes) ... })`
to avoid blocking the UI thread on multi-GB binary reads.

**(d) Streaming microphone capture.**
`audio_service.dart` uses `_recorder.startStream()` returning real-time
`Float32List` frames at 16 kHz. The streaming session pushes audio into a
C-side callback registered via `NativeCallable<...>.listener` (see §5.5);
because the callback is a `listener`, it does not return a value and is
safe to invoke from native threads.

### 4.5 Event-loop interaction with the FFI bridge

- The main-isolate FFI calls are wrapped in `Future` so the UI sees the
  call as async, but on a single audio file in a single isolate the
  underlying C work still pins one platform thread. The
  `transcription_worker_pool` is the mechanism by which the app extracts
  parallelism.
- Progress is reported through Dart-side callbacks
  (`void Function(double)? onProgress`) invoked from inside the FFI loop
  (the C side calls back through `NativeCallable.listener` trampolines).
- Segments stream back through `void Function(TranscriptionSegment)? onSegment`
  callbacks scoped to the active transcription call.
- Chunked Whisper (`_runChunkedWhisper`, ~lines 602–707 of
  `crispasr_engine.dart`) splits long audio into 30-second windows and
  re-yields progress (`onProgress?.call(remaining <= 0 ? 1.0 : (i - firstChunk + 1) / remaining)`)
  after each chunk so the UI sees movement on long files.

---

## 5. Dart FFI bridge to CrispASR

The FFI surface lives in **two files**:
1. The consumer-app wrapper, `lib/engines/crispasr_engine.dart`
   (45,351 bytes), which implements the app's `TranscriptionEngine`
   interface and delegates to:
2. The sibling-repo FFI binding package, `CrispStrobe/CrispASR` at
   `flutter/crispasr/lib/src/crispasr.dart` (106,317 bytes) plus
   `chat.dart` (15,543 bytes). The package's public export is
   `flutter/crispasr/lib/crispasr.dart` (139 bytes) which `export`s
   `src/crispasr.dart` and `src/chat.dart`.

The Dart consumer never calls `DynamicLibrary` directly. Library opening,
function lookup, and memory marshalling all live inside
`package:crispasr`.

### 5.1 The `CrispASREngine` class

Source: `lib/engines/crispasr_engine.dart` (HEAD,
`9b93f863d5cf70fcafd504ec3fa5711417c3071e`).

```dart
class CrispASREngine implements TranscriptionEngine {
  crispasr.CrispASR? _model;
  crispasr.CrispasrSession? _session;
  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _cancelRequested = false;
  String? _currentModelId;
  String? _currentModelPath;
  Map<String, dynamic> _config = {};
  ModelService? _modelService;
  AlignerService? _alignerService;
  LidService? _lidService;
  // ...
}
```

Imports (verbatim):
```dart
import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:crispasr/crispasr.dart' as crispasr;
import 'transcription_engine.dart';
import '../services/aligner_service.dart';
import '../services/lid_service.dart';
import '../services/log_service.dart';
import '../services/model_service.dart';
import '../services/transcription_service.dart' show AdvancedTranscribeOptions;
```

Public method surface (line ranges in the current file):

| Method | Approx. lines |
|---|---|
| `String get engineId` | 46 |
| `String get engineName` | 49 |
| `String get version` | 52 |
| `bool get supportsStreaming` | 55 |
| `bool get supportsLanguageDetection` | 58 |
| `bool get supportsWordTimestamps` | 61 |
| `bool get supportsSpeakerDiarization` | 64 |
| `List<String> get supportedLanguages` | 67–128 |
| `bool get isInitialized` | 131 |
| `bool get isProcessing` | 134 |
| `String? get currentModelId` | 137 |
| `Map<String, dynamic> get currentConfig` | 140 |
| `Future<bool> initialize({ModelService?, Map<String, dynamic>?})` | 143–166 |
| `Future<void> dispose()` | 169–174 |
| `Future<List<EngineModel>> getAvailableModels()` | 177–198 |
| `Future<bool> loadModel(String, {void Function(double)?})` | 201–301 |
| `Future<void> unloadModel()` | 304–310 |
| `Future<String?> detectLanguage(Float32List)` | 319–333 |
| `Future<TranscriptionResult> transcribe(...)` | 336–598 |
| `Future<List<TranscriptionSegment>> _runChunkedWhisper(...)` | 602–707 |
| `static TranscriptionSegment shiftSegmentByOffset(...)` | 535–560 |
| `static Float32List _trimLeadingSamples(...)` | 565–573 |
| `static TranscriptionSegment shiftSegmentForResume(...)` | 579–600 |
| `Future<List<crispasr.Segment>> _runTranscription(...)` | 709–756 |
| `Future<List<crispasr.SessionSegment>> _runSessionTranscription(...)` | 758–816 |
| `List<TranscriptionSegment> _mapWhisperSegments(...)` | 818–860 |
| `List<TranscriptionSegment> _mapSessionSegments(...)` | 862–909 |
| `Stream<TranscriptionSegment>? transcribeStream(...)` | 912–1013 |
| `Future<void> cancel()` | 1015–1017 |
| `Future<void> updateConfig(Map<String, dynamic>)` | 1019–1021 |

**`initialize()` verbatim** (`lib/engines/crispasr_engine.dart`, lines
~101–127 per WebFetch):

```dart
@override
Future<bool> initialize(
    {ModelService? modelService, Map<String, dynamic>? config}) async {
  try {
    _config = Map<String, dynamic>.from(config ?? const {});
    _modelService = modelService;
    if (_modelService != null) {
      await _modelService!.initialize();
      _lidService = LidService(_modelService!);
      _alignerService = AlignerService(modelService: _modelService);
    }
    _alignerService ??= AlignerService();
    _isInitialized = true;
    final libName = crispasr.CrispASR.defaultLibName();
    final backends = crispasr.CrispasrSession.availableBackends();
    Log.instance.i('crispasr', 'engine initialised', fields: {
      'lib': libName,
      'backends': backends.join(','),
      'count': backends.length,
    });
    return true;
  } catch (e, st) {
    Log.instance.e('crispasr', 'Initialize failed', error: e, stack: st);
    throw EngineInitializationException(
      'Failed to initialize CrispASR engine: $e',
      engineId,
      e,
    );
  }
}
```

**`loadModel()` backend-availability gate** (~lines 169–179):

```dart
final available = crispasr.CrispasrSession.availableBackends();
Log.instance.d('crispasr',
    'Available backends in libwhisper: ${available.join(", ")}');
if (!available.contains(def.backend)) {
  throw ModelLoadException(
    'Model uses the ${def.backend} backend. The bundled libwhisper '
    'was built with {${available.join(", ")}}. Rebuild CrispASR '
    'with the ${def.backend} backend linked in.',
    engineId,
    modelId,
  );
}
```

### 5.2 The `package:crispasr` FFI binding

Source: `flutter/crispasr/lib/src/crispasr.dart` in
`CrispStrobe/CrispASR`. Package version `0.5.7` per its own `pubspec.yaml`.
Dart SDK lower bound `3.1.0` because earlier Dart lacked
`NativeCallable.listener` and inline struct fields.

#### Public classes (skeleton, observed from WebFetch summary)

```dart
class CrispASR {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  // 30+ optional FFI function pointers (resolved lazily via
  // providesSymbol)
  late final _WhisperFull _full;
  late final _VoidPtr _free;
  late final _IntPtr _nSegments;
  // ...

  static String defaultLibName();
  static List<String> _libCandidates();

  List<Segment> transcribePcm(Float32List pcm, {/* TranscribeOptions */});
  LanguageDetection detectLanguage(Float32List pcm, {/* ... */});
  List<VadSpan> vad(Float32List pcm, {/* ... */});
  StreamingSession openStream({/* ... */});
  void dispose();
}

class CrispasrSession {
  final DynamicLibrary _lib;
  Pointer<Void> _handle;
  final String _backend;
  bool _closed = false;

  factory CrispasrSession.open(String modelPath, {/* ... */});
  factory CrispasrSession.openWithParams(String modelPath, {/* ... */});
  static List<String> availableBackends();

  List<SessionSegment> transcribe(Float32List pcm, {/* ... */});
  List<SessionSegment> transcribeVad(
      Float32List pcm, String vadModelPath, {/* ... */});
  void close();
}

class StreamingSession {
  final Pointer<Void> _handle;
  final _StreamFeed _feedFn;
  StreamingUpdate? feed(Float32List pcm);
  StreamingUpdate? flush();
  void close();
}
```

#### C-ABI signatures (verbatim from package source)

Session API (the canonical 0.4.0+ surface used by every backend):

```dart
// crispasr_session_open(const char* model_path, int32_t flags)
final open = lib.lookupFunction<
    Pointer<Void> Function(Pointer<Utf8>, Int32),
    Pointer<Void> Function(Pointer<Utf8>, int)>(
  'crispasr_session_open',
);

// crispasr_session_transcribe_lang(session, float* pcm, int32_t n,
//                                   const char* lang)
final fn = _lib.lookupFunction<
    Pointer<Void> Function(Pointer<Void>, Pointer<Float>,
        Int32, Pointer<Utf8>),
    Pointer<Void> Function(Pointer<Void>, Pointer<Float>,
        int, Pointer<Utf8>)>(
  'crispasr_session_transcribe_lang',
);

// crispasr_session_result_n_segments(session) -> int32_t
final nSegs = _lib.lookupFunction<
    Int32 Function(Pointer<Void>),
    int Function(Pointer<Void>)>(
  'crispasr_session_result_n_segments');
```

Legacy whisper.cpp API (still bound for backward compatibility):

```dart
typedef _WhisperFullNative = Int32 Function(
    Pointer<Void>, Pointer<Void>, Pointer<Float>, Int32);
typedef _WhisperFull = int Function(
    Pointer<Void>, Pointer<Void>, Pointer<Float>, int);

final _full = _lib.lookupFunction<_WhisperFullNative, _WhisperFull>(
    'whisper_full');
```

**Tolerant lookup** for backend-specific optional symbols (the package
binds 30+ optional functions only if they exist in the linked library):

```dart
if (_lib.providesSymbol('crispasr_params_set_language')) {
  _paramsSetLanguage = _lib.lookupFunction<
      _ParamsSetStringNative, _ParamsSetString>(
      'crispasr_params_set_language');
}
// later, at the call site:
if (opts.language != null) {
  langPtr = opts.language!.toNativeUtf8();
  _paramsSetLanguage?.call(params, langPtr);  // null-safe
}
```

### 5.3 Audio-buffer marshalling (verbatim from package source)

Dart → C copy via `calloc<Float>`:

```dart
final samples = calloc<Float>(pcm.length);
for (var i = 0; i < pcm.length; i++) {
  samples[i] = pcm[i];   // element-wise copy
}

try {
  final ret = _full(_ctx, params, samples, pcm.length);
  if (ret != 0) throw Exception('Transcription failed');
} finally {
  calloc.free(samples);
}
```

C → Dart for a buffer the C side allocated (e.g., audio-decoder output):

```dart
// C returns native buffer pointer; copy to Dart ownership
final copy = Float32List(n);
final srcView = ptr.asTypedList(n);
copy.setAll(0, srcView);   // detach from C lifetime
free(ptr);                 // free C buffer immediately
```

String marshalling:

```dart
// Dart → C
Pointer<Utf8> langPtr = language.toNativeUtf8();
_paramsSetLanguage?.call(params, langPtr);
calloc.free(langPtr);

// C → Dart
final textPtr = _getText(_ctx, i);
final text = textPtr == nullptr ? '' : textPtr.toDartString();
```

For Utf8 buffers needing native allocation the package documents
(`LEARNINGS.md`) that `calloc<Utf8>(n)` fails on newer Dart FFI with
*"'Utf8' is not a 'SizedNativeType'"*. The workaround is to allocate as
bytes and cast:

```dart
final outBuf = calloc<Uint8>(256);
final outCode = outBuf.cast<Utf8>();
// ... C call populates buffer ...
final code = outCode.toDartString();
calloc.free(outBuf);
```

### 5.4 Engine-level call into the package (from `crispasr_engine.dart`)

```dart
return _model!.transcribePcm(
  pcm,
  options: crispasr.TranscribeOptions(
    language: (language == null || language == 'auto') ? null : language,
    detectLanguage: false,
    wordTimestamps: wordTimestamps,
    silent: false,
    translate: translate,
    strategy: beamSearch ? 1 : 0,
    initialPrompt: prompt,
    bestOf: bestOf,
    vad: useVad,
    vadModelPath: useVad ? vadModelPath : null,
    vadThreshold: advanced.vadThreshold,
    vadMinSpeechMs: advanced.vadMinSpeechMs,
    vadMinSilenceMs: advanced.vadMinSilenceMs,
    tdrz: advanced.tdrz,
    maxLen: advanced.maxLen,
    splitOnWord: advanced.splitOnWord,
  ),
);
```

For session transcription:
`_session!.transcribe(pcm, language: langHint)` or
`_session!.transcribeVad(pcm, vadModelPath, ...)`.

### 5.5 `NativeCallable.listener` for streaming-mic callbacks

```dart
final trampoline = NativeCallable<_MicCallbackNative>.listener(
    (Pointer<Float> pcm, int n, Pointer<Void> _) {
  final view = pcm.asTypedList(n);
  final copy = Float32List.fromList(view);   // detach from C lifetime
  try {
    callback(copy);
  } catch (e) {
    // audio thread mustn't propagate Dart exceptions back to C
  }
});

final h = fn(sampleRate, channels, trampoline.nativeFunction, nullptr);
// Keep trampoline alive for session lifetime
return Mic._(lib, h, trampoline);

// On close:
_trampoline.close();   // release callback handle
```

### 5.6 Error propagation across the FFI boundary

`crispasr_engine.dart` wraps every FFI entry point in `try/catch` and
re-throws a typed exception:

```dart
} catch (e, st) {
  Log.instance.e('crispasr', 'Initialize failed', error: e, stack: st);
  throw EngineInitializationException(
    'Failed to initialize CrispASR engine: $e', engineId, e);
}
```

```dart
} catch (e, st) {
  _model = null;
  _session = null;
  _currentModelId = null;
  _currentModelPath = null;
  done(error: e);
  Log.instance.e('crispasr', 'Model load failed',
      error: e, stack: st,
      fields: {'model': modelId, 'backend': def.backend,
               'path': modelPath});
  throw ModelLoadException(
    'CrispASR failed to load $modelId: $e', engineId, modelId, e);
}
```

```dart
} catch (e, st) {
  if (e is EngineException) rethrow;
  Log.instance.e('crispasr', 'Transcription failed',
      error: e, stack: st);
  throw TranscriptionException(
      'CrispASR transcription failed: $e', engineId, e);
}
```

The C-ABI itself uses return codes (`Int32`) rather than exceptions. The
package code typically throws `Exception('...failed')` when the integer
return is non-zero (e.g., the `whisper_full` call shown above).

### 5.7 Async / isolate handling

- The transcription call **blocks** the platform thread for its full
  duration. The single `await Future<void>.delayed(Duration.zero)` before
  the FFI call yields the microtask queue once, then commits.
- For parallel multi-file workloads, `transcription_worker_pool.dart`
  spawns N isolates, each carrying its own `CrispASR` / `CrispasrSession`
  handle. FFI handles never cross the isolate port.
- Audio buffers cross isolate boundaries as `Float32List` via
  `SendPort.send()`; segments come back the same way.
- Progress callbacks are Dart-side (`onProgress?.call(...)`); the package
  also wires `NativeCallable.listener` trampolines for the C→Dart
  direction in mic streaming.

### 5.8 Memory ownership rules

| Buffer | Allocator | Freed by |
|---|---|---|
| Input PCM `Pointer<Float>` (one shot) | Dart (`calloc<Float>`) | Dart (`calloc.free` in `finally`) |
| Result segment text `Pointer<Utf8>` | C (engine-owned) | Engine; Dart copies to `String` via `toDartString()` |
| C-allocated audio-decoder output | C (`malloc`) | Dart calls `free(ptr)` immediately after copying into `Float32List` |
| Language string parameter `Pointer<Utf8>` | Dart (`.toNativeUtf8()`) | Dart (`calloc.free`) |
| `NativeCallable.listener` trampoline | Dart (`NativeCallable<>.listener(...)`) | Dart (`.close()` when session ends) |
| `CrispASR._ctx` opaque pointer | C (engine init) | Dart calls `_model.dispose()` → C cleanup |
| `CrispasrSession._handle` opaque pointer | C (session open) | Dart calls `_session.close()` → C cleanup |

`CrispASREngine.dispose()` calls `_model?.dispose()` and
`_session?.close()` (lines ~305–306).

### 5.9 macOS lifecycle race noted in `LEARNINGS.md`

> "Global `std::vector<ggml_metal_device>` destructors running during
> `exit()` cause assertion failures. Solution: dispose engine state in
> `AppLifecycleListener.onExitRequested` before returning
> `AppExitResponse.exit`, ensuring Metal cleanup completes before global
> destructors run."

---

## 6. Library-discovery fallback chain

Implemented entirely inside `package:crispasr` at
`flutter/crispasr/lib/src/crispasr.dart`. Verbatim:

```dart
static String defaultLibName() {
  for (final name in _libCandidates()) {
    try {
      DynamicLibrary.open(name);   // probe
      return name;
    } catch (_) { /* try next */ }
  }
  return _libCandidates().first;
}

static List<String> _libCandidates() {
  if (Platform.isAndroid || Platform.isLinux) {
    return ['libcrispasr.so', 'libwhisper.so'];
  }
  if (Platform.isIOS || Platform.isMacOS) {
    return [
      'libcrispasr.dylib',
      'crispasr.framework/crispasr',
      'libwhisper.dylib',
      'whisper.framework/whisper',
    ];
  }
  if (Platform.isWindows) {
    return ['crispasr.dll', 'whisper.dll'];
  }
  return ['libcrispasr.so', 'libwhisper.so'];
}
```

Search-path semantics rely on the platform dynamic loader's own resolution:

- **macOS / iOS:** the calls to `DynamicLibrary.open('libcrispasr.dylib')`
  use dyld's standard search — `Frameworks/` inside the `.app` bundle,
  `@rpath/` entries baked in by `install_name_tool -change ... @rpath/...`
  during bundling (see §3.2), then `DYLD_LIBRARY_PATH`, then standard
  paths.
- **Linux:** loader uses `LD_LIBRARY_PATH`, the Flutter bundle's
  `lib/` directory (set as `RPATH` by the Flutter runner build), and
  system paths.
- **Windows:** `LoadLibrary` searches the application directory first,
  which is where `bundle_windows_dlls.ps1` deposits `crispasr.dll` and
  `whisper.dll`.
- **Android:** Flutter's `jniLibs/<abi>/` is on the loader path; the
  `release.yml` Android job drops `libwhisper.so` (and sibling backend
  `.so` files) there.

**Behaviour on missing library:**
`defaultLibName()` returns `_libCandidates().first` after every probe
fails — so a downstream `DynamicLibrary.open(name)` will throw an
`ArgumentError` / `Invalid argument(s): Failed to load dynamic library`
exception. `CrispASREngine.initialize()` catches this and rethrows it
inside an `EngineInitializationException` (see §5.6). The README's
*Building* section also documents an additional informal layer:
`"At runtime, CrispASREngine resolves the library by probing
platform-specific names — crispasr.dll / libcrispasr.dylib / libcrispasr.so
first, then the whisper-named alias — under the bundle, the user's
CrispASR checkout, system lib dirs, and any user-supplied override
path."` However, the actual `defaultLibName()` implementation observed
above does not expose a user-override parameter; the probe list is
hard-coded. Any "user-supplied override path" referenced in the README
is not represented in the source extracted by WebFetch. See section 17.

**Aliasing strategy** (`LEARNINGS.md`): *"`libcrispasr.dylib` is a
symlink (Unix) or copy (Windows) of `libwhisper.dylib` — they're
identical, not independently versionable."* This is why probing
`libcrispasr.*` first works even when the actual file on disk is named
`libwhisper.*`.

---

## 7. Model management UI

Implementation: `lib/services/model_service.dart` (95,395 bytes — the
single largest service file) plus `lib/screens/model_management_screen.dart`
(18,915 bytes).

### 7.1 HuggingFace API probing

```dart
final url = 'https://huggingface.co/api/models/${repo.repoId}?blobs=true';
final resp = await _dio.get<dynamic>(url, options: Options(headers: headers));
```

The `?blobs=true` query parameter exposes per-file byte sizes in the
response, which the catalog uses to display "real" sizes alongside
estimates from the static `ModelDefinition` map. Authorization uses a
`Bearer $token` header when `_settingsService.hfToken` is non-empty.

Download URL convention:
`https://huggingface.co/{repoId}/resolve/main/{fileName}`

CoreML companion fetch for macOS/iOS Whisper:

```dart
final zipUrl =
    'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$stem-encoder.mlmodelc.zip';
final resp = await _dio.download(zipUrl, zipPath);
```

Gated by:

```dart
if (modelDef.backend == 'whisper' && modelDef.fileName.endsWith('.bin') &&
    (Platform.isMacOS || Platform.isIOS)) {
  await _maybeFetchCoreMLCompanion(modelDef, modelDir);
}
```

### 7.2 GGUF quantisation-variant discovery

Regex for quant-suffix detection:

```dart
final m = RegExp(r'-(q[0-9][a-z_0-9]*|f16|f32|bf16)$').firstMatch(stem);
String _inferQuant(String stem) => m == null ? 'f16' : m.group(1)!;
```

Filename parsing inside `_probeRepo`:

```dart
if (stem == repo.baseName) {
  quant = 'f16';
  modelNameKey = '${repo.baseName}-f16';
} else if (stem.startsWith('${repo.baseName}-')) {
  quant = stem.substring(repo.baseName.length + 1);
  modelNameKey = '${repo.baseName}-$quant';
}
```

The README states the picker supports `q4_0 / q5_0 / q4_k / q5_k / q6_k /
q8_0` plus `f16`/`f32`/`bf16`. The regex above captures everything in
that surface.

### 7.3 Parallel-download mechanics

```dart
final Map<String, CancelToken> _activeDowloads = {};
if (_activeDowloads.containsKey(modelName)) {
  throw ModelException('Download already in progress for $modelName');
}
final cancelToken = CancelToken();
_activeDowloads[modelName] = cancelToken;
```

There is **no global max-parallel cap**: one in-flight download per
`modelName` is permitted, and concurrent downloads of different models
proceed in parallel (Dio handles its own connection pooling). Cleanup
removes the entry on completion or error: `_activeDowloads.remove(modelName);`

### 7.4 SHA-1 checksum verification (verbatim)

```dart
Future<bool> _verifyChecksum(String filePath, String expectedChecksum) async {
  if (expectedChecksum.isEmpty) return true;
  final file = File(filePath);
  if (!await file.exists()) return false;
  final result = await Isolate.run(() async {
    final bytes = await File(filePath).readAsBytes();
    final digest = sha1.convert(bytes);
    return digest.toString();
  });
  return result.toLowerCase() == expectedChecksum.toLowerCase();
}
```

Uses the `crypto` package (`sha1.convert`) and runs inside `Isolate.run`
so multi-GB binaries do not block the UI isolate.

The README claims "SHA-1 verify"; the code matches.

### 7.5 Skip-checksum toggle

Settings integration:
```dart
bool get skipChecksum => _settingsService.skipChecksum;
```

Conditional verification at download time:

```dart
if (modelDef.checksum.isNotEmpty && !skipChecksum) {
  final isValid = await _verifyChecksum(tempPath, modelDef.checksum);
  if (!isValid) {
    await File(tempPath).delete();
    throw const ModelException(
        'Download verification failed. File may be corrupted. '
        'Enable "Skip checksum verification" in Settings → '
        'Debugging to bypass.');
  }
} else if (skipChecksum) {
  Log.instance.i('model',
      'Skipping checksum for $modelName (user override)');
}
```

And for the post-download integrity gate:
```dart
if (!skipChecksum && modelDef.checksum.isNotEmpty &&
    modelDef.sizeBytes > 100 * 1024 * 1024) {
  return await _verifyChecksum(localPath, modelDef.checksum);
}
```

### 7.6 Cache directory structure

```dart
Future<void> initialize() async {
  final appDir = await getApplicationDocumentsDirectory();
  _modelsDir = path.join(appDir.path, 'models');
}
```

Default subdirectory for Whisper ggml models: `<app-docs>/models/whisper_cpp`.

Custom-path override with silent fallback:

```dart
String whisperCppDir() {
  final override = _settingsService.customModelsDir;
  if (override.isNotEmpty) {
    try {
      final dir = Directory(override);
      if (!dir.existsSync()) dir.createSync(recursive: true);
      return override;
    } catch (e) {
      Log.instance.w('model',
          'customModelsDir unusable, falling back to sandbox');
    }
  }
  return path.join(_modelsDir, 'whisper_cpp');
}
```

### 7.7 Resume-download capability

```dart
int downloadedBytes = 0;
if (await file.exists()) {
  downloadedBytes = await file.length();
  onStatusChange?.call('Resuming download...');
}

final headers = <String, dynamic>{
  'Accept': '*/*',
  'Accept-Encoding': 'identity', // Disable compression for resume
};
if (downloadedBytes > 0 && downloadedBytes < expectedSize) {
  headers['Range'] = 'bytes=$downloadedBytes-';
}
```

The download call:

```dart
await _dio.download(
  url, savePath,
  options: Options(headers: headers),
  cancelToken: cancelToken,
  onReceiveProgress: (received, total) { /* ... */ },
);
```

Size-tolerance for "complete" determination (accepts 5% undershoot or
≥ 2 MB):

```dart
final finalSize = await file.length();
if (expectedSize > 0 && finalSize < expectedSize) {
  final diff = expectedSize - finalSize;
  final tolerance = (expectedSize * 0.05).ceil();
  final absTolerance = tolerance > 2 * 1024 * 1024 ? tolerance
                                                    : 2 * 1024 * 1024;
  if (diff > absTolerance) { /* error */ }
}
```

### 7.8 Catalog structure

Three static-or-near-static maps unified through a single lookup:

```dart
static const Map<String, ModelDefinition> whisperCppModels = {
  'large-v3': ModelDefinition(
    name: 'large-v3',
    displayName: 'Whisper Large v3',
    fileName: 'ggml-large-v3.bin',
    url: '$whisperCppBaseUrl/ggml-large-v3.bin',
    sizeBytes: 3000 * 1024 * 1024,
    checksum: 'ad82bf6a9043ceed055076d0fd39f5f186ff8062',
    // ...
  ),
  'large-v3-q5_0': ModelDefinition(/* ... */),
};

static const Map<String, ModelDefinition> crispasrBackendModels = {
  'parakeet-tdt-0.6b-v3-q4_k': ModelDefinition(
    backend: 'parakeet', kind: ModelKind.asr),
  'kokoro-82m-q8_0': ModelDefinition(
    backend: 'kokoro', kind: ModelKind.tts,
    companions: ['kokoro-voice-af_heart']),
};

static final Map<String, ModelDefinition> _ttsVoicepacks = () {
  const vibevoiceVoices = <List<String>>[
    ['en-Carter_man', 'en', 'English — Carter (m)'],
    // ...
  ];
  final out = <String, ModelDefinition>{};
  for (final v in vibevoiceVoices) {
    out['vibevoice-voice-${v[0]}'] = ModelDefinition(
      kind: ModelKind.voice,
    );
  }
  return out;
}();
```

HuggingFace repo metadata for live probing:

```dart
static const Map<String, BackendRepo> backendRepos = {
  'parakeet': BackendRepo(
    backend: 'parakeet',
    repoId: 'cstr/parakeet-tdt-0.6b-v3-GGUF',
    baseName: 'parakeet-tdt-0.6b-v3',
    displayPrefix: 'Parakeet TDT 0.6B v3',
    extension: '.gguf',
  ),
};
```

Unified lookup precedence (live-probed wins over static):

```dart
ModelDefinition? lookupDefinition(String name) {
  return _discoveredModels[name]
      ?? whisperCppModels[name]
      ?? crispasrBackendModels[name]
      ?? _ttsVoicepacks[name];
}
```

Per README, Whisper ggml files come from `ggerganov/whisper.cpp`, and
quantised variants from `cstr/whisper-ggml-quants` and other `cstr/*-GGUF`
repos.

---

## 8. Audio capture + file ingestion

`lib/services/audio_service.dart` (16,297 bytes).

- **File decoding:** primary path is `crispasr.decodeAudioFile()` —
  i.e. the FFI engine itself performs the decode. Fallback is a
  custom Dart WAV parser (`_basicWavProcessing`). `just_audio` is used
  only for *header-only* duration probing, not decoding. The CrispASR
  README mentions a `crisp_audio` decoder shipped inside the engine
  ("WAV / MP3 / FLAC decoded on-device, no ffmpeg required"); the
  CrisperWeaver wrapper calls into it via the package.
- **Sample-rate resampling:** none explicit in the service.
  Recording and streaming always capture at fixed 16 kHz. File decoding
  appears to assume the consumer (typically Whisper) tolerates the
  input rate, or that `crispasr.decodeAudioFile()` normalises to 16 kHz
  internally.
- **Microphone path:**
  - File-based recording: `_recorder.start()` with WAV encoding at a
    configurable bitrate (the `record: ^6.2.0` package).
  - Streaming: `_recorder.startStream()` returns realtime `Float32List`
    frames; int16 little-endian is converted to normalised float
    `[-1, 1]` in-stream.
- **URL download:** simple HTTP GET via the `http` package; entire
  response is buffered before write; output file is timestamp-named in
  the app-documents directory. No chunked transfer or resume here (the
  resumeable downloader is in `model_service`, not `audio_service`).
- **File drop / share intake:** handled by `lib/services/share_intake_service.dart`
  (7,342 bytes) + `desktop_drop: ^0.7.0` for desktop drag-drop +
  `receive_sharing_intent: ^1.8.0` for Android share sheet. The
  2026-05-11 commit `8a6e5df` extended this to parse SubRip + WebVTT +
  plain-text transcripts as review-mode input.
- **Large-file chunking:** no chunking inside `audio_service`. Full
  file is loaded into memory as `Float32List`. The chunking visible
  elsewhere — Whisper's 30-second window splitting in
  `_runChunkedWhisper` — runs *after* the full file is in memory.

---

## 9. Decoding-parameter surface

Source-of-truth for parameters: `AdvancedTranscribeOptions` in
`lib/services/transcription_service.dart`. Observed fields (extracted by
WebFetch):

| Field | Type |
|---|---|
| `vadBackend` | `VadBackend` |
| `vadThreshold` | `double` |
| `vadMinSpeechMs` | `int` |
| `vadMinSilenceMs` | `int` |
| `vadSpeechPadMs` | `int` |
| `diarizeMethod` | `crispasr.DiarizeMethod` |
| `lidMethod` | `crispasr.LidMethod` |
| `tdrz` | `bool` |
| `tokenTimestamps` | `bool` |
| `puncFamily` | `String` |
| `lidUseGpu` | `bool` |
| `lidFlashAttn` | `bool` |
| `nThreads` | `int` |
| `asrUseGpu` | `bool` |
| `asrFlashAttn` | `bool` |
| `asrNGpuLayers` | `int` |
| `maxLen` | `int` |
| `splitOnWord` | `bool` |

Additional decoding parameters surfaced through `crispasr.TranscribeOptions`
in the FFI call (extracted from `crispasr_engine.dart` verbatim, §5.4):

- `language` (string code, or `null` for `'auto'`)
- `detectLanguage` (bool)
- `wordTimestamps` (bool)
- `silent` (bool — suppress engine-internal logging)
- `translate` (bool — Whisper translate-to-English)
- `strategy` (int — `1` if beamSearch else `0`)
- `initialPrompt` (string — vocabulary bias, exposed in UI as the
  custom-vocabulary chip list per README)
- `bestOf` (int — README "best-of-N slider 1–10")
- `vad` (bool)
- `vadModelPath` (string)
- `vadThreshold`, `vadMinSpeechMs`, `vadMinSilenceMs` (from advanced)
- `tdrz` (bool — TinyDiarize / speaker-turn token)
- `maxLen` (int — token cap per segment, the SRT-friendly short-line
  toggle)
- `splitOnWord` (bool)

The README also enumerates UI-side controls:

- best-of-N slider (1–10, picks the highest-scoring decode)
- decoder temperature slider on every backend that honours
  `crispasr_session_set_temperature` (the README lists: canary, cohere,
  parakeet, moonshine, voxtral, qwen3, granite, glm-asr, gemma4,
  omniasr-llm, kyutai-stt)
- VAD picker: Silero (bundled), FireRedVAD, MarbleNet, Whisper-VAD-EncDec
- Diarisation method: vad-turns (default), pyannote, stereo energy,
  stereo cross-correlation
- LID method: Whisper-encoder, Silero 95-langs
- Source + target language pickers (Canary)
- Audio Q&A prompt (Voxtral / Qwen3 instruct mode)
- initial-prompt vocabulary bias (Whisper / Moonshine), `setAsk`-prefix
  variant for audio-LLM backends, no-op for CTC-style with explanatory
  helper text
- tokens-per-segment cap + split-on-word toggle (subtitle formatting)
- translate-to-English toggle (Whisper)
- punctuation restorer family picker (FireRedPunc vs fullstop-punc)

---

## 10. Telemetry surface

- The README states: *"See live performance numbers — real-time factor,
  words per second, wall-clock."*
- Direct numeric source: `crispasr_engine.dart` reports timing via the
  engine; `transcription_service.dart` aggregates and emits
  `onProgress?.call(p)` with `p ∈ [0.0, 1.0]` across stages. The flow,
  per WebFetch summary of `transcription_service.dart`:
  - audio load: 10%
  - engine transcription: 60% (scaled in-flight:
    `onProgress?.call(0.1 + progress * 0.6)`)
  - diarisation: 20%
  - punctuation: 5%
  - complete: 100%
- Per-chunk progress for chunked Whisper:
  `onProgress?.call(remaining <= 0 ? 1.0 : (i - firstChunk + 1) / remaining)`
  (`crispasr_engine.dart` line ~705).
- Segment-level streaming: `onSegment?.call(shifted)` (line ~697) fires
  per finished window in the chunked path; a `StreamController<TranscriptionSegment>`
  feeds true streaming mic transcription (line ~945).
- RTF / words-per-second numerics are computed Dart-side from
  `DateTime`-based wall-clock measurement around the FFI call plus the
  segment text length. WebFetch did not extract the exact Dart math; see
  section 17.
- Native-thread → UI delivery: progress callbacks are invoked from inside
  the FFI call. The actual blocking is on the platform thread the FFI is
  running on; Flutter's framework dispatches the callback to the UI
  thread via Dart's event loop without explicit `runOnUiThread`-style
  hops. For the streaming-mic case, the `NativeCallable.listener`
  trampoline allows the native audio thread to deliver buffers without
  blocking on a Dart return.

---

## 11. Backend coverage in UI

The README enumerates the full backend matrix exposed in the picker.
Combined ASR table (verbatim from README):

| Family | Sizes | Languages |
|---|---|---|
| Whisper | tiny → large-v3 + q4_0/q5_0/q8_0 | 99 |
| Parakeet (NVIDIA) | tdt-0.6b-v3 | 25 EU |
| Canary (NVIDIA) | 1b-v2 | 25 EU (explicit src/tgt) |
| Qwen3-ASR | 0.6b | 30 + 22 Chinese dialects |
| Cohere | 03-2026 | 13 |
| Granite Speech | 3.2-8b, 3.3-2b/8b, 4.0-1b, 4.1-2b | en fr de es pt ja |
| FastConformer-CTC | small → xxlarge | en |
| Canary-CTC | 1b | 25 EU |
| Voxtral Mini | 3B (2507), 4B realtime (2602) | 8 / 13 |
| Wav2Vec2 | large-xlsr-53-english + variants | per-model |
| OmniASR LLM | 300M v2 | multilingual |
| FireRed ASR2 | aed-2b | zh / en |
| Kyutai STT | 1b | en |
| GLM-ASR Nano | nano | multilingual |
| Moonshine | tiny / base + streaming | en |
| VibeVoice ASR | large (~4.5 GB) | multilingual |
| MiMo ASR | 2.5B + tokenizer companion | en zh |
| Gemma4-E2B | 2B (q4_k) | 140+ languages |
| OmniASR LLM unlim. | 300M v2 streaming | 1600+ |
| Granite Speech 4.1 | 2B / 4.1+ / 4.1-NAR | en fr de es pt ja |

TTS families: Kokoro, VibeVoice, Qwen3-TTS, Orpheus, Chatterbox,
Kartoffelbox (German Chatterbox derivative), IndexTTS.

Post-processors: FireRedPunc, Fullstop-punc.

VAD / LID / diarisation GGUFs: Pyannote v3, Silero LID 95, FireRedVAD,
MarbleNet VAD, Whisper-VAD-EncDec.

**Default selection.** From `settings_service.dart` (per WebFetch): the
keys `preferred_engine`, `default_model`, `default_backend`,
`default_language` are persisted. The README says: *"Default pick is
Whisper base (~140 MB, covers 99 languages)"* and *"CrispASR is the
default"* preferred engine. The actual default values stored in the
SharedPreferences map are not exposed by the WebFetch summary; see
section 17.

**Live registry probing.** Per README: *"The Model Management screen
also probes CrispASR's built-in C-side registry on every open, so any
backend the bundled libwhisper knows about appears even if it isn't
hardcoded in the app catalog."* The C call used for this is
`CrispasrSession.availableBackends()` (called in `initialize()` and
referenced in `loadModel()` — see §5.1).

**Per-backend UI differences.** The README documents three categories:

1. `whisper / moonshine` → "initial_prompt" path for custom-vocabulary
   chips.
2. Audio-LLM backends (Voxtral / Qwen3 / Granite) → `setAsk` prefix
   instead of `initial_prompt`.
3. CTC-style backends (wav2vec2 / fastconformer-ctc / firered-asr) →
   custom-vocabulary chip list is a no-op with explanatory helper text.

The dispatcher class name is `CrispasrSession` (see §5.2). All backends
flow through one engine wrapper (`CrispASREngine`), differentiated by
`def.backend` strings.

---

## 12. Cross-platform reach

Per README *"Platforms"* table:

| Platform | State |
|---|---|
| macOS | "Released — `.app.zip`, Metal-enabled, all 24+ backend dylibs bundled, espeak-ng auto-bundled for kokoro phonemisation" |
| Linux | "Released — `.tar.gz` bundle" |
| Windows | "Released — `.zip` with whisper.dll + sibling backend DLLs" |
| Android | "Released — real-ASR APK (arm64-v8a) with libwhisper.so cross-built in CI" |
| iOS | "Unsigned IPA — sideload via SideStore / AltStore / Feather" |

**Per-platform differences observed in source:**

- **macOS:** Metal + CoreML build flags
  (`DGGML_METAL=ON`, `DCRISPASR_COREML=ON`); CoreML companion auto-fetch
  for Whisper models; Services menu integration (commit `814407c`,
  2026-05-11); Open-With handler via `OpenWithReceiver.swift` (commit
  `7e5d2e2`); system-audio capture via ScreenCaptureKit (macOS 13+).
- **Linux:** GTK-3 desktop, libmpv-backed audio
  (`just_audio_media_kit` + `media_kit_libs_linux`), `parec` for
  system-audio capture, `.desktop` file with `Exec=crisper_weaver %F`
  for file-manager Open-With (commit `8a6e5df`),
  `libkeybinder-3.0-dev` for `hotkey_manager_linux` (commit
  `6b76203`, 2026-05-11), `libayatana-appindicator3-dev` slated for
  future tray-icon work.
- **Windows:** WASAPI loopback via ffmpeg for system-audio capture
  (per README), `media_kit_libs_windows_audio` for `just_audio` on
  Windows, no installer (zipped runner directory only).
- **Android:** real ASR uses cross-compiled `libwhisper.so` in
  `jniLibs/arm64-v8a/`; MediaProjection-based system-audio on Android
  10+; share-intent MIME filters include `application/x-subrip` and
  `text/vtt`; unsigned APK; arm64-v8a only.
- **iOS:** XCFramework slices (device arm64 + simulator arm64); Share
  Extension target files exist as templates in `ios/ShareExtension/`
  but per the 2026-05-11 commit message the Xcode target wiring is
  "the tracked follow-up"; CoreML deployment target 14.0+; espeak-ng
  disabled in the iOS build; no system-audio capture by Apple sandbox
  rule.

**Tested platforms.** The README marks all 5 as "Released" with
matching release assets on v0.5.0; the `release.yml` workflow builds
all 5. PLAN.md flags Windows as `continue-on-error` in CI "until
real-machine verification" and iOS as "⚠️" with multiple blockers
(mic permission prompt behaviour, streaming 16 kHz PCM16,
recording↔playback transitions, background audio survival, share-intake
security-scoped paths, CoreML companion loading,
`PrivacyInfo.xcprivacy` for App Store submission).

**v0.5.0 release asset sizes** (re-listed for cross-platform
comparison): macOS 29.2 MB, Linux 22.9 MB, Windows 28.3 MB, Android
36.3 MB, iOS 16.9 MB. The Android APK is the largest single artifact.

---

## 13. Persistence + settings

`lib/services/settings_service.dart` (17,331 bytes).

- **Storage:** `SharedPreferences` exclusively (no file-based persistence
  inside this service). `SharedPreferences` itself maps to
  `NSUserDefaults` on iOS/macOS, `SharedPreferences` Android API,
  `LocalStorage`-style files on desktop platforms — the per-platform
  path is not surfaced in the service.
- **Encryption / secure storage:** none in this service. The settings
  service contains a comment (per WebFetch summary): *"for real secret
  storage we'd reach for `flutter_secure_storage` — out of scope for
  the v1 opt-in cleanup feature"*. Sensitive fields (`hf_token`,
  `cloud_llm_api_key`) are logged only as `SET` / `EMPTY` to avoid leaks.
- **Schema versioning:** none detected.

**Persisted keys** (from WebFetch enumeration of
`settings_service.dart`):

- Transcription: `preferred_engine`, `default_model`, `default_backend`,
  `default_language`, `auto_detect_language`, `enable_word_timestamps`
- Audio: `audio_quality`, `keep_audio_files`
- Diarisation: `enable_diarization_by_default`
- Localisation: `app_locale`
- Debug / dev: `log_level`, `log_to_file`, `skip_checksum`, `hf_token`,
  `custom_models_dir`, `group_batch_by_backend`,
  `max_concurrent_transcriptions`, `max_concurrent_sessions`
- Cloud LLM: `cloud_llm_api_url`, `cloud_llm_api_key`, `cloud_llm_model`
- Local LLM: `llm_cleanup_mode`, `local_llm_model_path`,
  `local_llm_n_gpu_layers`, `local_llm_n_ctx`, `local_llm_n_threads`,
  `local_llm_max_tokens`, `local_llm_temperature`
- Hotkey: `hotkey_enabled`, `hotkey_combo`, `hotkey_action`
- UI state: `edit_audio_show_transcript`

Additional file-based persistence appears outside the settings service:
- Presets: `lib/services/preset_service.dart` (12,628 bytes) —
  "JSON-backed with schema-versioned migration" per the README.
- History: `lib/services/history_service.dart` (7,141 bytes) — every
  run persisted as JSON under `<app-docs>/history/` per the README.
- Batch state: `batch_persistence_service.dart` (13,620 bytes).

---

## 14. Logging + error surfaces

`lib/services/log_service.dart` (11,467 bytes).

- **Levels:** five — `trace`, `debug`, `info`, `warn`, `error` — each
  tagged TRC/DBG/INF/WRN/ERR. `LogLevel` enum supports min-level filtering.
- **Sinks:**
  - In-memory ring buffer (capacity 5000 by default) exposed via
    `snapshot()`.
  - Broadcast `Stream<LogEntry>` for live UI subscription.
  - Optional file sink at `<app-docs>/logs/session.log`, enabled via
    `Log.instance.enableFileSink(true)` from `main.dart`.
- **Windows-detached-stderr workaround:** `LEARNINGS.md` describes a
  `FileSystemException` that escapes try/catch when GUI-detached
  Windows builds write to an invalid stderr handle. Fix:
  `stdioType(stderr) != StdioType.other` guard before any `writeln`.
- **FFI / native error surfacing:** the log service has no FFI
  bindings of its own; native errors surface through standard `Log.e()`
  calls inside `crispasr_engine.dart` (see §5.6 verbatim). Each typed
  exception (`EngineInitializationException`, `ModelLoadException`,
  `TranscriptionException`) carries the underlying engine error
  message as its third positional argument.
- **In-app log viewer:** `lib/screens/logs_screen.dart` (6,997 bytes).
  Per the README: *"in-app viewer with filter / search / copy /
  export, optional file sink."*
- **Crash handler:** `LEARNINGS.md` says the logger does not hook
  `FlutterError.onError` or `platformDispatcher.onError`. No
  automatic crash interception; only explicitly-passed exceptions
  reach the logger.

---

## 15. Maintainer activity

- **Sole contributor:** `CrispStrobe` (Christian Ströbele, Stuttgart,
  Germany per README) with 137 contributions. The repo has 0 external
  contributors per `gh api repos/CrispStrobe/CrisperWeaver/contributors`.
- **Commit cadence (recent, from `gh api commits?per_page=5`):**
  - `9b93f86` 2026-05-12 06:08 UTC — `chore: bump version to 0.5.0`
  - `6b76203` 2026-05-11 21:04 UTC — CI fix for `libkeybinder-3.0-dev`
  - `814407c` 2026-05-11 18:26 UTC — macOS Services menu "Transcribe
    with CrisperWeaver"
  - `7e5d2e2` 2026-05-11 18:23 UTC — macOS Open-With bridge
  - `8a6e5df` 2026-05-11 17:46 UTC — Platform-native share / transcript
    intake
  - Every recent commit message includes `Co-Authored-By: Claude Opus 4.7
    (1M context) <noreply@anthropic.com>`.
- **Release cadence:** 10 versioned releases between 2026-04-18 and
  2026-05-12 — roughly one release every 2.4 days during the window.
  All releases are produced by `github-actions[bot]` from tag pushes,
  no manually-uploaded artifacts visible.
- **Issue response patterns:** only 1 issue in the repo's history
  (#1, 2026-04-24, `MissingPluginException` for `just_audio` —
  `disposeAllPlayers` channel method missing). The issue was closed by
  the reporter `BFG-BFG`; no comment thread is visible in the API
  response (a single state transition). The fix shipped in
  `LEARNINGS.md`'s `just_audio` note: route Windows / Linux through
  `just_audio_media_kit` before any player is constructed.
- **PR history:** zero pull requests in the entire repo lifetime
  (`gh api pulls?state=all` returns empty).
- **Compatibility / version-pinning posture with CrispASR:**
  - `pubspec.yaml` uses `crispasr: path: ../CrispASR/flutter/crispasr`
    — a local-path dependency, no version constraint.
  - The two CI workflows (`ci.yml` and `release.yml`) honour
    `CRISPASR_REPO` and `CRISPASR_REF` env vars at the top of each
    file; the default is `CrispStrobe/CrispASR@main`. No tag-locking
    by default — CrisperWeaver builds against whatever is on
    `CrispASR/main` at the time the workflow runs.
  - The README's roadmap section pins specific CrispASR symbol names
    introduced in 0.4.5 / 0.4.6 / 0.4.7 (`crispasr_diarize_segments_abi`,
    `crispasr_detect_language_pcm`, `crispasr_align_words_abi`)
    indicating the consumer tracks the engine version closely.
  - The package's own version is `0.5.7` per
    `flutter/crispasr/pubspec.yaml` (separate from the CrisperWeaver
    `0.5.0+1`).

---

## 16. Recent issues + PRs

- **Open issues:** 0.
- **Closed issues:** 1 total (#1, `MissingPluginException`, see §15).
- **PRs:** none (zero).
- **WIP signals visible in `PLAN.md` (summarised from WebFetch):**
  - Flash-attention wiring across the five non-Whisper backends that
    have the toggle but no kernel-level consumer (~2–3 days work).
  - GPU backend selector (blocked on multi-backend ggml dispatch).
  - Curated chat-model catalogue (download URLs + recommended params)
    extending the existing file-picker MVP (~half day).
  - Grammar-constrained sampling (GBNF) for Whisper structured output
    (~2–3 days, CrispASR-side work).
  - Beam search for `granite`, `voxtral`, `qwen3` (per-backend
    high-level API exposure, ~1–1.5 days).
  - CoreML pipeline cache via `MTLBinaryArchive` (test-suite §5.18,
    "38× cold-start improvement" already shipped per PLAN.md; further
    CoreML-on-iOS / Apple-Silicon Whisper is deferred).
  - Re-download q4_k variants for vibevoice/orpheus (blocked on HF
    availability).
- **Unresolved technical debt enumerated in PLAN.md:**
  - "37 package constraint overrides" awaiting Flutter 3.39.
  - macOS notarisation (currently ad-hoc only).
  - Android APK signing (currently unsigned).
  - Windows installer (no MSI/EXE step exists).
  - i18n migration incomplete (en/de scaffold done, older Settings and
    legacy-widget strings remain hard-coded English).
  - iOS device-test blockers (mic permission, streaming 16 kHz PCM16,
    background audio, share-intake security-scoped paths, CoreML
    companion loading, `PrivacyInfo.xcprivacy`).

---

## 17. Could not verify

The following claims and details could not be confirmed from primary
source within the WebFetch sample. Most are due to file size (files
> 50 KB returned only AI-summarised content rather than verbatim) or
the path not being explicit in the visible code.

1. **README claim of a "user-supplied override path" in the library
   resolver.** The README states *"any user-supplied override path"*
   is part of the runtime library probe order, but the
   `defaultLibName()` / `_libCandidates()` implementation extracted
   from `flutter/crispasr/lib/src/crispasr.dart` is hard-coded (no
   parameter, no env-var lookup). If the override exists it lives
   either in an unsurfaced API on `CrispASR` (perhaps a static setter)
   or further down in the 106 KB binding file beyond the slice
   WebFetch summarised.
2. **Exact CrispASR / CrisperWeaver version-pinning relationship.**
   `pubspec.yaml` uses a `path:` dependency only; CrisperWeaver
   `0.5.0+1` ships with whichever `flutter/crispasr/pubspec.yaml`
   exists on `CrispASR/main` at build time (currently package version
   `0.5.7`). No tag-locking is enforced in CI by default.
3. **Default values for persisted settings keys.** The list of keys
   in `settings_service.dart` was captured but per-key default
   (e.g., the default `preferred_engine`, `default_model`) is not
   visible in the WebFetch summary.
4. **Exact RTF / words-per-second computation site.** The README
   advertises "real-time factor, words per second, wall-clock" but the
   Dart code that computes them was not surfaced; it is most likely in
   `transcription_screen.dart` (94 KB) or `transcription_service.dart`.
5. **Per-platform `SharedPreferences` storage path.** The service uses
   `SharedPreferences` exclusively; the underlying per-platform file
   path was not surfaced (this is plugin-internal anyway).
6. **Cohere backend version note.** README lists "Cohere | 03-2026 |
   13 languages | High-accuracy Conformer decoder" but the model
   identifier and HF repo are not enumerated in the visible
   `model_service.dart` slice.
7. **iOS Share Extension wiring status.** The 2026-05-11 commit
   message says template files exist in `ios/ShareExtension/` but the
   Xcode target wiring is a "tracked follow-up". The actual
   `project.pbxproj` state was not inspected.
8. **Detailed `chat.dart` FFI surface** (the on-device chat-model
   ABI for cleanup / summarise) — the 15 KB
   `flutter/crispasr/lib/src/chat.dart` file was not fetched.
9. **`server_service.dart` route surface.** The README mentions an
   OpenAI-compatible HTTP server on
   `/v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/translations`,
   but the actual route handlers in the 18 KB service were not
   inspected in detail.
10. **Issue #1 comment thread.** The API response includes the issue
    but not its comments; whether the reporter received a maintainer
    response before closing it themselves is not verifiable here.
11. **Verbatim text of `LICENSE`** — only its size (34,523 bytes,
    consistent with full AGPL-3.0 text) and the GitHub API's
    `spdx_id: AGPL-3.0` classification were confirmed. The file
    content was not opened.
12. **Coverage of `analysis_options.yaml` strict rules.** The README
    enumerates `use_build_context_synchronously`, `avoid_print`,
    `unused_*`, `inference_failure_*`, `deprecated_member_use`
    erroring rules, but the analysis file itself was not inspected.
13. **System-audio capture per-platform implementation details.**
    `lib/services/system_audio_capture_service.dart` (15,817 bytes)
    was not opened beyond the README's high-level claim
    (ScreenCaptureKit / parec / WASAPI / MediaProjection).
14. **PLAN.md numerical roadmap items.** Specific section numbers
    referenced in commit messages (§5.1, §5.1.6, §5.1.11, §5.8,
    §5.18) were captured as themes but the literal section text was
    not pulled.
15. **Engine factory `EngineType` enum values and `engineFactoryProvider`
    construction.** Only the high-level shape (Mock + CrispASR enum,
    Riverpod `engineManagerProvider`) was surfaced. The verbatim
    file content was not pulled.
