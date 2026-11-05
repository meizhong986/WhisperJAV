# WhisperJAV 1.10.1

**Thank you, CrispASR.** [CrispASR](https://github.com/CrispStrobe/CrispASR) runs the GGUF pipelines and ChronosJAV's aligner and timing model. Its developer answered our requests quickly: CrispASR 0.8.50 added the sampling and beam size that MOSS uses in this release. Thank you for the rapid response and support.

These notes compare 1.10.1 with 1.10.0, the release before it.

**Coming from 1.9.x or earlier: do not upgrade. Do a complete, fresh install, as for 1.10.0** (see [Installation](#installation)). Your settings and downloaded models are kept.

**从 1.9.x 或更早版本而来：请不要升级，请像安装 1.10.0 时一样全新完整安装**（参见[安装](#installation)）。您的设置和已下载的模型会保留。

## Key points

- **Upgrading from 1.10.0:** close WhisperJAV and run `"%LOCALAPPDATA%\WhisperJAV\python.exe" -m whisperjav.upgrade` in Command Prompt, not `Scripts\whisperjav-upgrade.exe`. The steps are under [Installation](#upgrading-from-1100).
- **ChronosJAV (Qwen3-ASR, anime-whisper) now times each subtitle by its VAD group by default.** No aligner and no timing model run unless you choose them in Customize. ChronosJAV presets saved in 1.10.0 keep their aligner and timing model.
- **"Finish each file" is on by default**, in the Ensemble tab and on the command line. Each file is finished before the next one starts, and each pass's models are loaded again for every file.
- **Five new GGUF pipelines, experimental**, in the Ensemble tab (Pipeline menu, group "Experimental, open to feedback"): reazonspeech, kotoba, whisper-turbo, whisper-large and MOSS. Japanese only. CrispASR, which runs them, is updated to 0.8.50.
- **Fixed:** the Qwen3-ASR fine-tunes JA Anime-Galgame and JA-tuned (#494). The timing model is no longer given audio under 0.2 s (#498, #500). Customize Apply no longer changes values outside a slider's range. When an Ensemble pass crashes, the files it had finished are kept, and a ChronosJAV pass keeps its subtitles made so far once its ASR step is done.
- **Not fixed yet:** ZipEnhancer cleans only the first file of a single-pass run, and of an Ensemble run with "Finish each file" unticked. BS-RoFormer cannot start on the command line. Both are planned for 1.10.2; see [Limits](#limits-and-known-issues-in-1101).

## New in this release

- Added ASR pipeline reazonspeech-crispasr: ReazonSpeech NeMo v2 (GGUF)
- Added ASR pipeline kotoba-crispasr: Kotoba-Whisper v2.0
- Added ASR pipeline whisper-turbo-crispasr: Whisper large-v3-turbo
- Added ASR pipeline whisper-large-crispasr: Whisper large-v2 or large-v3
- Added ASR pipeline moss-crispasr: MOSS-Transcribe-Diarize 0.9B (GGUF)
- Added full-size (f16) model files for parakeet and cohere
- Added MOSS tab: instruction language, hotwords, token limit, sampling, beam size
- Added Timing tab for GGUF passes: word times from the model or the aligner
- Added `--pass1-word-timestamps` and `--pass2-word-timestamps` (model, aligner)
- Added speech segmenter None for GGUF pipelines: whole scenes go to the model
- Added Shortest Group (s) to FireRedVAD, TEN VAD, WhisperSeg, Silero v6.2
- Added GUI log file `raw_subs\whisperjav_run.log`; each run replaces the last

## Changed defaults (what you will notice after upgrading)

- ChronosJAV (Qwen3-ASR, anime-whisper): Timestamp Mode VAD Only by default
- ChronosJAV: no aligner and no timing model by default; both stay in Customize
- ChronosJAV presets saved in 1.10.0 keep their aligner and timing model
- `--qwen-aligner` without `--qwen-timestamp-mode` brings `aligner_vad_fallback`
- "Finish each file" is ticked by default; `--ensemble-serial` is on by default
- "Finish each file" loads each pass's models again for every file
- Command line: pass 2 Qwen3-ASR starts on TEN VAD (balanced), as in the GUI
- ChronosJAV, GGUF pipelines: VAD groups under 0.2 s are dropped, not transcribed
- Qwen3-ASR: a model that cannot give real text stops the run, with the reason
- The console shows fewer lines; `--debug` brings them back
- `--log-file` also records debug lines; each Ensemble pass writes its own file

To keep 1.10.0's behaviour:
- ChronosJAV timing of 1.10.0: in Customize, Alignment tab, set Timestamp Mode to "Aligner + VAD Fallback" and Timing Model to "Whisper small". On the command line: `--qwen-timestamp-mode aligner_vad_fallback --qwen-timing-model whisper-small`.
- Pass 1 on all files, then pass 2: untick "Finish each file"; on the command line, `--no-ensemble-serial`.
- Pass 2 Qwen3-ASR on WhisperSeg on the command line: `--pass2-speech-segmenter whisperseg`.

## Enhancements

- Updated CrispASR to 0.8.50 (from 0.8.44)
- GGUF model menus show each file's quantization and GPU memory
- GGUF pipelines (not MOSS): hallucination filter and repetition cleaner added
- cohere: lines of only 「あっ……」 or 「ああ……」 are removed
- ChronosJAV and GGUF pipelines keep the subtitles before clean-up in raw_subs
- GGUF pipelines: an artifacts file in raw_subs lists each line removed or changed
- Customize: scene-length sliders reach 600 s (Auditok, Silero, Semantic)
- "Finish each file" is kept by Remember settings
- GUI: the closing line gives the exit status; the counts are in RUN SUMMARY

## Fixed

- Fixed Qwen3-ASR JA Anime-Galgame and JA-tuned writing random text (#494)
- Timing model: audio under 0.2 s is no longer given to it (crash #498, #500)
- Fixed answer labels in other languages (原题>, 翻译>) left in translations (#460)
- Fixed untranslated subtitles going unreported; now named by number (#460)
- Translation ends with TRANSLATION INCOMPLETE when subtitles are missing (#460)
- Fixed Gemma 3 answers in Markdown code fences failing their last subtitle
- Fixed Customize Apply changing values outside a slider's range (e.g. 0.3→0.5 s)
- Fixed a crashed Ensemble pass losing the files it had already finished
- ChronosJAV pass: a crash after the ASR step keeps its subtitles so far
- Fixed red "Aligner native: 0.0%" in PIPELINE ANALYTICS under VAD Only
- Fixed "Run on the CPU this time" not showing its run's output

## Which pipeline for which computer

You choose your kind of computer once, when you install WhisperJAV.

| Your computer | What uses the graphics card |
|---|---|
| NVIDIA graphics card | Every pipeline. |
| AMD or Intel graphics card, or an older NVIDIA card (Windows) | The GGUF pipelines (Ensemble tab, Pipeline menu groups "GGUF via crispasr" and "Experimental, open to feedback"), through Vulkan. Experimental, not yet tested. The other pipelines run on the processor, several times slower. |
| Mac with Apple silicon | transformers asks for the Apple GPU; the GGUF pipelines and fidelity ask for it too (not yet tested on a Mac). balanced, fast, faster and ChronosJAV run on the processor. |
| No graphics card | None: every pipeline runs on the processor, several times slower than on a graphics card. |

## Limits and known issues in 1.10.1

**Experimental or not yet tested**
- The GGUF pipelines are experimental (early release). They run as Ensemble passes only, for Japanese only.
- MOSS: CrispASR decodes MOSS's audio in pieces of at most 30 seconds, cut at quiet points. Work on this and on MOSS's token limit is planned for 1.10.2.
- MOSS removes every subtitle made only of ああ, あっ, うん or えっ, real ones too.
- GPU through Vulkan and your own PyTorch are experimental and untested on AMD and Intel cards. Intel GPUs are not used through PyTorch. On Linux, the GGUF pipelines run on the CPU for AMD and Intel GPUs. ChronosJAV's aligner and timing model have not been tried through Vulkan.
- Nobody in the project has run 1.10.1 on a Mac or a Linux PC yet; the Apple GPU is not yet tested.
- The GGUF pipelines need folder names in plain Western letters (ASCII). When your user folder has Chinese, Japanese or Korean characters, WhisperJAV links or copies the model to a plain-named folder.
- The GGUF pipelines, and the CrispASR aligner and timing model, need a processor with AVX2 (Intel since 2013, AMD since 2015).
- `--device cpu` does not keep the CrispASR aligner and timing model on the CPU when CrispASR's build uses a GPU.

**Known faults, planned for 1.10.2**
- ZipEnhancer: from the second file of a single-pass run, or of an Ensemble run with "Finish each file" unticked, files get no clean-up. The run goes on and says so only in a warning line. Way round it: tick "Finish each file", or run one file at a time.
- BS-RoFormer cannot start on the command line. The GUI shows it as Coming Soon.
- Customize: if you change a pass's speech segmenter or scene detector after Apply, the new one runs with the earlier one's Customize values.
- `--async-processing` with `--mode transformers` or `--mode qwen` may stop with an error. Leave it out for these modes.

**Other known faults**
- Timing model, when chosen: a very short stretch inside a longer VAD group can still stop the process (the crash of #498). We have not seen this in our tests (16,574 VAD groups of six films). With "Finish each file" only that file is lost, and a ChronosJAV Ensemble pass keeps its subtitles made so far once its ASR step is done.
- Translation: subtitles the model did not translate are left out of the translated file. The run names them and ends with TRANSLATION INCOMPLETE, but its exit status is still 0.
- The upgrade tool's rollback (`--rollback`, which the tool suggests when it finishes, and the rollback it tries after a failed upgrade) does not work on installs made with the Windows installer: it stops with an error. If WhisperJAV no longer starts after an upgrade, uninstall it and install 1.10.1 with the Windows installer (see [Installation](#upgrading-from-1100)).
- `Scripts\whisperjav-upgrade.exe` of 1.10.0 and 1.10.1 stops halfway on an install on another drive than Windows' temporary folder (normally drive C:), and WhisperJAV then no longer starts. Upgrade with `"%LOCALAPPDATA%\WhisperJAV\python.exe" -m whisperjav.upgrade` instead.

**Still open from 1.10.0** (the 1.10.0 notes listed these for 1.10.1; they are not fixed):
- ChronosJAV: a run without `--keep-temp` may show "Error during temporary file cleanup" for each file. The subtitles are already written by then.
- ChronosJAV with the auditok or silero scene detector (the default is semantic): scenes shorter than 12 seconds may be dropped, and their speech with them. Keep the default.
- GGUF pipelines: a speech enhancer chosen for the pass is not checked when the run starts, as it is for the other pipelines.
- With the Silero v6.2, whisper-vad or nemo speech segmenter, a segmenter that fails is treated as finding no speech: the scene gets no subtitles and no error is shown.
- The CrispASR aligner and timing model are downloaded the first time they are used, during the run. If the download fails (for example with `--offline`), a GGUF pass goes on without its aligner.
- `whisperjav-translate --configure` stops with an error when you choose ollama or custom. Give the provider on the command line instead, for example `whisperjav-translate --provider ollama …`.
- On a CPU-only install on a PC with an NVIDIA card, `--model` with balanced, fast or faster may make the run try the GPU.
- XXL Faster Whisper: Windows paths in Extra Args lose their backslashes (`D:\models` becomes `D:models`).
- `install.py --minimal` followed by a GGUF pipeline may give no subtitles. Use the full install.
- XXL Faster Whisper as pass 1 crashed once when closing, on one scene in our tests.
- `whisperjav-upgrade` does not remove packages a new release no longer uses (1.10.1 drops none).

## Installation

### Before you start

- **Do not install or upgrade with `pip install -U whisperjav`.** On Windows it replaces your GPU build of PyTorch with a CPU-only one. Use the upgrade tool, the installer or the install scripts below.
- Your settings and downloaded models are kept in every case below; they are stored outside the install folder.
- Keep the internet connection on during the whole upgrade or installation. The first run of each GGUF pipeline downloads its model (up to about 3.1 GB for the default files). **Users in China:** the upgrade and the installers download from GitHub, PyPI, download.pytorch.org and Hugging Face; keep your VPN on throughout.

| Your computer | Method |
|---|---|
| Windows with 1.10.0 installed by the installer | **Upgrade from 1.10.0** (below) |
| Windows 10 or 11 (64-bit, Intel or AMD processor), new install | **Windows installer** (recommended) |
| Mac with Apple silicon (M1 or newer), macOS 14 or newer | macOS from source |
| Linux with glibc 2.35 or newer (Ubuntu 22.04, Debian 12, Fedora 36 or newer) | Linux from source |
| Windows, if you prefer a source install | Windows from source |

### Upgrading from 1.10.0

**Windows installer installs:** the GUI's update notice brings you to this page; the upgrade itself runs in Command Prompt.
1. Close WhisperJAV (every window of it).
2. Open Command Prompt (press Start, type `cmd`, press Enter) and run:
   ```
   "%LOCALAPPDATA%\WhisperJAV\python.exe" -m whisperjav.upgrade
   ```
   If you installed into another folder, put that folder in place of `%LOCALAPPDATA%\WhisperJAV`, here and in step 4.
3. When it asks `Continue? [y/N]:`, type `y` and press Enter. It installs 1.10.1 with its tested package versions, CrispASR 0.8.50 included, and keeps the accelerator of your install.
4. When it says "Upgrade complete!", check the version: `"%LOCALAPPDATA%\WhisperJAV\Scripts\whisperjav.exe" --version` prints `WhisperJAV 1.10.1`.

**Do not start this upgrade with `Scripts\whisperjav-upgrade.exe`,** although the 1.10.0 notes name it. When WhisperJAV is installed on another drive than Windows' temporary folder (normally drive C:), it stops halfway and WhisperJAV no longer starts.

**If WhisperJAV no longer starts after an upgrade:** uninstall it (`Uninstall-WhisperJAV.exe` in the install folder, or Windows **Settings > Apps**), then install 1.10.1 with the Windows installer (below). Your settings and downloaded models stay.

Adding `--wheel-only` to the command in step 2 updates WhisperJAV alone and keeps CrispASR 0.8.44. MOSS's Sampling and Beam size are then not applied, and MOSS says so when it runs.

**Source installs:** in the `WhisperJAV` folder, with the same environment active as at install (for `--local` on Windows, run `.venv\Scripts\activate` first), run `git fetch --tags`, then `git checkout v1.10.1`, then the install script again as at install (on Windows with `--local`). It offers your install's accelerator as the default. Do not use `whisperjav-upgrade` on a source install: in that folder it runs `git pull`, which fails on a clone of a release.

### Coming from 1.9.x

1.10 needs Python 3.13, so `whisperjav-upgrade` cannot take a 1.9.x install to it. Install 1.10.1 fresh with the method for your computer below.
- **Windows installer:** you can install into the same folder. The installer offers to replace the old version and removes everything in that folder first; your settings and downloaded models stay.
- **Source installs:** first rename or delete your 1.9.x `WhisperJAV` folder and your 1.9.x environment (`~/venvs/whisperjav` on macOS, `~/.venv/whisperjav` on Linux).

### Windows installer (recommended)

1. Download `WhisperJAV-1.10.1-Windows-x86_64.exe` from the [Releases page](https://github.com/meizhong986/WhisperJAV/releases/latest).
2. Run it and follow the steps. It brings its own Python and FFmpeg and installs every package WhisperJAV needs; you do not need to install Python yourself.
3. **What the install is made for:** a supported NVIDIA GPU (GTX 16 / RTX 20 or newer, driver 580 or newer) is used by itself. Otherwise a window asks: AMD, Intel or older NVIDIA card (these use GPU through Vulkan), your own PyTorch, or CPU only. With no NVIDIA card, CPU only is installed after 20 seconds without an answer. With an NVIDIA card 1.10 cannot use, the window names the reason and waits.
4. Wait until the installer says it has finished; the last step installs the packages and can take several minutes.
5. Start WhisperJAV from the desktop shortcut "WhisperJAV v1.10.1".
6. Check the install (see [Check your install](#check-your-install)).

To change what the install is made for later, run the installer again into the same folder.

### macOS (Apple silicon, macOS 14 or newer)

1. Install Xcode Command Line Tools: open Terminal and run `xcode-select --install`.
2. Install Homebrew from [brew.sh](https://brew.sh) (one command, shown on its home page). When it finishes, run the lines it prints under "Next steps" so that the `brew` command is found.
3. In Terminal, run these lines one by one:
   ```
   brew install python@3.13 ffmpeg portaudio git
   python3.13 -m venv ~/venvs/whisperjav
   source ~/venvs/whisperjav/bin/activate
   git clone --branch v1.10.1 https://github.com/meizhong986/WhisperJAV.git
   cd WhisperJAV
   chmod +x installer/install_mac.sh
   ./installer/install_mac.sh
   ```
   Git prints a note about "detached HEAD"; that is expected.
4. The install is made for the Apple GPU (Metal). To start the GUI, double-click `WhisperJAV.command` in the `WhisperJAV` folder.
5. Check the install (see [Check your install](#check-your-install)); in a new Terminal, run `source ~/venvs/whisperjav/bin/activate` first.

More detail: [macOS guide](https://meizhong986.github.io/WhisperJAV/guides/installation_mac_apple_silicon/).

### Linux (glibc 2.35 or newer)

1. Install Python 3.13 with `venv`, FFmpeg, Git and the system packages listed in the [Linux guide](https://meizhong986.github.io/WhisperJAV/guides/installation_linux/). For the NVIDIA GPU, NVIDIA driver 570 or newer.
2. In a terminal, run these lines one by one:
   ```
   python3.13 -m venv ~/.venv/whisperjav
   source ~/.venv/whisperjav/bin/activate
   git clone --branch v1.10.1 https://github.com/meizhong986/WhisperJAV.git
   cd WhisperJAV
   chmod +x installer/install_linux.sh
   ./installer/install_linux.sh
   ```
   Git prints a note about "detached HEAD"; that is expected.
3. With a supported NVIDIA GPU the install is made for it. Otherwise the installer asks: 1 = your own PyTorch (it must already be installed in this environment, for example ROCm for an AMD card; otherwise it stops and installs nothing), 2 = CPU only, 3 = stop.
4. For the GUI, add a window backend in the same terminal: `pip install "pywebview[qt]"` (see the Linux guide, "GUI backend").
5. Start WhisperJAV with `./WhisperJAV.sh` in the `WhisperJAV` folder, or `whisperjav-gui` with the environment active.
6. Check the install (see [Check your install](#check-your-install)).

### Windows from source

1. Install Python 3.13 from [python.org](https://www.python.org/downloads/) and tick **Add python.exe to PATH**. Install Git, FFmpeg (on your PATH), the Visual C++ Redistributable and WebView2 as the [Windows Python guide](https://meizhong986.github.io/WhisperJAV/guides/installation_windows_python/) shows.
2. Open Command Prompt (press Start, type `cmd`, press Enter) and run these lines one by one:
   ```
   python --version
   git clone --branch v1.10.1 https://github.com/meizhong986/WhisperJAV.git
   cd WhisperJAV
   installer\install_windows.bat --local
   ```
   The first line must print Python 3.13.x. Git then prints a note about "detached HEAD"; that is expected. `--local` installs everything into a `.venv` folder inside `WhisperJAV`. To choose the accelerator in advance, add it, for example `--local --accelerator vulkan`; your own PyTorch cannot be chosen this way.
3. Start WhisperJAV by double-clicking `WhisperJAV.bat` in the `WhisperJAV` folder.
4. Check the install (see [Check your install](#check-your-install)), running `.venv\Scripts\activate` in the `WhisperJAV` folder first.

### Check your install

On Windows (installer), open Command Prompt and run:
```
cd /d "%LOCALAPPDATA%\WhisperJAV"
Scripts\whisperjav.exe --version
Scripts\whisperjav.exe --check
```
(Use the folder you installed into if you changed it.) For a source install, run `whisperjav --version` and `whisperjav --check` with its environment active. The first prints `WhisperJAV 1.10.1`. The second names the install's accelerator and where each part runs, and lists CrispASR's models as downloaded or not downloaded (they download the first time you use them).

## Credits

- CrispASR, which runs the GGUF pipelines and ChronosJAV's aligner and timing model, is developed at [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR). Its version 0.8.50 brings the sampling and beam size that MOSS uses.
- Models of the new pipelines: ReazonSpeech NeMo v2 by reazon-research (Apache-2.0), in GGUF form by cstr; Kotoba-Whisper v2.0 by kotoba-tech (Apache-2.0); OpenAI's Whisper large-v2, large-v3 and large-v3-turbo (MIT), in the whisper.cpp files of ggerganov/whisper.cpp; MOSS-Transcribe-Diarize 0.9B by OpenMOSS-Team (Apache-2.0), in GGUF form by cstr.
- The reports behind this release's fixes: #494 and #499 (the Qwen3-ASR fine-tunes), #498, #499 and #500 (the timing-model crash), #460 (translation labels in Chinese and untranslated subtitles).
