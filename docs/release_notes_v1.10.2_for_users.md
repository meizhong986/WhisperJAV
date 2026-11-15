# WhisperJAV 1.10.2

These notes compare 1.10.2 with 1.10.1, the release before it.

**Coming from 1.9.x or earlier: do not upgrade. Do a complete, fresh install, as for 1.10.0** (see [Installation](#installation)). Your settings and downloaded models are kept.

**从 1.9.x 或更早版本而来：请不要升级，请像安装 1.10.0 时一样全新完整安装**（参见[安装](#installation)）。您的设置和已下载的模型会保留。

## Key points

- **Upgrading from 1.10.1:** close WhisperJAV and run `"%LOCALAPPDATA%\WhisperJAV\python.exe" -m whisperjav.upgrade` in Command Prompt, not `Scripts\whisperjav-upgrade.exe`. The steps are under [Installation](#upgrading-from-1101).
- **Two new GGUF pipelines, experimental**, in the Ensemble tab (Pipeline menu, group "Experimental, open to feedback"): haas (Haas-ja-v1.0) and qwen3-asr (Qwen3-ASR 1.7B, timed by the aligner). Japanese only. CrispASR, which runs them, is updated to 0.8.53.
- **New speech enhancers and speech segmenters, run by CrispASR.** Speech Enhancer menu: Mel-Band-RoFormer and Mini-BS-RoFormer; Mini-BS-RoFormer replaces BS-RoFormer, which could not start. Speech Segmenter menu, under "Experimental, open to feedback": Sortformer (NVIDIA Nemotron) and pyannote 3.0.
- **"For VAD only" is ticked by default** in the Speech Enhancer menu of fidelity, ChronosJAV and the GGUF pipelines: the cleaned-up audio is used to find the speech, and the original audio goes to the ASR model. Balanced, fast, faster and transformers do not offer it.
- **A run now downloads the aligner and timing model it uses before any audio is read** (the GGUF pipelines that run the aligner; ChronosJAV when you choose them), and the models of the new speech enhancers and speech segmenters, and checks WhisperSeg's packages. If one of these is missing or cannot run, the run stops at once and says why, instead of going on without it.
- **Claude translation:** the Anthropic package it needs is now installed; 1.10.0 and 1.10.1 lacked it. Claude's default model is Claude Haiku 5.5.
- **The Windows installer's changes come with a fresh install only:** FFmpeg 8.1.2, a WhisperJAV entry in the Start menu, and an uninstaller that removes this install's shortcuts. An install upgraded from 1.10.1 keeps FFmpeg 7.1.1, its Start menu and its uninstaller.
- **Not fixed yet:** ZipEnhancer cleans only the first file of a single-pass run, and of an Ensemble run with "Finish each file" unticked. MOSS's token limit and its 30-second pieces are unchanged. Both are planned for 1.10.3; see [Limits](#limits-and-known-issues-in-1102).

## New in this release

- Added ASR pipeline haas-crispasr: Haas-ja-v1.0 (GGUF)
- Added ASR pipeline qwen3-asr-crispasr: Qwen3-ASR 1.7B (GGUF), with the aligner
- Added speech enhancer Mel-Band-RoFormer: vocal separation, run by CrispASR
- Added speech enhancer Mini-BS-RoFormer: vocal separation, run by CrispASR
- Added speech segmenter Sortformer (NVIDIA Nemotron), experimental
- Added speech segmenter pyannote 3.0, experimental, on the CPU
- Moved "Enhance for VAD only" into the Speech Enhancer menu, as "For VAD only"
- Updated both Claude menus to Haiku 5.5, Sonnet 5.5, Opus 5.5 (from Claude 3.x)
- Added to `whisperjav --check`: version, where PyTorch loads from, network
- Added ChronosJAV's aligner and timing model to `--dump-params`

## Changed defaults (what you will notice after upgrading)

- "For VAD only" is ticked by default (fidelity, ChronosJAV, GGUF pipelines)
- Presets saved before 1.10.2 load with "For VAD only" ticked
- Claude's default model is claude-haiku-5-5 (was claude-3-5-haiku-20241022)
- A run stops at the start if its aligner or timing model cannot be downloaded
- ChronosJAV stops at the start if CrispASR cannot run its aligner or timing model
- WhisperSeg stops at the start when a package it needs is missing
- With `--skip-existing`, the start downloads run even when every file is done
- A speech enhancer WhisperJAV no longer offers (bs-roformer) stops the run
- `--async-processing` with `--mode transformers` or `--mode qwen` stops, saying why
- Merge strategy "Smart Merge (Best Quality)" is now called "Smart Merge"
- Kotoba Bilingual is asked for English with Direct to English (#469); see Limits
- Start is greyed while a run starts; the status line shows "Starting..."
- A run stops at the start if another run uses its working folder (`--temp-dir`)
- Closing the GUI window ends the running run

To keep 1.10.1's behaviour:
- Cleaned-up audio to the ASR model too: untick "For VAD only" in the Speech Enhancer menu. On the command line: `--no-pass1-enhance-for-vad`, `--no-pass2-enhance-for-vad`, or `--no-enhance-for-vad` for `--mode qwen`.

## Enhancements

- Updated CrispASR to 0.8.53 (from 0.8.50)
- Updated PySubtrans to 1.8.0 (from 1.6.0); a re-run retries batches with errors
- The Windows installer now brings FFmpeg 8.1.2 (before: 7.1.1).
- Changed the Speech Enhancer menu: Vocal Separation, Noise Suppression, DSP
- WhisperSeg's message names only the missing packages, with their install line
- A saved speech enhancer no longer offered is marked, and Start is refused
- anime-whisper's Text Cleaner names its own cleaner, not "Passthrough"
- ChronosJAV prints no console line per scene when no aligner runs
- GGUF pipelines' log says "the groups span N s of M s", not "N s of speech"

## Fixed

- Fixed Claude translation: the Anthropic package is now installed
- Fixed Test Connection saying "Invalid API key" for working keys (#465)
- Fixed ChronosJAV files marked "suspect" when the aligner timed part (#501)
- Fixed AI translation always asking the model to translate "from Japanese" (#508)
- Fixed a new speech segmenter or scene detector running with old Customize values
- Fixed "Error during temporary file cleanup" after each ChronosJAV file
- Fixed GGUF passes going on without the aligner after a failed download
- Fixed GGUF passes taking a Silero v6.2, whisper-vad or nemo failure as silence
- Fixed GGUF passes not checking their speech enhancer before the run starts
- Fixed WhisperSeg giving no subtitles on a minimal install; it stops at start
- Fixed `whisperjav-translate --configure` stopping for ollama and custom
- Fixed `whisperjav-translate --configure` and `--show-settings` asking for `-i`
- Fixed `--model` with balanced, fast or faster picking a GPU they cannot use
- Fixed XXL Faster Whisper Extra Args losing the backslashes of Windows paths
- Fixed two translations at once changing each other's instructions
- Fixed the uninstaller leaving the desktop shortcut behind (1.10.2 installer)
- Fixed the Start-menu folder holding only "Git Bash" (1.10.2 installer)
- Fixed console errors of `whisperjav.exe` when its output is redirected
- Fixed `whisperjav-translate --help` showing "usage: cli.py"
- Fixed the GUI starting several runs when Start was clicked again while starting
- Fixed two runs in one working folder deleting each other's files
- Fixed a run going on after the GUI window was closed
- Fixed the GUI console losing lines after a second start

## Which pipeline for which computer

You choose your kind of computer once, when you install WhisperJAV.

| Your computer | What uses the graphics card |
|---|---|
| NVIDIA graphics card | Every pipeline. |
| AMD or Intel graphics card, or an older NVIDIA card (Windows) | The GGUF pipelines (Ensemble tab, Pipeline menu groups "GGUF via crispasr" and "Experimental, open to feedback"), through Vulkan. Experimental, not yet tested. The other pipelines run on the processor, several times slower. |
| Mac with Apple silicon | transformers asks for the Apple GPU; the GGUF pipelines and fidelity ask for it too (not yet tested on a Mac). balanced, fast, faster and ChronosJAV run on the processor. |
| No graphics card | None: every pipeline runs on the processor, several times slower than on a graphics card. |

## Limits and known issues in 1.10.2

**Experimental or not yet tested**
- The GGUF pipelines are experimental (early release). They run as Ensemble passes only, for Japanese only.
- Sortformer and pyannote 3.0 are experimental and use the same settings at every sensitivity; a VAD threshold does not reach them. Sortformer runs where the install's CrispASR runs (on an NVIDIA install the GPU), also with `--device cpu`; pyannote 3.0 runs on the CPU. `whisperjav --check` does not list their models, nor those of the CrispASR speech enhancers.
- Mel-Band-RoFormer is fast on an NVIDIA GPU and very slow on the processor.
- MOSS: CrispASR decodes MOSS's audio in pieces of at most 30 seconds, cut at quiet points. Work on this and on MOSS's token limit is planned for 1.10.3.
- MOSS removes every subtitle made only of ああ, あっ, うん or えっ, real ones too.
- GPU through Vulkan and your own PyTorch are experimental and untested on AMD and Intel cards. Intel GPUs are not used through PyTorch. On Linux, the GGUF pipelines run on the CPU for AMD and Intel GPUs. ChronosJAV's aligner and timing model have not been tried through Vulkan.
- Nobody in the project has run 1.10.2 on a Mac or a Linux PC yet; the Apple GPU is not yet tested.
- The GGUF pipelines need folder names in plain Western letters (ASCII). When your user folder has Chinese, Japanese or Korean characters, WhisperJAV links or copies the model to a plain-named folder.
- The GGUF pipelines, the CrispASR speech enhancers and speech segmenters, and the CrispASR aligner and timing model need a processor with AVX2 (Intel since 2013, AMD since 2015). For the speech enhancers and speech segmenters this is not checked before the run starts.
- `--device cpu` does not keep the CrispASR aligner and timing model on the CPU when CrispASR's build uses a GPU.

**Known faults, planned for 1.10.3**
- ZipEnhancer: from the second file of a single-pass run, or of an Ensemble run with "Finish each file" unticked, files get no clean-up. The run goes on and says so only in a warning line. Way round it: tick "Finish each file", or run one file at a time.
- The upgrade removes every desktop shortcut named "WhisperJAV v…", those of other installs too, and makes one for 1.10.2.
- On an install on the drive of Windows' temporary folder, the upgrade leaves a copy of the old launcher (40 KB) in the temporary folder.
- The decoupled pipeline (`--pipeline decoupled`, command line only) downloads its aligner during the run, and goes on without it if the download fails.

**Other known faults**
- Kotoba Bilingual (the transformers pipeline's default model) with Direct to English: the model gives the English text no times, so the subtitles are unusable and the file may come out empty.
- Timing model, when chosen: a very short stretch inside a longer VAD group can still stop the process (the crash of #498). We have not seen this in our tests. With "Finish each file" only that file is lost, and a ChronosJAV Ensemble pass keeps its subtitles made so far once its ASR step is done.
- Translation: subtitles the model did not translate are left out of the translated file. The run names them and ends with TRANSLATION INCOMPLETE, but its exit status is still 0.
- ChronosJAV and the decoupled pipeline with the auditok or silero scene detector (the default is semantic): every scene shorter than 12 seconds is dropped, and its speech with it. Keep the default, or untick Safe Chunking in the pass's Customize window; on the command line, add `--no-qwen-safe-chunking`.
- ChronosJAV, the decoupled pipeline and fidelity: when the speech segmenter fails on a scene, an error line is printed, but the run summary has no note of it.
- The upgrade tool's rollback (`--rollback`, which the tool suggests when it finishes, and the rollback it tries after a failed upgrade) does not work on installs made with the Windows installer: it stops with an error. If WhisperJAV no longer starts after an upgrade, uninstall it and install 1.10.2 with the Windows installer (see [Installation](#upgrading-from-1101)).
- `Scripts\whisperjav-upgrade.exe` of 1.10.0, 1.10.1 and 1.10.2 stops halfway on an install on another drive than Windows' temporary folder (normally drive C:), and WhisperJAV then no longer starts. Upgrade with `"%LOCALAPPDATA%\WhisperJAV\python.exe" -m whisperjav.upgrade` instead.
- `whisperjav-upgrade` does not remove packages a new release no longer uses. 1.10.2 no longer uses five: bs-roformer-infer, rotary-embedding-torch, beartype, ml-collections and absl-py.
- XXL Faster Whisper as pass 1 crashed once when closing, on one of our test files.
- A run refused because another run uses its working folder is refused only after its start-of-run checks and model downloads are done.
- Closing the GUI window while a run is still starting: if the start takes longer than 30 seconds, that run goes on. End it in Task Manager.
- If a run's main process is ended from outside (for example in Task Manager) and one of its worker processes goes on, a new run can start in the same working folder.
- In the Chinese GUI, the text under "The run did not start" for a working folder in use is in English.

## Installation

### Before you start

- **Do not install or upgrade with `pip install -U whisperjav`.** On Windows it replaces your GPU build of PyTorch with a CPU-only one. Use the upgrade tool, the installer or the install scripts below.
- Your settings and downloaded models are kept in every case below; they are stored outside the install folder.
- Keep the internet connection on during the whole upgrade or installation. The first run of each GGUF pipeline downloads its model at the start of the run (up to about 4.7 GB for the default files; Cohere-CrispASR and Qwen3-ASR-CrispASR also fetch the 232 MB aligner). **Users in China:** the upgrade and the installers download from GitHub, PyPI, download.pytorch.org and Hugging Face; keep your VPN on throughout.

| Your computer | Method |
|---|---|
| Windows with 1.10.1 installed by the installer | **Upgrade from 1.10.1** (below) |
| Windows 10 or 11 (64-bit, Intel or AMD processor), new install | **Windows installer** (recommended) |
| Mac with Apple silicon (M1 or newer), macOS 14 or newer | macOS from source |
| Linux with glibc 2.35 or newer (Ubuntu 22.04, Debian 12, Fedora 36 or newer) | Linux from source |
| Windows, if you prefer a source install | Windows from source |

### Upgrading from 1.10.1

**Windows installer installs:** the GUI's update notice brings you to this page; the upgrade itself runs in Command Prompt.
1. Close WhisperJAV (every window of it).
2. Open Command Prompt (press Start, type `cmd`, press Enter) and run:
   ```
   "%LOCALAPPDATA%\WhisperJAV\python.exe" -m whisperjav.upgrade
   ```
   If you installed into another folder, put that folder in place of `%LOCALAPPDATA%\WhisperJAV`, here and in step 4.
3. When it asks `Continue? [y/N]:`, type `y` and press Enter. It installs 1.10.2 with its tested package versions, CrispASR 0.8.53, PySubtrans 1.8.0 and the Anthropic package (for Claude) included, and keeps the accelerator of your install. It does not change FFmpeg (it stays 7.1.1), the Start menu or the uninstaller; those come with the 1.10.2 Windows installer.
4. When it says "Upgrade complete!", check the version: `"%LOCALAPPDATA%\WhisperJAV\Scripts\whisperjav.exe" --version` prints `WhisperJAV 1.10.2`.

**Do not start this upgrade with `Scripts\whisperjav-upgrade.exe`.** When WhisperJAV is installed on another drive than Windows' temporary folder (normally drive C:), it stops halfway and WhisperJAV no longer starts.

**If WhisperJAV no longer starts after an upgrade:** uninstall it (`Uninstall-WhisperJAV.exe` in the install folder, or Windows **Settings > Apps**), then install 1.10.2 with the Windows installer (below). Your settings and downloaded models stay.

Adding `--wheel-only` to the command in step 2 updates WhisperJAV alone and keeps CrispASR 0.8.50 and PySubtrans 1.6.0; it ends with "Hot-patch complete!". Claude translation then stays unavailable, because the Anthropic package is not installed.

**Source installs:** in the `WhisperJAV` folder, with the same environment active as at install (for `--local` on Windows, run `.venv\Scripts\activate` first), run `git fetch --tags`, then `git checkout v1.10.2`, then the install script again as at install (on Windows with `--local`). It offers your install's accelerator as the default. Do not use `whisperjav-upgrade` on a source install: in that folder it runs `git pull`, which fails on a clone of a release.

### Coming from 1.9.x

1.10 needs Python 3.13, so `whisperjav-upgrade` cannot take a 1.9.x install to it. Install 1.10.2 fresh with the method for your computer below.
- **Windows installer:** you can install into the same folder. The installer offers to replace the old version and removes everything in that folder first; your settings and downloaded models stay.
- **Source installs:** first rename or delete your 1.9.x `WhisperJAV` folder and your 1.9.x environment (`~/venvs/whisperjav` on macOS, `~/.venv/whisperjav` on Linux).

### Windows installer (recommended)

1. Download `WhisperJAV-1.10.2-Windows-x86_64.exe` from the [Releases page](https://github.com/meizhong986/WhisperJAV/releases/latest).
2. Run it and follow the steps. It brings its own Python and FFmpeg and installs every package WhisperJAV needs; you do not need to install Python yourself.
3. **What the install is made for:** a supported NVIDIA GPU (GTX 16 / RTX 20 or newer, driver 580 or newer) is used by itself. Otherwise a window asks: AMD, Intel or older NVIDIA card (these use GPU through Vulkan), your own PyTorch, or CPU only. With no NVIDIA card, CPU only is installed after 20 seconds without an answer. With an NVIDIA card 1.10 cannot use, the window names the reason and waits.
4. Wait until the installer says it has finished; the last step installs the packages and can take several minutes.
5. Start WhisperJAV from the desktop shortcut "WhisperJAV v1.10.2", or from WhisperJAV in the Start menu.
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
   git clone --branch v1.10.2 https://github.com/meizhong986/WhisperJAV.git
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
   git clone --branch v1.10.2 https://github.com/meizhong986/WhisperJAV.git
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
   git clone --branch v1.10.2 https://github.com/meizhong986/WhisperJAV.git
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
(Use the folder you installed into if you changed it.) For a source install, run `whisperjav --version` and `whisperjav --check` with its environment active. The first prints `WhisperJAV 1.10.2`. The second names the install's accelerator and where each part runs, and lists the models of the GGUF pipelines and the CrispASR aligners and timing model as downloaded or not downloaded. It also names WhisperJAV's version, where PyTorch is loaded from, and whether the download addresses answer. When you report a problem, include this output.

## Credits

- CrispASR, which runs the GGUF pipelines, the new speech enhancers and speech segmenters, and ChronosJAV's aligner and timing model, is developed at [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR).
- PySubtrans, which runs the AI translation, is developed at [machinewrapped/llm-subtrans](https://github.com/machinewrapped/llm-subtrans). Its version 1.8.0 flags an answer label left in a subtitle, which we reported there (llm-subtrans issue #541).
- Models of the new pipelines: Haas-ja-v1.0 by third-intelligence (Apache-2.0); Qwen3-ASR 1.7B by Qwen (Apache-2.0); both in GGUF form by cstr.
- Models of the new speech enhancers and speech segmenters: NVIDIA Nemotron-3-Diarization (OpenMDW-1.1); pyannote segmentation-3.0, Mel-Band-RoFormer vocals and Mini-BS-RoFormer, in GGUF form by cstr (MIT).
- The reports behind this release's fixes: #469 (Kotoba bilingual), #465 (Test Connection), #501 (ChronosJAV's partial alignment), #508 (steve1316 found that the chosen source language did not reach the translator).
