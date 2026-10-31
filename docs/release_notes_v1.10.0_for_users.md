# WhisperJAV 1.10.0

**Coming from 1.9.x or earlier: do not upgrade — do a complete, fresh install of 1.10.0.** Do not run `whisperjav-upgrade` or the GUI's update button on a 1.9.x install; they cannot take it to 1.10.0. Your settings and downloaded models are kept.

**从 1.9.x 或更早版本而来：请不要升级，请全新完整安装 1.10.0。** 不要在 1.9.x 安装上运行 `whisperjav-upgrade` 或 GUI 中的更新按钮，它们无法升级到 1.10.0。您的设置和已下载的模型会保留。

**1.10.0 首次提供简体中文界面。** 点击窗口顶部的「中文」按钮即可将整个界面切换为中文，点击「English」切换回英文，您的选择会被记住。如果电脑的显示语言是中文，首次启动时直接显示中文界面。GUI 中的运行日志和 Qwen 指南页仍为英文。如果您发现更好的译法，欢迎在这里告诉我们：https://github.com/meizhong986/WhisperJAV/issues

## Key points

- **A fresh install for everyone.** 1.10.0 runs on Python 3.13 and PyTorch 2.11, so a 1.9.x install cannot be upgraded to it. Install it fresh with the steps below; your settings and downloaded models are kept.
- **Made for the graphics card you have.** You choose once, at install: NVIDIA GPU, another graphics card through Vulkan (Windows), the Apple GPU, your own PyTorch, or CPU only. Every run follows that choice. With an AMD or Intel card, only the GGUF pipelines and ChronosJAV's timing use the card.
- **Two new Japanese pipelines, experimental:** parakeet and cohere, run through CrispASR in Ensemble mode. The Ensemble tab now starts with them.
- **Better subtitle timing in ChronosJAV:** subtitles start where their first word starts, and two sentences in one piece of speech can become two subtitles.
- **The GUI in Simplified Chinese**, with a 中文 / English button.
- Removed: llama-cpp translation (use Ollama), ClearVoice, the 1.9.x CrispASR mode. Google Colab and Kaggle are not supported; the Colab notebook keeps installing 1.9.5.

## New in this release

- Added ASR pipeline cohere-crispasr: Cohere Transcribe ja models (GGUF)
- Added ASR pipeline parakeet-crispasr: NVIDIA Parakeet TDT 0.6B ja (GGUF)
- Added GGUF model support through CrispASR 0.8.44
- Added GGUF forced aligners: wav2vec2-ja (default), parakeet-ctc-ja, qwen3-fa
- Added timing model (Whisper small) to split ChronosJAV subtitles at sentences
- Added Chinese GUI (Simplified Chinese), 中文 / English button (#448)
- Added XXL Faster Whisper as Pass 1 (#451, #452)
- Added one accelerator per install: NVIDIA, Vulkan, Apple GPU, own PyTorch, CPU
- Added GPU through Vulkan on Windows: AMD, Intel, older NVIDIA (experimental)
- Added own-PyTorch install, e.g. ROCm (experimental)
- Added installer window to choose the accelerator when no NVIDIA GPU is found
- Added `install.py --accelerator nvidia|vulkan|own-pytorch|cpu`

## Enhancements

- Updated to Python 3.13 and PyTorch 2.11
- Updated to CUDA 13.0 on Windows (driver 580+); CUDA 12.8 on Linux (driver 570+)
- Updated Qwen3-ASR to transformers 5, through WhisperJAV's qwen-asr build
- Updated CTranslate2 to 4.8.3 for balanced, fast and faster (#459)
- Every install and upgrade gets the same locked package versions
- NVIDIA install needs GTX 16 / RTX 20 or newer; older cards: Vulkan or CPU
- Unsupported computers are refused before install, with the reason
- No CPU-mode question; the run shows where each part runs
- `--device auto|cuda|cpu` applies to every pipeline and speech enhancer
- `whisperjav --check` shows the accelerator, CrispASR and its models
- `whisperjav-upgrade` stays within 1.10.x and keeps the accelerator
- ChronosJAV subtitles start at their first word (aligner timing)
- anime-whisper uses the aligner, like Qwen3-ASR
- Adaptive Step-Down is off by default for every ChronosJAV model
- 1.9.5 timing kept as an option: Aligner Backend None, Timing Model None
- ChronosJAV and GGUF pipelines remove 「うん!」「え?」 as they did 「うん。」
- Ensemble tab opens on cohere (Pass 1) and parakeet (Pass 2)
- XXL: each pass runs the model chosen in its Model column
- XXL: the console names the failed pass and the reason
- Clearer messages when a run does not start

## Fixed

- Fixed XXL running large-v3 whatever model was chosen (#449)
- Fixed ChronosJAV Customize → Alignment settings being ignored (since 1.8.10)
- Fixed batch runs counting earlier files' scenes in the ChronosJAV summary
- Fixed wrong "Aligner native" and "Collapse rate" in the ChronosJAV summary
- Fixed answer labels (Translation>, Original>) left in translations (#460)
- Fixed old NVIDIA driver silently getting the CPU install; the installer asks
- Fixed "Failed to run post install script" on a deliberate installer stop
- Fixed the package carrying a developer XXL path and settings backups

## Removed

- **Local translation with llama-cpp.** Use Ollama.
- **ClearVoice.** If you used it, choose ZipEnhancer, htdemucs (Vocals) or None. In the GUI a saved ClearVoice choice marks the pass until you choose another; on the command line a ClearVoice value stops the run before any audio is read.
- **The 1.9.x CrispASR mode**, which ran a `crispasr` program you downloaded. `--mode crispasr` and its options stop with a message pointing to the new pipelines.
- The hidden Kotoba faster-whisper pipeline. Transformers mode and its Kotoba models are unchanged.
- **Google Colab and Kaggle** are not supported by 1.10.0; the Colab notebook keeps installing 1.9.5.

## Limits in 1.10.0

- The GGUF pipelines need folder names in plain Western letters (ASCII). When your user folder has Chinese, Japanese or Korean characters, WhisperJAV links or copies the model to a plain-named folder.
- The GGUF pipelines need a processor with AVX2 (Intel since 2013, AMD since 2015). So do ChronosJAV's aligner and timing model; without it ChronosJAV keeps its text and the times of its pieces of speech.
- `--device cpu` cannot keep ChronosJAV's aligner and timing model, or Cohere's aligner, on the CPU when CrispASR's build uses a GPU; the box at the start of the run says so.
- GPU through Vulkan and your own PyTorch are experimental and untested on AMD and Intel cards. Intel GPUs are not used through PyTorch. On Linux, the GGUF pipelines run on the CPU for AMD and Intel GPUs. Please report how they work: https://github.com/meizhong986/WhisperJAV/issues
- ChronosJAV's aligner and timing model have not been tried through Vulkan on any card, and how long the timing model takes on a computer without a GPU has not been measured.
- Nobody in the project has run 1.10.0 on a Mac or a Linux PC yet; the Apple GPU for fidelity, the GGUF pipelines and ChronosJAV's aligner and timing model is not yet tested.
- **Known faults** (to be fixed in 1.10.1):
  - BS-RoFormer cannot start on the command line (marked Coming Soon in the GUI); ZipEnhancer cleans only the first file of a run; an upgrade does not remove packages the new release no longer uses.
  - ChronosJAV: a run without `--keep-temp` may show "Error during temporary file cleanup" for each file and leave some temporary files behind. The subtitles are already written by then.
  - ChronosJAV with the auditok or silero scene detector (the default is semantic): scenes shorter than 12 seconds may be dropped, and their speech with them. Keep the default.
  - Parakeet-CrispASR and Cohere-CrispASR: a speech enhancer chosen for the pass is not checked when the run starts, as it is for the other pipelines.
  - With the silero-v6, whisper-vad or nemo speech segmenter, a segmenter that fails is treated as finding no speech: the scene gets no subtitles and no error is shown.
  - ChronosJAV's timing model may stop the run on a very short piece of speech. We have not seen this happen in our tests.
  - ChronosJAV's aligner and timing model, and Cohere-CrispASR's aligner, are downloaded the first time they are used, during the run. If the download fails (for example with `--offline`), Cohere-CrispASR continues without its aligner.
  - `whisperjav-translate --configure` stops with an error when you choose ollama or custom. Give the provider on the command line instead, for example `whisperjav-translate --provider ollama …`.
  - `--async-processing` with `--mode transformers` or `--mode qwen` may stop with an error. Leave it out for these modes.
  - On a CPU-only install on a PC with an NVIDIA card, `--model` with balanced, fast or faster may make the run try the GPU.
  - XXL Faster Whisper: Windows paths in Extra Args lose their backslashes (`D:\models` becomes `D:models`).
  - `install.py --minimal` followed by a GGUF pipeline may give no subtitles. Use the full install, as 1.10.0 asks of every user.
  - XXL Faster Whisper as pass 1 crashed once when closing, on one scene in our tests.
- balanced gives slightly different subtitles from 1.9.5 (CTranslate2 4.8.3): on one of our two reference films, slightly fewer character errors and two more missed lines.

## Installation

### Before you start

- **Do not install or upgrade with `pip install -U whisperjav`.** On Windows it replaces your GPU build of PyTorch with a CPU-only one. Use the installer or the install scripts below.
- Your settings and downloaded models are kept in every case below; they are stored outside the install folder.
- Keep the internet connection on during the whole installation; the first run of a GGUF pipeline or of ChronosJAV also downloads its models. **Users in China:** the installers download from GitHub, PyPI, download.pytorch.org and Hugging Face; keep your VPN on throughout.
- Pick your method:

| Your computer | Method |
|---|---|
| Windows 10 or 11 (64-bit, Intel or AMD processor) | **Windows installer** (recommended) |
| Mac with Apple silicon (M1 or newer), macOS 14 or newer | macOS from source |
| Linux with glibc 2.35 or newer (Ubuntu 22.04, Debian 12, Fedora 36 or newer) | Linux from source |
| Windows, if you prefer a source install | Windows from source |

### Coming from 1.9.x

1.10.0 needs Python 3.13, so `whisperjav-upgrade` cannot take a 1.9.x install to it. Install 1.10.0 fresh with the method for your computer below.

- **Windows installer:** you can install into the same folder. The installer offers to replace the old version and removes everything in that folder first; your settings and downloaded models stay.
- **Source installs:** 1.10.0 needs a new folder and a new Python environment. First rename or delete your 1.9.x `WhisperJAV` folder (the clone below fails if one is in the same place) and your 1.9.x environment: `~/venvs/whisperjav` on macOS, `~/.venv/whisperjav` on Linux (the 1.9.5 guides used the same names). Your settings and models are stored elsewhere and stay.

### Windows installer (recommended)

1. Download `WhisperJAV-1.10.0-Windows-x86_64.exe` from the [Releases page](https://github.com/meizhong986/WhisperJAV/releases/latest).
2. Run it and follow the steps. It brings its own Python and FFmpeg and installs every package WhisperJAV needs; you do not need to install Python yourself.
3. **What the install is made for:** a supported NVIDIA GPU (GTX 16 / RTX 20 or newer, driver 580 or newer) is used by itself. Otherwise a window asks: AMD, Intel or older NVIDIA card (these use GPU through Vulkan), your own PyTorch, or CPU only. With no NVIDIA card, CPU only is installed after 20 seconds without an answer; with an NVIDIA card 1.10 cannot use, the window names the reason and waits.
4. Wait until the installer says it has finished; the last step installs the packages and can take several minutes.
5. Start WhisperJAV from the desktop shortcut "WhisperJAV v1.10.0".
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
   git clone --branch v1.10.0 https://github.com/meizhong986/WhisperJAV.git
   cd WhisperJAV
   bash installer/install_mac.sh
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
   git clone --branch v1.10.0 https://github.com/meizhong986/WhisperJAV.git
   cd WhisperJAV
   bash installer/install_linux.sh
   ```
   Git prints a note about "detached HEAD"; that is expected.
3. With a supported NVIDIA GPU the install is made for it. Otherwise the installer asks: 1 = your own PyTorch (it must already be installed in this environment, for example ROCm for an AMD card; otherwise it stops and installs nothing), 2 = CPU only, 3 = stop.
4. For the GUI, add a window backend in the same terminal: `pip install "pywebview[qt]"` (see the Linux guide, "GUI backend").
5. Start WhisperJAV with `./WhisperJAV.sh` in the `WhisperJAV` folder, or `whisperjav-gui` with the environment active.
6. Check the install (see [Check your install](#check-your-install)).

Nobody in the project has run 1.10.0 on Linux yet; please report what you find.

### Windows from source

1. Install Python 3.13 from [python.org](https://www.python.org/downloads/) and tick **Add python.exe to PATH**; install Git, FFmpeg (on your PATH), the Visual C++ Redistributable and WebView2 as the [Windows Python guide](https://meizhong986.github.io/WhisperJAV/guides/installation_windows_python/) shows.
2. Open Command Prompt (press Start, type `cmd`, press Enter) and run these lines one by one:
   ```
   python --version
   git clone --branch v1.10.0 https://github.com/meizhong986/WhisperJAV.git
   cd WhisperJAV
   installer\install_windows.bat --local
   ```
   The first line must print Python 3.13.x. Git then prints a note about "detached HEAD"; that is expected. `--local` installs everything into a `.venv` folder inside `WhisperJAV`, even if another Python environment is active. To choose the accelerator in advance, add it, for example `--local --accelerator vulkan`; your own PyTorch cannot be chosen this way.
3. Start WhisperJAV by double-clicking `WhisperJAV.bat` in the `WhisperJAV` folder.
4. Check the install (see [Check your install](#check-your-install)), running `.venv\Scripts\activate` in the `WhisperJAV` folder first.

### Check your install

On Windows (installer), open Command Prompt and run:
```
cd /d "%LOCALAPPDATA%\WhisperJAV"
Scripts\whisperjav.exe --version
Scripts\whisperjav.exe --check
```
(Use the folder you installed into if you changed it.) For a source install, run `whisperjav --version` and `whisperjav --check` with its environment active. The first prints `WhisperJAV 1.10.0`; the second names the install's accelerator and where each part runs, and lists CrispASR's models as downloaded or not downloaded (they download the first time you use them).

### Later 1.10.x releases

- **Windows installer:** run `Scripts\whisperjav-upgrade.exe` in `%LOCALAPPDATA%\WhisperJAV` (or the folder you chose). It installs the newest 1.10.x release with its tested versions and keeps the accelerator of your install.
- **Source installs:** in the `WhisperJAV` folder, with the same environment active as at install (for `--local`, run `.venv\Scripts\activate` on Windows), run `git fetch --tags`, then `git checkout v1.10.x` (the release you want), then the install script again as at install (on Windows with `--local`); it offers your install's accelerator as the default. Do not use `whisperjav-upgrade` on a source install: in that folder it runs `git pull`, which fails on a clone of a release.

## Credits

- CrispASR, which runs the GGUF pipelines and ChronosJAV's aligner and timing model, is developed at [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR). Models: NVIDIA's parakeet-tdt_ctc-0.6b-ja (CC-BY-4.0) in GGUF form by cstr; Japanese fine-tunes of Cohere Transcribe by efwkjn, in GGUF form by TransWithAI and CKHO.
- WhisperJAV's build of `qwen-asr` is a fork of [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) with its pull request #125.
- The request for CTranslate2 4.8.3 came from #459; XXL as Pass 1 from #451 and #452; the Chinese GUI from #448.
