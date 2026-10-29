# WhisperJAV 1.10.0

## Key points

- **A fresh install for everyone.** 1.10.0 runs on Python 3.13 and PyTorch 2.11, so a 1.9.x install cannot be upgraded to it. Install it fresh with the steps below; your settings and downloaded models are kept.
- **Made for the graphics card you have.** You choose once, at install: NVIDIA GPU, another graphics card through Vulkan (Windows), the Apple GPU, your own PyTorch, or CPU only. Every run follows that choice. With an AMD or Intel card, only the GGUF pipelines and ChronosJAV's timing use the card.
- **Two new Japanese pipelines, experimental:** parakeet and cohere, run through CrispASR in Ensemble mode. The Ensemble tab now starts with them.
- **Better subtitle timing in ChronosJAV:** subtitles start where their first word starts, and two sentences in one piece of speech can become two subtitles.
- **The GUI in Simplified Chinese**, with a 中文 / English button.
- Removed: llama-cpp translation (use Ollama), ClearVoice, the 1.9.x CrispASR mode. Google Colab and Kaggle are not supported; the Colab notebook keeps installing 1.9.5.

## What is new and changed

### 1. Installing and running on your computer

Each install is now made for one accelerator (the graphics card, or the CPU, that it runs on) and gets exactly the package versions we tested for it. Computers 1.10.0 cannot support are refused before anything is installed, and a run no longer stops to ask about the CPU: it says where each part runs.

- **One accelerator per install:** NVIDIA GPU (CUDA); GPU through Vulkan (AMD, Intel or older NVIDIA card; Windows, experimental: only the GGUF pipelines and ChronosJAV's aligner and timing model use the card, the rest runs on the CPU); Apple GPU (Metal); your own PyTorch (experimental); or CPU only. The choice is recorded; every run and every upgrade follows it.
- **The Windows installer asks** what the install is for when it finds no supported NVIDIA GPU. With no NVIDIA card it installs CPU only after 20 seconds without an answer; with an NVIDIA card 1.10 cannot use, it names the reason and waits.
- **Source installs** take the same choice: `python install.py --accelerator nvidia|vulkan|own-pytorch|cpu`. Without it, a supported NVIDIA GPU or an Apple silicon Mac is used by itself; otherwise the installer asks.
- **Your own PyTorch** (for example ROCm for an AMD card): WhisperJAV installs everything except PyTorch and never replaces your PyTorch when it installs or upgrades. The Windows installer shows the one command that adds it; running that installer again replaces the whole folder, PyTorch included. A source install needs your PyTorch already installed in the active Python environment.
- **New versions:** Python 3.13 and PyTorch 2.11. On Windows, the NVIDIA install uses CUDA 13.0 and needs driver 580 or newer; where this PC cannot load the CUDA 12 library that balanced, fast and faster need, it uses CUDA 12.8. Linux uses CUDA 12.8, driver 570 or newer.
- **Older NVIDIA cards** (before GTX 16 / RTX 20) are no longer supported by the NVIDIA install. The installer says so, names 1.9.5 for such a card, and offers GPU through Vulkan (Windows), your own PyTorch or CPU only.
- **Unsupported computers are refused before anything is installed,** with the reason: Windows on ARM or older than Windows 10, Linux with glibc older than 2.35, Intel Macs and macOS 13 or older.
- **Exact versions everywhere:** the Windows installer, the source installers and `whisperjav-upgrade` install the same tested package versions for each accelerator, so two computers with the same accelerator get the same versions.
- **At the start of a run** that does not use a supported NVIDIA GPU or the Apple GPU, a box says where each part runs. An NVIDIA install whose GPU cannot be used now stops with the reason and the fix; in the GUI, **Run on the CPU this time** runs the job on the CPU.
- **No CPU question any more:** the GUI's "Accept CPU-only mode" box is gone. Scripts that pass `--accept-cpu-mode` still run (it does nothing); `--hf-device` and `--qwen-device` still work as other names for `--device`.
- **`--device auto|cuda|cpu` now applies to every pipeline**, the speech enhancers included; `--device cpu` runs one job on the CPU.
- **`whisperjav --check`** names the install's accelerator, where each part runs, CrispASR and its models.
- balanced, fast and faster no longer try a graphics card they cannot use (for example with your own ROCm PyTorch for an AMD card); on an Apple silicon Mac each pipeline says where it runs.
- When PyTorch is missing or damaged, the run stops at once and names the command that repairs the install.
- **`whisperjav-upgrade`** (installs made with the Windows installer) installs the newest 1.10.x release (never unreleased code), keeps the accelerator of your install, and never installs PyTorch on an own-PyTorch install. Source installs update with Git; see Installation.

### 2. New: GGUF pipelines through CrispASR

Two Japanese ASR pipelines that run through CrispASR, which comes with WhisperJAV. They run in Ensemble mode only and can use the NVIDIA GPU, a graphics card through Vulkan (Windows) or the Apple GPU (not yet tested on a Mac); otherwise the CPU. Both are experimental in 1.10.0.

- **parakeet** (Parakeet-CrispASR): NVIDIA's Parakeet TDT 0.6B ja model (674 MB). Within each piece of speech (at most about 6 s), a subtitle ends only at a sentence end, a pause, or a 、 or space before a pause, never by character count.
- **cohere** (Cohere-CrispASR): Japanese versions of Cohere Transcribe, `cohere-asr-ja-v0.1` (default) or `cohere-asr-ja` (2.4 GB each). An aligner (`wav2vec2-ja`, 232 MB) times the text; a subtitle ends only at a sentence end (。？！… ? !).
- **The Ensemble tab starts with them:** Pass 1 cohere (Balanced, FireRedVAD), Pass 2 parakeet (Aggressive, TEN VAD; off until you tick it); 1.9.5 opened on Qwen3-ASR and Anime-Whisper. In the Pass menus they are under **GGUF via crispasr**. With Remember settings on, your saved passes open instead.
- Both take the pass's scene detector, speech enhancer, Enhance for VAD only and Customize settings, and print the audio analytics report with the Semantic scene detector (the default).
- A status panel under the pass shows the model, whether it is downloaded, and where CrispASR runs. Models download on first use.
- On the command line: `--ensemble --pass1-pipeline parakeet-crispasr` (or `cohere-crispasr`); without `--ensemble` the run is a single Balanced pass. A Parakeet pass that names no speech segmenter or sensitivity runs TEN VAD at Aggressive; Cohere runs FireRedVAD at Balanced.

### 3. Better subtitle timing in ChronosJAV

ChronosJAV (Qwen3-ASR and anime-whisper) now starts each subtitle at its first word instead of at the start of the piece of speech, and can split one piece of speech in two at a sentence break. The last subtitle of a piece of speech still ends where that piece ends. On our 7 test scenes, more subtitles start within half a second of Netflix's Japanese subtitles than in 1.9.5, with the same text.

- **An aligner** (`wav2vec2-ja`, through CrispASR) times each character; each subtitle starts at its first character. Customize → Alignment offers two other CrispASR aligners and the PyTorch Qwen3-ForcedAligner.
- **A timing model** (Whisper small; its text is not used) splits a piece of speech at a punctuation mark or space near where Whisper starts anew, for example at a change of speaker. Never a split without a mark, by character count or by seconds.
- The aligner (232 MB) and the timing model (488 MB) download the first time ChronosJAV uses them.
- **Measured:** subtitles starting within 0.5 s of Netflix's Japanese subtitles: Qwen3-ASR 78% (1.9.5: 75%), anime-whisper 77% (1.9.5: 63%), with the same text.
- **The text is always kept:** if the aligner or timing model cannot be loaded, or gives no times, the subtitles keep the times of their piece of speech and the run summary says so.
- anime-whisper now uses the aligner like Qwen3-ASR (it was VAD-only), and Adaptive Step-Down is off by default for every ChronosJAV model.
- 1.9.5 timing remains available: `--qwen-timestamp-mode vad_only --qwen-timing-model none`, or in Customize → Alignment, Aligner Backend None and Timing Model None.

### 4. Cleaner subtitles

Fewer subtitles that hold nothing but a lone sound.

- ChronosJAV, parakeet and cohere: a subtitle that is only a lone sound such as 「あ」, 「うん」 or 「はい」 is now removed whatever punctuation follows it (「うん!」, 「え?」, 「あ……」), as it already was with 「。」.

### 5. XXL Faster Whisper

PurfView's XXL Faster Whisper (your own copy) can now be either pass, and it runs the model you choose.

- **XXL as Pass 1** (#451, #452), Pass 2 or both in Ensemble mode (`--ensemble --pass1-pipeline xxl --xxl-exe <path to XXL>`). One program path and one Extra Args box serve both passes.
- Each XXL pass picks its model in the Model column: Large V2 (default), Large V3, Large V3 Turbo or Medium.
- When XXL fails for a file, the console says which pass failed and why.

### 6. Translation

Ollama is now the only local translator, and translations that contain the model's answer labels are caught and asked for again (see Fixed).

- **Ollama is the local translator.** The llama-cpp provider is removed; a saved llama-cpp choice opens as Ollama, and `--provider local` (or `--translate-provider local`) stops with a message pointing to Ollama.

### 7. The GUI

The window can be switched to Simplified Chinese, and messages are clearer when a run does not start.

- **Simplified Chinese** (#448): the **中文** button in the header switches the window, tabs, menus, tooltips, pop-up boxes and Customize Parameters; **English** switches back. On Windows set to Chinese, the GUI opens in Chinese the first time. The console stays English.
- **Clearer messages** when a run does not start ("The NVIDIA GPU cannot be used", "The run did not start") or ends without a run summary.

### 8. Updated engines

Newer versions of the libraries behind Qwen3-ASR and behind balanced, fast and faster; results as in 1.9.5, except the small difference for balanced under Limits.

- Qwen3-ASR runs on transformers 5 through WhisperJAV's own build of `qwen-asr`; on our two reference clips its subtitles score the same as in 1.9.5.
- balanced, fast and faster run on CTranslate2 4.8.3 (#459).

## Fixed

- **In the GUI, XXL ran large-v3 whatever model you set** (#449). It now runs the model in the Model column; with none chosen, large-v2. A `--model` typed in Extra Args is not used, and the console says so.
- **ChronosJAV's Customize → Alignment settings** (and presets saved from that window) now reach the run; from 1.8.10.post3 to 1.9.5 they were ignored.
- **Batch runs:** each file's ChronosJAV run summary counts only its own scenes; since 1.9.5 the second and later files also counted the scenes of the files before them.
- The ChronosJAV run summary's figures "Aligner native" (subtitles the aligner timed) and "Collapse rate" (scenes where alignment failed) are now right; both showed a wrong figure before.
- **AI translation labels** (#460): a subtitle holding `Translation>`, `Original>` or a `#number` line is sent to the model again; if the answer is the same, the subtitles are kept and the console names them. All providers. A label in the middle of a line, or in lower case, is not caught.
- **An NVIDIA driver too old** for the GPU install: the installer now names your driver version and the 580 minimum, then asks what to install instead (Stop is pre-selected); 1.9.5 silently installed the CPU version.
- **A deliberate stop in the Windows installer** (old card, old driver) now ends with "WhisperJAV was not installed." instead of "Failed to run post install script".
- The WhisperJAV package no longer carries a path to the developer's XXL program or the developer's settings backups; your own settings backups now go to your settings folder.

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
- **Known faults:** BS-RoFormer cannot start on the command line (marked Coming Soon in the GUI); ZipEnhancer cleans only the first file of a run (both to be fixed in 1.10.1); an upgrade does not remove packages the new release no longer uses.
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
