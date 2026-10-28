# WhisperJAV 1.10.0

## Key points

- **A fresh install for everyone.** 1.10.0 runs on Python 3.13 with PyTorch 2.11. A 1.9.x install cannot be upgraded to it; install 1.10.0 fresh (steps below). Your settings are kept.
- **One accelerator per install.** You choose it once, when you install: NVIDIA GPU (CUDA), GPU through Vulkan (AMD, Intel or older NVIDIA card, Windows), Apple GPU (Metal), your own PyTorch, or CPU only. Runs no longer stop to ask about the CPU. A run that does not use a supported NVIDIA GPU or the Apple GPU starts with a box saying where each part runs.
- **Two new pipelines for Japanese, experimental:** Parakeet-CrispASR and Cohere-CrispASR run as Ensemble passes (menu group GGUF). CrispASR is installed with WhisperJAV; there is nothing to download separately.
- **ChronosJAV subtitle timing** (new, not yet measured): subtitles start where the speech starts, and two sentences in one piece of speech (for example two speakers) can become two subtitles, split at a punctuation mark or space in the text.
- **The GUI in Simplified Chinese:** the **中文** button in the header switches the whole window; **English** switches back.
- **Every install gets a fixed list of package versions for its accelerator**, so two computers with the same accelerator get the same versions.
- Removed: the llama-cpp translation provider (use Ollama), ClearVoice speech enhancement, and the 1.9.x CrispASR mode. Google Colab and Kaggle are not supported by 1.10.0.

## Changes

**New**

- **Parakeet-CrispASR** (Ensemble, Japanese, experimental): NVIDIA's Parakeet TDT 0.6B ja model (674 MB). Within each piece of speech (at most about 6 s), a subtitle ends only at a sentence end, a pause, or a 、 or space before a pause — never by character count.
- **Cohere-CrispASR** (Ensemble, Japanese, experimental): Japanese versions of Cohere Transcribe, `cohere-asr-ja-v0.1` (default) or `cohere-asr-ja` (2.4 GB each). An aligner (`wav2vec2-ja`, 232 MB, downloaded on first use) times the text; within each piece of speech a subtitle ends only at a sentence end (。？！… ? !), never by character count. A piece of speech the aligner cannot time stays one subtitle.
- **ChronosJAV timing** (Qwen3-ASR and anime-whisper): by default an aligner, `wav2vec2-ja` (232 MB), times each character of the text, and a timing model, Whisper small (488 MB; its text is not used), moves each subtitle's start later to where Whisper hears the speech begin, and splits a piece of speech at a punctuation mark or space near where Whisper starts anew — no mark, no split; never by number of characters or seconds. Both run in CrispASR and download on first use; if one cannot be loaded, the text is kept and the run summary says so. Settings: Customize → Alignment, or `--qwen-aligner`, `--qwen-timing-model`, `--qwen-timestamp-mode` (all choices on the [ChronosJAV page](https://meizhong986.github.io/WhisperJAV/pipelines/chronosjav/#subtitle-timing-1100)). New in 1.10.0; not yet measured against reference subtitles.
- **Faster Whisper XXL as Pass 1** (#451): your own copy of PurfView's XXL Faster Whisper can now be Pass 1, Pass 2 or both in Ensemble mode (`--pass1-pipeline xxl`). Each XXL pass picks its model in the Model column: Large V2 (default), Large V3, Large V3 Turbo or Medium.
- Both GGUF pipelines take the pass's scene detector, speech enhancer, Enhance for VAD only and Customize settings, and print the audio analytics report when the scene detector is Semantic (the default). A status panel under the pass shows the model and where CrispASR runs. The model downloads on first use.
- **The installers ask what the install is made for** when there is no supported NVIDIA GPU: the Windows installer in a window, `install.py` in the console, or `install.py --accelerator nvidia|vulkan|own-pytorch|cpu` in advance. A supported NVIDIA GPU, or an Apple silicon Mac, is used without a question.
- **Own PyTorch** (experimental): WhisperJAV installs everything except PyTorch and never replaces your PyTorch build when it installs or upgrades — for example ROCm for an AMD GPU. The Windows installer shows the one command that adds it; running that installer again replaces the whole install folder, PyTorch included, and shows the command again.
- **GPU through Vulkan** (Windows, experimental): with an AMD, Intel or older NVIDIA card, the GGUF pipelines and ChronosJAV's aligner and timing model ask for the graphics card through Vulkan (not tested on AMD or Intel cards; ChronosJAV's aligner and timing model not yet tried through Vulkan on any card).
- **At the start of a run**, when the install's GPU is not in use, WhisperJAV says where each part runs on this computer. An install made for the NVIDIA GPU whose GPU cannot be used now (for example after a driver change) stops with the reason and the fix; in the GUI, **Run on the CPU this time** runs the job on the CPU.
- **`--device auto|cuda|cpu` now sets the device for every pipeline**, the speech enhancers included. `--device cpu` runs one job on the CPU.
- **`whisperjav --check`** names the accelerator of the install, where each part runs, CrispASR and its models.
- **GUI in Simplified Chinese** (#448): the **中文** button in the header switches the window to Chinese — tabs, buttons, menus, tooltips, pop-up boxes, Customize Parameters and Translation Settings — and **English** switches back. The console and the messages written while files are processed stay English. The language is kept for the next start; on Windows set to Chinese, the GUI opens in Chinese the first time, also after upgrading.
- **GUI:** clear dialogs when a run does not start ("The NVIDIA GPU cannot be used", "The run did not start") or ends without a run summary.
- **Supported computers are checked before anything is installed.** On a computer 1.10.0 does not support, the installer says why and installs nothing.

**Changed**

- Python 3.13 and PyTorch 2.11. On Windows, the NVIDIA GPU install uses CUDA 13.0 and needs NVIDIA driver 580 or newer; where this PC cannot load the CUDA 12 library that balanced, fast and faster need, the installer uses CUDA 12.8 instead. Linux uses CUDA 12.8, driver 570 or newer.
- NVIDIA cards older than the GTX 16 / RTX 20 series are no longer supported by the NVIDIA GPU install. The installer says so, names 1.9.5 for such a card, and offers GPU through Vulkan (Windows), your own PyTorch or CPU only.
- balanced, fast and faster run on CTranslate2 4.8.3 (#459) and use a GPU only when CTranslate2 itself can use it.
- Qwen3-ASR runs on transformers 5 through WhisperJAV's own build of `qwen-asr`; on our two reference clips its subtitles score the same as in 1.9.5.
- On an install made with the Windows installer, `whisperjav-upgrade` installs the newest 1.10 release, keeps the accelerator of your install, and never installs PyTorch on an Own PyTorch install.
- The GUI's "Accept CPU-only mode" box is gone, and so is the stop-and-ask at the start of a run. Scripts that pass `--accept-cpu-mode` still run (it now does nothing); `--hf-device` and `--qwen-device` still work as other names for `--device`.
- On an Apple silicon Mac, each pipeline says where it runs: transformers on the Apple GPU (recommended on a Mac); fidelity, the GGUF pipelines and ChronosJAV's aligner and timing model ask for it (not yet tested on a Mac); balanced, fast, faster and ChronosJAV's ASR model on the CPU.
- ChronosJAV: anime-whisper now uses the aligner like Qwen3-ASR (it was VAD-only), and step-down is off by default for every ChronosJAV model — a scene whose alignment collapses takes the times of its pieces of speech instead.
- ChronosJAV times from the pieces of speech, as in 1.9.5: `--qwen-timestamp-mode vad_only --qwen-timing-model none`, or in the GUI, Aligner Backend None and Timing Model None.
- In the GUI, a ChronosJAV pass's Customize → Alignment settings (and a preset saved from that window) now reach the run; from 1.8.10.post3 to 1.9.5 they were ignored.
- XXL Faster Whisper runs the model you choose (#449): in the GUI, the Model column of the XXL pass; on the command line, `--pass1-model` or `--pass2-model`. Before, the GUI sent no model and XXL ran large-v3. With no model chosen, XXL now runs large-v2 (was large-v3). The Extra Args box no longer starts with `--model large-v3`; a `--model` typed there is not used, and the console says so. When XXL fails for a file, the console says which pass failed and why.
- The settings that come with WhisperJAV no longer hold a path to an XXL program on the developer's computer.
- AI translation (#460): when the model writes its answer labels (`Translation>`, `Original>` or a `#number` line) into a subtitle, WhisperJAV now asks the model again for that group of subtitles. If the second answer has the same fault, the subtitles are kept and the console names their numbers in the translated file, so you know which lines to check. This applies to every provider.

**Removed**

- **Local translation with llama-cpp.** Ollama is the local translator. A saved llama-cpp choice opens as Ollama; `--provider local` stops with a message pointing to `--provider ollama`.
- **ClearVoice.** ClearVoice speech enhancement is no longer included. If you used it, choose ZipEnhancer, htdemucs (Vocals) or None.
- **The 1.9.x CrispASR mode**, which ran a `crispasr` program you downloaded yourself. `--mode crispasr` and its options stop with a message that points to the two new pipelines.
- The hidden Kotoba faster-whisper pipeline. Transformers mode and its Kotoba models are unchanged.

**Limits in 1.10.0**

- The GGUF pipelines need folder names in plain Western letters (ASCII). When your user folder has Chinese, Japanese or Korean characters, WhisperJAV links or copies the model to a plain-named folder (the original stays where it is).
- The GGUF pipelines need a processor with AVX2 (Intel since 2013, AMD since 2015); every other pipeline runs without it. ChronosJAV's aligner and timing model need it too; without it ChronosJAV keeps its text and takes the times of its pieces of speech.
- `--device cpu` cannot hold ChronosJAV's aligner and timing model, or Cohere-CrispASR's aligner, on the CPU: where CrispASR's build uses a GPU they still ask for it, and the box at the start of the run says so. How long the timing model takes on a computer without a GPU has not been measured.
- GPUs through Vulkan and your own PyTorch are experimental: we could not test them on AMD or Intel graphics cards. Intel GPUs are not used through PyTorch in 1.10.0. On Linux, the GGUF pipelines run on the CPU for AMD and Intel GPUs. Please tell us how they work on your PC: https://github.com/meizhong986/WhisperJAV/issues
- **Known faults in 1.10.0:**
  - BS-RoFormer cannot start on the command line; the run goes on with the original audio and a warning in the log. In the GUI it is marked Coming Soon. (To be fixed in 1.10.1.)
  - ZipEnhancer cleans only the first file of a run; later files of the same run get no clean-up, with a warning in the log. (To be fixed in 1.10.1.)
  - An upgrade does not remove packages the new release no longer uses.
  - Balanced gives somewhat different subtitles from 1.9.5 (CTranslate2 4.8.3): on one of our two reference films, slightly fewer character errors and two more missed lines; the other was unchanged.
- Not supported: Windows on ARM, Windows older than 10; Linux with glibc older than 2.35 (Ubuntu 20.04, Debian 11, RHEL/Rocky/Alma 9); Intel Macs and macOS 13 or older; Google Colab and Kaggle.

## Installation

Do not install or upgrade with `pip install -U whisperjav`: on Windows it replaces your GPU build of PyTorch with a CPU-only one. Your settings and downloaded models are kept in every case below.

**Users in China:** the installers download from GitHub, PyPI, download.pytorch.org and Hugging Face. Keep your VPN on during the whole installation.

### Coming from 1.9.x

Install 1.10.0 fresh, as below. 1.10.0 needs Python 3.13, so `whisperjav-upgrade` cannot take a 1.9.x install to it. On Windows, you can install into the same folder: the installer offers to replace the old version, and removes everything in that folder first (your settings and downloaded models stay).

### Fresh install: Windows

1. Download `WhisperJAV-1.10.0-Windows-x86_64.exe` from the [Releases page](https://github.com/meizhong986/WhisperJAV/releases/latest).
2. Run it and follow the steps; keep the internet connection on. With a supported NVIDIA GPU (GTX 16 / RTX 20 or newer, driver 580 or newer) it installs for the GPU by itself. Otherwise a window asks what the install should be made for. With no NVIDIA card, the CPU-only version is installed after 20 seconds without an answer. With an NVIDIA card 1.10 cannot use (an older card, a driver below 580, or one it cannot reach), the window names the reason and waits for your answer.
3. Start WhisperJAV from the desktop shortcut "WhisperJAV v1.10.0".
4. To check the install: open Command Prompt (press Start, type `cmd`, press Enter) and run:
   ```
   cd /d "%LOCALAPPDATA%\WhisperJAV"
   Scripts\whisperjav.exe --version
   Scripts\whisperjav.exe --check
   ```
   The first should print `WhisperJAV 1.10.0`; the second names the accelerator of the install.

To change what the install is made for later, run the installer again into the same folder.

### Fresh install: macOS (Apple silicon, macOS 14 or newer)

Needs Xcode Command Line Tools and Homebrew ([macOS guide](https://meizhong986.github.io/WhisperJAV/guides/installation_mac_apple_silicon/), section 2). Then, in Terminal:

```
brew install python@3.13 ffmpeg portaudio git
python3.13 -m venv ~/venvs/whisperjav
source ~/venvs/whisperjav/bin/activate
git clone --branch v1.10.0 https://github.com/meizhong986/WhisperJAV.git
cd WhisperJAV
bash installer/install_mac.sh
```

The install is made for the Apple GPU (Metal). Double-click `WhisperJAV.command` in the `WhisperJAV` folder to start the GUI.

### Fresh install: Linux

Needs glibc 2.35 or newer (Ubuntu 22.04, Debian 12, Fedora 36 or newer), Python 3.13 with `venv`, FFmpeg, Git and the system packages of the [Linux guide](https://meizhong986.github.io/WhisperJAV/guides/installation_linux/); for the GPU, NVIDIA driver 570 or newer. Then:

```
python3.13 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate
git clone --branch v1.10.0 https://github.com/meizhong986/WhisperJAV.git
cd WhisperJAV
bash installer/install_linux.sh
```

Without a supported NVIDIA GPU, the installer asks: your own PyTorch, CPU only, or stop. For the GUI, also install a window backend into the environment: `pip install "pywebview[qt]"` (see the Linux guide, "GUI backend"). Start WhisperJAV with `./WhisperJAV.sh` in the `WhisperJAV` folder, or `whisperjav-gui`. Nobody in the project has run 1.10.0 on Linux yet.

### Fresh install: Windows from source

Needs Python 3.13, Git, FFmpeg on your PATH, the Visual C++ Redistributable and WebView2 ([Windows Python guide](https://meizhong986.github.io/WhisperJAV/guides/installation_windows_python/)). In a Command Prompt, with your Python 3.13 environment active:

```
git clone --branch v1.10.0 https://github.com/meizhong986/WhisperJAV.git
cd WhisperJAV
installer\install_windows.bat
```

Start WhisperJAV with `whisperjav-gui`, with the same environment active.

### Google Colab and Kaggle

Not supported by 1.10.0. The Colab expert notebook keeps installing 1.9.5.

## Credits

- CrispASR, which runs the GGUF pipelines, is developed at [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR). Models: NVIDIA's parakeet-tdt_ctc-0.6b-ja (CC-BY-4.0) in GGUF form by cstr; Japanese fine-tunes of Cohere Transcribe by efwkjn, in GGUF form by TransWithAI and CKHO.
- WhisperJAV's build of `qwen-asr` is a fork of [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) with its pull request #125.
- The request for CTranslate2 4.8.3 came from #459.
