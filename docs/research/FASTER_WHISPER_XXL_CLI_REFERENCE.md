# Faster Whisper XXL — CLI Reference & Integration Research

> Created: 2026-03-15 | Source: R1 research (GitHub releases, issue #223/#224, web docs)

## Overview

Faster-Whisper-XXL is a standalone executable (no Python required) by PurfView, built on CTranslate2-optimized Whisper models. It is a **pure CLI tool** — no GUI. Runs headlessly and is used by Subtitle Edit, PotPlayer, and pyvideotrans as an external process.

**Two editions:**
- **Free** (r245.4) — mdx_kim2 vocal separation, silero_v4 VAD
- **Pro** (r3.256.1, donation-based >=50 GBP) — MB-RoFormer, silero_v6, ten VAD, realignment

**Repository**: https://github.com/Purfview/whisper-standalone-win

---

## Invocation

```bash
faster-whisper-xxl.exe <input_path> [options]
```

- **Input**: Accepts video files (mkv, mp4) OR audio files (wav, mp3). FFmpeg is bundled internally.
- **Batch mode**: Pass a folder path with `--batch_recursive`.
- **Help**: `faster-whisper-xxl.exe --help`

### Minimum Viable Command (Japanese)

```bash
faster-whisper-xxl.exe "video.mkv" -l ja -m large-v3 --output_dir /path/to/output
```

### With Vocal Separation (key quality advantage)

```bash
faster-whisper-xxl.exe "video.mkv" -l ja -m large-v3 --compute_type float16 --ff_vocal_extract mdx_kim2 -f srt -o source
```

---

## Complete CLI Arguments

### Core Parameters

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | `large-v2` | Model name (tiny, base, small, medium, large-v2, large-v3, turbo, distil-large-v3.5) |
| `--language` | `-l` | auto-detect | Language code (ja, en, etc.) |
| `--task` | | `transcribe` | `transcribe` or `translate` (to English) |
| `--device` | | `cuda` | `cuda` or `cpu` |
| `--compute_type` | | `default` (CUDA) / `auto` (CPU) | `float16`, `int8_float16`, `int8`, `auto`, `default` |

### Decoding Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--beam_size` | 5 | Beam search width |
| `--best_of` | 5 | Number of candidates for best-of sampling |
| `--temperature` | 0 | Sampling temperature (0 = greedy) |
| `-fallback` | enabled | Temperature fallback; `None` to disable |
| `--initial_prompt` | None | Text to condition the model |
| `--hotwords` | | Hotwords/phrases to boost |
| `--repetition_penalty` | 1.0 | Penalty for repeated tokens (>1 penalizes) |
| `--no_repeat_ngram_size` | 0 | Prevent repeated n-grams |
| `--compression_ratio_threshold` | 2.4 | Threshold for failed transcription detection |
| `--log_prob_threshold` | -1.0 | Min average log probability |
| `--hallucination_silence_threshold` | | Silence-based hallucination detection |
| `--without_timestamps` | false | Disable timestamp generation |
| `--clip_timestamps` | | Process specific time ranges |

### Output Parameters

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output_dir` | `-o` | current dir / `source` | Output directory; `source` = same as input file |
| `--output_format` | `-f` | `srt` | `srt`, `vtt`, `json`, `txt`, `tsv`, `lrc`, `all` |
| `--verbose` | `-v` | false | Verbose logging |
| `--print_progress` | `-pp` | | Print progress bar |

### Subtitle Formatting

| Flag | Description |
|------|-------------|
| `--sentence` | Sentence-based segmentation |
| `--standard` | Standard subtitle formatting |
| `--standard_asia` | Asia-optimized formatting (16 chars/line, smart comma breaking) |
| `--one_word` | One word per subtitle |
| `--max_line_width` | Maximum characters per line |
| `--max_line_count` | Maximum lines per subtitle block |
| `--japanese` / `-ja` | Japanese-specific formatting |

### VAD (Voice Activity Detection)

| Flag | Default | Description |
|------|---------|-------------|
| `--vad_filter` | true | Enable VAD filtering |
| `--vad_method` | `ten` (Pro) / `silero_v4` (free) | VAD method selection |
| `--vad_threshold` | | Detection threshold |
| `--vad_min_speech_duration_ms` | | Minimum speech duration |
| `--vad_max_speech_duration_s` | | Maximum speech segment length |
| `--vad_min_silence_duration_ms` | | Minimum silence to split |
| `--vad_speech_pad_ms` | | Padding around detected speech |

**Available VAD methods:**

| Method | Edition | Notes |
|--------|---------|-------|
| `silero_v3` | Free | Older, fewer quirks |
| `silero_v4` / `silero_v4_fw` | Free | Default in free version |
| `silero_v5` / `silero_v5_fw` | Free | Poor accuracy (not recommended) |
| `silero_v6` / `silero_v6_fw` | Pro | Patched |
| `pyannote_v3` | Free | Best accuracy, supports CUDA |
| `pyannote_onnx_v3` | Free | Lighter |
| `nemo_v2` | Pro | |
| `ten` | Pro | Default in Pro |
| `auditok` | Free | Audio Activity Detection |

### Vocal Separation (MDX/UVR)

| Flag | Description |
|------|-------------|
| `--ff_vocal_extract <model>` | Enable vocal separation before ASR |
| `--voc_device` | Device for vocal separation (`cuda` / `cpu`) |

**Available vocal extraction models:**

| Model | Edition | Notes |
|-------|---------|-------|
| `mdx_kim2` | Free | MDX23 Kim vocal v2 (good for music/BGM) |
| `mdx1_kim2` | Pro | Reworked, better quality, less RAM |
| `mdx2_kim2` | Pro | Alternative MDX implementation |
| `mb-roformer` | Pro | SOTA quality, best vocal separation |

### Audio Filters (FFmpeg-based)

| Flag | Description |
|------|-------------|
| `--ff_rnndn_sh` | RNNoise denoising (SH) |
| `--ff_rnndn_xiph` | RNNoise denoising (Xiph) |
| `--ff_fftdn` | FFT-based denoising |
| `--ff_loudnorm` | Loudness normalization |
| `--ff_speechnorm` | Speech normalization |
| `--ff_gate` | Noise gate |
| `--ff_lc` | Level compressor |
| `--ff_lowhighpass` | Low/high pass filter |
| `--ff_tempo` | Tempo adjustment |
| `--ff_track` | Select audio track |

### Speaker Diarization

| Flag | Choices |
|------|---------|
| `--diarize` | `pyannote_v3.0`, `pyannote_v3.1`, `reverb_v1`, `reverb_v2` |

### Batch & Performance

| Flag | Description |
|------|-------------|
| `--batch_recursive` | Process folders recursively |
| `--batched` | Enable batched inference |
| `--batch_size` | Batch size for batched mode |
| `--beep_off` | Disable completion beep |

---

## Output Format

- **Default**: SRT file, saved to `--output_dir` or next to input file
- **File naming**: `<input_stem>.srt` (or `<input_stem>.<lang>.srt` depending on config)
- **Supported formats**: SRT, VTT, JSON, TXT, TSV, LRC, or `all`
- **JSON output** includes word-level timestamps and segment metadata

---

## Key Differences from Standard Faster-Whisper

1. **Built-in vocal separation** (`--ff_vocal_extract mdx_kim2`)
2. **Multiple VAD methods** (pyannote, auditok, silero v3-v6, ten, nemo)
3. **FFmpeg audio filters** built-in
4. **Speaker diarization** built-in
5. **Japanese-specific flags** (`-ja`, `--standard_asia`)
6. **Sentence-based formatting** (`--sentence`, `--standard`)
7. **Standalone executable** — no Python environment needed

---

## Quality Gap Analysis (from Issues #223, #224)

The quality advantage of XXL over WhisperJAV balanced mode comes from:

**Layer 1 — Parameter defaults (~70% of gap, fixable in WhisperJAV):**
- Model: XXL uses large-v3, WhisperJAV uses large-v2
- Compute type: XXL uses float16, WhisperJAV resolves to int8_float16
- Beam search: XXL uses beam_size=5/best_of=5, WhisperJAV uses 2/1

**Layer 2 — Vocal separation (~30% of gap, architectural):**
- XXL's `--ff_vocal_extract mdx_kim2` extracts clean vocals before ASR
- WhisperJAV has speech enhancement (ClearVoice, BS-RoFormer) but not the same MDX-Net vocal extraction
- This is the key advantage for noisy content (BGM, sound effects)

---

## Sources

- [GitHub Repository](https://github.com/Purfview/whisper-standalone-win)
- [Discussion #231 — Feature Documentation](https://github.com/Purfview/whisper-standalone-win/discussions/231)
- [Discussion #456 — XXL Pro Features](https://github.com/Purfview/whisper-standalone-win/discussions/456)
- [Release r245.4 (Free)](https://github.com/Purfview/whisper-standalone-win/releases/tag/Faster-Whisper-XXL)
- [Changelog](https://raw.githubusercontent.com/Purfview/whisper-standalone-win/main/changelog.txt)
