# Quality-aware ASR pipeline using brouhaha VAD/SNR/C50

This document captures a two-stage pipeline design proposed for WhisperJAV that leverages brouhaha VAD/SNR/C50 to route segments to the optimal processing path.

## Goals
- Maximize ASR accuracy and throughput by routing clean speech directly to ASR
- Improve challenging segments via targeted enhancement prior to ASR
- Remain modular, configurable, and data-driven

## High-level flow
1. Diarize audio and extract segments (existing pyannote pipeline)
2. For each segment:
   - Run brouhaha inference to obtain per-frame (vad, snr, c50)
   - Aggregate per-segment metrics (e.g., mean/median over vad>threshold frames)
   - Classify segment as clean or challenging based on configurable rules
3. Route:
   - Clean → Whisper ASR as-is
   - Challenging → Enhancement stack → Whisper ASR
4. Merge ASR outputs and preserve provenance/metadata (path taken, metrics)

## Segment classification
- Inputs: list of frames with (vad, snr, c50)
- Steps:
  - Keep frames with vad >= VAD_MIN
  - Compute metrics: avg_snr, p50_snr, p10_snr, avg_c50, speech_coverage (ratio of speech frames)
- Decision policies (configurable):
  - Clean if avg_snr >= SNR_CLEAN && avg_c50 >= C50_CLEAN && speech_coverage >= MIN_COVERAGE
  - Challenging otherwise, with sub-classes (noisy, reverberant, low-speech-coverage)
- Optional: score = w_snr * avg_snr + w_c50 * avg_c50; compare with SCORE_THRESH

## Enhancement stack (challenging segments)
- Noisy: denoise → loudness norm → (optional) bandwidth EQ
- Reverberant: dereverb → denoise → norm
- Music/background: vocal separation → denoise → norm
- Low-energy: dynamic range compression/AGC → norm

## Libraries/algorithms (Python-first)
- Denoising:
  - torchaudio.functional.spectral_gate
  - RNNoise (via rnnoise-py)
  - NVIDIA Maxine/RTX Voice (Windows, optional, external)
  - SpeechBrain/asteroid (DPRNN/ConvTasNet/FullSubNet2)
- Vocal separation:
  - demucs (facebookresearch/demucs)
  - spleeter (deezer/spleeter)
- Dereverberation / clarity:
  - Weighted prediction error (WPE) via nara_wpe
  - Kaggle/pyroomacoustics-based dereverb filters
  - DCCRN/DCCRNet models
- Loudness/dynamics:
  - pyloudnorm for EBU R128 normalization
  - librosa effects (preemphasis), simple compressors (custom)
- Utility:
  - pyannote.audio for diarization/embedding
  - brouhaha-vad for VAD/SNR/C50

## Data structures
- SegmentMetrics
  - start: float
  - end: float
  - avg_snr: float
  - avg_c50: float
  - coverage: float
  - label: Literal["clean", "noisy", "reverberant", "music", "low_energy"]
- SegmentDecision
  - route: Literal["direct", "enhance"]
  - reasons: list[str]
  - params: dict[str, Any] (e.g., chosen enhancement chain)

## Config
- YAML/JSON with thresholds and weights
- Example:
  - VAD_MIN: 0.5
  - SNR_CLEAN: 7.5
  - C50_CLEAN: 0.5
  - MIN_COVERAGE: 0.4
  - SCORE: { w_snr: 0.7, w_c50: 0.3, thresh: 6.0 }
  - Chains:
    - noisy: [denoise, norm]
    - reverberant: [dereverb, denoise, norm]
    - music: [separate, denoise, norm]
    - low_energy: [agc, norm]

## Integration notes
- Keep current diarization/embedding flow unchanged
- Insert a segment routing phase prior to embedding ASR or directly prior to Whisper
- Enhancement should be stateless, batch-friendly, and GPU-optional
- Cache enhanced waveforms for reproducibility; tag outputs with metadata

## Testing
- Unit tests for routing decisions given synthetic metrics
- Golden audio samples with known issues to verify improvement
- A/B test: baseline vs two-step routing on WER/CER and speaker attribution

## Future
- Adaptive thresholds learned from data
- Confidence-driven re-segmentation for borderline cases
- On-device light-weight enhancement for real-time
