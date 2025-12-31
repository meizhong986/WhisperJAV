#!/usr/bin/env python3
"""omniASR LLM chunk-to-SRT (Reference Implementation)

New feature reference implementation (not gap filling).

What it does
- Takes a list of *pre-cut* audio chunks (expected <= 30s each).
- Assumes Japanese audio, 16 kHz mono WAV.
- Transcribes each chunk with Meta's OmniASR LLM models:
  - Default: `omniASR_LLM_1B_v2`
  - Optional: `omniASR_LLM_300M_v2`
- Writes one SRT file per chunk.

Timestamps / forced alignment
- OmniASR's public `ASRInferencePipeline.transcribe()` returns only text.
- Word/segment timestamps are NOT provided directly.
- This reference implementation provides an OPTIONAL forced-alignment step
  using a *separate* CTC model (default: ReazonSpeech Wav2Vec2 CTC) and
  `torchaudio.functional.forced_align` if available.

Notes
- For Japanese, "word-level" segmentation is ambiguous (no spaces). This script
  outputs subtitle cues grouped from aligned CTC tokens (typically character-like
  tokens), not true linguistic "words".
- If alignment is unavailable or fails (e.g., missing dependencies, tokenizer
  cannot encode the transcript), it falls back to a single full-span cue.

Example
  python scripts/omnilingual_asr_llm_chunk_to_srt_reference.py \
    --inputs input_chunks/*.wav \
    --output-dir output_srt \
    --omni-model omniASR_LLM_1B_v2 \
    --lang jpn_Jpan \
    --align

VRAM guidance
- With ~7GB VRAM, `omniASR_LLM_1B_v2` should generally be the upper bound.
  Use `--dtype float16` and `--batch-size 1` to reduce memory.

"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


logger = logging.getLogger("omniasr_chunk_srt")


@dataclass(frozen=True)
class SrtCue:
    index: int
    start_s: float
    end_s: float
    text: str


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_srt(path: Path, cues: Sequence[SrtCue]) -> None:
    lines: List[str] = []
    for cue in cues:
        lines.append(str(cue.index))
        lines.append(f"{_format_srt_timestamp(cue.start_s)} --> {_format_srt_timestamp(cue.end_s)}")
        lines.append(cue.text)
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _expand_inputs(patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for item in patterns:
        p = Path(item)
        if any(ch in item for ch in ["*", "?", "["]):
            out.extend(sorted(Path().glob(item)))
        elif p.is_dir():
            out.extend(sorted(p.glob("*.wav")))
        else:
            out.append(p)

    # Deduplicate while preserving order.
    seen = set()
    uniq: List[Path] = []
    for p in out:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)

    return uniq


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            return 0.0
        return frames / float(rate)


def _read_wav_mono_16k(path: Path) -> Tuple["torch.Tensor", int]:
    """Read PCM WAV via stdlib `wave`.

    Returns (waveform, sample_rate) where waveform is float32 torch tensor shape [T].
    """
    import numpy as np
    import torch

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM WAV supported for alignment read: {path} (sampwidth={sampwidth})")

    audio_i16 = np.frombuffer(pcm, dtype=np.int16)

    if n_channels == 2:
        audio_i16 = audio_i16.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif n_channels != 1:
        raise ValueError(f"Unsupported channel count: {n_channels} for {path}")

    waveform = torch.from_numpy(audio_i16.astype(np.float32) / 32768.0)
    return waveform, int(sample_rate)


def _normalize_transcript(text: str) -> str:
    # OmniASR outputs spoken-form; keep it as-is, but collapse whitespace.
    t = re.sub(r"\s+", " ", text).strip()
    return t


def _load_omniasr_pipeline(model_card: str, device: str, dtype_name: str):
    import torch

    try:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    except Exception as e:
        raise RuntimeError(
            "Failed to import omnilingual_asr. Install it first (and fairseq2 deps)."
        ) from e

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}. Choose from {sorted(dtype_map)}")

    dtype = dtype_map[dtype_name]
    return ASRInferencePipeline(model_card=model_card, device=device, dtype=dtype)


class _ForcedAligner:
    """CTC forced aligner (optional) using HuggingFace Wav2Vec2ForCTC + torchaudio."""

    def __init__(
        self,
        model_id: str,
        device: str,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None
        self._torch_device = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self._torch_device = torch.device(self.device)
        logger.info(f"Loading alignment CTC model: {self.model_id} on {self._torch_device}")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id).to(self._torch_device)
        self._model.eval()

    def can_align(self) -> bool:
        try:
            import torchaudio.functional as AF  # noqa: F401

            return hasattr(AF, "forced_align") and hasattr(AF, "merge_tokens")
        except Exception:
            return False

    def align_to_cues(
        self,
        wav_path: Path,
        transcript: str,
        *,
        max_chars_per_cue: int,
        gap_break_s: float,
        min_cue_dur_s: float,
    ) -> Optional[List[Tuple[float, float, str]]]:
        """Return list of (start_s, end_s, text) cues or None if alignment fails."""

        if not self.can_align():
            return None

        self._load()

        import torch
        import torchaudio.functional as AF

        assert self._processor is not None
        assert self._model is not None

        waveform, sr = _read_wav_mono_16k(wav_path)
        if sr != 16000:
            # For reference impl: require correct format.
            raise ValueError(f"Expected 16kHz WAV for alignment, got {sr}Hz: {wav_path}")

        transcript = _normalize_transcript(transcript)
        if not transcript:
            return []

        tokenized = self._processor.tokenizer(transcript)
        target_ids = list(getattr(tokenized, "input_ids", []) or [])
        if not target_ids:
            return None

        unk_id = getattr(self._processor.tokenizer, "unk_token_id", None)
        if unk_id is not None and any(t == unk_id for t in target_ids):
            # Tokenizer can't represent transcript → alignment won't be meaningful.
            return None

        with torch.inference_mode():
            inputs = self._processor(
                waveform.numpy(),
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            logits = self._model(inputs.input_values.to(self._torch_device)).logits  # type: ignore[attr-defined]
            emission = torch.log_softmax(logits[0], dim=-1).to("cpu")

        blank_id = getattr(self._processor.tokenizer, "pad_token_id", 0) or 0
        targets = torch.tensor(target_ids, dtype=torch.int32)

        # torchaudio forced alignment API: returns (path, scores)
        path, scores = AF.forced_align(emission, targets, blank=blank_id)
        spans = AF.merge_tokens(path, scores, blank=blank_id)

        # Map frame indices to seconds.
        audio_dur = float(len(waveform) / sr)
        n_frames = int(emission.shape[0])
        if n_frames <= 0 or audio_dur <= 0:
            return None
        frame_dur = audio_dur / n_frames

        def tok_to_text(token_id: int) -> str:
            tok = self._processor.tokenizer.convert_ids_to_tokens(int(token_id))
            if tok == "|":
                return " "
            # Common CTC tokenizers use special markers; keep it minimal.
            return tok

        # Build token segments.
        token_items: List[Tuple[float, float, str]] = []
        for sp in spans:
            start_s = float(sp.start) * frame_dur
            end_s = float(sp.end) * frame_dur
            if end_s <= start_s:
                end_s = start_s + min_cue_dur_s

            text_piece = tok_to_text(sp.token)
            if not text_piece.strip():
                continue

            token_items.append((start_s, end_s, text_piece))

        if not token_items:
            return None

        # Group token segments into subtitle cues.
        cues: List[Tuple[float, float, str]] = []
        cur_start, cur_end, cur_text = token_items[0][0], token_items[0][1], token_items[0][2]

        def flush():
            nonlocal cur_start, cur_end, cur_text
            t = re.sub(r"\s+", " ", cur_text).strip()
            if t:
                cues.append((cur_start, cur_end, t))

        for start_s, end_s, piece in token_items[1:]:
            gap = start_s - cur_end
            next_text = cur_text + piece
            too_long = len(next_text) > max_chars_per_cue
            break_on_gap = gap_break_s > 0 and gap >= gap_break_s

            if too_long or break_on_gap:
                flush()
                cur_start, cur_end, cur_text = start_s, end_s, piece
            else:
                cur_end = max(cur_end, end_s)
                cur_text = next_text

        flush()
        return cues


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reference: OmniASR LLM chunk transcription → per-chunk SRT (optional forced alignment)",
    )

    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input chunk paths, directories, or globs (e.g. chunks/*.wav).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write per-chunk .srt files.",
    )

    p.add_argument(
        "--omni-model",
        default="omniASR_LLM_1B_v2",
        choices=["omniASR_LLM_300M_v2", "omniASR_LLM_1B_v2"],
        help="OmniASR LLM model card to use.",
    )
    p.add_argument(
        "--lang",
        default="jpn_Jpan",
        help="OmniASR language id (used by LLM models), e.g. jpn_Jpan.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Device for OmniASR LLM inference (e.g. cuda, cpu, cuda:0).",
    )
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype for OmniASR inference. float16 is recommended for ~7GB VRAM.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size. Keep at 1 if VRAM is tight.",
    )

    align_group = p.add_mutually_exclusive_group()
    align_group.add_argument(
        "--align",
        dest="align",
        action="store_true",
        default=True,
        help="Enable forced-alignment timestamps (default: enabled; requires torchaudio + CTC model).",
    )
    align_group.add_argument(
        "--no-align",
        dest="align",
        action="store_false",
        help="Disable forced alignment and emit a single full-span cue per chunk.",
    )
    p.add_argument(
        "--align-model",
        default="reazon-research/reazonspeech-wav2vec2-large-rs35kh",
        help="HuggingFace Wav2Vec2 CTC model for forced alignment.",
    )
    p.add_argument(
        "--align-device",
        default="cpu",
        help="Device for forced-alignment model (cpu recommended to save VRAM).",
    )

    p.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="Reject chunks longer than this (seconds).",
    )
    p.add_argument(
        "--max-chars-per-cue",
        type=int,
        default=16,
        help="When alignment is enabled, group token spans into cues up to this length.",
    )
    p.add_argument(
        "--gap-break-seconds",
        type=float,
        default=0.6,
        help="When alignment is enabled, break cues if silence gap >= this.",
    )
    p.add_argument(
        "--min-cue-dur-seconds",
        type=float,
        default=0.20,
        help="Minimum cue duration when alignment spans collapse.",
    )

    p.add_argument(
        "--write-txt",
        action="store_true",
        help="Also write a .txt transcript file per chunk.",
    )

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    inputs = _expand_inputs(args.inputs)
    if not inputs:
        logger.error("No input files found.")
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-check lengths to fail early and avoid GPU load.
    for p in inputs:
        if not p.exists():
            logger.error(f"Missing input: {p}")
            return 2
        dur = _wav_duration_seconds(p)
        if dur > args.max_seconds:
            logger.error(f"Chunk too long ({dur:.2f}s > {args.max_seconds:.2f}s): {p}")
            return 2

    # Load OmniASR pipeline.
    pipe = _load_omniasr_pipeline(args.omni_model, device=args.device, dtype_name=args.dtype)

    # Language list must match inputs length.
    lang_list = [args.lang] * len(inputs)

    logger.info(f"Transcribing {len(inputs)} chunks with {args.omni_model}...")
    try:
        texts: List[str] = pipe.transcribe([str(p) for p in inputs], lang=lang_list, batch_size=int(args.batch_size))
    except Exception as e:
        logger.exception(f"OmniASR transcription failed: {e}")
        return 1

    if len(texts) != len(inputs):
        logger.error(f"Unexpected output length: got {len(texts)} texts for {len(inputs)} inputs")
        return 1

    aligner: Optional[_ForcedAligner] = None
    if args.align:
        aligner = _ForcedAligner(model_id=args.align_model, device=args.align_device)
        if not aligner.can_align():
            logger.warning(
                "Alignment requested, but torchaudio forced-align is unavailable. "
                "Falling back to single-cue SRT per chunk."
            )

    written = 0

    for p, raw in zip(inputs, texts):
        transcript = _normalize_transcript(raw)
        dur = _wav_duration_seconds(p)

        cues: List[Tuple[float, float, str]]
        if aligner is not None and aligner.can_align() and transcript:
            try:
                aligned = aligner.align_to_cues(
                    p,
                    transcript,
                    max_chars_per_cue=int(args.max_chars_per_cue),
                    gap_break_s=float(args.gap_break_seconds),
                    min_cue_dur_s=float(args.min_cue_dur_seconds),
                )
            except Exception as e:
                logger.warning(f"Alignment failed for {p.name}: {e}")
                aligned = None

            if aligned is not None and len(aligned) > 0:
                cues = aligned
            else:
                cues = [(0.0, max(dur, float(args.min_cue_dur_seconds)), transcript or "")]
        else:
            cues = [(0.0, max(dur, float(args.min_cue_dur_seconds)), transcript or "")]

        srt_cues = [SrtCue(i + 1, a, b, t) for i, (a, b, t) in enumerate(cues) if t.strip()]
        if not srt_cues:
            # Write an empty-but-valid SRT with a single blank cue to keep tooling happy.
            srt_cues = [SrtCue(1, 0.0, max(dur, 0.2), "")]

        out_srt = output_dir / f"{p.stem}.srt"
        _write_srt(out_srt, srt_cues)

        if args.write_txt:
            (output_dir / f"{p.stem}.txt").write_text(transcript + "\n", encoding="utf-8")

        written += 1

    logger.info(f"Wrote {written} SRT files to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
