"""R-SCN analysis: the sound's measures over time, the change strength, digital silence and quiet stretches.

Adapted from the owner's reference R-SCN 1.0 (D:\\Git\\MOSS-Transcrive-Concept-Implementation\\r_scn_detector_v1_0.py:
extract_features, digital_silence_spans, quiet_intervals). One change, by the owner's requirements (study 5a (2), 5b):
the change strength is absolute. Each measure is divided by a fixed unit (settings.unit_*), and the strength at a moment
is the root-mean-square difference of the measures' means before and after it, so the threshold means the same on every
film; the reference scaled every film to its own 95th percentile. Design: docs/plans/design/2026-11-02-rscn-design.md
sections 5.1-5.2.

numpy and scipy only (both WhisperJAV dependencies); no model.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

from .settings import RSCNSettings

MEASURE_NAMES = ("level_db", "floor_db", "above_floor_db", "spectral_centre", "flatness", "spectral_change",
                 "speech_band_share")


@dataclass
class Measures:
    """The measures of one file at the analysis rate."""
    fine_level_db: np.ndarray      # every feature hop (20 ms): the level, for quiet stretches and placement
    fine_hop_s: float
    coarse: np.ndarray             # every novelty hop (100 ms): the seven measures, divided by their units
    coarse_hop_s: float
    coarse_level_db: np.ndarray    # every novelty hop: the level (dB), for marks

    @property
    def duration_s(self) -> float:
        return len(self.fine_level_db) * self.fine_hop_s


def measures(audio: np.ndarray, s: RSCNSettings) -> Measures:
    """The seven measures of mono float32 audio at ``s.analysis_rate`` (the reference's extract_features, in
    bounded batches, every sample analysed)."""
    rate = s.analysis_rate
    n = len(audio)
    hop = round(s.feature_hop_ms * rate / 1000)
    frame_len = round(s.frame_ms * rate / 1000)
    n_frames = max(1, (n + hop - 1) // hop)
    level = np.empty(n_frames, dtype=np.float32)
    centre = np.empty(n_frames, dtype=np.float32)
    flat = np.empty(n_frames, dtype=np.float32)
    change = np.empty(n_frames, dtype=np.float32)
    band = np.empty(n_frames, dtype=np.float32)
    window = np.hanning(frame_len).astype(np.float32)
    freqs = sfft.rfftfreq(frame_len, 1.0 / rate).astype(np.float32)
    in_band = (freqs >= s.speech_band_low_hz) & (freqs <= min(s.speech_band_high_hz, rate / 2))
    previous = None
    batch = 2048
    for first in range(0, n_frames, batch):
        count = min(batch, n_frames - first)
        start = first * hop
        need = (count - 1) * hop + frame_len
        block = np.asarray(audio[start:min(n, start + need)], dtype=np.float32)
        if len(block) < need:
            block = np.pad(block, (0, need - len(block)))
        frames = np.lib.stride_tricks.sliding_window_view(block, frame_len)[::hop][:count]
        lengths = np.maximum(1, np.minimum(frame_len, n - (first + np.arange(count)) * hop))
        energy = np.sum(np.square(frames, dtype=np.float32), axis=1, dtype=np.float64) / lengths
        level[first:first + count] = np.maximum(10.0 * np.log10(np.maximum(energy, 1e-18)), s.level_floor_db)
        magnitude = np.abs(sfft.rfft(frames * window[None, :], axis=1, workers=1)).astype(np.float32)
        power = magnitude * magnitude
        total = np.sum(magnitude, axis=1, dtype=np.float64)
        centre[first:first + count] = (np.sum(magnitude * freqs[None, :], axis=1, dtype=np.float64)
                                       / np.maximum(total, 1e-10)) / (rate / 2)
        flat[first:first + count] = (np.exp(np.mean(np.log(np.maximum(power, 1e-14)), axis=1))
                                     / np.maximum(np.mean(power, axis=1), 1e-14))
        band[first:first + count] = (np.sum(power[:, in_band], axis=1, dtype=np.float64)
                                     / np.maximum(np.sum(power, axis=1, dtype=np.float64), 1e-12))
        shape = magnitude / np.maximum(total[:, None], 1e-8)
        change[first] = 0.0 if previous is None else float(np.sum(np.abs(shape[0] - previous)))
        if count > 1:
            change[first + 1:first + count] = np.sum(np.abs(shape[1:] - shape[:-1]), axis=1)
        previous = shape[-1].copy()

    group = round(s.novelty_hop_ms / s.feature_hop_ms)
    starts = np.arange(0, n_frames, group)
    counts = np.minimum(group, n_frames - starts)

    def coarse(x: np.ndarray) -> np.ndarray:
        return (np.add.reduceat(x.astype(np.float64), starts) / counts).astype(np.float32)

    c_level = coarse(level)
    size = max(3, int(round(s.floor_window_s * 1000 / s.novelty_hop_ms)) | 1)
    floor = ndimage.percentile_filter(c_level, percentile=s.floor_percentile, size=size, mode="nearest")
    columns = [c_level / s.unit_level_db, floor / s.unit_floor_db, (c_level - floor) / s.unit_above_floor_db,
               coarse(centre) / s.unit_centroid, coarse(flat) / s.unit_flatness, coarse(change) / s.unit_flux,
               coarse(band) / s.unit_band]
    return Measures(level, hop / rate, np.column_stack(columns).astype(np.float32), s.novelty_hop_ms / 1000.0,
                    c_level)


def strength(coarse: np.ndarray, s: RSCNSettings) -> Tuple[np.ndarray, List[np.ndarray]]:
    """The change strength of one region (rows of ``Measures.coarse``): per scale, at each moment, the RMS over the
    measures of (mean over the window before - mean over the window after); then the weighted mean over the scales.
    Near the region's edges a window is shortened to what fits, down to half its length (at least one second); a
    scale that does not fit is left out of the mean there, so a change close to an edge still shows."""
    n = len(coarse)
    hop_s = s.novelty_hop_ms / 1000.0
    per_scale: List[np.ndarray] = []
    if n < 2:
        return np.zeros(n, dtype=np.float32), [np.zeros(n, dtype=np.float32) for _ in s.novelty_scales_s]
    integral = np.zeros((n + 1, coarse.shape[1]), dtype=np.float64)
    np.cumsum(coarse, axis=0, dtype=np.float64, out=integral[1:])
    t = np.arange(n + 1)
    shortest = max(1, round(s.edge_min_window_s / hop_s))
    valid: List[np.ndarray] = []
    for seconds in s.novelty_scales_s:
        w = max(1, round(seconds / hop_s))
        width = np.minimum(w, np.minimum(t, n - t))
        curve = np.zeros(n + 1, dtype=np.float32)
        # A scale counts where at least half its window fits on each side (at least one second): near an edge the
        # long scales drop out instead of turning into short ones.
        ok = width >= max(min(shortest, w), (w + 1) // 2)
        valid.append(ok[:n])
        idx = t[ok]
        wd = width[ok]
        before = (integral[idx] - integral[idx - wd]) / wd[:, None]
        after = (integral[idx + wd] - integral[idx]) / wd[:, None]
        curve[ok] = np.sqrt(np.mean((before - after) ** 2, axis=1)).astype(np.float32)
        curve = curve[:n]  # the value at boundary k is the change between rows < k and rows >= k
        if s.novelty_smooth_frames > 1 and n >= s.novelty_smooth_frames:
            curve = ndimage.median_filter(curve, size=int(s.novelty_smooth_frames), mode="nearest")
        per_scale.append(curve.astype(np.float32))
    weights = np.asarray(s.novelty_scale_weights, dtype=np.float64)[:, None]
    mask = np.stack(valid).astype(np.float64)
    total = (weights * mask).sum(axis=0)
    combined = np.where(total > 0, (weights * mask * np.stack(per_scale)).sum(axis=0) / np.maximum(total, 1e-12), 0.0)
    return combined.astype(np.float32), per_scale


def digital_silence(samples: np.ndarray, rate: int, min_s: float) -> List[Tuple[int, int]]:
    """Runs of exact-zero samples of at least ``min_s`` seconds, in the samples' own units (the reference's
    digital_silence_spans, on the WAV's own samples: a resampled copy would blur the zeros' edges)."""
    minimum = int(np.ceil(min_s * rate))
    spans: List[Tuple[int, int]] = []
    active: Optional[int] = None
    previous = False
    block = 1_000_000
    total = len(samples)
    for base in range(0, total, block):
        zero = np.asarray(samples[base:min(base + block, total)] == 0, dtype=np.bool_)
        if zero.size == 0:
            continue
        if bool(zero[0]) != previous:
            if zero[0]:
                active = base
            else:
                if active is not None and base - active >= minimum:
                    spans.append((active, base))
                active = None
        for point in np.flatnonzero(zero[1:] != zero[:-1]) + 1:
            at = base + int(point)
            if zero[point]:
                active = at
            else:
                if active is not None and at - active >= minimum:
                    spans.append((active, at))
                active = None
        previous = bool(zero[-1])
    if previous and active is not None and total - active >= minimum:
        spans.append((active, total))
    return spans


def quiet_stretches(level_db: np.ndarray, hop_s: float, s: RSCNSettings) -> List[Tuple[float, float]]:
    """Low-level stretches (seconds) that may hold a pause: candidate cut points only, never proof of no speech
    (the reference's quiet_intervals, its fixed values made settings)."""
    if len(level_db) < 2:
        return []
    smooth = ndimage.median_filter(level_db, size=int(s.quiet_median_frames), mode="nearest")
    p05, p80, p95 = np.percentile(level_db, [s.quiet_low_percentile, s.quiet_mid_percentile, s.quiet_high_percentile])
    mask = np.zeros(len(smooth), dtype=bool)
    if p95 - p05 >= s.quiet_spread_db:
        mask |= smooth <= min(p05 + s.quiet_above_p05_db, p80 - s.quiet_below_p80_db)
    context = ndimage.uniform_filter1d(smooth, size=max(5, round(s.quiet_context_s / hop_s)), mode="nearest")
    mask |= smooth <= context - s.quiet_drop_db
    edges = np.flatnonzero(np.diff(np.pad(mask.astype(np.int8), (1, 1))) != 0)
    minimum = max(1, round(s.quiet_min_s / hop_s))
    return [(i * hop_s, j * hop_s) for i, j in zip(edges[::2], edges[1::2]) if j - i >= minimum]


def to_analysis_rate(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    """A float32 copy of the audio at the analysis rate (polyphase resampling; the same audio when equal)."""
    samples = np.asarray(samples, dtype=np.float32)
    if rate == target:
        return samples
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(rate), int(target))
    return resample_poly(samples, int(target) // g, int(rate) // g).astype(np.float32)
