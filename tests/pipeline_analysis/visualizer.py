"""
Interactive matplotlib visualization for the pipeline analysis test suite.

Displays waveform, segment bars, coverage heatmap, and ground truth overlay
with synchronized audio playback and cursor.

No WhisperJAV imports — receives data via models.
"""

import time
from typing import Dict, List, Optional

import numpy as np

from .analyzer import compute_time_coverage_array
from .models import AnalysisResult, BackendRunResult, SegmentInfo


# ---------------------------------------------------------------------------
# Audio display helpers
# ---------------------------------------------------------------------------


def downsample_for_display(
    audio_data: np.ndarray, max_points: int = 2000
) -> np.ndarray:
    """Downsample audio for waveform display using max-pooling.

    Preserves peaks for better waveform visualization.
    """
    if len(audio_data) <= max_points:
        return audio_data

    chunk_size = len(audio_data) // max_points
    n_chunks = len(audio_data) // chunk_size
    trimmed = audio_data[: n_chunks * chunk_size]
    reshaped = trimmed.reshape(n_chunks, chunk_size)

    max_vals = np.max(np.abs(reshaped), axis=1)
    signs = np.sign(reshaped[:, chunk_size // 2])
    return (max_vals * signs).astype(np.float32)


# ---------------------------------------------------------------------------
# Playback state
# ---------------------------------------------------------------------------


class PlaybackState:
    """Track audio playback state for interactive player."""

    def __init__(self, audio_data: np.ndarray, sample_rate: int):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.duration = len(audio_data) / sample_rate
        self.is_playing = False
        self.position = 0.0
        self.start_time: Optional[float] = None

    def get_current_position(self) -> float:
        if self.is_playing and self.start_time is not None:
            elapsed = time.time() - self.start_time
            return min(elapsed, self.duration)
        return self.position


def _play_audio(state: PlaybackState) -> None:
    """Start audio playback from current position."""
    try:
        import sounddevice as sd
    except ImportError:
        return

    state.is_playing = True
    state.start_time = time.time() - state.position

    start_sample = int(state.position * state.sample_rate)
    audio_to_play = state.audio_data[start_sample:]
    sd.play(audio_to_play, state.sample_rate)


def _stop_audio(state: PlaybackState) -> None:
    """Stop audio playback and save position."""
    try:
        import sounddevice as sd

        sd.stop()
    except ImportError:
        pass

    if state.is_playing and state.start_time is not None:
        state.position = time.time() - state.start_time
        state.position = min(state.position, state.duration)

    state.is_playing = False
    state.start_time = None


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

# Scene detection backends
_SCENE_COLORS = {
    "auditok": "#E65100",  # Deep orange
    "silero": "#1565C0",  # Deep blue
    "semantic": "#2E7D32",  # Deep green
    "none": "#9E9E9E",  # Grey
}

# Speech segmentation backends
_SEG_COLORS = {
    "silero-v4.0": "#2196F3",  # Blue
    "silero-v3.1": "#03A9F4",  # Light blue
    "silero-v6.2": "#0D47A1",  # Dark blue
    "ten": "#FF9800",  # Orange
    "nemo-lite": "#4CAF50",  # Green
    "nemo": "#4CAF50",
    "nemo-diarization": "#8BC34A",  # Light green
    "whisper-vad": "#9C27B0",  # Purple
    "none": "#9E9E9E",  # Grey
}

_DEFAULT_COLOR = "#673AB7"  # Purple fallback


def _get_color(result: BackendRunResult) -> str:
    """Get color for a backend result based on type and name."""
    if result.backend_type == "scene_detection":
        return _SCENE_COLORS.get(result.backend_name, _DEFAULT_COLOR)
    else:
        return _SEG_COLORS.get(result.backend_name, _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Interactive player
# ---------------------------------------------------------------------------


def create_interactive_player(
    audio_data: np.ndarray,
    sample_rate: int,
    results: Dict[str, BackendRunResult],
    analyses: Dict[str, AnalysisResult],
    title: str = "Pipeline Analysis",
    ground_truth: Optional[List[SegmentInfo]] = None,
    show_heatmap: bool = True,
) -> None:
    """Create interactive visualization with audio playback.

    Layout (top to bottom):
    1. Waveform with amplitude envelope
    2. Coverage heatmap (if show_heatmap=True)
    3. Ground truth bar (if provided)
    4. One bar per backend result

    Controls:
        SPACE: Play/Stop toggle
        R: Reset to beginning
        Click: Seek to clicked position
        Q/ESC: Quit
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        return

    # Check sounddevice availability (optional)
    has_sd = False
    try:
        import sounddevice as sd

        has_sd = True
    except ImportError:
        print(
            "sounddevice not available — visualization without audio playback"
        )

    # Filter to successful results only
    successful = {k: v for k, v in results.items() if v.success}
    if not successful:
        print("No successful results to visualize.")
        return

    duration = len(audio_data) / sample_rate
    state = PlaybackState(audio_data, sample_rate)

    # Determine layout
    has_gt = ground_truth is not None and len(ground_truth) > 0
    num_backends = len(successful)
    has_heatmap = show_heatmap and num_backends > 0

    # Row count: waveform + heatmap? + ground_truth? + backends
    num_rows = 1 + (1 if has_heatmap else 0) + (1 if has_gt else 0) + num_backends
    height_ratios = (
        [2.5]
        + ([1.2] if has_heatmap else [])
        + ([0.8] if has_gt else [])
        + [0.8] * num_backends
    )
    fig_height = max(4, 2.5 + (num_rows - 1) * 0.9)

    fig, axes = plt.subplots(
        num_rows,
        1,
        figsize=(16, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    # Ensure axes is always a list
    if num_rows == 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = list(axes)

    ax_idx = 0

    # --- Row 1: Waveform ---
    ax_wave = axes[ax_idx]
    display_audio = downsample_for_display(audio_data, max_points=3000)
    time_axis = np.linspace(0, duration, len(display_audio))
    ax_wave.plot(time_axis, display_audio, color="#424242", linewidth=0.4, alpha=0.8)
    ax_wave.fill_between(time_axis, display_audio, alpha=0.25, color="#757575")
    ax_wave.set_ylabel("Amp", fontsize=8)
    ax_wave.set_title(title, fontsize=11, fontweight="bold")
    ax_wave.set_xlim(0, duration)
    ax_wave.grid(True, alpha=0.2)

    # Overlay ground truth on waveform if available
    if has_gt:
        for seg in ground_truth:
            ax_wave.axvspan(
                seg.start_sec, seg.end_sec, alpha=0.15, color="gold"
            )

    ax_idx += 1

    # --- Row 2: Coverage heatmap (optional) ---
    if has_heatmap:
        ax_heat = axes[ax_idx]
        resolution_ms = 10
        num_bins = max(1, int(duration * 1000 / resolution_ms))

        # Build coverage matrix: each row is one backend
        backend_names = list(successful.keys())
        coverage_matrix = np.zeros((len(backend_names), num_bins), dtype=np.float32)
        for i, key in enumerate(backend_names):
            arr = compute_time_coverage_array(
                successful[key].segments, duration, resolution_ms
            )
            # Ensure arrays match (handle rounding)
            min_len = min(len(arr), num_bins)
            coverage_matrix[i, :min_len] = arr[:min_len].astype(np.float32)

        # Sum across backends to get density (0 = no coverage, N = all backends agree)
        density = np.sum(coverage_matrix, axis=0)

        # Display as heatmap (single row image)
        density_2d = density.reshape(1, -1)
        time_edges = np.linspace(0, duration, num_bins + 1)
        row_edges = np.array([0, 1])

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "coverage",
            ["#FFEBEE", "#FFCDD2", "#EF9A9A", "#66BB6A", "#2E7D32"],
        )
        ax_heat.pcolormesh(
            time_edges,
            row_edges,
            density_2d,
            cmap=cmap,
            vmin=0,
            vmax=max(1, len(backend_names)),
            shading="flat",
        )
        ax_heat.set_ylim(0, 1)
        ax_heat.set_yticks([])
        label = f"Coverage\n({len(backend_names)} backends)"
        ax_heat.set_ylabel(
            label, fontsize=7, rotation=0, ha="right", va="center", labelpad=70
        )
        ax_heat.set_xlim(0, duration)

        ax_idx += 1

    # --- Row 3: Ground truth (optional) ---
    if has_gt:
        ax_gt = axes[ax_idx]
        gt_coverage_sec = sum(s.duration_sec for s in ground_truth)
        gt_ratio = gt_coverage_sec / duration if duration > 0 else 0

        for seg in ground_truth:
            ax_gt.barh(
                0,
                seg.duration_sec,
                left=seg.start_sec,
                height=0.6,
                color="#FFD700",
                alpha=0.9,
                edgecolor="#B8860B",
                linewidth=0.5,
            )

        ax_gt.set_xlim(0, duration)
        ax_gt.set_ylim(-0.5, 0.5)
        ax_gt.set_yticks([])
        stats = f"Ground Truth\n{len(ground_truth)} segs | {gt_ratio:.1%}"
        ax_gt.set_ylabel(
            stats, fontsize=7, rotation=0, ha="right", va="center", labelpad=70
        )
        ax_gt.set_facecolor("#FFFACD")
        ax_gt.grid(True, axis="x", alpha=0.2)

        ax_idx += 1

    # --- Rows 4+: Backend results ---
    for i, (key, result) in enumerate(successful.items()):
        ax = axes[ax_idx + i]
        color = _get_color(result)

        for seg in result.segments:
            ax.barh(
                0,
                seg.duration_sec,
                left=seg.start_sec,
                height=0.6,
                color=color,
                alpha=0.8,
            )

        ax.set_xlim(0, duration)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])

        # Build label with metrics
        analysis = analyses.get(key)
        if analysis:
            stats = (
                f"{result.display_name}\n"
                f"{analysis.num_segments} segs | "
                f"{analysis.coverage_ratio:.1%}"
            )
        else:
            stats = f"{result.display_name}\n{len(result.segments)} segs"

        ax.set_ylabel(
            stats, fontsize=7, rotation=0, ha="right", va="center", labelpad=70
        )
        ax.grid(True, axis="x", alpha=0.2)

        # Label x-axis on last row
        if i == len(successful) - 1:
            ax.set_xlabel("Time (seconds)", fontsize=9)

    # --- Cursor lines ---
    cursor_lines = []
    for ax in axes:
        line = ax.axvline(
            x=0, color="red", linewidth=1.5, linestyle="-", alpha=0.9, zorder=100
        )
        cursor_lines.append(line)

    # --- Status and time text ---
    controls = "SPACE: Play/Stop | R: Reset | Click: Seek | Q: Quit"
    if not has_sd:
        controls = "Click: Seek | Q: Quit (no audio — install sounddevice)"
    status_text = fig.text(
        0.5,
        0.01,
        controls,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#666666",
        style="italic",
    )

    time_text = fig.text(
        0.02,
        0.01,
        "0:00 / 0:00",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#333333",
        family="monospace",
    )

    def _fmt_time(seconds: float) -> str:
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m}:{s:02d}"

    def _animate(frame: int):
        if state.is_playing:
            position = state.get_current_position()
            if position >= duration:
                _stop_audio(state)
                state.position = 0
                position = 0
            for line in cursor_lines:
                line.set_xdata([position, position])
        else:
            position = state.position

        time_text.set_text(f"{_fmt_time(position)} / {_fmt_time(duration)}")
        return cursor_lines + [time_text]

    def _on_key(event):
        if event.key == " ":
            if state.is_playing:
                _stop_audio(state)
                status_text.set_text("PAUSED | " + controls)
            else:
                _play_audio(state)
                status_text.set_text("PLAYING | " + controls)
            fig.canvas.draw_idle()
        elif event.key == "r":
            _stop_audio(state)
            state.position = 0
            for line in cursor_lines:
                line.set_xdata([0, 0])
            status_text.set_text("RESET | " + controls)
            fig.canvas.draw_idle()
        elif event.key in ("q", "escape"):
            _stop_audio(state)
            plt.close(fig)

    def _on_click(event):
        if event.inaxes and event.button == 1:
            pos = event.xdata
            if pos is not None and 0 <= pos <= duration:
                was_playing = state.is_playing
                if was_playing:
                    _stop_audio(state)
                state.position = pos
                for line in cursor_lines:
                    line.set_xdata([pos, pos])
                if was_playing:
                    _play_audio(state)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("button_press_event", _on_click)

    _ani = FuncAnimation(
        fig, _animate, interval=50, blit=False, cache_frame_data=False
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)
    plt.show()

    _stop_audio(state)
