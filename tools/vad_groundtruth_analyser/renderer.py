"""Interactive Plotly HTML renderer — N-track DAW-style layout.

Layout:
    Row 0: Waveform (always)
    Row 1: Ground truth (only if GT provided)
    Rows 2..N: one per successful backend, with metrics badge in y-axis label

Shared x-axis across all rows; single range slider on the bottom. Dark theme
for readability alongside matplotlib outputs. Hover tooltips show per-segment
details. Offline self-contained HTML (no CDN calls).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .models import AnalysisReport, BackendReport, GtSegment, VadSegment

logger = logging.getLogger("whisperjav.vad_gt")

# Color palette — distinct per backend, deterministic assignment
COLORS = {
    # Reserved
    "waveform":       "#3498db",                    # blue line
    "waveform_fill":  "rgba(52, 152, 219, 0.5)",
    "ground_truth":   "#2ecc71",                    # bright green (reserved)
    "ground_truth_fill": "rgba(46, 204, 113, 0.70)",

    # Backend-specific (matched to the HTML dropdown intent)
    "silero":         ("#2196F3", "rgba( 33, 150, 243, 0.70)"),
    "silero-v4.0":    ("#2196F3", "rgba( 33, 150, 243, 0.70)"),
    "silero-v3.1":    ("#03A9F4", "rgba(  3, 169, 244, 0.70)"),
    "silero-v6.2":    ("#1976D2", "rgba( 25, 118, 210, 0.70)"),
    "ten":            ("#FF9800", "rgba(255, 152,   0, 0.70)"),
    "whisperseg":     ("#9b59b6", "rgba(155,  89, 182, 0.75)"),
    "nemo":           ("#4CAF50", "rgba( 76, 175,  80, 0.70)"),
    "nemo-lite":      ("#4CAF50", "rgba( 76, 175,  80, 0.70)"),
    "nemo-diarization": ("#8BC34A", "rgba(139, 195,  74, 0.70)"),
    "whisper-vad":    ("#FF5722", "rgba(255,  87,  34, 0.70)"),
    "none":           ("#9E9E9E", "rgba(158, 158, 158, 0.70)"),
}

_FALLBACK_PALETTE = [
    ("#673AB7", "rgba(103,  58, 183, 0.70)"),
    ("#009688", "rgba(  0, 150, 136, 0.70)"),
    ("#E91E63", "rgba(233,  30,  99, 0.70)"),
    ("#795548", "rgba(121,  85,  72, 0.70)"),
]

BG_DARK = "#1a1a2e"
GRID = "#333355"
TEXT_LIGHT = "#ffffff"


def _backend_colors(name: str, idx: int = 0):
    """Return (line_hex, fill_rgba) for a backend."""
    entry = COLORS.get(name)
    if isinstance(entry, tuple):
        return entry
    return _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)]


def _downsample_waveform(audio: np.ndarray, max_points: int = 10000) -> np.ndarray:
    """Max-pool downsample for peak-preserving waveform display.

    Copied pattern from scripts/visualization/waveform_processor.py to keep
    this tool self-contained.
    """
    if len(audio) <= max_points:
        return audio
    chunk = len(audio) // max_points
    n_chunks = len(audio) // chunk
    trimmed = audio[: n_chunks * chunk]
    reshaped = trimmed.reshape(n_chunks, chunk)
    max_vals = np.max(np.abs(reshaped), axis=1)
    signs = np.sign(reshaped[:, chunk // 2])
    # Handle zero-sign edge (pure silence) by treating as positive
    signs = np.where(signs == 0, 1, signs)
    return (max_vals * signs).astype(np.float32)


def _shorten_text(text: str, max_chars: int = 40) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _format_gt_hover(seg: GtSegment) -> str:
    dur = seg.end_sec - seg.start_sec
    return (
        f"<b>GT #{seg.index}</b><br>"
        f"{seg.start_sec:.3f}s — {seg.end_sec:.3f}s (dur {dur:.3f}s)<br>"
        f"<i>{_shorten_text(seg.text, 80)}</i>"
    )


def _format_vad_hover(seg: VadSegment, backend_name: str) -> str:
    dur = seg.end_sec - seg.start_sec
    parts = [
        f"<b>{backend_name}</b>",
        f"{seg.start_sec:.3f}s — {seg.end_sec:.3f}s (dur {dur:.3f}s)",
        f"confidence: {seg.confidence:.3f}",
    ]
    if seg.metadata:
        interesting = {k: v for k, v in seg.metadata.items() if not k.startswith("_")}
        if interesting:
            parts.append(
                "metadata: " + ", ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in list(interesting.items())[:3]
                )
            )
    return "<br>".join(parts)


# (previous _metrics_badge helper removed — label composition moved inline
# into render_html where it's attached as a y-axis title per row.)


def _add_segment_bars(
    fig,
    segments: List,
    row: int,
    line_color: str,
    fill_color: str,
    hover_texts: List[str],
    name: str,
) -> None:
    """Draw segments as filled rectangles on a subplot row.

    Uses Plotly Bar horizontal; each bar is centred at y=0 with height 0.8.
    """
    import plotly.graph_objects as go  # lazy import

    if not segments:
        # Still add an empty trace so legend entry exists and row renders
        fig.add_trace(
            go.Bar(
                x=[0], y=[0],
                base=[0],
                orientation="h",
                marker=dict(color=fill_color, line=dict(color=line_color, width=0)),
                hoverinfo="skip",
                showlegend=False,
                name=name,
            ),
            row=row, col=1,
        )
        return

    widths = [max(0.001, s.end_sec - s.start_sec) for s in segments]
    bases = [s.start_sec for s in segments]
    ys = [0] * len(segments)

    fig.add_trace(
        go.Bar(
            x=widths, y=ys, base=bases,
            orientation="h",
            marker=dict(color=fill_color, line=dict(color=line_color, width=1)),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=True,
            name=name,
        ),
        row=row, col=1,
    )


def render_html(
    report: AnalysisReport,
    waveform: np.ndarray,
    output_path: Path,
    title: Optional[str] = None,
    waveform_points: int = 10000,
    gt_segments: Optional[List[GtSegment]] = None,
) -> None:
    """Render the analysis to an interactive HTML file.

    Args:
        report: populated AnalysisReport.
        waveform: full-resolution audio for waveform downsampling.
        output_path: HTML destination.
        title: optional custom title (defaults to media filename).
        waveform_points: max points in waveform trace after downsampling.
        gt_segments: full list of GT segments (for per-segment hover text).
            If None, no GT track is drawn even when report.ground_truth is set.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError(
            "plotly is required for HTML rendering. "
            "Install with: pip install plotly>=5.0.0"
        ) from e

    duration = max(report.audio_duration_sec, 0.001)

    # Determine row layout
    has_gt_row = bool(report.ground_truth and gt_segments)
    successful_backends = [
        (name, rep) for name, rep in report.backends.items() if rep.success
    ]
    n_backend_rows = len(successful_backends)

    if n_backend_rows == 0:
        logger.warning("No successful backends; HTML will render waveform only")

    n_rows = 1 + (1 if has_gt_row else 0) + n_backend_rows

    # Row heights: waveform gets ~0.35; backend rows ~0.12 each; GT ~0.13
    row_heights: List[float] = [0.35]
    if has_gt_row:
        row_heights.append(0.13)
    row_heights.extend([0.12] * n_backend_rows)
    # Normalise
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    # NOTE: We intentionally do NOT use subplot_titles. Plotly places them in
    # the narrow gap between rows, which (with tight vertical_spacing) makes
    # each label visually read as if it belonged to the row ABOVE it. Instead,
    # we attach labels as y-axis titles — these are anchored to their own
    # subplot and render to the LEFT of each row, unambiguously.
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # Collected (row_idx, label_text, colour) for horizontal annotations
    # rendered in the left margin once the layout is finalised.
    row_labels: List[tuple] = []

    # --- Row 1: Waveform ---------------------------------------------------
    ds = _downsample_waveform(waveform, max_points=waveform_points)
    x_wave = np.linspace(0.0, duration, len(ds))
    fig.add_trace(
        go.Scatter(
            x=x_wave, y=ds,
            mode="lines",
            line=dict(color=COLORS["waveform"], width=0.5),
            fill="tozeroy",
            fillcolor=COLORS["waveform_fill"],
            name="Waveform",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1,
    )
    row_labels.append((1, "<b>Waveform</b><br>amplitude", COLORS["waveform"]))

    current_row = 2

    # --- Optional GT row --------------------------------------------------
    if has_gt_row:
        hover_gt = [_format_gt_hover(g) for g in gt_segments]
        _add_segment_bars(
            fig,
            gt_segments,
            row=current_row,
            line_color=COLORS["ground_truth"],
            fill_color=COLORS["ground_truth_fill"],
            hover_texts=hover_gt,
            name="Ground Truth",
        )
        fig.update_yaxes(
            showticklabels=False,
            range=[-0.5, 0.5],
            row=current_row, col=1,
        )
        # Record this row's label + colour for horizontal annotation pass below.
        gt_n = report.ground_truth.get("num_segments", len(gt_segments)) if report.ground_truth else len(gt_segments)
        row_labels.append((
            current_row,
            f"<b>Ground Truth</b><br>{gt_n} segments",
            COLORS["ground_truth"],
        ))
        current_row += 1

    # --- Backend rows -----------------------------------------------------
    for idx, (name, rep) in enumerate(successful_backends):
        line_color, fill_color = _backend_colors(name, idx)
        hover_texts = [_format_vad_hover(s, rep.display_name) for s in rep.segments]
        _add_segment_bars(
            fig,
            rep.segments,
            row=current_row,
            line_color=line_color,
            fill_color=fill_color,
            hover_texts=hover_texts,
            name=rep.display_name,
        )
        fig.update_yaxes(
            showticklabels=False,
            range=[-0.5, 0.5],
            row=current_row, col=1,
        )
        # Build per-row label (will be rendered horizontally via annotation below)
        label_lines = [f"<b>{rep.display_name}</b>"]
        if rep.metrics is not None:
            m = rep.metrics
            label_lines.append(
                f"F1={m.frame_f1:.2f}  P={m.frame_precision:.2f}  R={m.frame_recall:.2f}"
            )
            label_lines.append(
                f"miss={m.missed_speech_pct:.1f}%  FA={m.false_alarm_pct:.1f}%"
            )
        else:
            label_lines.append(
                f"{rep.num_segments} segs  cov={rep.coverage_ratio * 100:.1f}%"
            )
        row_labels.append((current_row, "<br>".join(label_lines), line_color))
        current_row += 1

    # --- Layout & cosmetics -----------------------------------------------
    if title is None:
        title = f"VAD Comparison — {Path(report.media_file).name}"

    header_parts = [
        f"sensitivity={report.sensitivity}",
        f"duration={report.audio_duration_sec:.2f}s",
    ]
    if report.ground_truth:
        header_parts.append(
            f"GT: {report.ground_truth.get('num_segments', 0)} segments"
        )
    else:
        header_parts.append("GT: none")
    header_parts.append(f"backends={n_backend_rows}")
    subtitle = "  •  ".join(header_parts)

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub style='color:#bbb'>{subtitle}</sub>",
            x=0.02, xanchor="left",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.12,
            xanchor="left", x=0,
            bgcolor="rgba(37, 37, 64, 0.6)",
            font=dict(color=TEXT_LIGHT, size=10),
        ),
        plot_bgcolor=BG_DARK,
        paper_bgcolor=BG_DARK,
        font=dict(color=TEXT_LIGHT, family="Arial, sans-serif"),
        # Wider left margin hosts horizontal row labels.
        # Wider bottom margin hosts the metric-glossary footer.
        margin=dict(l=240, r=30, t=80, b=120),
        height=max(400, 260 + n_rows * 100),
        bargap=0,
    )

    # Waveform y-axis: amplitude ticks + gridlines (label handled by annotation).
    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        row=1, col=1,
    )

    # Shared x-axis; NO rangeslider (Plotly's native box-zoom / drag-pan / scroll
    # zoom already cover navigation, and a rangeslider on the last row shows a
    # misleading miniature of only that backend's trace).
    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        range=[0, duration],
    )
    fig.update_xaxes(
        title_text="Time (seconds)",
        row=n_rows, col=1,
    )

    # --- Horizontal row labels in the left margin ----------------------------
    # Each row's yaxis domain is in paper coordinates after make_subplots.
    # We anchor each label's right edge just outside the plot area (paper x≈-0.005)
    # and its vertical midpoint at the row's centre, with textangle=0 (horizontal).
    for row_idx, label_text, colour in row_labels:
        axis_key = "yaxis" if row_idx == 1 else f"yaxis{row_idx}"
        try:
            domain = fig.layout[axis_key].domain
            y_center = (float(domain[0]) + float(domain[1])) / 2.0
        except (AttributeError, KeyError, TypeError):
            # Fallback: evenly divide [0, 1] if domain not populated.
            y_center = 1.0 - (row_idx - 0.5) / n_rows
        fig.add_annotation(
            text=label_text,
            xref="paper", yref="paper",
            x=-0.005, y=y_center,
            xanchor="right", yanchor="middle",
            showarrow=False,
            textangle=0,
            font=dict(size=11, color=colour),
            align="right",
        )

    # --- Metric-glossary footer (bottom margin) ------------------------------
    # Explains the per-row abbreviations so the chart is self-documenting.
    # Different content for GT vs GT-less modes.
    if report.ground_truth:
        footer_text = (
            "<b>Metrics glossary</b>   "
            f"F1 = 2·P·R / (P+R), harmonic mean of precision and recall   "
            f"&nbsp;•&nbsp;   P (precision) = backend-speech frames that are also GT-speech   "
            f"&nbsp;•&nbsp;   R (recall) = GT-speech frames covered by backend<br>"
            f"miss% = GT-speech frames not covered by backend   "
            f"&nbsp;•&nbsp;   FA% (false-alarm) = backend-speech frames with no GT overlap   "
            f"&nbsp;•&nbsp;   frame grid = {report.frame_ms} ms"
        )
    else:
        footer_text = (
            "<b>GT-less mode</b> — no ground-truth SRT was provided.<br>"
            "Per backend: segs = segment count,  cov% = fraction of audio marked as speech.   "
            "Agreement matrix (in console summary + JSON) = pairwise F1 between successful backends."
        )
    fig.add_annotation(
        text=footer_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        xanchor="center", yanchor="top",
        showarrow=False,
        textangle=0,
        font=dict(size=10, color="#bbbbbb"),
        align="center",
    )

    # Write self-contained offline HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs=True,       # embed plotly.min.js (self-contained, no CDN)
        full_html=True,
        auto_open=False,
    )
    logger.info("Wrote HTML: %s (%d tracks)", output_path, n_rows)
