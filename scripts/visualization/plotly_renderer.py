"""
Plotly-based renderer for multi-track DAW-style visualization.

Generates interactive HTML files with 5 horizontal tracks:
1. Waveform (16kHz Mono) - Blue amplitude envelope
2. Scene Detect Pass 1 - Olive/yellow blocks (coarse boundaries)
3. Scene Detect Pass 2 - Olive/yellow blocks (fine boundaries)
4. VAD Speech Segments - Green blocks with "SPEECH" labels
5. SRT Subtitles - Purple blocks with timestamp + text preview

Features:
- Timeline ruler at top (hh:mm:ss format)
- Multi-track timeline layout (DAW-style)
- Shared x-axis with zoom/pan
- Filled blocks with compact labels
- Hover tooltips
- Range slider for navigation
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError(
        "Plotly is required for visualization. "
        "Install with: pip install -r scripts/visualization/requirements.txt"
    )

from .data_loader import VisualizationData, SceneInfo, VadSegment, Subtitle


# Color scheme matching mockup
COLORS = {
    "waveform": "#3498db",           # Blue
    "waveform_fill": "rgba(52, 152, 219, 0.6)",
    "scene_pass1": "#8B8B3D",        # Olive/yellow-green (from mockup)
    "scene_pass1_fill": "rgba(139, 139, 61, 0.7)",
    "scene_pass2": "#9B9B4D",        # Slightly lighter olive
    "scene_pass2_fill": "rgba(155, 155, 77, 0.7)",
    "vad": "#2ecc71",                # Green
    "vad_fill": "rgba(46, 204, 113, 0.7)",
    "subtitle": "#9b59b6",           # Purple/magenta
    "subtitle_fill": "rgba(155, 89, 182, 0.7)",
    "background": "#1a1a2e",         # Dark background
    "track_bg": "#252540",           # Slightly lighter for tracks
    "grid": "#333355",
    "text": "#ffffff",
    "text_dark": "#000000",
    "label_bg": "#2d2d4a"
}

# Row heights (waveform gets more space)
ROW_HEIGHTS = [0.35, 0.16, 0.16, 0.16, 0.17]


def _format_time_hms(seconds: float) -> str:
    """Format seconds as hh:mm:ss."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def create_visualization(
    data: VisualizationData,
    time_array: np.ndarray,
    min_envelope: np.ndarray,
    max_envelope: np.ndarray,
    title: str = "Audio Analysis Timeline - WhisperJAV"
) -> go.Figure:
    """
    Create multi-track DAW-style Plotly visualization.

    Args:
        data: VisualizationData container
        time_array: Downsampled time points
        min_envelope: Waveform min envelope
        max_envelope: Waveform max envelope
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Create subplots with 5 rows, shared x-axis
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=ROW_HEIGHTS,
        subplot_titles=None
    )

    # Track 1: Waveform
    _add_waveform_track(fig, time_array, min_envelope, max_envelope, row=1)

    # Track 2: Scene Detection Pass 1 (coarse boundaries)
    # Use coarse_boundaries if available, otherwise fall back to filtering scenes
    if data.coarse_boundaries:
        pass1_scenes = data.coarse_boundaries
    else:
        # Fallback: filter scenes with detection_pass=1 or unknown
        pass1_scenes = [s for s in data.scenes if s.detection_pass == 1]
        if not pass1_scenes and data.scenes:
            unknown_scenes = [s for s in data.scenes if s.detection_pass is None]
            if unknown_scenes:
                pass1_scenes = unknown_scenes
    _add_scene_track(fig, pass1_scenes, data.duration_seconds, row=2, pass_num=1)

    # Track 3: Scene Detection Pass 2 (all final scenes)
    # Show all final scenes - they represent the fine-grained result after splitting
    pass2_scenes = data.scenes  # All final scenes go in Pass 2 track
    _add_scene_track(fig, pass2_scenes, data.duration_seconds, row=3, pass_num=2)

    # Track 4: VAD Speech Segments
    _add_vad_track(fig, data.vad_segments, data.duration_seconds, row=4)

    # Track 5: Subtitles
    _add_subtitle_track(fig, data.subtitles, data.duration_seconds, row=5)

    # Configure layout
    _configure_layout(fig, data.duration_seconds, title)

    return fig


def _add_waveform_track(
    fig: go.Figure,
    time_array: np.ndarray,
    min_envelope: np.ndarray,
    max_envelope: np.ndarray,
    row: int
):
    """Add waveform envelope to track."""
    # Upper envelope
    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=max_envelope,
            mode='lines',
            name='Waveform',
            line=dict(color=COLORS["waveform"], width=0.5),
            fill=None,
            legendgroup="waveform",
            showlegend=True,
            hovertemplate="Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>"
        ),
        row=row, col=1
    )

    # Lower envelope (fills to upper)
    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=min_envelope,
            mode='lines',
            line=dict(color=COLORS["waveform"], width=0.5),
            fill='tonexty',
            fillcolor=COLORS["waveform_fill"],
            legendgroup="waveform",
            showlegend=False,
            hoverinfo='skip'
        ),
        row=row, col=1
    )


def _add_scene_track(
    fig: go.Figure,
    scenes: List[SceneInfo],
    duration: float,
    row: int,
    pass_num: int
):
    """Add scene detection blocks to track."""
    if not scenes:
        # Add empty placeholder
        fig.add_trace(
            go.Scatter(
                x=[0, duration],
                y=[0.5, 0.5],
                mode='lines',
                line=dict(color=COLORS["track_bg"], width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=1
        )
        return

    color_fill = COLORS["scene_pass1_fill"] if pass_num == 1 else COLORS["scene_pass2_fill"]
    color_line = COLORS["scene_pass1"] if pass_num == 1 else COLORS["scene_pass2"]
    legend_name = f"Scene Pass {pass_num}"

    for i, scene in enumerate(scenes):
        # Create filled rectangle for scene block
        x_coords = [
            scene.start_time_seconds,
            scene.end_time_seconds,
            scene.end_time_seconds,
            scene.start_time_seconds,
            scene.start_time_seconds
        ]
        y_coords = [0.1, 0.1, 0.9, 0.9, 0.1]

        # Generate compact scene label (no "SCENE" prefix)
        if pass_num == 1:
            # Pass 1: A, B, C, etc. (use sequential index i, not scene.scene_index)
            label = chr(65 + i) if i < 26 else str(i)
        else:
            # Pass 2: A.1, A.2, B.1, etc. (use sequential index i)
            parent_letter = chr(65 + (i // 10)) if i < 260 else str(i // 10)
            sub_index = (i % 10) + 1
            label = f"{parent_letter}.{sub_index}"

        # Full label for hover
        full_label = f"Scene {label}"

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                fill='toself',
                fillcolor=color_fill,
                line=dict(color=color_line, width=1),
                name=legend_name,
                legendgroup=f"pass{pass_num}",
                showlegend=(i == 0),
                hovertemplate=(
                    f"<b>{full_label}</b><br>"
                    f"Start: {_format_time_hms(scene.start_time_seconds)} ({scene.start_time_seconds:.1f}s)<br>"
                    f"End: {_format_time_hms(scene.end_time_seconds)} ({scene.end_time_seconds:.1f}s)<br>"
                    f"Duration: {scene.duration_seconds:.1f}s"
                    f"<extra></extra>"
                )
            ),
            row=row, col=1
        )

        # Add compact text label inside block (only if block is wide enough)
        block_duration = scene.end_time_seconds - scene.start_time_seconds
        # Calculate minimum width based on duration (scale with zoom)
        min_width_for_label = duration / 500  # Show labels when block is > 0.2% of total duration
        if block_duration > min_width_for_label:
            fig.add_annotation(
                x=(scene.start_time_seconds + scene.end_time_seconds) / 2,
                y=0.5,
                text=label,
                showarrow=False,
                font=dict(size=9, color=COLORS["text"]),
                xref=f"x{row}" if row > 1 else "x",
                yref=f"y{row}" if row > 1 else "y"
            )


def _add_vad_track(
    fig: go.Figure,
    vad_segments: List[VadSegment],
    duration: float,
    row: int
):
    """Add VAD speech segments as green blocks."""
    if not vad_segments:
        # Add empty placeholder
        fig.add_trace(
            go.Scatter(
                x=[0, duration],
                y=[0.5, 0.5],
                mode='lines',
                line=dict(color=COLORS["track_bg"], width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=1
        )
        return

    for i, seg in enumerate(vad_segments):
        x_coords = [seg.start_sec, seg.end_sec, seg.end_sec, seg.start_sec, seg.start_sec]
        y_coords = [0.1, 0.1, 0.9, 0.9, 0.1]

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                fill='toself',
                fillcolor=COLORS["vad_fill"],
                line=dict(color=COLORS["vad"], width=1),
                name="VAD Speech",
                legendgroup="vad",
                showlegend=(i == 0),
                hovertemplate=(
                    f"<b>Speech #{i+1}</b><br>"
                    f"Start: {_format_time_hms(seg.start_sec)} ({seg.start_sec:.1f}s)<br>"
                    f"End: {_format_time_hms(seg.end_sec)} ({seg.end_sec:.1f}s)<br>"
                    f"Duration: {seg.end_sec - seg.start_sec:.1f}s"
                    f"<extra></extra>"
                )
            ),
            row=row, col=1
        )

        # No labels inside VAD blocks - too many segments, just use hover


def _add_subtitle_track(
    fig: go.Figure,
    subtitles: List[Subtitle],
    duration: float,
    row: int
):
    """Add subtitle blocks with timestamp and text preview."""
    if not subtitles:
        # Add empty placeholder
        fig.add_trace(
            go.Scatter(
                x=[0, duration],
                y=[0.5, 0.5],
                mode='lines',
                line=dict(color=COLORS["track_bg"], width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=1
        )
        return

    for i, sub in enumerate(subtitles):
        x_coords = [sub.start_sec, sub.end_sec, sub.end_sec, sub.start_sec, sub.start_sec]
        y_coords = [0.1, 0.1, 0.9, 0.9, 0.1]

        # Format timestamp compactly
        timestamp = f"[{_format_time_hms(sub.start_sec)}]"

        # Truncate text for display
        text_preview = sub.text[:15] + "..." if len(sub.text) > 15 else sub.text

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                fill='toself',
                fillcolor=COLORS["subtitle_fill"],
                line=dict(color=COLORS["subtitle"], width=1),
                name="Subtitles",
                legendgroup="subtitles",
                showlegend=(i == 0),
                hovertemplate=(
                    f"<b>#{sub.index}</b><br>"
                    f"Time: {_format_time_hms(sub.start_sec)} - {_format_time_hms(sub.end_sec)}<br>"
                    f"Text: {sub.text[:100]}"
                    f"<extra></extra>"
                )
            ),
            row=row, col=1
        )

        # Add compact label inside block (only if wide enough)
        block_duration = sub.end_sec - sub.start_sec
        min_width_for_label = duration / 300
        if block_duration > min_width_for_label:
            fig.add_annotation(
                x=(sub.start_sec + sub.end_sec) / 2,
                y=0.5,
                text=f"{timestamp}<br>{text_preview}",
                showarrow=False,
                font=dict(size=7, color=COLORS["text"]),
                xref=f"x{row}" if row > 1 else "x",
                yref=f"y{row}" if row > 1 else "y",
                align="center"
            )


def _configure_layout(fig: go.Figure, duration: float, title: str):
    """Configure multi-track layout with shared x-axis and timeline."""

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=COLORS["text"]),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        height=750,  # Taller to accommodate tracks

        # Legend at bottom (horizontal)
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=COLORS["grid"],
            borderwidth=1
        ),

        hovermode="x unified",
        dragmode="pan",  # Default to pan mode for easier horizontal scrolling
    )

    # Generate tick values and labels in hh:mm:ss format
    # Create ticks every appropriate interval based on duration
    if duration <= 60:
        tick_interval = 10  # Every 10 seconds
    elif duration <= 600:
        tick_interval = 60  # Every minute
    elif duration <= 3600:
        tick_interval = 300  # Every 5 minutes
    else:
        tick_interval = 600  # Every 10 minutes

    tick_vals = list(range(0, int(duration) + 1, tick_interval))
    tick_text = [_format_time_hms(t) for t in tick_vals]

    # Configure x-axis for top row (row 1) - show timeline at TOP
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=0,
        tickfont=dict(size=10),
        side="top",  # Timeline at top
        showticklabels=True,
        row=1, col=1
    )

    # Configure x-axis (bottom) with range slider
    # Vivid gray colors for better visibility
    fig.update_xaxes(
        title_text="Time",
        showgrid=True,
        gridcolor=COLORS["grid"],
        range=[0, min(duration, 300)],  # Default view: first 5 minutes
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=0,
        tickfont=dict(size=10),
        rangeslider=dict(
            visible=True,
            thickness=0.12,  # Thicker for easier interaction
            bgcolor="#4a4a6a",  # Vivid gray background
            bordercolor="#8888aa",  # Lighter border
            borderwidth=2,
            yaxis=dict(rangemode="auto")
        ),
        row=5, col=1
    )

    # Set the main x-axis range (affects the rangeslider behavior)
    fig.update_layout(
        xaxis5=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.12,
                bgcolor="#4a4a6a",
                bordercolor="#8888aa",
                borderwidth=2
            )
        )
    )

    # Configure middle x-axes (no tick labels, just grid)
    for i in range(2, 5):
        fig.update_xaxes(
            showgrid=True,
            gridcolor=COLORS["grid"],
            showticklabels=False,
            row=i, col=1
        )

    # Configure y-axes for each track
    y_axis_config = dict(
        showgrid=False,
        showticklabels=False,
        range=[0, 1],
        fixedrange=True
    )

    # Track 1: Waveform (different range)
    fig.update_yaxes(
        title_text="Waveform",
        title_font=dict(size=10),
        title_standoff=5,
        showgrid=False,
        range=[-1, 1],
        fixedrange=True,
        row=1, col=1
    )

    # Track 2-5: Block tracks
    track_labels = ["Pass 1", "Pass 2", "VAD", "Subs"]
    for i, label in enumerate(track_labels, start=2):
        fig.update_yaxes(
            title_text=label,
            title_font=dict(size=10),
            title_standoff=5,
            **y_axis_config,
            row=i, col=1
        )

    # Modebar configuration - add useful buttons
    fig.update_layout(
        modebar=dict(
            orientation="h",
            bgcolor=COLORS["background"],
            color=COLORS["text"],
            activecolor=COLORS["waveform"]
        )
    )


def render_to_html(
    data: VisualizationData,
    time_array: np.ndarray,
    min_envelope: np.ndarray,
    max_envelope: np.ndarray,
    output_path: Path,
    title: str = "Audio Analysis Timeline - WhisperJAV"
) -> Path:
    """
    Render visualization to standalone HTML file.

    Args:
        data: VisualizationData container
        time_array: Downsampled time points
        min_envelope: Waveform min envelope
        max_envelope: Waveform max envelope
        output_path: Output HTML file path
        title: Chart title

    Returns:
        Path to generated HTML file
    """
    fig = create_visualization(data, time_array, min_envelope, max_envelope, title)

    # Generate standalone HTML
    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d'],
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'whisperjav_visualization',
                'height': 800,
                'width': 1600,
                'scale': 2
            }
        }
    )

    print(f"Visualization saved to: {output_path}")
    return output_path
