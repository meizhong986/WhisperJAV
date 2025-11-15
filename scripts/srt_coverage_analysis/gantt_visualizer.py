"""
Timeline Visualizer Module
===========================

Generate horizontal timeline visualizations showing reference segments
and test segment coverage for easy visual inspection.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pathlib import Path
from typing import List, Optional, Dict, Any
import json


def extract_scene_boundaries(metadata: Dict[str, Any]) -> List[Dict]:
    """
    Extract scene boundaries from WhisperJAV metadata.

    Args:
        metadata: Metadata dictionary (may be pipeline metadata or analysis metadata)

    Returns:
        List of scene boundary dicts with index, start, end times
    """
    # Try to get scenes from metadata
    scenes_data = metadata.get('scenes_detected', [])

    if not scenes_data:
        return []

    boundaries = []
    for scene in scenes_data:
        boundaries.append({
            'index': scene.get('scene_index', -1),
            'start': scene.get('start_time_seconds', 0.0),
            'end': scene.get('end_time_seconds', 0.0),
        })

    return boundaries


def generate_timeline_chart(
    coverage_results: List,
    test_segments: List,
    metadata: Optional[Dict[str, Any]],
    output_path: str,
    figure_width: int = 24,
    figure_height: int = 6
) -> str:
    """
    Generate horizontal timeline showing Reference and Test segments.

    Args:
        coverage_results: List of CoverageResult objects
        test_segments: List of all test Segment objects
        metadata: Metadata dict containing scene boundaries (optional)
        output_path: Path to save PNG file
        figure_width: Figure width in inches (default 24)
        figure_height: Figure height in inches (default 6)

    Returns:
        Path to generated PNG file

    Timeline Layout:
        Row 0 (bottom): Test segments (blue bars)
        Row 1 (top): Reference segments (colored by coverage)
        Scene boundaries: Vertical dashed lines
        Time axis: Top and bottom
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not coverage_results:
        print("Warning: No segments to plot")
        return None

    # Extract scene boundaries if metadata provided
    scene_boundaries = []
    if metadata:
        scene_boundaries = extract_scene_boundaries(metadata)

    # Calculate timeline extent
    all_times = []
    for result in coverage_results:
        all_times.extend([result.ref_segment.start, result.ref_segment.end])
    for test_seg in test_segments:
        all_times.extend([test_seg.start, test_seg.end])

    if not all_times:
        print("Warning: No timeline data")
        return None

    time_min = max(0, min(all_times) - 10)
    time_max = max(all_times) + 10
    time_span = time_max - time_min

    # Create figure
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    # Color mapping for coverage status
    status_colors = {
        'COVERED': '#28a745',  # Green
        'PARTIAL': '#ffc107',  # Yellow/Orange
        'MISSING': '#dc3545',  # Red
    }

    # Row positions
    REF_ROW = 1.0  # Reference row (top)
    TEST_ROW = 0.0  # Test row (bottom)
    BAR_HEIGHT = 0.6

    # ===== Draw Reference Segments (Row 1) =====
    for result in coverage_results:
        ref_seg = result.ref_segment
        color = status_colors.get(result.status, '#6c757d')

        # Draw reference segment bar
        ref_rect = Rectangle(
            (ref_seg.start, REF_ROW - BAR_HEIGHT/2),
            ref_seg.duration,
            BAR_HEIGHT,
            linewidth=0.5,
            edgecolor='black',
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(ref_rect)

    # ===== Draw Test Segments (Row 0) =====
    for test_seg in test_segments:
        test_rect = Rectangle(
            (test_seg.start, TEST_ROW - BAR_HEIGHT/2),
            test_seg.duration,
            BAR_HEIGHT,
            linewidth=0.5,
            edgecolor='black',
            facecolor='#007bff',  # Blue
            alpha=0.7
        )
        ax.add_patch(test_rect)

    # ===== Draw Scene Boundaries =====
    if scene_boundaries:
        for scene in scene_boundaries:
            scene_start = scene['start']
            scene_index = scene['index']

            # Vertical dashed line
            ax.axvline(
                x=scene_start,
                ymin=0,
                ymax=1,
                color='gray',
                linestyle='--',
                linewidth=1.5,
                alpha=0.6,
                zorder=5
            )

            # Scene label at top
            ax.text(
                scene_start,
                2.2,
                f"S{scene_index}",
                ha='left',
                va='bottom',
                fontsize=8,
                color='gray',
                fontweight='bold',
                rotation=0
            )

    # ===== Axis Configuration =====
    ax.set_xlim(time_min, time_max)
    ax.set_ylim(-0.8, 2.5)

    # X-axis (time) - show at top and bottom
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(axis='x', which='both', labeltop=True, labelbottom=True, top=True, bottom=True)

    # Calculate appropriate time tick intervals
    if time_span <= 300:  # 5 minutes
        tick_interval = 30
    elif time_span <= 600:  # 10 minutes
        tick_interval = 60
    elif time_span <= 1800:  # 30 minutes
        tick_interval = 180
    elif time_span <= 3600:  # 1 hour
        tick_interval = 300
    else:  # > 1 hour
        tick_interval = 600

    ax.set_xticks(range(int(time_min), int(time_max) + 1, tick_interval))

    # Y-axis - row labels
    ax.set_yticks([TEST_ROW, REF_ROW])
    ax.set_yticklabels(['Test SRT', 'Reference SRT'], fontsize=11, fontweight='bold')

    # Title
    ax.set_title(
        'SRT Coverage Analysis - Horizontal Timeline\n'
        'Blue bars: Test segments | Green overlays: Test segments',
        fontsize=14,
        fontweight='bold',
        pad=30
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=status_colors['COVERED'], alpha=0.7,
                      edgecolor='black', linewidth=0.5, label='Covered (≥60%)'),
        mpatches.Patch(facecolor=status_colors['PARTIAL'], alpha=0.7,
                      edgecolor='black', linewidth=0.5, label='Partial (<60%)'),
        mpatches.Patch(facecolor=status_colors['MISSING'], alpha=0.7,
                      edgecolor='black', linewidth=0.5, label='Missing (0%)'),
        mpatches.Patch(facecolor='#007bff', alpha=0.7,
                      edgecolor='black', linewidth=0.5, label='Test segment'),
    ]

    if scene_boundaries:
        legend_elements.append(
            Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5,
                   alpha=0.6, label='Scene boundary')
        )

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

    # Grid for readability (vertical lines only)
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

    # Clean up borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Timeline chart saved: {output_file.name}")
    print(f"    Timeline span: {time_min:.1f}s - {time_max:.1f}s ({time_span:.1f}s total)")
    print(f"    Reference segments: {len(coverage_results)}")
    print(f"    Test segments: {len(test_segments)}")
    if scene_boundaries:
        print(f"    Scene boundaries: {len(scene_boundaries)}")

    return str(output_file)


def generate_interactive_gantt(
    coverage_results: List,
    test_segments: List,
    metadata: Optional[Dict[str, Any]],
    output_path: str
) -> Optional[str]:
    """
    Generate interactive timeline using Plotly.

    Args:
        coverage_results: List of CoverageResult objects
        test_segments: List of all test segments
        metadata: Metadata dict containing scene boundaries (optional)
        output_path: Path to save HTML file

    Returns:
        Path to generated HTML file, or None if Plotly unavailable
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Warning: Plotly not installed. Skipping interactive chart.")
        print("Install with: pip install plotly")
        return None

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not coverage_results:
        print("Warning: No segments to plot")
        return None

    # Extract scene boundaries
    scene_boundaries = []
    if metadata:
        scene_boundaries = extract_scene_boundaries(metadata)

    # Color mapping
    status_colors = {
        'COVERED': '#28a745',
        'PARTIAL': '#ffc107',
        'MISSING': '#dc3545',
    }

    # Row positions
    REF_ROW = 1.0
    TEST_ROW = 0.0
    BAR_HEIGHT = 0.6

    # Create figure
    fig = go.Figure()

    # Add reference segments
    for result in coverage_results:
        ref_seg = result.ref_segment
        color = status_colors.get(result.status, 'gray')

        # Reference segment bar
        fig.add_trace(go.Scatter(
            x=[ref_seg.start, ref_seg.end, ref_seg.end, ref_seg.start, ref_seg.start],
            y=[REF_ROW - BAR_HEIGHT/2, REF_ROW - BAR_HEIGHT/2,
               REF_ROW + BAR_HEIGHT/2, REF_ROW + BAR_HEIGHT/2, REF_ROW - BAR_HEIGHT/2],
            fill='toself',
            fillcolor=color,
            opacity=0.7,
            line=dict(color='black', width=0.5),
            mode='lines',
            name=f'Ref {ref_seg.index}',
            hovertemplate=(
                f'<b>Reference Segment {ref_seg.index}</b><br>'
                f'Time: {ref_seg.start:.2f}s - {ref_seg.end:.2f}s<br>'
                f'Duration: {ref_seg.duration:.2f}s<br>'
                f'Coverage: {result.coverage_percent:.1f}%<br>'
                f'Status: {result.status}<br>'
                f'Text: {ref_seg.text[:60]}...<br>'
                '<extra></extra>'
            ),
            showlegend=False
        ))

    # Add test segments
    for test_seg in test_segments:
        fig.add_trace(go.Scatter(
            x=[test_seg.start, test_seg.end, test_seg.end, test_seg.start, test_seg.start],
            y=[TEST_ROW - BAR_HEIGHT/2, TEST_ROW - BAR_HEIGHT/2,
               TEST_ROW + BAR_HEIGHT/2, TEST_ROW + BAR_HEIGHT/2, TEST_ROW - BAR_HEIGHT/2],
            fill='toself',
            fillcolor='#007bff',
            opacity=0.7,
            line=dict(color='black', width=0.5),
            mode='lines',
            name=f'Test {test_seg.index}',
            hovertemplate=(
                f'<b>Test Segment {test_seg.index}</b><br>'
                f'Time: {test_seg.start:.2f}s - {test_seg.end:.2f}s<br>'
                f'Duration: {test_seg.duration:.2f}s<br>'
                f'Text: {test_seg.text[:60]}...<br>'
                '<extra></extra>'
            ),
            showlegend=False
        ))

    # Add scene boundaries
    if scene_boundaries:
        for scene in scene_boundaries:
            fig.add_vline(
                x=scene['start'],
                line_dash="dash",
                line_color="gray",
                line_width=1.5,
                opacity=0.6,
                annotation_text=f"Scene {scene['index']}",
                annotation_position="top",
                annotation_font_size=10
            )

    # Update layout
    fig.update_layout(
        title='SRT Coverage Analysis - Interactive Timeline',
        xaxis=dict(
            title='Time (seconds)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title='',
            tickmode='array',
            tickvals=[TEST_ROW, REF_ROW],
            ticktext=['Test SRT', 'Reference SRT'],
            range=[-0.8, 2.5],
        ),
        hovermode='closest',
        height=600,
        showlegend=False,
        plot_bgcolor='white',
    )

    # Save as HTML
    fig.write_html(str(output_file))

    print(f"  Interactive timeline saved: {output_file.name}")

    return str(output_file)


def generate_gantt_chart(
    coverage_results: List,
    test_segments: List,
    output_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
    generate_static: bool = True,
    generate_interactive: bool = True
) -> dict:
    """
    Generate both static and interactive timeline charts.

    Args:
        coverage_results: List of CoverageResult objects
        test_segments: List of all test segments
        output_dir: Directory to save chart files
        metadata: Metadata dict containing scene boundaries (optional)
        generate_static: Whether to generate static PNG
        generate_interactive: Whether to generate interactive HTML

    Returns:
        Dictionary with paths to generated files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        'static_chart': None,
        'interactive_chart': None,
    }

    if generate_static:
        static_path = output_path / "timeline.png"
        results['static_chart'] = generate_timeline_chart(
            coverage_results,
            test_segments,
            metadata,
            str(static_path)
        )

    if generate_interactive:
        interactive_path = output_path / "timeline_interactive.html"
        results['interactive_chart'] = generate_interactive_gantt(
            coverage_results,
            test_segments,
            metadata,
            str(interactive_path)
        )

    return results


if __name__ == "__main__":
    # Test module
    import sys
    from .srt_parser import parse_srt_file
    from .coverage_calculator import analyze_all_segments

    if len(sys.argv) < 4:
        print("Usage: python gantt_visualizer.py <reference.srt> <test.srt> <output_dir>")
        sys.exit(1)

    ref_path = sys.argv[1]
    test_path = sys.argv[2]
    output_dir = sys.argv[3]

    print(f"Loading SRT files...")
    ref_segments = parse_srt_file(ref_path)
    test_segments = parse_srt_file(test_path)

    print(f"Analyzing coverage...")
    results = analyze_all_segments(ref_segments, test_segments)

    print(f"Generating charts...")
    chart_files = generate_gantt_chart(results, test_segments, output_dir)

    print(f"\nGenerated charts:")
    if chart_files['static_chart']:
        print(f"  Static: {chart_files['static_chart']}")
    if chart_files['interactive_chart']:
        print(f"  Interactive: {chart_files['interactive_chart']}")
