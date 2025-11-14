"""
Gantt Visualizer Module
========================

Generate Gantt chart timeline visualizations showing reference segments
and test segment coverage.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import List, Optional
import json


def generate_static_gantt(
    coverage_results: List,
    test_segments: List,
    output_path: str,
    max_segments: int = 50
) -> str:
    """
    Generate static PNG Gantt chart using matplotlib.

    Args:
        coverage_results: List of CoverageResult objects
        test_segments: List of all test segments
        output_path: Path to save PNG file
        max_segments: Maximum number of segments to display (default 50)

    Returns:
        Path to generated PNG file

    Chart Layout:
        - X-axis: Timeline in seconds
        - Y-axis: Each reference segment as a row
        - Reference segments: Horizontal bars
        - Test segment overlays: Semi-transparent green bars
        - Color coding by coverage status:
            * Green outline: COVERED (≥60%)
            * Yellow outline: PARTIAL (<60%)
            * Red outline: MISSING (0%)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Limit segments to avoid overcrowding
    results_to_plot = coverage_results[:max_segments]

    if not results_to_plot:
        print("Warning: No segments to plot")
        return None

    # Calculate figure size based on number of segments
    num_segments = len(results_to_plot)
    fig_height = max(8, num_segments * 0.3)  # At least 8 inches, scale with segments
    fig_width = 16

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Color mapping for status
    status_colors = {
        'COVERED': '#28a745',  # Green
        'PARTIAL': '#ffc107',  # Yellow/Orange
        'MISSING': '#dc3545',  # Red
    }

    # Plot each reference segment
    for i, result in enumerate(results_to_plot):
        ref_seg = result.ref_segment
        y_pos = num_segments - i - 1  # Reverse order (latest at top)

        # Determine color based on status
        edge_color = status_colors.get(result.status, '#6c757d')

        # Draw reference segment bar
        ref_rect = Rectangle(
            (ref_seg.start, y_pos - 0.4),
            ref_seg.duration,
            0.8,
            linewidth=2,
            edgecolor=edge_color,
            facecolor='lightblue',
            alpha=0.5,
            label=f'Ref {ref_seg.index}'
        )
        ax.add_patch(ref_rect)

        # Draw overlapping test segments
        for overlap_info in result.overlapping_segments:
            test_seg = overlap_info.segment
            overlap_start = overlap_info.overlap_start
            overlap_duration = overlap_info.overlap_duration

            test_rect = Rectangle(
                (overlap_start, y_pos - 0.3),
                overlap_duration,
                0.6,
                linewidth=1,
                edgecolor='darkgreen',
                facecolor='green',
                alpha=0.6
            )
            ax.add_patch(test_rect)

        # Add segment index label
        ax.text(
            ref_seg.start - 0.5,
            y_pos,
            f"{ref_seg.index}",
            ha='right',
            va='center',
            fontsize=8,
            color='black'
        )

        # Add coverage percentage label
        coverage_text = f"{result.coverage_percent:.0f}%"
        ax.text(
            ref_seg.end + 0.5,
            y_pos,
            coverage_text,
            ha='left',
            va='center',
            fontsize=8,
            color=edge_color,
            fontweight='bold'
        )

    # Set axis limits
    all_times = [r.ref_segment.start for r in results_to_plot] + \
                [r.ref_segment.end for r in results_to_plot]
    time_min = min(all_times) - 5
    time_max = max(all_times) + 5

    ax.set_xlim(time_min, time_max)
    ax.set_ylim(-1, num_segments)

    # Labels and title
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reference Segment Index', fontsize=12, fontweight='bold')
    ax.set_title(
        'SRT Coverage Analysis - Gantt Chart\n'
        'Blue bars: Reference segments | Green overlays: Test segments',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Add legend for status colors
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor=status_colors['COVERED'],
                      linewidth=2, label='COVERED (≥60%)'),
        mpatches.Patch(facecolor='none', edgecolor=status_colors['PARTIAL'],
                      linewidth=2, label='PARTIAL (<60%)'),
        mpatches.Patch(facecolor='none', edgecolor=status_colors['MISSING'],
                      linewidth=2, label='MISSING (0%)'),
        mpatches.Patch(facecolor='lightblue', alpha=0.5, label='Reference segment'),
        mpatches.Patch(facecolor='green', alpha=0.6, label='Test segment overlap'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Grid for readability
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    return str(output_file)


def generate_interactive_gantt(
    coverage_results: List,
    test_segments: List,
    output_path: str
) -> Optional[str]:
    """
    Generate interactive Gantt chart using Plotly.

    Args:
        coverage_results: List of CoverageResult objects
        test_segments: List of all test segments
        output_path: Path to save HTML file

    Returns:
        Path to generated HTML file, or None if Plotly unavailable
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Warning: Plotly not installed. Skipping interactive chart.")
        print("Install with: pip install plotly")
        return None

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not coverage_results:
        print("Warning: No segments to plot")
        return None

    # Color mapping
    status_colors = {
        'COVERED': 'green',
        'PARTIAL': 'orange',
        'MISSING': 'red',
    }

    # Create figure
    fig = go.Figure()

    # Add reference segments
    for i, result in enumerate(coverage_results):
        ref_seg = result.ref_segment
        color = status_colors.get(result.status, 'gray')

        # Reference segment bar
        fig.add_trace(go.Scatter(
            x=[ref_seg.start, ref_seg.end, ref_seg.end, ref_seg.start, ref_seg.start],
            y=[i, i, i + 0.8, i + 0.8, i],
            fill='toself',
            fillcolor='lightblue',
            line=dict(color=color, width=2),
            mode='lines',
            name=f'Ref {ref_seg.index}',
            hovertemplate=(
                f'<b>Ref Segment {ref_seg.index}</b><br>'
                f'Time: {ref_seg.start:.2f}s - {ref_seg.end:.2f}s<br>'
                f'Duration: {ref_seg.duration:.2f}s<br>'
                f'Coverage: {result.coverage_percent:.1f}%<br>'
                f'Status: {result.status}<br>'
                f'Text: {ref_seg.text[:50]}...<br>'
                '<extra></extra>'
            ),
            showlegend=False
        ))

        # Overlapping test segments
        for overlap_info in result.overlapping_segments:
            overlap_start = overlap_info.overlap_start
            overlap_end = overlap_info.overlap_end

            fig.add_trace(go.Scatter(
                x=[overlap_start, overlap_end, overlap_end, overlap_start, overlap_start],
                y=[i + 0.1, i + 0.1, i + 0.7, i + 0.7, i + 0.1],
                fill='toself',
                fillcolor='green',
                opacity=0.6,
                line=dict(color='darkgreen', width=1),
                mode='lines',
                name=f'Test overlap',
                hovertemplate=(
                    f'<b>Test Overlap</b><br>'
                    f'Overlap: {overlap_start:.2f}s - {overlap_end:.2f}s<br>'
                    f'Duration: {overlap_info.overlap_duration:.2f}s<br>'
                    '<extra></extra>'
                ),
                showlegend=False
            ))

    # Update layout
    fig.update_layout(
        title='SRT Coverage Analysis - Interactive Gantt Chart',
        xaxis=dict(
            title='Time (seconds)',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title='Segment Index',
            showgrid=False,
            tickmode='linear',
            tick0=0,
            dtick=1,
        ),
        hovermode='closest',
        height=max(600, len(coverage_results) * 20),
        showlegend=False,
        plot_bgcolor='white',
    )

    # Save as HTML
    fig.write_html(str(output_file))

    return str(output_file)


def generate_gantt_chart(
    coverage_results: List,
    test_segments: List,
    output_dir: str,
    generate_static: bool = True,
    generate_interactive: bool = True,
    max_segments_static: int = 50
) -> dict:
    """
    Generate both static and interactive Gantt charts.

    Args:
        coverage_results: List of CoverageResult objects
        test_segments: List of all test segments
        output_dir: Directory to save chart files
        generate_static: Whether to generate static PNG
        generate_interactive: Whether to generate interactive HTML
        max_segments_static: Max segments for static chart

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
        results['static_chart'] = generate_static_gantt(
            coverage_results,
            test_segments,
            str(static_path),
            max_segments=max_segments_static
        )

    if generate_interactive:
        interactive_path = output_path / "timeline_interactive.html"
        results['interactive_chart'] = generate_interactive_gantt(
            coverage_results,
            test_segments,
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
