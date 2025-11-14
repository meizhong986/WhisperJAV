"""
Report Generator Module
=======================

Generate comprehensive HTML and JSON reports for SRT coverage analysis.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


def generate_json_report(
    reference_segments: List,
    test_segments: List,
    coverage_results: List,
    trace_results: List,
    summary_stats: Dict[str, Any],
    root_cause_stats: Dict[str, Any],
    extraction_stats: Dict[str, Any],
    metadata: Dict[str, Any],
    output_path: str
) -> str:
    """
    Generate machine-readable JSON report.

    Args:
        reference_segments: List of reference Segment objects
        test_segments: List of test Segment objects
        coverage_results: List of CoverageResult objects
        trace_results: List of TraceResult objects
        summary_stats: Summary statistics dictionary
        root_cause_stats: Root cause analysis dictionary
        extraction_stats: Media extraction statistics
        metadata: Analysis metadata (file paths, timestamps, etc.)
        output_path: Path to save JSON file

    Returns:
        Path to generated JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build trace results mapping
    trace_map = {tr.ref_segment.index: tr for tr in trace_results}

    # Build segments list with full details
    segments_data = []
    for result in coverage_results:
        seg = result.ref_segment
        seg_data = {
            'index': seg.index,
            'start': seg.start,
            'end': seg.end,
            'duration': seg.duration,
            'text': seg.text,
            'coverage_percent': round(result.coverage_percent, 2),
            'status': result.status,
            'total_overlap_seconds': round(result.total_overlap_seconds, 2),
            'overlapping_test_segments': len(result.overlapping_segments),
        }

        # Add trace information if available
        if seg.index in trace_map:
            trace = trace_map[seg.index]
            seg_data['root_cause'] = trace.root_cause
            seg_data['trace_details'] = trace.details
            seg_data['containing_scenes'] = [
                {
                    'scene_index': scene.scene_index,
                    'start_time': scene.start_time,
                    'end_time': scene.end_time,
                    'transcribed': scene.transcribed
                }
                for scene in trace.containing_scenes
            ]

        # Add media chunk path if extracted
        if result.needs_review and 'missing_files' in extraction_stats:
            # Try to find corresponding chunk file
            chunk_files = extraction_stats.get('missing_files', []) + \
                         extraction_stats.get('partial_files', [])
            for chunk_file in chunk_files:
                if f"seg_{seg.index:04d}_" in chunk_file:
                    seg_data['media_chunk'] = chunk_file
                    break

        segments_data.append(seg_data)

    # Build complete report structure
    report = {
        'metadata': metadata,
        'summary': summary_stats,
        'root_cause_analysis': root_cause_stats,
        'extraction_stats': extraction_stats,
        'segments': segments_data,
    }

    # Write JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return str(output_file)


def generate_html_report(
    reference_segments: List,
    test_segments: List,
    coverage_results: List,
    trace_results: List,
    summary_stats: Dict[str, Any],
    root_cause_stats: Dict[str, Any],
    extraction_stats: Dict[str, Any],
    metadata: Dict[str, Any],
    timeline_png_path: Optional[str],
    timeline_html_path: Optional[str],
    output_path: str
) -> str:
    """
    Generate comprehensive HTML report with embedded visualizations.

    Args:
        reference_segments: List of reference Segment objects
        test_segments: List of test Segment objects
        coverage_results: List of CoverageResult objects
        trace_results: List of TraceResult objects
        summary_stats: Summary statistics dictionary
        root_cause_stats: Root cause analysis dictionary
        extraction_stats: Media extraction statistics
        metadata: Analysis metadata
        timeline_png_path: Path to static timeline PNG (relative to report)
        timeline_html_path: Path to interactive timeline HTML (relative to report)
        output_path: Path to save HTML file

    Returns:
        Path to generated HTML file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build trace map
    trace_map = {tr.ref_segment.index: tr for tr in trace_results}

    # Filter segments that need review (missing + partial)
    needs_review = [r for r in coverage_results if r.needs_review]

    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SRT Coverage Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            padding: 15px;
            border-radius: 6px;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .stat-card.covered {{
            border-left-color: #28a745;
        }}
        .stat-card.partial {{
            border-left-color: #ffc107;
        }}
        .stat-card.missing {{
            border-left-color: #dc3545;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .status-covered {{
            background: #d4edda;
            color: #155724;
        }}
        .status-partial {{
            background: #fff3cd;
            color: #856404;
        }}
        .status-missing {{
            background: #f8d7da;
            color: #721c24;
        }}
        .timeline-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .timeline-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .button {{
            display: inline-block;
            padding: 10px 20px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 5px;
            font-weight: 600;
            transition: background 0.3s;
        }}
        .button:hover {{
            background: #5568d3;
        }}
        .metadata-table {{
            display: grid;
            grid-template-columns: 200px 1fr;
            gap: 10px;
            margin: 15px 0;
        }}
        .metadata-label {{
            font-weight: 600;
            color: #666;
        }}
        .metadata-value {{
            color: #333;
        }}
        .root-cause {{
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <header>
        <h1>SRT Coverage Analysis Report</h1>
        <div class="subtitle">Generated: {metadata.get('analysis_timestamp', 'N/A')}</div>
    </header>

    <!-- Metadata Section -->
    <section class="section">
        <h2>Analysis Configuration</h2>
        <div class="metadata-table">
            <div class="metadata-label">Reference SRT:</div>
            <div class="metadata-value">{metadata.get('reference_srt', 'N/A')}</div>

            <div class="metadata-label">Test SRT:</div>
            <div class="metadata-value">{metadata.get('test_srt', 'N/A')}</div>

            <div class="metadata-label">Media File:</div>
            <div class="metadata-value">{metadata.get('media_file', 'N/A')}</div>

            <div class="metadata-label">Coverage Threshold:</div>
            <div class="metadata-value">{metadata.get('coverage_threshold', 0.6) * 100:.0f}%</div>

            <div class="metadata-label">Padding:</div>
            <div class="metadata-value">{metadata.get('padding_seconds', 1.0)}s</div>
        </div>
    </section>

    <!-- SRT File Statistics -->
    {_generate_srt_stats_section(metadata) if 'reference_stats' in metadata and 'test_stats' in metadata else ''}

    <!-- Summary Statistics -->
    <section class="section">
        <h2>Coverage Analysis Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Segments</div>
                <div class="stat-value">{summary_stats.get('total_segments', 0)}</div>
            </div>
            <div class="stat-card covered">
                <div class="stat-label">Covered</div>
                <div class="stat-value">{summary_stats.get('covered_segments', 0)} ({summary_stats.get('covered_percent', 0):.1f}%)</div>
            </div>
            <div class="stat-card partial">
                <div class="stat-label">Partial</div>
                <div class="stat-value">{summary_stats.get('partial_segments', 0)} ({summary_stats.get('partial_percent', 0):.1f}%)</div>
            </div>
            <div class="stat-card missing">
                <div class="stat-label">Missing</div>
                <div class="stat-value">{summary_stats.get('missing_segments', 0)} ({summary_stats.get('missing_percent', 0):.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Coverage</div>
                <div class="stat-value">{summary_stats.get('average_coverage_percent', 0):.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Duration</div>
                <div class="stat-value">{summary_stats.get('total_reference_duration', 0):.1f}s</div>
            </div>
        </div>
    </section>

    <!-- Root Cause Analysis -->
    {_generate_root_cause_section(root_cause_stats) if root_cause_stats else ''}

    <!-- Timeline Visualization -->
    <section class="section">
        <h2>Timeline Visualization</h2>
        <div style="text-align: center; margin-bottom: 20px;">
            {f'<a href="{timeline_html_path}" class="button" target="_blank">📊 View Interactive Timeline</a>' if timeline_html_path else ''}
        </div>
        <div class="timeline-container">
            {f'<img src="{timeline_png_path}" alt="Coverage Timeline">' if timeline_png_path else '<p>Timeline visualization not available</p>'}
        </div>
    </section>

    <!-- Missing and Partial Segments -->
    <section class="section">
        <h2>Segments Needing Review ({len(needs_review)})</h2>
        <table>
            <thead>
                <tr>
                    <th>Index</th>
                    <th>Time</th>
                    <th>Text</th>
                    <th>Status</th>
                    <th>Coverage</th>
                    <th>Root Cause</th>
                    <th>Media Chunk</th>
                </tr>
            </thead>
            <tbody>
                {_generate_segments_table_rows(needs_review, trace_map, extraction_stats)}
            </tbody>
        </table>
    </section>

    <!-- Extraction Statistics -->
    <section class="section">
        <h2>Media Extraction Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Extracted</div>
                <div class="stat-value">{extraction_stats.get('successful', 0)}/{extraction_stats.get('total', 0)}</div>
            </div>
            <div class="stat-card missing">
                <div class="stat-label">Missing Chunks</div>
                <div class="stat-value">{len(extraction_stats.get('missing_files', []))}</div>
            </div>
            <div class="stat-card partial">
                <div class="stat-label">Partial Chunks</div>
                <div class="stat-value">{len(extraction_stats.get('partial_files', []))}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Failed Extractions</div>
                <div class="stat-value">{extraction_stats.get('failed', 0)}</div>
            </div>
        </div>
    </section>

    <footer style="text-align: center; margin-top: 40px; padding: 20px; color: #666;">
        <p>Generated by WhisperJAV SRT Coverage Analysis Tool</p>
    </footer>
</body>
</html>
    """

    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    return str(output_file)


def _generate_srt_stats_section(metadata: Dict[str, Any]) -> str:
    """Generate HTML for SRT file statistics section."""
    ref_stats = metadata.get('reference_stats', {})
    test_stats = metadata.get('test_stats', {})

    return f"""
    <section class="section">
        <h2>SRT File Statistics</h2>

        <h3 style="color: #667eea; margin-top: 20px; margin-bottom: 15px;">Reference SRT</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Lines</div>
                <div class="stat-value">{ref_stats.get('total_lines', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Duration</div>
                <div class="stat-value">{ref_stats.get('total_duration_formatted', '00:00:00.000')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Duration</div>
                <div class="stat-value">{ref_stats.get('average_duration', 0):.2f}s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Timeline Span</div>
                <div class="stat-value">{ref_stats.get('timeline_start', 0):.1f}s - {ref_stats.get('timeline_end', 0):.1f}s</div>
            </div>
        </div>

        <h3 style="color: #667eea; margin-top: 30px; margin-bottom: 15px;">Test SRT</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Lines</div>
                <div class="stat-value">{test_stats.get('total_lines', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Duration</div>
                <div class="stat-value">{test_stats.get('total_duration_formatted', '00:00:00.000')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Duration</div>
                <div class="stat-value">{test_stats.get('average_duration', 0):.2f}s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Timeline Span</div>
                <div class="stat-value">{test_stats.get('timeline_start', 0):.1f}s - {test_stats.get('timeline_end', 0):.1f}s</div>
            </div>
        </div>
    </section>
    """


def _generate_root_cause_section(root_cause_stats: Dict[str, Any]) -> str:
    """Generate HTML for root cause analysis section."""
    return f"""
    <section class="section">
        <h2>Root Cause Analysis</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Not in Scene</div>
                <div class="stat-value">{root_cause_stats.get('not_in_scene', 0)} ({root_cause_stats.get('not_in_scene_percent', 0):.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Scene Not Transcribed</div>
                <div class="stat-value">{root_cause_stats.get('scene_not_transcribed', 0)} ({root_cause_stats.get('scene_not_transcribed_percent', 0):.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Filtered/Failed</div>
                <div class="stat-value">{root_cause_stats.get('filtered_or_failed', 0)} ({root_cause_stats.get('filtered_or_failed_percent', 0):.1f}%)</div>
            </div>
        </div>
    </section>
    """


def _generate_segments_table_rows(needs_review: List, trace_map: Dict, extraction_stats: Dict) -> str:
    """Generate HTML table rows for segments needing review."""
    rows = []

    for result in needs_review:
        seg = result.ref_segment
        status_class = f"status-{result.status.lower()}"

        # Get trace information
        trace = trace_map.get(seg.index)
        root_cause = trace.root_cause if trace else 'N/A'

        # Find media chunk
        chunk_files = extraction_stats.get('missing_files', []) + \
                     extraction_stats.get('partial_files', [])
        chunk_path = 'N/A'
        for chunk_file in chunk_files:
            if f"seg_{seg.index:04d}_" in chunk_file:
                chunk_filename = Path(chunk_file).name
                chunk_path = f'<a href="{chunk_file}" style="text-decoration: none;">📹 {chunk_filename}</a>'
                break

        row = f"""
                <tr>
                    <td>{seg.index}</td>
                    <td>{seg.start:.2f}s - {seg.end:.2f}s</td>
                    <td>{seg.text[:80]}{'...' if len(seg.text) > 80 else ''}</td>
                    <td><span class="status-badge {status_class}">{result.status}</span></td>
                    <td>{result.coverage_percent:.1f}%</td>
                    <td><span class="root-cause">{root_cause}</span></td>
                    <td>{chunk_path}</td>
                </tr>
        """
        rows.append(row)

    return '\n'.join(rows)


if __name__ == "__main__":
    print("Report Generator Module - Run via test_srt_coverage.py")
