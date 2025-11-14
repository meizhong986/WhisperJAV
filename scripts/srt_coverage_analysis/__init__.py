"""
SRT Coverage Analysis Tool
===========================

A diagnostic tool for analyzing SRT subtitle coverage by comparing
reference subtitles against WhisperJAV test outputs.

Modules:
    srt_parser: Parse SRT files into structured data
    coverage_calculator: Calculate temporal overlap between segments
    metadata_tracer: Trace missing segments through pipeline metadata
    media_extractor: Extract media chunks for manual review
    gantt_visualizer: Generate timeline visualizations
    report_generator: Generate HTML and JSON reports
"""

__version__ = "1.0.0"
__author__ = "WhisperJAV Development Team"

from .srt_parser import Segment, parse_srt_file, calculate_srt_statistics
from .coverage_calculator import (
    CoverageResult,
    calculate_coverage,
    analyze_all_segments,
    calculate_summary_statistics
)
from .metadata_tracer import (
    TraceResult,
    load_metadata,
    trace_in_metadata,
    trace_all_segments,
    analyze_root_causes
)
from .media_extractor import extract_media_chunk, extract_multiple_chunks
from .gantt_visualizer import generate_gantt_chart
from .report_generator import generate_json_report, generate_html_report

__all__ = [
    'Segment',
    'parse_srt_file',
    'calculate_srt_statistics',
    'CoverageResult',
    'calculate_coverage',
    'analyze_all_segments',
    'calculate_summary_statistics',
    'TraceResult',
    'load_metadata',
    'trace_in_metadata',
    'trace_all_segments',
    'analyze_root_causes',
    'extract_media_chunk',
    'extract_multiple_chunks',
    'generate_gantt_chart',
    'generate_json_report',
    'generate_html_report',
]
