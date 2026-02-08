"""
Qwen Pipeline Diagnostic & Benchmarking System.

Compare Qwen input modes (assembly, context_aware, vad_slicing) against
ground-truth SRT to determine text and timing accuracy.

Usage:
    whisperjav-bench --ground-truth gt.srt --test path1 "Label 1" --test path2 "Label 2"
"""
