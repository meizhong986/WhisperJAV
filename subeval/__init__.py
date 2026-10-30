"""subeval: evaluate a test subtitle file against a reference subtitle file.

A standalone research and measurement utility (Python standard library only). It reads SRT files, aligns the
characters of the test subtitles to the characters of the reference subtitles with a time-aware edit-distance
alignment, and derives every indicator - text, missed lines, suspected hallucination lines, line openings and
endings, timing - from that one alignment. METHOD.md defines each step and each indicator exactly.
"""

__version__ = "0.1.0"
