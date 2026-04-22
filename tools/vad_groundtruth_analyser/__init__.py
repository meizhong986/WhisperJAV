"""
VAD Ground-Truth Analyser — multi-backend speech-segmenter evaluation tool.

Compare VAD backends side-by-side on a media file. Ground truth is optional:

- **With GT (SRT file)**: computes frame-level F1/precision/recall, segment IoU,
  boundary drift, missed-speech% and false-alarm% per backend.
- **Without GT**: still produces per-backend segments + an inter-backend
  agreement matrix (pairwise F1) as a consensus proxy.

Outputs: interactive Plotly HTML, JSON, CSV (any subset via CLI flags).

Example (CLI):

    python -m tools.vad_groundtruth_analyser media.mp4 \\
        --ground-truth gt.srt \\
        --sensitivity aggressive \\
        --backends silero-v3.1,silero-v6.2,ten,whisperseg

Programmatic use (advanced):

    from tools.vad_groundtruth_analyser import BackendRunner, analyse
    report = analyse(media_path="media.wav", gt_path="gt.srt",
                     backends=["whisperseg", "ten"], sensitivity="aggressive")
    print(report.backends["whisperseg"].metrics.frame_f1)
"""

__version__ = "1.0.0"

# Public API re-exports (keep minimal; expand as needed)
from .models import (  # noqa: F401
    GtSegment,
    VadSegment,
    BackendMetrics,
    BackendReport,
    AgreementMatrix,
    AnalysisReport,
)
from .runner import BackendRunner  # noqa: F401

__all__ = [
    "__version__",
    "GtSegment",
    "VadSegment",
    "BackendMetrics",
    "BackendReport",
    "AgreementMatrix",
    "AnalysisReport",
    "BackendRunner",
    "analyse",
]


def analyse(**kwargs):
    """High-level entry point for programmatic use.

    Thin wrapper around cli.run_analysis — accepts the same keyword arguments
    as the CLI (media_path, gt_path, backends, sensitivity, frame_ms, ...).
    """
    from .cli import run_analysis
    return run_analysis(**kwargs)
