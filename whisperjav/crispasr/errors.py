"""The two failures of the CrispASR pipelines, and the prefix the run summary looks for (1.10.0).

One home for what install.py, engine.py, the pipelines, the Ensemble pass worker and main.py import
(route B design 5.3). Standard library only.
"""


class CrispASRSetupError(RuntimeError):
    """CrispASR cannot run here, or the model cannot be made ready or opened.

    In the main process the run stops before audio extraction (red box, exit status 1); in a pass the
    file fails, and the error is kept and raised again for the pass's other files.
    """


class CrispASRRunError(RuntimeError):
    """Every CrispASR call of a file failed: that file fails."""


# A pass error that starts with this is a plain-language reason, and the end-of-run summary shows it
# rather than "ensemble pass failed". The Ensemble pass worker adds it; main.py looks for it.
FAILURE_PREFIX = "CrispASR: "
