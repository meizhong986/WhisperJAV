"""The Qwen3-ASR model failure that stops the run (R-QFT, #494; owner W491 2).

Standard library only, so main.py and the Ensemble pass worker can name it without importing the ASR module.
"""


# Every message of the stop starts with this, so the run summary can name the reason for either Ensemble pass
# (main.py) without also showing other "stopped before transcription" texts (the damaged-audio stop).
PREFIX = "stopped before transcription: the ASR model "


class QwenModelUnusable(RuntimeError):
    """A Qwen3-ASR model that cannot give real text: the run stops with this plain message (owner W491 2).

    Raised by ``QwenASR._ensure_output_layer()`` (``modules/qwen_asr.py``). Its message starts with
    "stopped before transcription: ", which the Ensemble's run summary shows as the file's reason (``main.py``).
    In a single-pass run the files after it are not started. In an Ensemble pass every file the pass worker was
    given fails with the same message, without loading the model again: all the run's files in the normal mode, one
    file with --ensemble-serial ("Finish each file"), where each file's worker loads the model and fails anew.
    """
