"""The Qwen3-ASR model failure that stops the run (R-QFT, #494; owner W491 2).

Standard library only, so main.py and the Ensemble pass worker can name it without importing the ASR module.
"""


# Every message of the stop starts with this, so the run summary can name the reason for either Ensemble pass
# (main.py) without also showing other "stopped before transcription" texts (the damaged-audio stop).
PREFIX = "stopped before transcription: the ASR model "

# The reason a file after the stop shows in the run summary, state "skipped" (single pass since W491 2; the Ensemble
# since owner W501: once pass 1 stops for this reason no further file starts, in both Ensemble modes; a pass-2 stop
# keeps every file's pass-1 subtitles, owner W505 1).
NOT_STARTED = "not started: the run stopped because its ASR model cannot be used"


class QwenModelUnusable(RuntimeError):
    """A Qwen3-ASR model that cannot give real text: the run stops with this plain message (owner W491 2).

    Raised by ``QwenASR._ensure_output_layer()`` (``modules/qwen_asr.py``). Its message starts with
    "stopped before transcription: ", which the Ensemble's run summary shows as the file's reason (``main.py``).
    In a single-pass run the files after it are not started. In an Ensemble pass the pass worker fails the file with
    this message and gives every later file of its list the same message without loading the model again. When pass 1
    stops, the Ensemble orchestrator skips pass 2 for that file and starts no further file (owner W501): the later
    files show as "skipped", NOT_STARTED, in both modes. When pass 2 stops, every file keeps its pass-1 subtitles,
    marked "pass 2 failed (<this message>); output is pass 1 alone" (state "suspect"), in both modes: in the normal
    mode pass 1 has finished them all; with "Finish each file" (--ensemble-serial) the later files get pass 1 alone
    and pass 2 is not tried again (owner W505 1).
    """
