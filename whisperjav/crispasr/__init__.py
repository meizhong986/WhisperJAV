"""The parts every CrispASR pipeline shares (1.10.0, item 4; master plan 3.7, owner decision Q20 = A, W118 1).

WhisperJAV runs CrispASR's own program (``crispasr.exe``) and reads its ``-ojf`` JSON. Four pipelines
use it, one per engine: Parakeet-CrispASR and Cohere-CrispASR in 1.10.0, VibeVoice-CrispASR in 1.10.1,
MOSS-CrispASR in 1.10.2. They share only what the program's own formats make the same for every engine:

    manifest.py   the pinned program builds and the pinned model files
    install.py    downloading, checking and unpacking them; the network-free status for --check
    runner.py     starting the program for one film and judging its result
    ojf.py        reading its JSON, placing segments in their slices, writing its cut-point file

Everything that differs per engine -- the slice rule, the parser, line building and post-processing --
lives in that engine's own pipeline class (whisperjav/pipelines/*_crispasr_pipeline.py).

Two small modules serve WhisperJAV's command line and Ensemble pass worker:

    legacy.py         the removed 1.9.x "bring your own CrispASR" mode: what stops the run (exit 2)
    pass_settings.py  what a CrispASR pass accepts, refuses and does not use

Nothing in this package imports torch or an ASR module at import time.
"""
