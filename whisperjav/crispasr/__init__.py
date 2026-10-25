"""The parts every CrispASR pipeline shares (1.10.0; route B, owner W176; code structure S2, W175).

WhisperJAV runs CrispASR through its Python package, ``crispasr``, one call per VAD group. Two
pipelines use it in 1.10.0, Parakeet-CrispASR and Cohere-CrispASR (Ensemble passes only); VibeVoice
and MOSS are planned for 1.10.1 and 1.10.2 (W114). Their one chain is
whisperjav/pipelines/crispasr_pipeline.py; this package holds what it calls:

    errors.py         the two failures and the prefix the run summary looks for
    manifest.py       the pinned model files and the models each pipeline offers
    install.py        whether CrispASR can run here and on which device; downloading, checking and
                      staging the model files; the network-free status for --check
    engine.py         the ASR call behind one contract, and WhisperJAV's own result types
    lines.py          taking one VAD group's result apart and building its subtitles, per model
    pass_settings.py  what a CrispASR pass accepts, refuses and uses, and its speech segmenter settings
    legacy.py         the removed 1.9.x "bring your own CrispASR" mode: what stops the run (exit 2)

Nothing in this package imports torch, crispasr or an ASR module at import time.
"""
