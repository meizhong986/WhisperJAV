"""
Silero VAD version selection for faster-whisper's built-in VAD (v1.9.2, S6-S8).

WHAT THIS DOES
--------------
The balanced pipeline runs faster-whisper's OWN VAD (``transcribe(vad_filter=True)``).
faster-whisper ships exactly one Silero build inside its package and gives no way to
choose another.  This module lets the user pick between Silero **3.1**, **4.0** and
**6.2** by rebinding ``faster_whisper.vad.get_vad_model`` to an adapter that runs the
requested build.  Everything above the model — ``get_speech_timestamps``, the
hysteresis, the padding, the chunk assembly — is faster-whisper's own code, untouched.

WHY NOT THE CACHE HACK (requirement S6B1/S6B2)
----------------------------------------------
The owner's premise was that faster-whisper picks its Silero model out of a cache that
can be pre-populated.  It does not.  ``faster_whisper/vad.py`` builds a hard-coded path
inside its own package::

    path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")

Verified in faster-whisper 1.0.2, the 1.2.1 release, SYSTRAN master (the pinned build) and the
2.1.1 fork —
same pattern in all four.  There is no environment variable, no cache directory and no
hook.  Nor can the file simply be overwritten: the three Silero generations have
different ONNX input signatures (see the table below).  So the selection is made one
level up, at the single function that produces the model object.

The reference script the owner cited pre-populates *stable-ts*'s
``cached_model_instances['silero_vad']`` — a different library on a different code path.
Balanced drives ``faster_whisper.WhisperModel`` directly.

THE THREE TRAPS (each produces plausible-looking wrong output, not an error)
---------------------------------------------------------------------------
1. **v3.1's ONNX output is TWO-CLASS.**  Silero's own ``utils_vad.py`` takes
   ``squeeze(2)[:, 1]``.  Index 0 gives a smooth, believable series that is *not*
   speech probability.
2. **All three builds are stateful and batch-1.**  The LSTM state carries from window
   to window and must be reset per audio.  v3.1 raises "Onnx model does not support
   batching" outright.  They cannot use the one-shot batched call faster-whisper makes
   against its own bundled model (which was exported with a sequence dimension).
3. **The 64-sample context concatenation is a v5/v6 input convention only.**  Applying
   it to v3.1 or v4.0 corrupts the input; omitting it for 6.2 does the same.

MODEL SIGNATURES (read from the bundled files)
----------------------------------------------
==========  =========================================  ==============================
build       inputs                                     output
==========  =========================================  ==============================
v3.1        ``input`` (1,512), ``h0``/``c0`` (2,1,64)   ``output`` (1,2,1) -> ``[0,1,0]``
v4.0        ``input`` (1,512), ``sr``, ``h``/``c``      ``output`` (1,1)
6.2         ``input`` (1,576), ``state`` (2,1,128),     ``output`` (1,1)
            ``sr``
==========  =========================================  ==============================

Use the ONNX files, never the torch-JIT ones: measured on this machine per 60 s of
audio, single thread, JIT v3.1 12x realtime vs ONNX v3.1 87x, JIT v4.0 22x vs ONNX
v4.0 152x.  On a two-hour film that is ~83 s of CPU instead of ~10 minutes.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from whisperjav.utils.logger import logger

# The user-selectable Silero builds, in the order the owner specified (S7).
VAD_VERSIONS = ("3.1", "4.0", "6.2")

# S8: the default VAD version for the balanced pipeline.
# 4.0 since 2026-09-12 (owner, release-candidate testing); it was 3.1 from the
# v1.9.2 development cycle up to that point. Read by the Pydantic
# FasterWhisperVADOptions default, by main.py / pass_worker.py help text and by
# the GUI's Customize > Segmenter info line, so this is the only place it is set
# on the Python side. The GUI Ensemble tab carries its own copy -- see
# VAD_VERSION_LABELS below.
DEFAULT_VAD_VERSION = "4.0"

# Human labels for the Silero builds.
#
# NOTHING CURRENTLY DISPLAYS THESE STRINGS (checked 2026-09-12). The one reader,
# webview_gui/api.py:1295, does `", ".join(VAD_VERSION_LABELS)` -- joining a dict
# iterates its KEYS, so the Customize > Segmenter info line renders "3.1, 4.0, 6.2".
# The "(default)" marker below is therefore dead text kept only so the dict does not
# contradict DEFAULT_VAD_VERSION; do not cite it as a consumer of that constant.
# The CLI help builds its list from VAD_VERSIONS, and the Ensemble-tab dropdown carries
# its own copy at app.js:vadVersionOptions -- keep that copy in step by hand.
VAD_VERSION_LABELS: Dict[str, str] = {
    "3.1": "Internal FW Silero VAD 3.1",
    "4.0": "Internal FW Silero VAD 4.0 (default)",
    "6.2": "Internal FW Silero VAD 6.2 (latest)",
}

# faster-whisper's get_speech_timestamps hard-codes a 512-sample window at 16 kHz
# and indexes the probability series by it (`window_size_samples * i`), so the
# adapter must emit exactly one probability per 512 samples.
_WINDOW_SAMPLES = 512
_SAMPLE_RATE = 16000

_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "vad"

# Guards the module-level install: the recognizer can be constructed from more than
# one thread in the async pipeline, and rebinding a third-party global is not atomic.
#
# NOTE: the adapter's __call__ is NOT reentrant -- it carries LSTM state on `self`, and
# one instance per version is shared process-wide. The code it replaces
# (faster_whisper.vad.SileroVADModel) keeps its state in locals and is reentrant. This is
# safe only because the pipeline drives VAD from one thread at a time
# (utils/async_processor.py pins max_workers=1). Revisit before that changes.
_install_lock = threading.Lock()
_active_version: Optional[str] = None
_model_cache: Dict[str, "_SileroONNXAdapter"] = {}


def model_path(version: str) -> Path:
    """Absolute path of the bundled ONNX file for ``version``."""
    return _ASSET_DIR / f"silero_vad_v{version}.onnx"


def normalise_version(version: Optional[str]) -> str:
    """
    Resolve a version string, defaulting when it is absent and warning when it is wrong.

    Every producer is constrained -- argparse `choices`, the Pydantic validator and the
    GUI dropdown all emit one of VAD_VERSIONS -- so this exists for None (the common
    case: no override) and as a last resort for a hand-edited config file. It never
    raises: a bad VAD-version string must not end a transcription run.
    """
    if version is None:
        return DEFAULT_VAD_VERSION
    token = str(version).strip()
    if token in VAD_VERSIONS:
        return token
    logger.warning(
        "Unknown VAD version %r; falling back to Silero v%s.", version, DEFAULT_VAD_VERSION
    )
    return DEFAULT_VAD_VERSION


def _make_session(path: Path):
    """Create a single-threaded CPU onnxruntime session, as faster-whisper does."""
    try:
        import onnxruntime
    except ImportError as e:  # pragma: no cover - onnxruntime is a faster-whisper dep
        raise RuntimeError(
            "Selecting a Silero VAD version requires the onnxruntime package"
        ) from e

    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    opts.log_severity_level = 4
    return onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"], sess_options=opts
    )


class _SileroONNXAdapter:
    """
    Drop-in replacement for ``faster_whisper.vad.SileroVADModel``.

    faster-whisper calls the model exactly once per audio, as ``model(padded_audio)``
    where ``padded_audio`` is 1-D and a whole multiple of 512 samples, and expects one
    probability per window back.  The keyword arguments mirror faster-whisper's own
    signature so this stays a drop-in even though its own code never passes them.
    """

    def __init__(self, version: str):
        self.version = version
        path = model_path(version)
        if not path.is_file():
            raise FileNotFoundError(
                f"Bundled Silero VAD model for v{version} is missing: {path}"
            )
        self.session = _make_session(path)
        self._sr = np.array(_SAMPLE_RATE, dtype="int64")

    # -- per-generation state and one-window step -------------------------------

    def _reset(self):
        if self.version == "3.1":
            self._h = np.zeros((2, 1, 64), dtype="float32")
            self._c = np.zeros((2, 1, 64), dtype="float32")
        elif self.version == "4.0":
            self._h = np.zeros((2, 1, 64), dtype="float32")
            self._c = np.zeros((2, 1, 64), dtype="float32")
        else:  # 6.2
            self._state = np.zeros((2, 1, 128), dtype="float32")
            # Silero 5.x/6.x prepend the previous window's last 64 samples; the very
            # first window sees zeros (silero_vad/utils_vad.py reset_states()).
            self._context = np.zeros((1, 64), dtype="float32")

    def _step(self, window: np.ndarray) -> float:
        """One 512-sample window in, one speech probability out."""
        if self.version == "3.1":
            out, self._h, self._c = self.session.run(
                None, {"input": window, "h0": self._h, "c0": self._c}
            )
            # TRAP 1: v3.1 emits (1, 2, 1) — [not-speech, speech]. Index 1 is speech.
            return float(out[0, 1, 0])

        if self.version == "4.0":
            out, self._h, self._c = self.session.run(
                None, {"input": window, "sr": self._sr, "h": self._h, "c": self._c}
            )
            return float(out[0, 0])

        # 6.2 — TRAP 3: this generation, and only this one, takes the 64-sample context.
        padded = np.concatenate([self._context, window], axis=1)
        out, self._state = self.session.run(
            None, {"input": padded, "state": self._state, "sr": self._sr}
        )
        self._context = window[:, -64:]
        return float(out[0, 0])

    # -- faster-whisper entry point ---------------------------------------------

    def __call__(
        self,
        audio: np.ndarray,
        num_samples: int = _WINDOW_SAMPLES,
        context_size_samples: int = 64,
    ) -> np.ndarray:
        assert audio.ndim == 1, "Input should be a 1D array"
        assert (
            audio.shape[0] % num_samples == 0
        ), "Input size should be a multiple of num_samples"

        # TRAP 2: stateful, batch-1, strictly sequential. State resets per audio.
        self._reset()

        windows = audio.reshape(-1, num_samples).astype(np.float32, copy=False)
        probs = np.empty((windows.shape[0], 1), dtype=np.float32)
        for i in range(windows.shape[0]):
            probs[i, 0] = self._step(windows[i : i + 1])
        return probs


def get_model(version: str) -> _SileroONNXAdapter:
    """Return the (cached) adapter for ``version``, loading the ONNX file on first use."""
    version = normalise_version(version)
    model = _model_cache.get(version)
    if model is None:
        model = _SileroONNXAdapter(version)
        _model_cache[version] = model
    return model


def active_version() -> Optional[str]:
    """The version currently installed into faster-whisper, or None if untouched."""
    return _active_version


def install(version: Optional[str]) -> str:
    """
    Make faster-whisper's built-in VAD run the requested Silero build.

    Rebinds the module-level ``faster_whisper.vad.get_vad_model``.  ``get_speech_timestamps``
    resolves that name through ``vad.py``'s own globals at call time, and it is the only
    consumer in the package, so the rebind reaches every caller (``WhisperModel.transcribe``
    and ``BatchedInferencePipeline.transcribe``).  The ``lru_cache`` sits on the function
    being replaced, so it is irrelevant.

    Process-global by nature: call it in whichever process hosts the recognizer.  Returns
    the version actually installed.  On any failure the run continues on faster-whisper's
    own bundled model rather than dying — a VAD-version preference must never be fatal.
    """
    global _active_version

    version = normalise_version(version)
    with _install_lock:
        if _active_version == version:
            return version
        try:
            import faster_whisper.vad as fw_vad

            model = get_model(version)
            fw_vad.get_vad_model = lambda: model
            _active_version = version
            logger.debug(
                "faster_whisper.vad.get_vad_model rebound to bundled %s",
                model_path(version).name,
            )
            return version
        except Exception as e:
            logger.warning(
                "Could not select Silero VAD v%s (%s); falling back to the model "
                "bundled with faster-whisper.", version, e,
            )
            _active_version = None
            return version
