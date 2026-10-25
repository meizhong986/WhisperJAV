"""The model files WhisperJAV downloads for its CrispASR pipelines, pinned (1.10.0).

CrispASR itself is a Python package, ``crispasr``, pinned in uv.lock like every other dependency
(route B, owner W176; design 11). The model files are not: WhisperJAV downloads them itself, each by
repository, file name and commit, with its size and SHA-256 as Hugging Face lists them (read
2026-10-17). Each pipeline offers only the models listed for it here.

Standard library only.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelFile:
    name: str            # the short name; the --pass1-model / --pass2-model value
    label: str           # plain words for messages
    backend: str         # the model family CrispASR reports when it opens the file (Session.backend)
    repo_id: str
    filename: str
    revision: str        # the pinned commit
    size: int            # bytes
    sha256: str
    licence: str


MODELS = {m.name: m for m in (
    ModelFile("parakeet-tdt-0.6b-ja", "Parakeet TDT 0.6B ja (q8_0)", "parakeet",
              "cstr/parakeet-tdt-0.6b-ja-GGUF", "parakeet-tdt-0.6b-ja-q8_0.gguf",
              "d9e3ba65a6579796389ea89e5939509ed257f972", 673_554_880,
              "5a61e6c7d956c3c72a76fafcd798cac0c9ea66d0e29b3910cd04865a1e42cc17",
              "CC-BY-4.0 (nvidia/parakeet-tdt_ctc-0.6b-ja)"),
    ModelFile("cohere-asr-ja-v0.1", "Cohere Transcribe ja v0.1 (q8_0)", "cohere",
              "TransWithAI/cohere-transcribe-ja-v0.1-GGUF", "cohere-asr-ja-v0.1-q8_0.gguf",
              "c5d36d66dc7d92ade00e01862a95b571c04b446b", 2_423_806_080,
              "2583437aacf2f6d59fb10737b3c82c48941f064c58e10601287b188ed9545b63",
              "none declared by the fine-tune (efwkjn/cohere-asr-ja-v0.1); base model Apache-2.0"),
    ModelFile("cohere-asr-ja", "Cohere ASR ja (q8_0)", "cohere",
              "CKHO/cohere-asr-ja-GGUF", "cohere-asr-ja-q8_0.gguf",
              "c0f805718baa2daa065f0a8438256f764ee22144", 2_423_806_080,
              "f02cb2935f804d1d4d982ea6551aa7169a6c265795fbbe353310a3928f160586",
              "Apache-2.0 per the GGUF repository; the fine-tune (efwkjn/cohere-asr-ja) declares none"),
)}

# The models each 1.10.0 pipeline offers its users (W118 4), and its default. Cohere-CrispASR's default
# is v0.1 (W129 1); cohere-asr-ja stays a choice. The Parakeet CTC model is not offered (W129 2). These
# lists feed the command line (--passN-model), --help, --check and the GUI's model menus.
PIPELINE_MODELS = {
    "parakeet-crispasr": ("parakeet-tdt-0.6b-ja",),
    "cohere-crispasr": ("cohere-asr-ja-v0.1", "cohere-asr-ja"),
}
PIPELINE_DEFAULT_MODEL = {
    "parakeet-crispasr": "parakeet-tdt-0.6b-ja",
    "cohere-crispasr": "cohere-asr-ja-v0.1",
}
CRISPASR_PIPELINES = tuple(PIPELINE_MODELS)

# The source-language values that mean Japanese, lower-cased: the command line's --language value
# ("japanese") and the codes the pass worker passes on ("ja").
JAPANESE_NAMES = frozenset({"ja", "jp", "jpn", "japanese"})


def model_for(pipeline: str, name: Optional[str]) -> ModelFile:
    """The pinned model a pipeline runs: the named one, else the pipeline's default.

    Raises ValueError, with the offered choices in its text, for a name the pipeline does not offer.
    """
    choices = PIPELINE_MODELS[pipeline]
    if not name:
        name = PIPELINE_DEFAULT_MODEL[pipeline]
    if name not in choices:
        raise ValueError(f"'{name}' is not a model of {pipeline}; choose one of: {', '.join(choices)}")
    return MODELS[name]
