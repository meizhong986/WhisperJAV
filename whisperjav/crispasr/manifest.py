"""The model files WhisperJAV downloads for its CrispASR pipelines, aligners and timing model, pinned (1.10.0).

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

# The aligners: models that time text already written, run by CrispASR's align_words (item 12, C1; design
# note docs/plans/design/2026-10-28-chronosjav-timing-c3-c1.md section 4). Kept apart from MODELS, which are
# the CrispASR pipelines' ASR models. CrispASR picks the aligner family from the file's name ("forced-aligner"
# -> Qwen3-ForcedAligner, "wav2vec2" -> CTC with the model's own vocabulary, anything else -> canary-ctc), so
# a file is always opened under its own name (install.stage_model keep_name=True). ``backend`` names that
# family, for messages only. Read from Hugging Face on 2026-10-27 (all three on the owner's word, W248 2;
# wav2vec2-ja the default).
ALIGNER_MODELS = {m.name: m for m in (
    ModelFile("wav2vec2-ja", "wav2vec2 large XLSR-53 ja (q4_k)", "wav2vec2",
              "cstr/wav2vec2-large-xlsr-53-japanese-GGUF", "wav2vec2-large-xlsr-53-japanese-q4_k.gguf",
              "16c3b21fe7858426806f4bf795578497e0b5dad2", 231_505_920,
              "080cf27f2ab700a296ceb9bccd61275ec29424a7eb9a80009f3dd0b10d52f29b",
              "Apache-2.0 (jonatasgrosman/wav2vec2-large-xlsr-53-japanese)"),
    ModelFile("parakeet-ctc-ja", "Parakeet CTC 1.1B ja (q8_0)", "canary-ctc",
              "cstr/parakeet-ctc-1.1b-ja-GGUF", "parakeet-ctc-1.1b-ja-q8_0.gguf",
              "033e8bf08d60a582b47a92bba07ca9e392982861", 1_137_330_464,
              "4fca7a83833012d515b4a71ab3df9a0abcf3362107369850d89dda962dfd34b4",
              "Apache-2.0 (grider-transwithai/parakeet-ctc-1.1b-ja)"),
    ModelFile("qwen3-fa-gguf", "Qwen3-ForcedAligner 0.6B (q4_k)", "qwen3-fa",
              "cstr/qwen3-forced-aligner-0.6b-GGUF", "qwen3-forced-aligner-0.6b-q4_k.gguf",
              "1ec5110602ccab18c878ddebedab0891e290a95c", 529_001_216,
              "a7bb4cbeacc6414f11a5d23dc7661a51a941a71e6d559dc7b408b52473f2ae84",
              "Apache-2.0 (Qwen/Qwen3-ForcedAligner-0.6B)"),
)}
DEFAULT_ALIGNER = "wav2vec2-ja"
# What a user is told when choosing an aligner WhisperJAV's tests found wanting (W255: "Wire it, labelled").
# Text approved by the owner as drafted (W268); design note section 9.
ALIGNER_WARNINGS = {
    "parakeet-ctc-ja": ("The Parakeet CTC ja aligner's times came back evenly spread in WhisperJAV's tests "
                        "(2026-10-27); it is not recommended."),
}


def aligner_model_for(name: Optional[str]) -> ModelFile:
    """The pinned aligner model: the named one, else the default.

    Raises ValueError, with the choices in its text, for a name that is not offered.
    """
    name = name or DEFAULT_ALIGNER
    if name not in ALIGNER_MODELS:
        raise ValueError(f"'{name}' is not a CrispASR aligner; choose one of: {', '.join(ALIGNER_MODELS)}")
    return ALIGNER_MODELS[name]


# The timing model: a small Whisper whose DTW token times say where speech starts and where a new sentence
# starts inside a VAD group; its text is thrown away (item 12, C3; design note section 5). Whisper small only
# (owner W255). A whisper.cpp file (.bin), read by CrispASR's Whisper-only CrispASR class.
TIMING_MODELS = {m.name: m for m in (
    ModelFile("whisper-small", "Whisper small", "whisper",
              "ggerganov/whisper.cpp", "ggml-small.bin",
              "5359861c739e955e79d9a303bcbc70fb988958b1", 487_601_967,
              "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
              "MIT (OpenAI Whisper; whisper.cpp conversion)"),
)}
DEFAULT_TIMING_MODEL = "whisper-small"


def timing_model_for(name: Optional[str]) -> ModelFile:
    """The pinned timing model: the named one, else the default.

    Raises ValueError, with the choices in its text, for a name that is not offered.
    """
    name = name or DEFAULT_TIMING_MODEL
    if name not in TIMING_MODELS:
        raise ValueError(f"'{name}' is not a timing model; choose one of: {', '.join(TIMING_MODELS)}")
    return TIMING_MODELS[name]


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
