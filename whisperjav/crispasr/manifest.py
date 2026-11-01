"""The model files WhisperJAV downloads for its CrispASR pipelines, aligners and timing model, pinned (1.10.0; 1.10.1).

CrispASR itself is a Python package, ``crispasr``, pinned in uv.lock like every other dependency
(route B, owner W176; design 11). The model files are not: WhisperJAV downloads them itself, each by
repository, file name and commit, with its size and SHA-256 as Hugging Face lists them (read
2026-10-17; the 1.10.1 files 2026-10-31). Each pipeline offers only the models listed for it here.

1.10.1 (owner W397, W401): five more pipelines, and larger files for every pipeline. Users have 8-16 GB of GPU
memory, about 2 GB of it taken, so each pipeline offers its largest files that fit 6-14 GB, and the menus name
each file's quantization and the GPU memory it needs (W400).

Standard library only.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelFile:
    name: str            # the short name; the --pass1-model / --pass2-model value
    label: str           # plain words for messages, with the file's quantization in brackets
    backend: str         # the model family CrispASR reports when it opens the file (Session.backend)
    repo_id: str
    filename: str
    revision: str        # the pinned commit
    size: int            # bytes
    sha256: str
    licence: str
    # The GPU memory the model needs while it transcribes, in MiB: the card's peak use over its use before the
    # model was opened, measured on the Netflix reference clip's VAD groups through CrispASR 0.8.44 (CUDA
    # build, RTX 3060) on 2026-10-31 (D:\WJ110_measure\gguf6_2026-10-31\sizes). 0 for the aligners and the
    # timing model, which are not offered in a model menu.
    vram_mib: int = 0


def menu_label(model: ModelFile) -> str:
    """The model's name in the GUI's model menus and in --help: its label, which names the quantization, and the
    GPU memory it needs, rounded up to a tenth of a GB (W400). E.g. "Whisper large-v2 (f16), ~4.3 GB VRAM"."""
    return f"{model.label}, ~{math.ceil(model.vram_mib / 1024 * 10) / 10:.1f} GB VRAM"


MODELS = {m.name: m for m in (
    ModelFile("parakeet-tdt-0.6b-ja", "Parakeet TDT 0.6B ja (q8_0)", "parakeet",
              "cstr/parakeet-tdt-0.6b-ja-GGUF", "parakeet-tdt-0.6b-ja-q8_0.gguf",
              "d9e3ba65a6579796389ea89e5939509ed257f972", 673_554_880,
              "5a61e6c7d956c3c72a76fafcd798cac0c9ea66d0e29b3910cd04865a1e42cc17",
              "CC-BY-4.0 (nvidia/parakeet-tdt_ctc-0.6b-ja)",
              vram_mib=885),
    ModelFile("cohere-asr-ja-v0.1", "Cohere Transcribe ja v0.1 (q8_0)", "cohere",
              "TransWithAI/cohere-transcribe-ja-v0.1-GGUF", "cohere-asr-ja-v0.1-q8_0.gguf",
              "c5d36d66dc7d92ade00e01862a95b571c04b446b", 2_423_806_080,
              "2583437aacf2f6d59fb10737b3c82c48941f064c58e10601287b188ed9545b63",
              "none declared by the fine-tune (efwkjn/cohere-asr-ja-v0.1); base model Apache-2.0",
              vram_mib=2539),
    ModelFile("cohere-asr-ja", "Cohere ASR ja (q8_0)", "cohere",
              "CKHO/cohere-asr-ja-GGUF", "cohere-asr-ja-q8_0.gguf",
              "c0f805718baa2daa065f0a8438256f764ee22144", 2_423_806_080,
              "f02cb2935f804d1d4d982ea6551aa7169a6c265795fbbe353310a3928f160586",
              "Apache-2.0 per the GGUF repository; the fine-tune (efwkjn/cohere-asr-ja) declares none",
              vram_mib=2539),
    # 1.10.1 (W400, W401): the full-size (f16) files of the 1.10.0 models.
    ModelFile("parakeet-tdt-0.6b-ja-f16", "Parakeet TDT 0.6B ja (f16)", "parakeet",
              "cstr/parakeet-tdt-0.6b-ja-GGUF", "parakeet-tdt-0.6b-ja.gguf",
              "d9e3ba65a6579796389ea89e5939509ed257f972", 1_246_932_800,
              "374eb0132eebaec4df77a9631cbbeb03790be48a4a517f6cc8e8bdb38fe9a584",
              "CC-BY-4.0 (nvidia/parakeet-tdt_ctc-0.6b-ja)",
              vram_mib=1517),
    ModelFile("cohere-asr-ja-v0.1-f16", "Cohere Transcribe ja v0.1 (f16)", "cohere",
              "TransWithAI/cohere-transcribe-ja-v0.1-GGUF", "cohere-asr-ja-v0.1.gguf",
              "c5d36d66dc7d92ade00e01862a95b571c04b446b", 4_136_507_456,
              "8dd381a24175029632e4eb18dfb53734671226b43fd2d8e40a48f55a11ac5967",
              "none declared by the fine-tune (efwkjn/cohere-asr-ja-v0.1); base model Apache-2.0",
              vram_mib=4179),
    ModelFile("cohere-asr-ja-f16", "Cohere ASR ja (f16)", "cohere",
              "CKHO/cohere-asr-ja-GGUF", "cohere-asr-ja-f16.gguf",
              "c0f805718baa2daa065f0a8438256f764ee22144", 4_136_507_456,
              "c38929d242ef8cff014d13ad298dfe4c1364a0f2617d79168101cc20cd52ea71",
              "Apache-2.0 per the GGUF repository; the fine-tune (efwkjn/cohere-asr-ja) declares none",
              vram_mib=4179),
    # 1.10.1 (W397, W401): the models of the five new pipelines. ReazonSpeech is a NeMo transducer, which
    # CrispASR opens as the parakeet family. The Whisper files are whisper.cpp files (.bin); "f16" is the
    # repository's full-size file.
    ModelFile("reazonspeech-nemo-v2-f16", "ReazonSpeech NeMo v2 (f16)", "parakeet",
              "cstr/reazonspeech-nemo-v2-GGUF", "reazonspeech-nemo-v2-f16.gguf",
              "22799a5919ea26e3c5293fe0e68846fe7918a234", 1_240_435_584,
              "aa44f0e9adb44301fe247be98ff4e8a7065437f96831940e19af38b9ecbb25d1",
              "Apache-2.0 (reazon-research/reazonspeech-nemo-v2)",
              vram_mib=1513),
    ModelFile("reazonspeech-nemo-v2-q8_0", "ReazonSpeech NeMo v2 (q8_0)", "parakeet",
              "cstr/reazonspeech-nemo-v2-GGUF", "reazonspeech-nemo-v2-q8_0.gguf",
              "22799a5919ea26e3c5293fe0e68846fe7918a234", 667_147_072,
              "20b828d05f859a4b0ea0bdcc232cb6e02543d6ddd0b3a1ad1ce37aa56fd7cfd2",
              "Apache-2.0 (reazon-research/reazonspeech-nemo-v2)",
              vram_mib=879),
    ModelFile("kotoba-whisper-v2.0-f16", "Kotoba-Whisper v2.0 (f16)", "whisper",
              "kotoba-tech/kotoba-whisper-v2.0-ggml", "ggml-kotoba-whisper-v2.0.bin",
              "e3a0cf6a62b95911703cfb97d819292e058f12c3", 1_519_521_155,
              "eff70a8a236e731abba774ba71e1f6d0fce53302137208c32207e694e0bf4546",
              "Apache-2.0 (kotoba-tech/kotoba-whisper-v2.0)",
              vram_mib=1869),
    ModelFile("kotoba-whisper-v2.0-q5_0", "Kotoba-Whisper v2.0 (q5_0)", "whisper",
              "kotoba-tech/kotoba-whisper-v2.0-ggml", "ggml-kotoba-whisper-v2.0-q5_0.bin",
              "e3a0cf6a62b95911703cfb97d819292e058f12c3", 537_819_875,
              "4a3b92192b5d3578ff854a5876213e2e27af0c2d357492c2d14271e82c303658",
              "Apache-2.0 (kotoba-tech/kotoba-whisper-v2.0)",
              vram_mib=917),
    ModelFile("whisper-large-v3-turbo-f16", "Whisper large-v3-turbo (f16)", "whisper",
              "ggerganov/whisper.cpp", "ggml-large-v3-turbo.bin",
              "5359861c739e955e79d9a303bcbc70fb988958b1", 1_624_555_275,
              "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
              "MIT (OpenAI Whisper; whisper.cpp conversion)",
              vram_mib=2015),
    ModelFile("whisper-large-v3-turbo-q8_0", "Whisper large-v3-turbo (q8_0)", "whisper",
              "ggerganov/whisper.cpp", "ggml-large-v3-turbo-q8_0.bin",
              "5359861c739e955e79d9a303bcbc70fb988958b1", 874_188_075,
              "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
              "MIT (OpenAI Whisper; whisper.cpp conversion)",
              vram_mib=1286),
    ModelFile("whisper-large-v2-f16", "Whisper large-v2 (f16)", "whisper",
              "ggerganov/whisper.cpp", "ggml-large-v2.bin",
              "5359861c739e955e79d9a303bcbc70fb988958b1", 3_094_623_691,
              "9a423fe4d40c82774b6af34115b8b935f34152246eb19e80e376071d3f999487",
              "MIT (OpenAI Whisper; whisper.cpp conversion)",
              vram_mib=4152),
    ModelFile("whisper-large-v2-q8_0", "Whisper large-v2 (q8_0)", "whisper",
              "ggerganov/whisper.cpp", "ggml-large-v2-q8_0.bin",
              "5359861c739e955e79d9a303bcbc70fb988958b1", 1_656_129_691,
              "fef54e6d898246a65c8285bfa83bd1807e27fadf54d5d4e81754c47634737e8c",
              "MIT (OpenAI Whisper; whisper.cpp conversion)",
              vram_mib=2743),
    ModelFile("whisper-large-v3-f16", "Whisper large-v3 (f16)", "whisper",
              "ggerganov/whisper.cpp", "ggml-large-v3.bin",
              "5359861c739e955e79d9a303bcbc70fb988958b1", 3_095_033_483,
              "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
              "MIT (OpenAI Whisper; whisper.cpp conversion)",
              vram_mib=4353),
    ModelFile("whisper-large-v3-q5_0", "Whisper large-v3 (q5_0)", "whisper",
              "ggerganov/whisper.cpp", "ggml-large-v3-q5_0.bin",
              "5359861c739e955e79d9a303bcbc70fb988958b1", 1_081_140_203,
              "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1",
              "MIT (OpenAI Whisper; whisper.cpp conversion)",
              vram_mib=2187),
    ModelFile("moss-transcribe-diarize-0.9b-f16", "MOSS-Transcribe-Diarize 0.9B (f16)", "moss-diarize",
              "cstr/MOSS-Transcribe-Diarize-GGUF", "moss-transcribe-diarize-0.9b-f16.gguf",
              "17b039a39e4cfd4e5f24c26c2a53c4e7ac28f651", 1_823_188_512,
              "37800de328dac1689fd8716ede706809fce4c61921d5fb55100783edeb24e4f3",
              "Apache-2.0 (OpenMOSS-Team/MOSS-Transcribe-Diarize)",
              vram_mib=2087),
    ModelFile("moss-transcribe-diarize-0.9b-q8_0", "MOSS-Transcribe-Diarize 0.9B (q8_0)", "moss-diarize",
              "cstr/MOSS-Transcribe-Diarize-GGUF", "moss-transcribe-diarize-0.9b-q8_0.gguf",
              "17b039a39e4cfd4e5f24c26c2a53c4e7ac28f651", 1_410_311_776,
              "80cb31198a07e0c456460658e3264b9a829095c07ad8454bdc9f1f999c277154",
              "Apache-2.0 (OpenMOSS-Team/MOSS-Transcribe-Diarize)",
              vram_mib=1673),
)}

# The models each pipeline offers its users, and its default. 1.10.0: Cohere-CrispASR's default is v0.1 (W129 1);
# cohere-asr-ja stays a choice; the Parakeet CTC model is not offered (W129 2). 1.10.1: the full-size files join
# the menus (W401 1) and the defaults of these two pipelines stay; each new pipeline's largest file is its default
# (W400). These lists feed the command line (--passN-model), --help, --check and the GUI's model menus; the order
# is the menu's.
PIPELINE_MODELS = {
    "parakeet-crispasr": ("parakeet-tdt-0.6b-ja", "parakeet-tdt-0.6b-ja-f16"),
    "cohere-crispasr": ("cohere-asr-ja-v0.1", "cohere-asr-ja-v0.1-f16", "cohere-asr-ja", "cohere-asr-ja-f16"),
    "reazonspeech-crispasr": ("reazonspeech-nemo-v2-f16", "reazonspeech-nemo-v2-q8_0"),
    "kotoba-crispasr": ("kotoba-whisper-v2.0-f16", "kotoba-whisper-v2.0-q5_0"),
    "whisper-turbo-crispasr": ("whisper-large-v3-turbo-f16", "whisper-large-v3-turbo-q8_0"),
    "whisper-large-crispasr": ("whisper-large-v2-f16", "whisper-large-v2-q8_0",
                               "whisper-large-v3-f16", "whisper-large-v3-q5_0"),
    "moss-crispasr": ("moss-transcribe-diarize-0.9b-f16", "moss-transcribe-diarize-0.9b-q8_0"),
}
PIPELINE_DEFAULT_MODEL = {
    "parakeet-crispasr": "parakeet-tdt-0.6b-ja",
    "cohere-crispasr": "cohere-asr-ja-v0.1",
    "reazonspeech-crispasr": "reazonspeech-nemo-v2-f16",
    "kotoba-crispasr": "kotoba-whisper-v2.0-f16",
    "whisper-turbo-crispasr": "whisper-large-v3-turbo-f16",
    # large-v2 before large-v3: the fewer invented lines and the lower character error rate of the two on the
    # Netflix reference clip (2026-10-31, step 1: 4 and 0.520 against 14 and 0.574, both q5_0).
    "whisper-large-crispasr": "whisper-large-v2-f16",
    "moss-crispasr": "moss-transcribe-diarize-0.9b-f16",
}
CRISPASR_PIPELINES = tuple(PIPELINE_MODELS)
# The pipelines whose text the CrispASR aligner times (wav2vec2-ja, DEFAULT_ALIGNER) by default: Cohere-CrispASR
# (item 12; W248 3, Q7) and MOSS-CrispASR, whose model gives no word times. The others use their models' own
# token times (owner W411, W412): Parakeet, Kotoba, Whisper-Turbo, Whisper-Large starts and ends; ReazonSpeech
# starts only, its ends at the VAD group's end. Each pipeline class's ENGINE.aligner says the same
# (tests/test_crispasr_pipelines.py); the GUI's status panel reads it here.
ALIGNED_PIPELINES = frozenset({"cohere-crispasr", "moss-crispasr"})

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
