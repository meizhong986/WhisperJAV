"""What WhisperJAV downloads for its CrispASR pipelines, pinned (master plan 1.2, 3.3, 3.7).

The program ships as CrispASR's release files, not as a Python package, so uv.lock cannot pin it and
the installer does not carry it. This manifest is its pin instead: it changes once per release, on
purpose, like the lock. Every install -- the Windows installer, WJ110, a source install,
whisperjav-upgrade -- downloads exactly these bytes the first time a CrispASR pipeline runs.

Hosting is CrispASR's own GitHub release, pinned by address and SHA-256 (Q8 (a), W118 5). The sizes
and SHA-256 values below are the release's own record for v0.8.43; the two Windows files were also
hashed from the copies the 2026-10-17 studies downloaded, and matched.

The model files are pinned by repository, file name and commit, with their size and SHA-256 as Hugging
Face lists them (read 2026-10-17). Each pipeline offers only the models listed for it here.
"""

import platform
import sys
from dataclasses import dataclass, field
from typing import Optional

CRISPASR_VERSION = "0.8.43"
CRISPASR_GIT = "eb633d90"  # in the program's banner: "crispasr 0.8.43 (git eb633d90, Release) [backends: ...]"
RELEASE_BASE_URL = f"https://github.com/CrispStrobe/CrispASR/releases/download/v{CRISPASR_VERSION}/"

# The CUDA 13 build carries no CUDA runtime. These three files come from the user's own PyTorch
# (cu130 wheels hold them in torch/lib) and are put beside the program (plan 1.3).
CUDA13_RUNTIME_FILES = ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll")


@dataclass(frozen=True)
class ProgramBuild:
    key: str             # its folder under ~/.cache/whisperjav/crispasr/<version>/
    label: str           # plain words for messages
    platform: str        # sys.platform
    machines: tuple      # platform.machine() values it runs on
    asset: str           # the release file
    size: int            # bytes
    sha256: str
    gpu: Optional[str] = None      # "cuda13" for the CUDA 13 build
    needs_avx2: bool = False       # x86-64 builds stop without AVX2 (plan 1.4, Q23)
    extra_args: tuple = field(default_factory=tuple)  # added to every run (Apple: --no-gpu)
    base_url: str = RELEASE_BASE_URL

    @property
    def url(self) -> str:
        return self.base_url + self.asset

    @property
    def exe_name(self) -> str:
        return "crispasr.exe" if self.platform == "win32" else "crispasr"


PROGRAM_BUILDS = (
    ProgramBuild("cuda13", "GPU build (CUDA 13)", "win32", ("AMD64", "x86_64"),
                 "crispasr-windows-x86_64-cuda13-non-cuda.zip", 167_721_415,
                 "28ae0a75e6fbac5142be2bd1ca5b9c20440f8fb0dc532ec1becbf985ef35f40e",
                 gpu="cuda13", needs_avx2=True),
    ProgramBuild("cpu", "CPU build", "win32", ("AMD64", "x86_64"),
                 "crispasr-windows-x86_64-cpu.zip", 9_088_774,
                 "997bc39f541e5d5d0bd14144159c038bd95d6ab8b523718c34a54e03ab404f47",
                 needs_avx2=True),
    # Apple silicon: the build can use Metal, but runs CPU-only in 1.10.0 (plan 1.1; Metal is 1.10.2+).
    ProgramBuild("cpu", "CPU build", "darwin", ("arm64",),
                 "crispasr-macos-arm64.tar.gz", 18_012_924,
                 "5c13fbbc12eb45c3f01823a6bbfbbb22e0682bc76c6b78a0d9511da1aaf88831",
                 extra_args=("--no-gpu",)),
    ProgramBuild("cpu", "CPU build", "darwin", ("x86_64",),
                 "crispasr-macos-x86_64.tar.gz", 18_500_090,
                 "26c7ed772472330a354f08122a59107fff6830f9e7a784df8ab9c1d7dfc4b023",
                 needs_avx2=True),
    ProgramBuild("cpu", "CPU build", "linux", ("x86_64", "AMD64"),
                 "crispasr-linux-x86_64.tar.gz", 42_822_913,
                 "e81b597cb80e49d8687a4bceb1eb24bf55928c55ff1292a9cde12e9b4fe0eb57",
                 needs_avx2=True),
)


def builds_for_this_machine(sys_platform: Optional[str] = None, machine: Optional[str] = None) -> list:
    """The builds that can run here, GPU build first. Empty when CrispASR has no build for this machine."""
    sys_platform = sys_platform or sys.platform
    machine = machine or platform.machine()
    found = [b for b in PROGRAM_BUILDS if b.platform == sys_platform and machine in b.machines]
    return sorted(found, key=lambda b: b.gpu is None)


def build_by_key(key: str, sys_platform: Optional[str] = None, machine: Optional[str] = None):
    return next((b for b in builds_for_this_machine(sys_platform, machine) if b.key == key), None)


@dataclass(frozen=True)
class ModelFile:
    name: str            # the short name; the --pass1-model / --pass2-model value (Q22)
    label: str           # plain words for messages
    backend: str         # CrispASR --backend
    repo_id: str
    filename: str
    revision: str        # the pinned commit
    size: int            # bytes
    sha256: str
    licence: str
    extra_args: tuple = field(default_factory=tuple)


MODELS = {m.name: m for m in (
    ModelFile("parakeet-tdt-0.6b-ja", "Parakeet TDT 0.6B ja (q8_0)", "parakeet",
              "cstr/parakeet-tdt-0.6b-ja-GGUF", "parakeet-tdt-0.6b-ja-q8_0.gguf",
              "d9e3ba65a6579796389ea89e5939509ed257f972", 673_554_880,
              "5a61e6c7d956c3c72a76fafcd798cac0c9ea66d0e29b3910cd04865a1e42cc17",
              "CC-BY-4.0 (nvidia/parakeet-tdt_ctc-0.6b-ja)"),
    # Without --punc-model none the program fetches a 58 MB punctuation model by itself (run P1).
    # That the switch stops the download is not yet tested (U7).
    ModelFile("parakeet-ctc-1.1b-ja", "Parakeet CTC 1.1B ja (q8_0)", "fastconformer-ctc",
              "cstr/parakeet-ctc-1.1b-ja-GGUF", "parakeet-ctc-1.1b-ja-q8_0.gguf",
              "033e8bf08d60a582b47a92bba07ca9e392982861", 1_137_330_464,
              "4fca7a83833012d515b4a71ab3df9a0abcf3362107369850d89dda962dfd34b4",
              "Apache-2.0 (grider-transwithai/parakeet-ctc-1.1b-ja)",
              extra_args=("--punc-model", "none")),
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

# The models each 1.10.0 pipeline offers its users (W118 4, plan 3.3), and its default. Cohere-CrispASR
# has no default until the step 2b measurement picks one of its two candidates; until then a run must
# name its model. These lists feed the command line (--passN-model), --help and --check.
PIPELINE_MODELS = {
    "parakeet-crispasr": ("parakeet-tdt-0.6b-ja",),
    "cohere-crispasr": ("cohere-asr-ja-v0.1", "cohere-asr-ja"),
}
PIPELINE_DEFAULT_MODEL = {
    "parakeet-crispasr": "parakeet-tdt-0.6b-ja",
    "cohere-crispasr": None,
}
# Measured in step 2b but not offered: the plan offers the Parakeet CTC 1.1B "only if step 2b supports
# it" (plan 3.3, Q13). Only the pipeline constructor reaches it -- the step 2b measurement script --
# never the command line, --help or --check.
MEASUREMENT_MODELS = {
    "parakeet-crispasr": ("parakeet-ctc-1.1b-ja",),
    "cohere-crispasr": (),
}
CRISPASR_PIPELINES = tuple(PIPELINE_MODELS)


def model_for(pipeline: str, name: Optional[str], include_measurement: bool = False) -> ModelFile:
    """The pinned model a pipeline runs: the named one, else the pipeline's default.

    ``include_measurement`` also accepts the models measured in step 2b but not yet offered.
    Raises ValueError, with the offered choices in its text, for an unknown name or a missing choice.
    """
    choices = PIPELINE_MODELS[pipeline]
    allowed = choices + (MEASUREMENT_MODELS[pipeline] if include_measurement else ())
    if not name:
        name = PIPELINE_DEFAULT_MODEL[pipeline]
        if name is None:
            raise ValueError(f"{pipeline} has no default model yet; choose one of: {', '.join(choices)}")
    if name not in allowed:
        raise ValueError(f"'{name}' is not a model of {pipeline}; choose one of: {', '.join(choices)}")
    return MODELS[name]
