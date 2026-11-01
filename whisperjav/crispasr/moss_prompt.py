"""MOSS-CrispASR's model settings: the instruction MOSS is given, built as the owner's reference script builds it (1.10.1).

The owner's reference (W423, W428 I3): ``moss_transcribe_diarize_hybrid_v3_grok.py --preset jav-optimized
--chunk-minutes 0 --language-hint Japanese --do-sample --temperature 0.7 --hotwords "..."``. What those options switch
on, and where each one is in WhisperJAV (owner W428 I1 4: "I want those relevant settings to be exposed for
adjustments"):

  prompt_lang "en" (jav-optimized)   -> prompt_language, default "en"
  --language-hint Japanese           -> language_hint, default "Japanese"
  --hotwords "..."                   -> hotwords, default none
  max_new_tokens 65536 per chunk     -> max_new_tokens, default 0 = by the length of the audio, at the reference's
                                        rate (65,536 tokens per 50-minute chunk, about 21.8 a second; owner W406 i1)
  --chunk-minutes 0                  -> the stretches MOSS hears: the Segmenter tab (silero-v6.2, 0.7, 5 s, 120 s,
                                        70 s) and the Scene tab (silero, 420 s); crispasr/pass_settings.py (W426)
  include_speaker_labels             -> not used: no diarization in 1.10.x (W113 X7); the labels are dropped
  --do-sample --temperature 0.7      -> not available: CrispASR's MOSS decoder (0.8.44, and its main branch on
                                        2026-11-01) always takes the most likely next token; it has no sampling

The four settings are keys of the pass's Customize values: the GUI's MOSS tab and ``--passN-params`` on the command
line carry them, one mechanism for both. ``build_prompt`` is the reference's ``build_prompt``; the text goes to
CrispASR whole, through ``Session.set_ask`` (which replaces CrispASR's own Chinese instruction).

Light: standard library only.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

# The reference's DEFAULT_PROMPT_EN, and its DEFAULT_PROMPT_ZH (the MOSS-Transcribe-Diarize package's DEFAULT_PROMPT,
# which is also CrispASR's own instruction, moss_transcribe_diarize.cpp).
PROMPT_EN = ("Transcribe the audio. For each segment, start with the timestamp and speaker ID "
             "([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp.")
PROMPT_ZH = ("请将音频转写为文本，每一段需以起始时间戳和说话人编号"
             "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
             "并在段末标注结束时间戳，以清晰标明该段语音范围。")
PROMPT_LANGUAGES = ("en", "zh")

# The reference's rate: 65,536 tokens per 50-minute chunk (owner W406 i1).
TOKENS_PER_SECOND = 65536 / 3000

DEFAULTS: Dict[str, Any] = {"prompt_language": "en", "language_hint": "Japanese", "hotwords": "", "max_new_tokens": 0}
PARAMS = tuple(DEFAULTS)

# The MOSS tab of the Customize window, in the Segmenter tab's schema form (app.js generateSchemaControl). Drafted
# under W428 I2; the texts are the owner's to approve.
GUI_SCHEMA: Dict[str, Dict[str, Any]] = {
    "prompt_language": {"type": "dropdown", "label": "Instruction language",
                        "options": [{"value": "en", "label": "English (as the jav-optimized preset)"},
                                    {"value": "zh", "label": "Chinese (the model's own)"}],
                        "description": "The language of the instruction MOSS is given with the audio.",
                        "original_type": "str"},
    "language_hint": {"type": "text", "label": "Language hint",
                      "description": "Added to the instruction (\"Language hint: ...\"; in Chinese \"语言提示：...\"). "
                                     "Empty = none.",
                      "original_type": "str"},
    "hotwords": {"type": "text", "label": "Hotwords",
                 "description": "Names and words MOSS should expect, separated by commas. Added to the instruction "
                                "(\"Hotwords: ...\"; in Chinese \"热词提示：...\"). Empty = none.",
                 "original_type": "str"},
    "max_new_tokens": {"type": "number", "label": "Token limit per call", "min": 0, "max": 65536, "step": 1,
                       "description": "The most tokens MOSS may write for one stretch. 0 = by the stretch's length "
                                      "(about 22 a second, the reference's rate).",
                       "original_type": "int"},
}


@dataclass(frozen=True)
class MossSettings:
    """One pass's MOSS settings (DEFAULTS, then the pass's Customize values)."""
    prompt_language: str = DEFAULTS["prompt_language"]
    language_hint: str = DEFAULTS["language_hint"]
    hotwords: str = DEFAULTS["hotwords"]
    max_new_tokens: int = DEFAULTS["max_new_tokens"]

    @classmethod
    def from_params(cls, params: Optional[Mapping[str, Any]]) -> "MossSettings":
        """The settings from a pass's Customize values (``--passN-params``); keys of other tabs are ignored.
        Raises ValueError for a value that cannot be used, naming the key."""
        values = dict(DEFAULTS)
        for key in PARAMS:
            if params and params.get(key) is not None:
                values[key] = params[key]
        language = str(values["prompt_language"]).strip().lower()
        if language not in PROMPT_LANGUAGES:
            raise ValueError(f"prompt_language '{values['prompt_language']}' is not one of: "
                             f"{', '.join(PROMPT_LANGUAGES)}")
        try:
            tokens = int(values["max_new_tokens"])
        except (TypeError, ValueError):
            raise ValueError(f"max_new_tokens '{values['max_new_tokens']}' is not a whole number") from None
        if tokens < 0:
            raise ValueError(f"max_new_tokens must be 0 (by the audio's length) or more, not {tokens}")
        return cls(language, str(values["language_hint"] or "").strip(), str(values["hotwords"] or "").strip(),
                   tokens)

    def prompt(self) -> str:
        """The instruction, as the reference's build_prompt makes it (no custom prompt)."""
        base = PROMPT_EN if self.prompt_language == "en" else PROMPT_ZH
        extras = []
        if self.language_hint:
            extras.append(f"语言提示：{self.language_hint}" if self.prompt_language == "zh"
                          else f"Language hint: {self.language_hint}.")
        if self.hotwords:
            extras.append(f"热词提示：{self.hotwords}" if self.prompt_language == "zh"
                          else f"Hotwords: {self.hotwords}")
        return base.rstrip() + (" " + " ".join(extras) if extras else "")

    def fixed_token_limit(self) -> Optional[int]:
        """A fixed token limit for every call, or None: the limit then follows the audio's length at
        TOKENS_PER_SECOND (the pipeline's row, EngineRow.tokens_per_second)."""
        return self.max_new_tokens or None

    def as_dict(self) -> Dict[str, Any]:
        return {"prompt_language": self.prompt_language, "language_hint": self.language_hint,
                "hotwords": self.hotwords, "max_new_tokens": self.max_new_tokens}


def help_text() -> str:
    """The MOSS keys for --passN-params' help."""
    return ('For moss-crispasr also its MOSS settings: "prompt_language" (en, the default, or zh), "language_hint" '
            '(default Japanese), "hotwords" (comma-separated, default none) and "max_new_tokens" (0, the default, '
            '= by the length of the audio).')
