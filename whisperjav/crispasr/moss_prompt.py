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
  --chunk-minutes 0                  -> the stretches MOSS hears: the Segmenter tab (firered-vad, a group ends at a
                                        silence over 5 s, at most 240 s, no shortest group) and the Scene tab
                                        (semantic, 90-240 s); crispasr/pass_settings.py (W482; before it W426)
  include_speaker_labels             -> not used: no diarization in 1.10.x (W113 X7); the labels are dropped
  --do-sample --temperature 0.7      -> do_sample (default on), temperature (default 0.7), with top_k 50 (HF's
                                        default, which CrispASR asks for, #564) and a fixed seed 42 so a run can be
                                        repeated (0 = a new seed each run); since 1.10.1 with crispasr 0.8.50 (owner
                                        W492 M1, W501, W502). Greedy (do_sample off) stays a choice.
  (beam search)                      -> beam_size, default 1 (off); above 1 CrispASR runs beam search and does not
                                        sample (owner W501: shown for the user to change)

The settings are keys of the pass's Customize values: the GUI's MOSS tab and ``--passN-params`` on the command
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

DEFAULTS: Dict[str, Any] = {"prompt_language": "en", "language_hint": "Japanese", "hotwords": "", "max_new_tokens": 0,
                            # The decoding (owner W492 M1: "do_sample=True, temperature=0.7, top_k=50"; W502: seed 42)
                            "do_sample": True, "temperature": 0.7, "top_k": 50, "seed": 42, "beam_size": 1}
PARAMS = tuple(DEFAULTS)
DECODING = ("do_sample", "temperature", "top_k", "seed", "beam_size")
MAX_BEAM = 8
MAX_SEED = 2 ** 31 - 1
MAX_TEMPERATURE = 1.0

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
    # The decoding (owner W501 M1, W502; texts drafted with the step budget after W501).
    "do_sample": {"type": "toggle", "label": "Sampling",
                  "description": "On: MOSS picks each next piece of text from the likely ones (temperature 0.7, "
                                 "top-k 50, as the reference script). Repeats become shorter and rarer. Off: MOSS "
                                 "always takes the most likely piece (greedy).",
                  "original_type": "bool"},
    # At most 1.0: the GUI's shared limits table holds every "temperature" control at 0-1.0 on Apply (app.js).
    "temperature": {"type": "number", "label": "Temperature", "min": 0.05, "max": 1.0, "step": 0.05,
                    "description": "How freely MOSS picks. Lower stays closer to the most likely text. Used only "
                                   "when Sampling is on.",
                    "original_type": "float"},
    "top_k": {"type": "number", "label": "Top-k", "min": 0, "max": 200, "step": 1,
              "description": "MOSS picks only among this many most likely pieces of text; 0 = no limit. Used only "
                             "when Sampling is on.",
              "original_type": "int"},
    "seed": {"type": "number", "label": "Seed", "min": 0, "max": MAX_SEED, "step": 1,
             "description": "The same seed and settings give the same text again; 0 = a new seed every run. Used "
                            "only when Sampling is on.",
             "original_type": "int"},
    "beam_size": {"type": "number", "label": "Beam size", "min": 1, "max": MAX_BEAM, "step": 1,
                  "description": "1 = off. Above 1, MOSS weighs that many candidate texts and keeps the best; "
                                 "Sampling is then not used. Slower.",
                  "original_type": "int"},
}


@dataclass(frozen=True)
class MossSettings:
    """One pass's MOSS settings (DEFAULTS, then the pass's Customize values)."""
    prompt_language: str = DEFAULTS["prompt_language"]
    language_hint: str = DEFAULTS["language_hint"]
    hotwords: str = DEFAULTS["hotwords"]
    max_new_tokens: int = DEFAULTS["max_new_tokens"]
    do_sample: bool = DEFAULTS["do_sample"]
    temperature: float = DEFAULTS["temperature"]
    top_k: int = DEFAULTS["top_k"]
    seed: int = DEFAULTS["seed"]
    beam_size: int = DEFAULTS["beam_size"]

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
        do_sample = _flag(values["do_sample"])
        temperature = _number(values["temperature"], "temperature", float)
        if do_sample and not 0 < temperature <= MAX_TEMPERATURE:
            raise ValueError(f"temperature must be more than 0 and at most {MAX_TEMPERATURE}, not {temperature}")
        top_k = _number(values["top_k"], "top_k", int)
        if top_k < 0:
            raise ValueError(f"top_k must be 0 (no limit) or more, not {top_k}")
        seed = _number(values["seed"], "seed", int)
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be 0 (a new seed each run) up to {MAX_SEED}, not {seed}")
        beam = _number(values["beam_size"], "beam_size", int)
        if not 1 <= beam <= MAX_BEAM:
            raise ValueError(f"beam_size must be 1 (off) up to {MAX_BEAM}, not {beam}")
        return cls(language, str(values["language_hint"] or "").strip(), str(values["hotwords"] or "").strip(),
                   tokens, do_sample, temperature, top_k, seed, beam)

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

    def decoding(self) -> Dict[str, Any]:
        """How MOSS picks its next token, for the engine (crispasr/engine.py BindingEngine, CrispASR 0.8.50)."""
        return {"do_sample": self.do_sample, "temperature": self.temperature, "top_k": self.top_k, "seed": self.seed,
                "beam_size": self.beam_size}

    def as_dict(self) -> Dict[str, Any]:
        return {"prompt_language": self.prompt_language, "language_hint": self.language_hint,
                "hotwords": self.hotwords, "max_new_tokens": self.max_new_tokens, **self.decoding()}


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"do_sample '{value}' is not true or false")


def _number(value: Any, key: str, kind):
    if isinstance(value, bool):
        raise ValueError(f"{key} '{value}' is not a {'whole ' if kind is int else ''}number")
    try:
        number = kind(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} '{value}' is not a {'whole ' if kind is int else ''}number") from None
    if kind is int and isinstance(value, float) and value != int(value):
        raise ValueError(f"{key} '{value}' is not a whole number")
    return number


def help_text() -> str:
    """The MOSS keys for --passN-params' help."""
    return ('For moss-crispasr also its MOSS settings: "prompt_language" (en, the default, or zh), "language_hint" '
            '(default Japanese), "hotwords" (comma-separated, default none), "max_new_tokens" (0, the default, '
            '= by the length of the audio), "do_sample" (true, the default; false = greedy), "temperature" (0.7), '
            '"top_k" (50; 0 = no limit), "seed" (42; 0 = a new seed each run) and "beam_size" (1 = off; above 1 '
            'beam search, without sampling).')
