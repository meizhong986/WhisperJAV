"""R-SCN settings: every value the scene detector decides with, in one typed, checked object (owner W450 1).

Design: docs/plans/design/2026-11-02-rscn-design.md, section 4. The first preset is the MOSS-CrispASR one (W450 2e,
W451 C2): scenes of 70-420 s, preferably 120-360 s. It is the first iteration's preset, not a proven optimum.

The pass's overrides arrive as ``rscn_<name>`` keys of the Customize values (``--passN-params``); ``from_params``
takes those keys, refuses an unknown ``rscn_`` key or a value that cannot be used, and ignores every other key (the
other tabs' values). The correctness rules (every sample in exactly one scene, no gap, no overlap) are not settings.

Light: standard library only.
"""

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, Mapping, Optional, Tuple

PREFIX = "rscn_"
PICTURE_MODES = ("auto", "on", "off")


@dataclass(frozen=True)
class RSCNSettings:
    # --- Length (W451 C2, W450 2e, W426 4) ---------------------------------------------------------------------
    min_scene_s: float = 70.0
    preferred_min_s: float = 120.0
    preferred_max_s: float = 360.0
    max_scene_s: float = 420.0
    # Cost of a scene outside the preferred range: length_weight * (relative distance to the range)^2.
    length_weight: float = 8.0

    # --- Analysis (the reference's values) ---------------------------------------------------------------------
    analysis_rate: int = 16000
    frame_ms: float = 40.0
    feature_hop_ms: float = 20.0
    novelty_hop_ms: float = 100.0
    speech_band_low_hz: float = 120.0
    speech_band_high_hz: float = 3900.0
    floor_window_s: float = 4.0
    floor_percentile: float = 10.0
    level_floor_db: float = -90.0

    # --- Change strength (absolute: each measure divided by a fixed unit; study 5a (2)) -------------------------
    # MOSS reads the change at a coarse scale (v6 8.2): breaths and backchannels inside a conversation do not count.
    novelty_scales_s: Tuple[float, ...] = (20.0, 60.0)
    novelty_scale_weights: Tuple[float, ...] = (1.0, 2.0)
    # The shortest window used near an edge of a region (each side).
    edge_min_window_s: float = 1.0
    unit_level_db: float = 6.0
    unit_floor_db: float = 6.0
    unit_above_floor_db: float = 6.0
    unit_centroid: float = 0.05
    unit_flatness: float = 0.1
    unit_flux: float = 0.1
    unit_band: float = 0.1
    novelty_smooth_frames: int = 3
    # NOT calibrated (design section 9): on seven Netflix clips joined end to end, 1.7 caught 3 of 6 true joins with
    # about 10 peaks per hour inside scenes reaching it; no value separated the two. A first preset to tune.
    change_threshold: float = 1.7
    peak_min_distance_s: float = 5.0
    # How far a change must stand above its surroundings to count (a shoulder of a larger change does not).
    peak_prominence: float = 0.3
    candidate_merge_s: float = 1.0

    # --- Placement of a cut ----------------------------------------------------------------------------------
    max_cut_shift_s: float = 8.0
    quiet_min_s: float = 0.3
    quiet_context_s: float = 20.0
    quiet_drop_db: float = 8.0
    quiet_spread_db: float = 7.0
    quiet_above_p05_db: float = 6.0
    quiet_below_p80_db: float = 5.0
    quiet_median_frames: int = 5
    quiet_low_percentile: float = 5.0
    quiet_mid_percentile: float = 80.0
    quiet_high_percentile: float = 95.0
    speech_guard_s: float = 0.1
    # With no pause near a change: among points this close to the lowest level in reach, the nearest is taken.
    placement_level_tolerance_db: float = 3.0

    # --- Costs of the choice of cuts -------------------------------------------------------------------------
    # A cut at a strong change (r = strength / threshold >= 1) costs -change_reward * min(r, cap): at most -1.0 with
    # the defaults, so it never pays for a scene far outside the preferred range or for a cut in speech.
    change_reward: float = 0.5
    change_reward_cap: float = 2.0
    forced_cut_cost: float = 2.0
    forced_strength_bonus: float = 0.5
    in_speech_cost: float = 5.0
    grid_step_s: float = 5.0

    # --- Digital silence ---------------------------------------------------------------------------------------
    digital_silence_min_s: float = 2.0
    # Audio shorter than this next to digital silence joins the silence's scene (marked).
    tiny_region_s: float = 1.0

    # --- Speech measure (W451 C3, W450 2f) -----------------------------------------------------------------------
    speech_measure: str = "firered-vad"
    speech_measure_piece_s: float = 180.0
    # The pieces overlap by this much, so speech across a piece's edge is not split there.
    speech_measure_overlap_s: float = 5.0
    speech_measure_params: Dict[str, Any] = field(default_factory=dict)
    little_speech_share: float = 0.05

    # --- Picture (W450 2g) ---------------------------------------------------------------------------------------
    picture: str = "auto"
    black_min_s: float = 1.0
    black_pixel_threshold: float = 0.10
    black_picture_threshold: float = 0.98
    black_analysis_width: int = 320
    black_confirm_margin_s: float = 1.0
    black_confirm_change: float = 0.85
    # Black within this of a digital-silence run is that run's fade, not a break of its own.
    black_silence_merge_s: float = 2.0
    # An unconfirmed (or too close) black stretch counts this many thresholds stronger.
    black_strength_bonus: float = 0.5

    def __post_init__(self) -> None:
        problem = self.problem()
        if problem:
            raise ValueError(problem)

    # ------------------------------------------------------------------------------------------------------------

    def problem(self) -> Optional[str]:
        """Why these settings cannot be used, in plain words naming the setting; None when they can."""
        p = PREFIX
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("speech_measure", "picture"):
                if not isinstance(value, str):
                    return f"{p}{f.name} must be a word, not {value!r}"
            elif f.name == "speech_measure_params":
                if not isinstance(value, dict):
                    return f"{p}{f.name} must be a JSON object, not {value!r}"
            elif f.name in ("novelty_scales_s", "novelty_scale_weights"):
                if not value or any(not _is_number(v) or v <= 0 for v in value):
                    return f"{p}{f.name} must be a list of numbers above 0, not {value!r}"
            elif not _is_number(value):
                return f"{p}{f.name} must be a number, not {value!r}"
        if not 0 < self.min_scene_s <= self.preferred_min_s <= self.preferred_max_s <= self.max_scene_s:
            return (f"the scene lengths must be in order, 0 < {p}min_scene_s <= {p}preferred_min_s <= "
                    f"{p}preferred_max_s <= {p}max_scene_s; they are {self.min_scene_s:g}, {self.preferred_min_s:g}, "
                    f"{self.preferred_max_s:g}, {self.max_scene_s:g}")
        if self.max_scene_s < 2 * self.min_scene_s:
            return (f"{p}max_scene_s ({self.max_scene_s:g}) must be at least twice {p}min_scene_s "
                    f"({self.min_scene_s:g}), so that any long stretch can be divided within the bounds")
        if not 8000 <= self.analysis_rate <= 48000:
            return f"{p}analysis_rate must be between 8000 and 48000, not {self.analysis_rate:g}"
        if not 0 < self.feature_hop_ms <= self.frame_ms:
            return f"{p}feature_hop_ms must be above 0 and at most {p}frame_ms (no gaps in the analysis)"
        for name in ("frame_ms", "feature_hop_ms", "novelty_hop_ms"):
            samples = self.analysis_rate * getattr(self, name) / 1000.0
            if abs(samples - round(samples)) > 1e-6:
                return f"{p}{name} must be a whole number of samples at {p}analysis_rate"
        ratio = self.novelty_hop_ms / self.feature_hop_ms
        if ratio < 1 or abs(ratio - round(ratio)) > 1e-6:
            return f"{p}novelty_hop_ms must be a whole multiple of {p}feature_hop_ms"
        if not 0 < self.speech_band_low_hz < self.speech_band_high_hz:
            return f"{p}speech_band_low_hz must be above 0 and below {p}speech_band_high_hz"
        if not 0 <= self.floor_percentile <= 100:
            return f"{p}floor_percentile must be between 0 and 100"
        if len(self.novelty_scale_weights) != len(self.novelty_scales_s):
            return f"{p}novelty_scale_weights must have one weight per scale in {p}novelty_scales_s"
        positive = ("length_weight", "frame_ms", "novelty_hop_ms", "floor_window_s", "unit_level_db", "unit_floor_db",
                    "unit_above_floor_db", "unit_centroid", "unit_flatness", "unit_flux", "unit_band",
                    "change_threshold", "peak_min_distance_s", "peak_prominence", "quiet_min_s", "quiet_context_s", "change_reward",
                    "change_reward_cap", "grid_step_s", "digital_silence_min_s", "speech_measure_piece_s",
                    "black_min_s", "black_confirm_change", "edge_min_window_s")
        for name in positive:
            if getattr(self, name) <= 0:
                return f"{p}{name} must be above 0, not {getattr(self, name):g}"
        not_negative = ("candidate_merge_s", "max_cut_shift_s", "quiet_drop_db", "quiet_spread_db",
                        "quiet_above_p05_db", "quiet_below_p80_db", "speech_guard_s", "placement_level_tolerance_db", "forced_cut_cost",
                        "forced_strength_bonus", "in_speech_cost", "black_confirm_margin_s", "black_analysis_width",
                        "tiny_region_s", "speech_measure_overlap_s", "black_silence_merge_s", "black_strength_bonus")
        for name in not_negative:
            if getattr(self, name) < 0:
                return f"{p}{name} must be 0 or more, not {getattr(self, name):g}"
        for name in ("novelty_smooth_frames", "quiet_median_frames"):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                return f"{p}{name} must be a whole number of 1 or more"
        if not 0 <= self.quiet_low_percentile < self.quiet_mid_percentile < self.quiet_high_percentile <= 100:
            return f"{p}quiet_low_percentile < {p}quiet_mid_percentile < {p}quiet_high_percentile, between 0 and 100"
        if max(self.novelty_scales_s) / 2 > self.min_scene_s:
            return (f"half the longest of {p}novelty_scales_s must be at most {p}min_scene_s: a change is read over "
                    f"windows that fit inside a scene")
        if self.speech_measure_overlap_s >= self.speech_measure_piece_s:
            return f"{p}speech_measure_overlap_s must be below {p}speech_measure_piece_s"
        if self.grid_step_s > self.min_scene_s / 2:
            return (f"{p}grid_step_s ({self.grid_step_s:g}) must be at most half of {p}min_scene_s, so that a "
                    f"division within the bounds always exists")
        if not 0 <= self.little_speech_share <= 1:
            return f"{p}little_speech_share must be between 0 and 1"
        if not (0 <= self.black_pixel_threshold <= 1 and 0 <= self.black_picture_threshold <= 1):
            return f"{p}black_pixel_threshold and {p}black_picture_threshold must be between 0 and 1"
        if self.picture.strip().lower() not in PICTURE_MODES:
            return f"{p}picture must be one of {', '.join(PICTURE_MODES)}, not {self.picture!r}"
        if not self.speech_measure.strip():
            return f"{p}speech_measure must name a speech segmenter backend, or none"
        return None

    # ------------------------------------------------------------------------------------------------------------

    @property
    def picture_mode(self) -> str:
        return self.picture.strip().lower()

    @property
    def speech_measure_name(self) -> str:
        return self.speech_measure.strip().lower()

    def as_dict(self) -> Dict[str, Any]:
        """The values in force, for the run record and --dump-params."""
        values = asdict(self)
        for key in ("novelty_scales_s", "novelty_scale_weights"):
            values[key] = list(values[key])
        return values

    @classmethod
    def names(cls) -> Tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def for_pipeline(cls, pipeline: Optional[str] = None,
                     params: Optional[Mapping[str, Any]] = None) -> "RSCNSettings":
        """The pipeline's preset with the pass's ``rscn_`` overrides. Raises ValueError naming the key."""
        base = PRESETS.get(str(pipeline or "").strip().lower(), PRESETS["default"])
        return base.with_params(params)

    def with_params(self, params: Optional[Mapping[str, Any]]) -> "RSCNSettings":
        overrides = overrides_from_params(params)
        if not overrides:
            return self
        return replace(self, **overrides)


def overrides_from_params(params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The ``rscn_`` keys of a pass's Customize values, as typed settings. Other keys are ignored.
    Raises ValueError for an unknown ``rscn_`` key or a value of the wrong kind, naming the key."""
    if not params:
        return {}
    known = {f.name: f for f in fields(RSCNSettings)}
    out: Dict[str, Any] = {}
    for key, value in params.items():
        if not isinstance(key, str) or not key.startswith(PREFIX):
            continue
        name = key[len(PREFIX):]
        if name not in known:
            raise ValueError(f"{key} is not an R-SCN setting; known settings: "
                             + ", ".join(PREFIX + n for n in sorted(known)))
        if value is None:
            continue
        out[name] = _coerce(key, name, value, getattr(RSCNSettings(), name))
    return out


def _coerce(key: str, name: str, value: Any, default: Any) -> Any:
    if isinstance(default, tuple):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = [value]
        if not isinstance(value, (list, tuple)) or not all(_is_number(v) for v in value):
            raise ValueError(f"{key} must be a list of numbers, not {value!r}")
        return tuple(float(v) for v in value)
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a JSON object, not {value!r}")
        return dict(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a word, not {value!r}")
        return value.strip().lower()
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{key} must be a number, not {value!r}")
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"{key} must be a number, not {value!r}") from None
    if isinstance(default, int):
        if number != int(number):
            raise ValueError(f"{key} must be a whole number, not {value!r}")
        return int(number)
    return number


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value \
        and value not in (float("inf"), float("-inf"))


# The first preset (W450 2e): MOSS-CrispASR's. In this build rscn is accepted for moss-crispasr only
# (crispasr/pass_settings.rscn_problem); "default" serves direct callers and tests.
PRESETS: Dict[str, RSCNSettings] = {
    "default": RSCNSettings(),
    "moss-crispasr": RSCNSettings(),
}
