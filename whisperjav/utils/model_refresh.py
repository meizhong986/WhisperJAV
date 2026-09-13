"""
Model-refresh policy (v1.9.2, owner CFF1): reload the recogniser after it has been
handed a budget of scene audio.

Background (#394 family). Two recorded observations, both same-instance: in #302 the
reporter's control showed four minutes of audio that returned nothing inside a long run
transcribing normally as a separate job; the fixed probe (#394, 2026-09-02) showed one
instance transcribing identical audio for 111 iterations and then returning zero
segments for the remaining 89. Hypothesis (not established): cumulative use of one
instance is the trigger; refreshing bounds the exposure. Containment, not a fix.

This module only decides *when* a refresh is due; each pipeline
performs the refresh itself at the next scene boundary (Balanced: a fresh worker
process behind ``RemoteFasterWhisperASR``; Fidelity: a fresh ``WhisperProASR``).

Budget granularity is the scene (owner D2): the sum of the durations of the scenes
handed to the instance, measured in minutes of audio. Nothing finer.
"""
from __future__ import annotations

# Owner CFF1: "after a model instance been used for more than 20 minutes".
DEFAULT_MODEL_REFRESH_AUDIO_MINUTES = 20.0


class ModelRefreshPolicy:
    """Counts scene audio handed to one recogniser instance and says when to refresh.

    ``budget_audio_s <= 0`` disables the policy: ``due()`` is always False.
    """

    def __init__(self, budget_audio_s: float):
        try:
            budget = float(budget_audio_s or 0.0)
        except (TypeError, ValueError):
            budget = 0.0
        self.budget_audio_s = max(0.0, budget)
        self.enabled = self.budget_audio_s > 0.0
        self.consumed_audio_s = 0.0
        self.epoch = 1            # 1-based generation of the current instance
        self.refresh_count = 0    # how many refreshes have happened so far

    @classmethod
    def from_minutes(cls, minutes: float) -> "ModelRefreshPolicy":
        try:
            m = float(minutes or 0.0)
        except (TypeError, ValueError):
            m = 0.0
        return cls(m * 60.0)

    def record(self, audio_s: float) -> None:
        """Account for one scene's audio (seconds) handed to the current instance."""
        if not self.enabled:
            return
        try:
            self.consumed_audio_s += max(0.0, float(audio_s or 0.0))
        except (TypeError, ValueError):
            pass

    def due(self) -> bool:
        """True when the current instance has consumed at least the budget."""
        return self.enabled and self.consumed_audio_s >= self.budget_audio_s

    def reset(self, refresh: bool = True) -> None:
        """Start a new instance generation.

        ``refresh=True`` (default): the budget was spent and a fresh instance replaced
        the old one — counts as a refresh. ``refresh=False``: a new instance for another
        reason (the worker died and was restarted) — new generation, budget restarts,
        but it is not counted as a refresh.
        """
        self.consumed_audio_s = 0.0
        self.epoch += 1
        if refresh:
            self.refresh_count += 1

    @property
    def consumed_minutes(self) -> float:
        return self.consumed_audio_s / 60.0

    @property
    def budget_minutes(self) -> float:
        return self.budget_audio_s / 60.0
