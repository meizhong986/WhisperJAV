"""The ensemble rule that lowers a typed "aggressive" after a Fidelity pass 1.

When pass 1 is Fidelity and pass 2 is Balanced at aggressive, pass 2 produced
empty or badly truncated subtitles in about two thirds of the trials that led to
this rule. The rule runs pass 2 at balanced instead and says so on the console.

The owner decided on 11 September 2026 to keep it for v1.9.2. These tests hold it
to its exact shape: it must fire for that combination and no other, it must
change nothing else, and the user must be told when it fires -- a setting that is
silently overridden is worse than one that is not overridden at all.
"""

import logging

import pytest

from whisperjav.ensemble.safety_caps import apply_ensemble_safety_caps, list_active_caps


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def log():
    logger = logging.getLogger("whisperjav.test.safety_caps")
    handler = _Collector()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield logger, handler.records
    finally:
        logger.removeHandler(handler)


def _pass(pipeline, sensitivity):
    return {"pipeline": pipeline, "sensitivity": sensitivity, "model": "large-v2"}


class TestTheRuleFires:
    def test_fidelity_then_balanced_aggressive_is_lowered_to_balanced(self, log):
        logger, records = log
        pass1 = _pass("fidelity", "balanced")
        pass2 = _pass("balanced", "aggressive")

        _p1, capped = apply_ensemble_safety_caps(pass1, pass2, logger=logger)

        assert capped["sensitivity"] == "balanced"
        assert pass2["sensitivity"] == "aggressive", "the caller's dict must not be mutated"

    def test_the_user_is_told_when_it_fires(self, log):
        """A silently overridden setting is worse than one that is not overridden."""
        logger, records = log
        apply_ensemble_safety_caps(
            _pass("fidelity", "balanced"), _pass("balanced", "aggressive"), logger=logger
        )

        warnings = [r for r in records if r.levelno >= logging.WARNING]
        assert warnings, "the downgrade must be announced at WARNING or above"

        message = warnings[0].getMessage()
        assert "aggressive" in message and "balanced" in message, message
        assert "pass2" in message or "pass 2" in message, message

    def test_it_falls_back_to_stderr_when_no_logger_is_given(self, capsys):
        apply_ensemble_safety_caps(
            _pass("fidelity", "balanced"), _pass("balanced", "aggressive"), logger=None
        )
        assert "aggressive" in capsys.readouterr().err

    def test_nothing_else_in_the_pass_config_is_changed(self, log):
        logger, _records = log
        pass2 = _pass("balanced", "aggressive")
        pass2["speech_segmenter"] = "whisperseg"

        _p1, capped = apply_ensemble_safety_caps(_pass("fidelity", "balanced"), pass2, logger=logger)

        assert capped["model"] == "large-v2"
        assert capped["speech_segmenter"] == "whisperseg"
        assert set(capped) == set(pass2)


class TestTheRuleDoesNotFire:
    """Every neighbouring combination must be left exactly as the user typed it."""

    @pytest.mark.parametrize("pass1_pipeline,pass2_pipeline,pass2_sensitivity", [
        # The owner's own acceptance runs: Balanced first. No evidence covers these.
        ("balanced", "balanced", "aggressive"),
        ("balanced", "fidelity", "aggressive"),
        # Fidelity first, but pass 2 is not Balanced
        ("fidelity", "fidelity", "aggressive"),
        ("fidelity", "fast", "aggressive"),
        # Fidelity then Balanced, but not at aggressive
        ("fidelity", "balanced", "balanced"),
        ("fidelity", "balanced", "conservative"),
    ])
    def test_other_combinations_keep_the_typed_sensitivity(
        self, log, pass1_pipeline, pass2_pipeline, pass2_sensitivity
    ):
        logger, records = log
        pass2 = _pass(pass2_pipeline, pass2_sensitivity)

        _p1, capped = apply_ensemble_safety_caps(
            _pass(pass1_pipeline, "balanced"), pass2, logger=logger
        )

        assert capped["sensitivity"] == pass2_sensitivity, (
            f"{pass1_pipeline} -> {pass2_pipeline} at {pass2_sensitivity} must be left alone"
        )
        assert not [r for r in records if r.levelno >= logging.WARNING], (
            "nothing fired, so nothing should be announced"
        )

    def test_a_single_pass_ensemble_is_left_alone(self, log):
        logger, records = log
        pass1 = _pass("fidelity", "aggressive")

        p1, p2 = apply_ensemble_safety_caps(pass1, None, logger=logger)

        assert p1 is pass1
        assert p2 is None
        assert not records


class TestTheRuleIsStillDeclared:
    def test_exactly_one_cap_rule_is_active(self):
        """New rules need measured evidence; this test makes adding one deliberate."""
        caps = list_active_caps()
        assert len(caps) == 1, f"expected one cap rule, found: {[c['name'] for c in caps]}"

    def test_the_rule_records_the_decision_to_keep_it(self):
        """v1.9.2: the note used to say the rule's future was an open question.

        The owner settled it on 11 September 2026, and the rationale is printed to
        the user when the rule fires, so it has to read as a decision, not a doubt.
        """
        rationale = list_active_caps()[0]["rationale"]

        assert "open question" not in rationale.lower(), rationale
        assert "2026" in rationale, "the rationale should date the decision"
        assert "KEPT" in rationale or "kept" in rationale, rationale
