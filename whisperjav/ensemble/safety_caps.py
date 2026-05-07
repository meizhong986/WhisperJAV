"""
Ensemble configuration safety caps.

This module enforces overrides on known-unstable ensemble pipeline combinations
at the orchestration layer, BEFORE any pass runs. The single source of truth
for "which combinations are auto-corrected and why" lives in this file.

WHY THIS EXISTS
---------------
Some pass1 + pass2 + sensitivity combinations have been empirically shown to
produce non-deterministic catastrophic ASR truncation in pass 2 — pass 2 returns
~14 entries instead of the expected ~50 on a 293-second JAV reference clip,
intermittently across runs (~67% catastrophic rate observed in 3 trials).

The full investigation (T142 / T143 / T144_α / T144_β / T144_δ / T146 tests +
forensic log/code analysis + git archaeology + external agent reviews + agent
falsification) is captured in:

    docs/plans/V1814_T142_NONDETERMINISM_INVESTIGATION.md

The architectural root cause is non-determinism in CTranslate2's GPU operations
(cuBLAS workspace allocation, multi-stream kernel scheduling, sampling-fallback
RNG) that propagates through faster-whisper's `temperature=(0.0, 0.17)` fallback
path and the aggressive sensitivity preset's `no_speech_threshold=0.84` gate.
The architectural fix is deferred to v1.9.0+ pending deeper investigation; for
v1.8.14 we apply a configuration-level cap that avoids triggering the unstable
code path.

SCOPE OF v1.8.14 CAP
--------------------
ONE cap, narrowly scoped, empirically grounded:

    pass1=fidelity AND pass2=balanced AND pass2_sensitivity=aggressive
        ==> auto-downgrade pass2_sensitivity to "balanced"

Why this specific combination:
  - T142 (fidelity → balanced + aggressive): catastrophic (14 entries)
  - T146 (fidelity → balanced + aggressive): catastrophic (13 entries)
  - T143 (balanced → balanced + aggressive): correct (50 entries) — same audio
  - T144_α (fidelity → balanced + aggressive): correct (50 entries) — non-determ
  - T144_δ (fast/faster → balanced + aggressive): correct
  - The "balanced" sensitivity preset has temperature=[0.0] (no fallback)
    and no_speech_threshold=0.71 (vs 0.84) — this combination has never
    catastrophed in our evidence.

Why we do NOT extend to other pass1 modes:
  - qwen → balanced + aggressive: not yet empirically tested
  - transformers → balanced + aggressive: not yet empirically tested
  - We add caps only with empirical evidence, not on architectural speculation.
  - Future cap additions: edit only the CAP_RULES list below.

WHO THIS AFFECTS
----------------
- GUI users running ensemble mode with default sensitivity (aggressive)
- CLI users invoking `--ensemble --pass1-pipeline fidelity --pass2-pipeline balanced
  --pass2-sensitivity aggressive`
- Notebook users (Kaggle, Colab) using ensemble mode with the same combination

Single mode (`whisperjav --mode balanced --sensitivity aggressive`) does NOT
go through ensemble pass1→pass2 dynamics and is NOT affected.

VALIDATION
----------
The v1.8.14 fix can be empirically validated by running:

    python tools/ensemble_failure_rate_suite.py --media <clip> --validate-safety-cap --runs 10

Expected outcome:
  - A_fid_bal (uncapped, baseline):  some % catastrophic
  - E_fid_bal_BAL (capped):          0% catastrophic
  - F_fid_bal_CON (extra-safe):      0% catastrophic

The diff between A and E demonstrates the cap is necessary AND sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Cap definitions — single source of truth
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CapRule:
    """A single safety cap rule. Append a new instance to CAP_RULES to add a cap."""
    name: str
    pass1_pipeline: str
    pass2_pipeline: str
    pass2_sensitivity_match: str       # only triggers if pass2 sensitivity equals this
    pass2_sensitivity_replacement: str  # auto-replace with this
    rationale: str                      # one-line summary for log + tracker
    memo_section: str                   # § reference in V1814_T142_NONDETERMINISM_INVESTIGATION.md


CAP_RULES: list[_CapRule] = [
    _CapRule(
        name="fid_bal_aggressive_downgrade",
        pass1_pipeline="fidelity",
        pass2_pipeline="balanced",
        pass2_sensitivity_match="aggressive",
        pass2_sensitivity_replacement="balanced",
        rationale=(
            "pass1=fidelity + pass2=balanced + sensitivity=aggressive is empirically "
            "known to produce intermittent catastrophic ASR truncation in pass 2 "
            "(~67% rate in early trials). Auto-downgrading sensitivity to 'balanced' "
            "removes the temperature=0.17 fallback path and the high no_speech_threshold, "
            "eliminating the catastrophic-empty-scene manifestation."
        ),
        memo_section="§15",
    ),
    # Future caps go here — append _CapRule instances. Keep one rule per
    # empirically-verified unstable combination. Do NOT add caps based on
    # architectural speculation; wait for empirical evidence.
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_ensemble_safety_caps(
    pass1_config: dict,
    pass2_config: Optional[dict],
    logger=None,
) -> tuple[dict, Optional[dict]]:
    """
    Apply known-unstable-combination overrides to the ensemble pass configs.

    Called once at orchestration time, AFTER pass1_config and pass2_config are
    fully built from CLI args / GUI selection / config files, BEFORE the
    EnsembleOrchestrator is constructed.

    Each cap rule in CAP_RULES is checked in order. When a rule matches, the
    matching field is auto-replaced and a WARNING is logged (visible to all
    users — GUI, CLI, notebook).

    Args:
        pass1_config: dict of pass-1 settings (must contain 'pipeline' key).
        pass2_config: dict of pass-2 settings, or None if ensemble has no pass 2.
        logger: optional logger. If provided, fired warnings go through it; else
                stderr fallback via print(file=sys.stderr).

    Returns:
        Tuple of (pass1_config, pass2_config) with any caps applied.
        Pass1_config is returned for symmetry — currently no caps modify it.

    Notes:
        - This function does NOT mutate the input dicts. It returns new dicts
          when caps fire, or the original dicts when no caps match.
        - Adding a new cap is one new entry in CAP_RULES. No call-site changes.
        - The logger.warning calls are intentionally verbose (multi-line) so
          users see the override clearly in the console.
    """
    if pass2_config is None:
        # No pass 2 configured — ensemble degenerate to single pass, no caps apply
        return pass1_config, pass2_config

    pass1_pipeline = pass1_config.get("pipeline")
    pass2_pipeline = pass2_config.get("pipeline")
    pass2_sensitivity = pass2_config.get("sensitivity")

    # Iterate over rules in declaration order. Multiple rules could in principle
    # match for the same call; we apply ALL matching rules sequentially. In
    # current v1.8.14 there is one rule, but the structure supports growth.
    new_pass2_config = pass2_config
    for rule in CAP_RULES:
        if (pass1_pipeline == rule.pass1_pipeline
                and pass2_pipeline == rule.pass2_pipeline
                and pass2_sensitivity == rule.pass2_sensitivity_match):

            # Build a NEW dict (no mutation of caller's input)
            new_pass2_config = dict(new_pass2_config)
            old_value = new_pass2_config["sensitivity"]
            new_pass2_config["sensitivity"] = rule.pass2_sensitivity_replacement

            msg = (
                f"\n[Ensemble safety cap '{rule.name}'] "
                f"Auto-downgrading pass2 sensitivity: "
                f"'{old_value}' → '{rule.pass2_sensitivity_replacement}'.\n"
                f"  Reason: {rule.rationale}\n"
                f"  See: docs/plans/V1814_T142_NONDETERMINISM_INVESTIGATION.md "
                f"{rule.memo_section}"
            )

            if logger is not None:
                logger.warning(msg)
            else:
                # Fallback if no logger provided — print to stderr so users still see it
                import sys
                print(msg, file=sys.stderr)

            # Update working values so subsequent rules see post-cap state
            pass2_sensitivity = rule.pass2_sensitivity_replacement

    return pass1_config, new_pass2_config


# ---------------------------------------------------------------------------
# Introspection helpers (for tracker / debugging / future expansion)
# ---------------------------------------------------------------------------

def list_active_caps() -> list[dict]:
    """Return a serializable list of all active cap rules. Used for diagnostics."""
    return [
        {
            "name": rule.name,
            "pass1_pipeline": rule.pass1_pipeline,
            "pass2_pipeline": rule.pass2_pipeline,
            "pass2_sensitivity_match": rule.pass2_sensitivity_match,
            "pass2_sensitivity_replacement": rule.pass2_sensitivity_replacement,
            "rationale": rule.rationale,
            "memo_section": rule.memo_section,
        }
        for rule in CAP_RULES
    ]
