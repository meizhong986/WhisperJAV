"""The removed 1.9.x "bring your own CrispASR" mode, as typed on the command line (1.10.0).

WhisperJAV 1.9.x ran a crispasr.exe the user supplied: ``--mode crispasr``, the pass value
``crispasr`` and the flags ``--crispasr-exe``, ``--crispasr-backend`` and ``--crispasr-args``.
1.10.0 removed that mode completely (owner, W118 3). CrispASR now runs as the Ensemble pipelines
parakeet-crispasr and cohere-crispasr, through CrispASR's Python package, which WhisperJAV installs.

The old usage stops the run with exit status 2 and a message that names what the user typed and
points to Ensemble Mode. ``whisperjav/main.py`` looks for it in the raw command line when it is
imported, beside its --offline scan:
  - argparse no longer knows these flags and values, so its own errors (an unknown argument, an
    invalid choice) could neither name the old mode nor say where CrispASR went;
  - the run must stop before main.py's start-up GPU check, which may ask a question first.

Standard library only: main.py imports this before anything heavy.
"""

from typing import Optional, Sequence

# The old pass and mode value, and the prefix every old flag shares.
LEGACY_VALUE = "crispasr"
LEGACY_FLAG_PREFIX = "--crispasr"
# The flags whose value "crispasr" chose the old mode (--mode) or the old pass pipeline.
VALUE_FLAGS = ("--mode", "--pass1-pipeline", "--pass2-pipeline")

# Route B design 12, T14 (W193 5, W203 I2). {flag} is what the user typed: a flag, or a flag and its value.
LEGACY_CRISPASR_MESSAGE = (
    "{flag} is no longer available: WhisperJAV 1.10 removed the \"bring your own CrispASR\" mode.\n"
    "CrispASR now runs in Ensemble Mode as two pipelines, Parakeet-CrispASR and Cohere-CrispASR, and "
    "WhisperJAV installs it itself.\n"
    "Example:\n"
    "    whisperjav --ensemble --pass1-pipeline parakeet-crispasr <your video>\n"
    "In the GUI: open the Ensemble Mode tab and choose parakeet or cohere under GGUF via crispasr in "
    "the Pass 1 or Pass 2 menu."   # the menu names since W343
)


def legacy_crispasr_use(argv: Sequence[str]) -> Optional[str]:
    """The old CrispASR usage in ``argv`` (the arguments after the program name), or None.

    Returns the first one found, as the user typed it:
      - a flag starting with ``--crispasr``, in any letter case: ``--crispasr-exe``,
        ``--crispasr-backend``, ``--crispasr-args``, an abbreviation such as ``--crispasr-e``,
        or the ``--crispasr-exe=<value>`` form. Only the flag is returned, never the value: the
        flag is what was removed, and the value may be a long path;
      - ``--mode``, ``--pass1-pipeline`` or ``--pass2-pipeline`` with the value ``crispasr`` in
        any letter case, as two tokens (returned joined by one space) or as ``--mode=crispasr``.

    Read as argparse reads a command line: nothing after a bare ``--`` is an option (a file may be
    named ``--crispasr.mp4``), and a token with a space before any ``=`` is a value, not a flag.
    The new pass values parakeet-crispasr and cohere-crispasr, and a file named crispasr.mp4, are
    not old usage. An abbreviated --mode or --passN-pipeline is not caught here; argparse then
    refuses "crispasr" itself with its own "invalid choice" message, which lists the new values.
    """
    tokens = list(argv)
    for i, token in enumerate(tokens):
        if token == "--":
            break
        flag, has_value, value = token.partition("=")
        if " " in flag:
            continue
        if flag.lower().startswith(LEGACY_FLAG_PREFIX):
            return flag
        if flag not in VALUE_FLAGS:
            continue
        if has_value:
            if value.strip().lower() == LEGACY_VALUE:
                return token
        elif i + 1 < len(tokens) and tokens[i + 1].strip().lower() == LEGACY_VALUE:
            return f"{token} {tokens[i + 1]}"
    return None


def legacy_crispasr_message(used: str) -> str:
    """The stop message for one old usage, as returned by legacy_crispasr_use."""
    return LEGACY_CRISPASR_MESSAGE.format(flag=used)
