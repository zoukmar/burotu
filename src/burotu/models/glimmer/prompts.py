"""Text around the methylation tokens, and the answers Muse is trained to give.

The sample itself never appears as text: it enters the model as soft tokens placed
between PREFIX and QUESTION.
"""

from __future__ import annotations

from burotu.config import HEALTHY

PREFIX = "Cell-free DNA methylation profile of a blood sample:"
QUESTION = (
    "\nQuestion: Is there a cancer signal in this blood sample, and which organ does it most likely come from?"
    "\nAnswer: Cancer signal:"
)
YES, NO = " yes", " no"
ORIGIN_LEAD = " yes. Most likely origin:"


def target_text(label: str) -> str:
    """What Muse should write after QUESTION for a sample with this label."""
    if label == HEALTHY:
        return f"{NO}."
    return f"{ORIGIN_LEAD} {label}."


def origin_option(label: str) -> str:
    """Continuation scored after QUESTION + ORIGIN_LEAD when ranking tissues of origin."""
    return f" {label}."
