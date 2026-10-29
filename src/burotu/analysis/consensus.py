"""The fifth opinion: what the independent models say together.

Burotu's reported signal stays the classic model's (its threshold is the one calibrated to
99.5% specificity on new people). The consensus is a cross-check shown next to it: how many
models see a signal, and which tissue most of them name. Disagreement is a reason for the
lab to look again, not a different result.
"""

from __future__ import annotations

from collections import Counter

ORDER = ("classic", "logreg", "meta", "cpgpt")


def consensus(models: dict[str, dict]) -> dict | None:
    present = [k for k in ORDER if models.get(k)]
    if len(present) < 2:
        return None
    votes = [k for k in present if models[k]["cancer_signal"]]
    tissues = Counter(models[k]["origin"][0]["class"] for k in present)
    tissue, n_tissue = tissues.most_common(1)[0]
    return {
        "models": present,
        "signal_votes": len(votes),
        "voted_signal": votes,
        "n": len(present),
        "unanimous": len(votes) in (0, len(present)),
        "tissue": tissue,
        "tissue_votes": n_tissue,
        "tissue_by_model": {k: models[k]["origin"][0]["class"] for k in present},
    }
