"""What every Burotu model shares: the answer format, and the panel the answer is read on.

Each model package (classic, logreg, glimmer, cpgpt, consensus) has an `engine.py` whose
`analyse(sample)` returns a `Detection`, so the analysis pipeline (burotu.analysis.service)
never depends on how a model works or where it runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from burotu.core.labformat import LabSample

MODELS_DIR = Path(__file__).resolve().parent


def asset_dir(model_id: str, version: str | None = None) -> Path:
    """Where a model's weights live in the store: <store>/models/<id>/<version>.

    The version defaults to the one the registry marks as in use (registry.toml), so serving
    a new version is a one-line registry change once its weights are in the store.
    """
    from burotu.config import MODELS_STORE
    from burotu.models.registry import get

    return MODELS_STORE / model_id / (version or get(model_id).version)


def panel() -> list[str]:
    """The 3,148 CpG sites every model reads, in order (the classic model's weights define it)."""
    return json.loads((asset_dir("classic") / "meta.json").read_text())["panel"]


@dataclass
class Detection:
    engine: str  # stable id, e.g. "muse-glimmer" or "baseline"
    engine_label: str  # shown to people
    cancer_signal: bool
    score: float
    threshold: float
    origin: list[dict]  # [{"class": ..., "probability": ...}] best first, top 3

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def counts_on_panel(sample: LabSample, panel: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read counts in panel order; CpGs missing from the file count as unread."""
    c = sample.counts.reindex(panel).fillna(0)
    return c["meth"].to_numpy(np.int64)[None, :], c["total"].to_numpy(np.int64)[None, :]


def top_origins(proba: np.ndarray, classes: list[str], k: int = 3) -> list[dict]:
    order = np.argsort(proba)[::-1][:k]
    return [{"class": classes[i], "probability": round(float(proba[i]), 4)} for i in order]


class Engine(Protocol):
    """A model ready to answer for one sample."""

    def analyse(self, sample: LabSample) -> Detection: ...
