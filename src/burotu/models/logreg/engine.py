"""The logistic-regression ensembles at analysis time (numpy only, on CPU)."""

from __future__ import annotations

from pathlib import Path

from burotu.core.labformat import LabSample
from burotu.models.base import Detection, counts_on_panel, top_origins
from burotu.models.base import panel as base_panel

ASSETS = Path(__file__).resolve().parent / "assets"


class LogRegEngine:
    """Ensembles of logistic regressions, GRAIL-style (burotu.models.logreg.model), run locally on CPU."""

    def __init__(self, directory: Path = ASSETS, panel: list[str] | None = None) -> None:
        from burotu.models.logreg.model import LogRegEnsemble

        self.model = LogRegEnsemble.load(directory)
        self.panel = panel or base_panel()

    def analyse(self, sample: LabSample) -> Detection:
        meth, total = counts_on_panel(sample, self.panel)
        score = float(self.model.cancer_score(meth, total)[0])
        proba = self.model.origin_proba(meth, total)[0]
        return Detection(
            engine="logreg",
            engine_label="Logistic regression ensembles (GRAIL-style)",
            cancer_signal=score > self.model.threshold,
            score=round(score, 4),
            threshold=round(self.model.threshold, 4),
            origin=top_origins(proba, self.model.classes),
        )
