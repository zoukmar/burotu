"""The classic model at analysis time: LightGBM text models on CPU, plus its explanation."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from burotu.core.labformat import LabSample
from burotu.models.base import Detection, counts_on_panel, top_origins

ASSETS = Path(__file__).resolve().parent / "assets"


class BaselineEngine:
    """LightGBM benchmark exported as portable text models (burotu.models.classic.export)."""

    def __init__(self, directory: Path = ASSETS) -> None:
        import lightgbm as lgb

        from burotu.core.features import Featurizer

        meta = json.loads((directory / "meta.json").read_text())
        self.panel: list[str] = meta["panel"]
        self.classes: list[str] = meta["classes"]
        self.threshold: float = meta["threshold"]
        self.featurizer = Featurizer(mu=np.load(directory / "mu.npy"))
        self.detector = lgb.Booster(model_str=gzip.decompress((directory / "detector.txt.gz").read_bytes()).decode())
        self.origin = lgb.Booster(model_str=gzip.decompress((directory / "origin.txt.gz").read_bytes()).decode())
        explain = directory / "explain.json"
        self.explainer = None
        if explain.exists():
            from burotu.models.classic.explain import Explainer

            self.explainer = Explainer(explain)

    def explain(self, sample: LabSample) -> dict | None:
        """Plain counts behind the classic model's answer (see burotu.models.classic.explain)."""
        if self.explainer is None:
            return None
        meth, total = counts_on_panel(sample, self.panel)
        x = self.featurizer.transform(meth, total)
        score = float(self.detector.predict(x)[0])
        contrib = np.asarray(self.detector.predict(x, pred_contrib=True))[0]
        proba = np.asarray(self.origin.predict(x))[0]
        return self.explainer.explain(x[0, : len(self.panel)], total[0], contrib, score, self.classes, proba)

    def analyse(self, sample: LabSample) -> Detection:
        meth, total = counts_on_panel(sample, self.panel)
        x = self.featurizer.transform(meth, total)
        score = float(self.detector.predict(x)[0])
        proba = np.asarray(self.origin.predict(x))[0]
        return Detection(
            engine="baseline",
            engine_label="Benchmark classifier (gradient boosting)",
            cancer_signal=score > self.threshold,
            score=round(score, 6),
            threshold=round(self.threshold, 6),
            origin=top_origins(proba, self.classes),
        )
