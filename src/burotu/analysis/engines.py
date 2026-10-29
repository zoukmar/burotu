"""Analysis engines: what turns read counts into a cancer signal and a tissue of origin.

  * MuseEngine: the fine-tuned Muse Glimmer, served on Modal GPUs (see scripts/modal_muse.py).
  * BaselineEngine: the gradient-boosted benchmark classifier, run locally on CPU.

Both return the same structure, so the rest of Burotu does not care which one answered.
The tracking site uses Muse when it is configured and reachable, and says so.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from burotu.labformat import LabSample

ASSETS = Path(__file__).resolve().parent / "assets"


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


class BaselineEngine:
    """LightGBM benchmark exported as portable text models (scripts/export_baseline.py)."""

    def __init__(self, directory: Path = ASSETS / "baseline") -> None:
        import lightgbm as lgb

        from burotu.model.features import Featurizer

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
            from burotu.analysis.explain import Explainer

            self.explainer = Explainer(explain)

    def explain(self, sample: LabSample) -> dict | None:
        """Plain counts behind the classic model's answer (see burotu.analysis.explain)."""
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


class LogRegEngine:
    """Ensembles of logistic regressions, GRAIL-style (burotu.model.logreg), run locally on CPU."""

    def __init__(self, directory: Path = ASSETS / "logreg", panel: list[str] | None = None) -> None:
        from burotu.model.logreg import LogRegEnsemble

        self.model = LogRegEnsemble.load(directory)
        self.panel = panel or json.loads((ASSETS / "baseline" / "meta.json").read_text())["panel"]

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


class CpGPTHeads:
    """Burotu's heads on a CpGPT sample embedding (the embedding itself comes from the GPU server)."""

    def __init__(self, directory: Path = ASSETS / "cpgpt") -> None:
        import lightgbm as lgb

        meta = json.loads((directory / "meta.json").read_text())
        self.classes: list[str] = meta["classes"]
        self.threshold: float = meta["threshold"]
        self.dim: int = meta["dim"]
        self.detector = lgb.Booster(model_str=gzip.decompress((directory / "detector.txt.gz").read_bytes()).decode())
        self.origin = lgb.Booster(model_str=gzip.decompress((directory / "origin.txt.gz").read_bytes()).decode())

    def analyse(self, embedding: list[float]) -> Detection:
        x = np.asarray(embedding, dtype=np.float32)[None, :]
        if x.shape[1] != self.dim:
            raise ValueError(f"CpGPT embedding has {x.shape[1]} values, heads expect {self.dim}")
        score = float(self.detector.predict(x)[0])
        return Detection(
            engine="cpgpt",
            engine_label="CpGPT 100M (methylation foundation model) + Burotu heads",
            cancer_signal=score > self.threshold,
            score=round(score, 6),
            threshold=round(self.threshold, 6),
            origin=top_origins(np.asarray(self.origin.predict(x))[0], self.classes),
        )


def cpgpt_heads() -> CpGPTHeads | None:
    return CpGPTHeads() if (ASSETS / "cpgpt" / "meta.json").exists() else None


class MuseEngine:
    """Fine-tuned Muse Glimmer behind an authenticated Modal web endpoint."""

    def __init__(self, url: str, token: str, panel: list[str], timeout: float = 180.0) -> None:
        self.url, self.token, self.panel, self.timeout = url, token, panel, timeout

    def analyse(self, sample: LabSample) -> Detection:
        return self.analyse_all(sample)["meta"]

    def analyse_all(self, sample: LabSample) -> dict[str, Detection]:
        """Muse Glimmer's answer, plus CpGPT's when the same GPU server also runs it."""
        meth, total = counts_on_panel(sample, self.panel)
        r = httpx.post(
            self.url,
            json={"sample_id": sample.sample_id, "meth": meth[0].tolist(), "total": total[0].tolist()},
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        d = r.json()
        out = {
            "meta": Detection(
                engine="muse-glimmer",
                engine_label=d.get("engine_label", "Muse Glimmer 30B, fine-tuned (Meta open weights)"),
                cancer_signal=bool(d["cancer_signal"]),
                score=float(d["score"]),
                threshold=float(d["threshold"]),
                origin=d["origin"],
            )
        }
        heads = cpgpt_heads()
        if d.get("cpgpt_embedding") and heads is not None:
            out["cpgpt"] = heads.analyse(d["cpgpt_embedding"])
        return out
