"""CpGPT at analysis time: Burotu's heads on the sample embedding sent back by the GPU server."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from burotu.models.base import Detection, top_origins

ASSETS = Path(__file__).resolve().parent / "assets"


class CpGPTHeads:
    """Burotu's heads on a CpGPT sample embedding (the embedding itself comes from the GPU server)."""

    def __init__(self, directory: Path = ASSETS) -> None:
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
    return CpGPTHeads() if (ASSETS / "meta.json").exists() else None
