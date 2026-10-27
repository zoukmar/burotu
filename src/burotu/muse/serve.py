"""Serve a fine-tuned Muse Glimmer run as a small authenticated web API.

    POST /predict   {"sample_id": "...", "meth": [...], "total": [...]}   (panel order)
    GET  /health

The API answers in the same shape as the tracking site's engines (burotu.analysis), so
admin.burotu.com only needs BUROTU_MUSE_URL and BUROTU_MUSE_TOKEN to switch to Muse.
Deployed on Modal GPUs by scripts/modal_muse.py; runs on CPU with the tiny test model.
"""

import hmac
import json
import threading
from pathlib import Path

import numpy as np
import torch

from burotu.model.features import Featurizer
from burotu.muse.reader import MuseReader
from burotu.muse.train import TrainConfig, load_lm

ENGINE_LABEL = "Muse Glimmer 30B, fine-tuned (Meta open weights)"


class MuseService:
    """A trained run directory (trainable.pt, serving.json, mu.npy) loaded for inference."""

    def __init__(self, run_dir: str | Path) -> None:
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "serving.json").read_text())
        self.model: str = meta["model"]
        self.panel: list[str] = meta["panel"]
        self.classes: list[str] = meta["classes"]
        self.threshold: float = meta["threshold"]
        self.featurizer = Featurizer(mu=np.load(run_dir / "mu.npy"))
        cfg = TrainConfig(
            model=meta["model"],
            revision=meta["revision"],
            load_4bit=meta["load_4bit"],
            lora_r=meta["lora_r"],
            n_tokens=meta["n_tokens"],
        )
        lm, tokenizer = load_lm(cfg)
        self.reader = MuseReader(lm, tokenizer, len(self.panel) + 3, self.classes, n_tokens=cfg.n_tokens)
        self.reader.load_trainable(run_dir / "trainable.pt")
        self.reader.eval()
        self._lock = threading.Lock()  # one GPU, one forward pass at a time

    @property
    def label(self) -> str:
        return ENGINE_LABEL if "Glimmer" in self.model else f"{self.model}, fine-tuned"

    @torch.no_grad()
    def predict(self, meth: np.ndarray, total: np.ndarray) -> dict:
        from burotu.analysis.engines import top_origins

        x = torch.from_numpy(self.featurizer.transform(meth[None, :], total[None, :]))
        with self._lock, torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            score = float(self.reader.cancer_score(x)[0])
            proba = self.reader.origin_proba(x)[0].float().numpy()
        return {
            "engine_label": self.label,
            "cancer_signal": score > self.threshold,
            "score": round(score, 6),
            "threshold": round(self.threshold, 6),
            "origin": top_origins(proba, self.classes),
        }


def make_app(service: MuseService, token: str):
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field

    if len(token) < 24:
        raise ValueError("BUROTU_MUSE_TOKEN must be at least 24 characters")
    n = len(service.panel)

    class Sample(BaseModel):
        sample_id: str = Field(max_length=100)
        meth: list[int] = Field(min_length=n, max_length=n)
        total: list[int] = Field(min_length=n, max_length=n)

    app = FastAPI(title="Burotu Muse", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model": service.model, "cpgs": n, "classes": service.classes}

    @app.post("/predict")
    def predict(sample: Sample, authorization: str = Header(default="")) -> dict:
        if not hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
            raise HTTPException(status_code=401, detail="invalid token")
        meth, total = np.asarray(sample.meth), np.asarray(sample.total)
        if (meth < 0).any() or (meth > total).any():
            raise HTTPException(status_code=422, detail="counts must satisfy 0 <= meth <= total")
        return {"sample_id": sample.sample_id, **service.predict(meth, total)}

    return app
